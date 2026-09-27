"""R5: fine-tune the SPECTRE parameters (only) on WikiText-2.

Schedule per plan D8/D9: AdamW lr 3e-4, 100-step linear warmup, cosine decay
to 2e-5, weight decay 0.05 (0 for biases/LayerNorm), grad clip 1.0, effective
batch 32, <=10 epochs, early stop after 2 consecutive validation-loss
increases, 30-minute hard stop. Best checkpoint (by val loss) saved to
realmodel/checkpoints/best.pt.

Micro-batch is 8 with gradient accumulation x4: a full 32-sequence batch
through the SPECTRE FFT path spikes MPS memory well past what a 48GB machine
comfortably holds, while 8x4 keeps the optimizer math identical to batch 32.

--cotraining: unfreeze the backbone at a 30x lower peak LR (1e-5) so the
pretrained MLPs/LayerNorms can adapt around the transplanted SPECTRE layers.
The R6 diagnosis attributed the 1.62x PPL gap to the backbone never having
been co-trained with SPECTRE; this mode tests that hypothesis directly.
Checkpoint goes to checkpoints/best-cotraining.pt so the frozen-backbone
best.pt is never clobbered.
"""
import math
import os
import sys
import time

import torch
from transformers import GPT2LMHeadModel

from .data import blocks
from .evaluate import perplexity
from .memlog import MemLogger
from .surgery import freeze_backbone, swap_gpt2_attention

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EFFECTIVE_BATCH = 32
MICRO_BATCH = 8
ACCUM_STEPS = EFFECTIVE_BATCH // MICRO_BATCH
MAX_EPOCHS = 10
WARMUP = 100
LR_PEAK = 3e-4
LR_MIN = 2e-5
BACKBONE_LR_PEAK = 1e-5
BACKBONE_LR_MIN = 1e-6
WD = 0.05
CLIP = 1.0
TIME_BUDGET_S = 30 * 60
PATIENCE = 2
LOG_EVERY = 25
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def _lr_at(step: int, total_steps: int, peak: float = LR_PEAK, floor: float = LR_MIN) -> float:
    if step < WARMUP:
        return peak * (step + 1) / WARMUP
    t = (step - WARMUP) / max(total_steps - WARMUP, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def _param_groups(model, backbone_lr_scale: float = 0.0) -> list:
    """SPECTRE params at full LR; backbone params at backbone_lr_scale x of it.

    backbone_lr_scale=0 reproduces the frozen-backbone plan behavior (the
    backbone params are simply excluded, so AdamW never sees them).
    """
    spectre_decay, spectre_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = ".attn.spectre." not in name
        decay = p.ndim > 1 and "ln." not in name
        if is_backbone:
            (backbone_decay if decay else backbone_no_decay).append(p)
        else:
            (spectre_decay if decay else spectre_no_decay).append(p)
    groups = [
        {"params": spectre_decay, "weight_decay": WD},
        {"params": spectre_no_decay, "weight_decay": 0.0},
    ]
    if backbone_lr_scale > 0 and (backbone_decay or backbone_no_decay):
        groups.append({"params": backbone_decay, "weight_decay": WD, "lr_scale": backbone_lr_scale})
        groups.append({"params": backbone_no_decay, "weight_decay": 0.0, "lr_scale": backbone_lr_scale})
    return groups


def _val_loss(model, val_blocks: torch.Tensor) -> float:
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, val_blocks.shape[0], MICRO_BATCH):
            batch = val_blocks[i : i + MICRO_BATCH].to(DEVICE)
            logits = model(batch).logits
            nll = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total += nll.item()
            tokens += batch[:, 1:].numel()
    model.train()
    return total / tokens


def _heartbeat(start: float, step: int, total_steps: int, epoch: int, batch_loss: float, mem: MemLogger) -> None:
    now = time.time()
    if now - HEARTBEAT_LAST[0] < 5.0:
        return
    HEARTBEAT_LAST[0] = now
    elapsed = now - start
    pct = 100.0 * step / total_steps
    eta_s = elapsed / max(step, 1) * (total_steps - step)
    eta_m, eta_s = divmod(int(eta_s), 60)
    print(
        f"[heartbeat] {pct:5.1f}% step {step}/{total_steps} epoch {epoch} "
        f"loss {batch_loss:.4f} elapsed {int(elapsed)}s ETA {eta_m}m{eta_s:02d}s",
        flush=True,
    )
    mem.log(f"step {step}")


HEARTBEAT_LAST = [0.0]


def main() -> None:
    resume = "--resume" in sys.argv
    cotraining = "--cotraining" in sys.argv
    torch.manual_seed(42)
    mem = MemLogger("train")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    freeze_backbone(model)
    if cotraining:
        for p in model.parameters():
            p.requires_grad_(True)
    model.to(DEVICE)
    mem.log("after model load", force=True)

    train_blocks = blocks("train")
    val_blocks = blocks("validation")
    steps_per_epoch = train_blocks.shape[0] // EFFECTIVE_BATCH
    total_steps = steps_per_epoch * MAX_EPOCHS
    print(f"train blocks: {train_blocks.shape[0]}, val blocks: {val_blocks.shape[0]}, steps/epoch: {steps_per_epoch} (micro {MICRO_BATCH} x {ACCUM_STEPS})")
    if cotraining:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"cotraining mode: backbone unfrozen at {BACKBONE_LR_PEAK:.1e} peak ({n_trainable:,} trainable params)")
    mem.log("after data load", force=True)

    backbone_scale = BACKBONE_LR_PEAK / LR_PEAK if cotraining else 0.0
    opt = torch.optim.AdamW(_param_groups(model, backbone_scale), lr=LR_PEAK)
    os.makedirs(CKPT_DIR, exist_ok=True)
    best_path = os.path.join(CKPT_DIR, "best-cotraining.pt" if cotraining else "best.pt")
    time_budget = 3 * 60 * 60 if cotraining else TIME_BUDGET_S
    patience = 3 if cotraining else PATIENCE

    best_val = float("inf")
    bad_epochs = 0
    step = 0
    start_epoch = 0
    if resume:
        if not os.path.exists(best_path):
            sys.exit(f"--resume requested but {best_path} not found")
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            print("checkpoint has no optimizer state (pre-resume-support format); warm-starting AdamW")
        best_val = ckpt["val_loss"]
        step = ckpt["step"]
        start_epoch = ckpt["epoch"] + 1
        bad_epochs = ckpt.get("bad_epochs", 0)
        print(f"resumed from epoch {ckpt['epoch']} (step {step}, best val {best_val:.4f})")
    tokens_seen = 0
    start = time.time()
    stopped = "completed"

    for epoch in range(start_epoch, MAX_EPOCHS):
        perm = torch.randperm(train_blocks.shape[0], generator=torch.Generator().manual_seed(42 + epoch))
        for i in range(steps_per_epoch):
            if time.time() - start > time_budget:
                stopped = f"time budget hit at step {step}"
                break
            lr = _lr_at(step, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for micro in range(ACCUM_STEPS):
                idx = perm[i * EFFECTIVE_BATCH + micro * MICRO_BATCH : i * EFFECTIVE_BATCH + (micro + 1) * MICRO_BATCH]
                batch = train_blocks[idx].to(DEVICE)
                logits = model(batch).logits
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                    batch[:, 1:].reshape(-1),
                ) / ACCUM_STEPS
                loss.backward()
                batch_loss += loss.item()
                tokens_seen += batch.numel()
                _heartbeat(start, step, total_steps, epoch, batch_loss, mem)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
            opt.step()
            step += 1
            if step % LOG_EVERY == 0:
                elapsed = time.time() - start
                print(f"step {step} epoch {epoch} loss {batch_loss:.4f} lr {lr:.2e} tok/s {tokens_seen/elapsed:.0f}")
                mem.log(f"step {step}", force=True)
        else:
            val = _val_loss(model, val_blocks)
            print(f"epoch {epoch} val loss {val:.4f} (ppl {math.exp(val):.2f})")
            mem.log(f"epoch {epoch} val", force=True)
            if val < best_val:
                best_val = val
                bad_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "val_loss": val,
                        "step": step,
                        "epoch": epoch,
                        "bad_epochs": bad_epochs,
                    },
                    best_path,
                )
                print(f"saved best (val {val:.4f})")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    stopped = f"early stop after epoch {epoch}"
                    break
            continue
        break  # time budget hit inside the inner loop

    wall = time.time() - start
    print(f"TRAIN done: {stopped}, steps {step}, wall {wall/60:.1f} min, best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()