"""R6: evaluate the best R5 checkpoint and write docs/realmodel-report.md.

Loads the fine-tuned SPECTRE-GPT-2, measures test perplexity, runs gate
adaptivity diagnostics (10 diverse inputs, init vs trained), and emits the
report with the final PASS/FAIL line (ratio <= 1.10 x baseline).
"""
import math
import os

import torch
from transformers import GPT2LMHeadModel

from .data import blocks, loaders
from .evaluate import perplexity
from .surgery import swap_gpt2_attention

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASELINE_PPL = 30.3261  # R4, token-level WikiText-2 test
PASS_RATIO = 1.10
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
CKPT_FROZEN = os.path.join(_CKPT_DIR, "best.pt")
CKPT_COTRAIN = os.path.join(_CKPT_DIR, "best-cotraining.pt")
REPORT_PATH = os.path.join(REPO_ROOT, "docs", "realmodel-report.md")


def _load_at_init() -> GPT2LMHeadModel:
    """Post-surgery, pre-training model (all-pass gates, warm wq/wv/wo)."""
    torch.manual_seed(42)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def _layer_inputs(model, inputs: torch.Tensor) -> list[torch.Tensor]:
    """Run one forward pass and capture each SPECTRE layer's input hidden states.

    The gate consumes the layer input (B, n, d_model), not the raw token ids,
    so we hook every block's attention module and record what it received.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, args, _kwargs, _output):
        captured.append(args[0].detach())
        return None

    handles = [block.attn.register_forward_hook(hook, with_kwargs=True) for block in model.transformer.h]
    try:
        model(inputs.to(DEVICE))
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def _gate_magnitude_stats(model, inputs: torch.Tensor) -> dict:
    """Per-gate |g| mean and bin-std, averaged over layers/heads/inputs."""
    all_mean, all_std = [], []
    for block, x in zip(model.transformer.h, _layer_inputs(model, inputs)):
        for head in block.attn.spectre.heads:
            q = x @ head.wq + head.bq  # (B, n, d)
            q_mean = q.mean(dim=1)  # (B, d) global descriptor
            g = head.gate(q_mean)  # (B, F) complex
            mag = g.abs()
            all_mean.append(mag.mean().item())
            all_std.append(mag.std(dim=1).mean().item())
    return {"mean_abs_g": sum(all_mean) / len(all_mean), "bin_std": sum(all_std) / len(all_std)}


@torch.no_grad()
def _gate_pairwise_distance(model, inputs: torch.Tensor) -> float:
    """Mean pairwise cosine distance between gate vectors of different inputs."""
    dists = []
    for block, x in zip(model.transformer.h, _layer_inputs(model, inputs)):
        for head in block.attn.spectre.heads:
            q = x @ head.wq + head.bq
            q_mean = q.mean(dim=1)
            g = head.gate(q_mean)  # (B, F) complex
            gv = torch.view_as_real(g).reshape(g.shape[0], -1)  # (B, 2F)
            gv = gv / gv.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cos = gv @ gv.t()  # (B, B)
            n = cos.shape[0]
            off = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
            dists.append((1.0 - off).mean().item())
    return sum(dists) / len(dists)


def _eval_checkpoint(path: str, diag_inputs: torch.Tensor) -> tuple[float, dict, dict]:
    """Load a checkpoint, measure test PPL and gate diagnostics."""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    ppl = perplexity(model, loaders(8, "test"), DEVICE)
    stats = _gate_magnitude_stats(model, diag_inputs)
    stats["pairwise_cos_dist"] = _gate_pairwise_distance(model, diag_inputs)
    del model
    return ppl, ckpt, stats


def main() -> None:
    test_blocks = blocks("test")
    diag_inputs = test_blocks[:10]  # 10 diverse 1024-token inputs, one per batch element

    frozen_ppl, frozen_ckpt, frozen_stats = _eval_checkpoint(CKPT_FROZEN, diag_inputs)
    frozen_ratio = frozen_ppl / BASELINE_PPL

    cotrain_ppl, cotrain_ckpt, cotrain_stats = None, {}, {}
    if os.path.exists(CKPT_COTRAIN):
        cotrain_ppl, cotrain_ckpt, cotrain_stats = _eval_checkpoint(CKPT_COTRAIN, diag_inputs)
    cotrain_ratio = cotrain_ppl / BASELINE_PPL if cotrain_ppl else None

    init_model = _load_at_init()
    init_stats = _gate_magnitude_stats(init_model, diag_inputs)
    init_stats["pairwise_cos_dist"] = _gate_pairwise_distance(init_model, diag_inputs)
    del init_model

    verdict = "PASS" if frozen_ratio <= PASS_RATIO else f"FAIL (ratio {frozen_ratio:.3f})"

    ppl_rows = [
        "| model | PPL | ratio vs baseline |",
        "| --- | --- | --- |",
        "| GPT-2 small baseline (R4) | {:.4f} | 1.000 |".format(BASELINE_PPL),
        "| SPECTRE, frozen backbone (R5) | {:.4f} | {:.3f} |".format(frozen_ppl, frozen_ratio),
    ]
    if cotrain_ppl:
        ppl_rows.append("| SPECTRE, co-trained backbone (R5b) | {:.4f} | {:.3f} |".format(cotrain_ppl, cotrain_ratio))

    gate_rows = [
        "| metric | at init (all-pass) | frozen (R5) | co-trained (R5b) |",
        "| --- | --- | --- | --- |",
        "| mean abs(g) | {:.4f} | {:.4f} | {} |".format(
            init_stats["mean_abs_g"], frozen_stats["mean_abs_g"],
            "{:.4f}".format(cotrain_stats["mean_abs_g"]) if cotrain_stats else "—",
        ),
        "| bin std of abs(g) | {:.4f} | {:.4f} | {} |".format(
            init_stats["bin_std"], frozen_stats["bin_std"],
            "{:.4f}".format(cotrain_stats["bin_std"]) if cotrain_stats else "—",
        ),
        "| pairwise cosine distance | {:.4f} | {:.4f} | {} |".format(
            init_stats["pairwise_cos_dist"], frozen_stats["pairwise_cos_dist"],
            "{:.4f}".format(cotrain_stats["pairwise_cos_dist"]) if cotrain_stats else "—",
        ),
    ]

    cotrain_summary = ""
    if cotrain_ckpt:
        cotrain_summary = (
            "\n## Co-training summary (R5b)\n"
            "\n"
            "| item | value |\n"
            "| --- | --- |\n"
            "| best val loss | {:.4f} |\n"
            "| best epoch | {} |\n"
            "| schedule | SPECTRE 3e-4 + backbone 1e-5 (30x lower), cosine, patience 3 |\n"
            "| trainable | 127,647,408 (all params) |\n"
            "| wall | 84.6 min, 10/10 epochs, no early stop |\n".format(
                cotrain_ckpt.get("val_loss", float("nan")), cotrain_ckpt.get("epoch", "?")
            )
        )

    lines = [
        "# SPECTRE real-model validation report",
        "",
        "## Setup",
        "",
        "| item | value |",
        "| --- | --- |",
        "| base model | GPT-2 small (124M, 12 layers, 12 heads, d=768) |",
        "| surgery | all 12 attention blocks -> SPECTRE layers (warm init, D6) |",
        "| trainable (R5) | 31,556,016 (SPECTRE only; backbone frozen) |",
        "| trainable (R5b) | 127,647,408 (all params; backbone at 30x lower LR) |",
        "| data | WikiText-2 raw, 1024-token non-overlapping blocks |",
        "| hardware | Apple Silicon, MPS, float32 |",
        "",
        "## Cross-validation (R2, vs Rust PoC)",
        "",
        "| check | rel diff | verdict |",
        "| --- | --- | --- |",
        "| full layer | 2.630e-07 | PASS |",
        "| rfft | 7.199e-08 | PASS |",
        "",
        "## Perplexity (WikiText-2 test, token-level)",
        "",
        *ppl_rows,
        "",
        "## Training summary (R5, frozen backbone)",
        "",
        "| item | value |",
        "| --- | --- |",
        "| best val loss | {:.4f} |".format(frozen_ckpt.get("val_loss", float("nan"))),
        "| best step | {} |".format(frozen_ckpt.get("step", "?")),
        "| best epoch | {} |".format(frozen_ckpt.get("epoch", "?")),
        "| schedule | AdamW 3e-4, warmup 100, cosine to 2e-5, wd 0.05, clip 1.0 |",
        "| effective batch | 32 (micro 8 x grad-accum 4) |",
        cotrain_summary,
        "## Gate adaptivity diagnostics (10 diverse inputs)",
        "",
        *gate_rows,
        "",
        "Soft expectation: after training, higher bin-std (more spectrally",
        "selective) and higher pairwise distance (more input-discriminative).",
        "",
        "**Overall: {}**".format(verdict),
        "",
    ]
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"SPECTRE (frozen backbone) ppl: {frozen_ppl:.4f} (baseline {BASELINE_PPL:.4f}, ratio {frozen_ratio:.3f})")
    if cotrain_ppl:
        print(f"SPECTRE (co-trained) ppl: {cotrain_ppl:.4f} (ratio {cotrain_ratio:.3f})")
    print(f"report written to {REPORT_PATH}")
    print(f"**Overall: {verdict}**")


if __name__ == "__main__":
    main()