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
CKPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "best.pt")
REPORT_PATH = os.path.join(REPO_ROOT, "docs", "realmodel-report.md")


def _load_trained() -> tuple[GPT2LMHeadModel, dict]:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention(model)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, ckpt


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


def main() -> None:
    test_blocks = blocks("test")
    diag_inputs = test_blocks[:10]  # 10 diverse 1024-token inputs, one per batch element

    trained, ckpt = _load_trained()
    spectre_ppl = perplexity(trained, loaders(8, "test"), DEVICE)
    ratio = spectre_ppl / BASELINE_PPL

    init_model = _load_at_init()
    init_stats = _gate_magnitude_stats(init_model, diag_inputs)
    init_stats["pairwise_cos_dist"] = _gate_pairwise_distance(init_model, diag_inputs)
    del init_model

    trained_stats = _gate_magnitude_stats(trained, diag_inputs)
    trained_stats["pairwise_cos_dist"] = _gate_pairwise_distance(trained, diag_inputs)

    verdict = "PASS" if ratio <= PASS_RATIO else f"FAIL (ratio {ratio:.3f})"

    lines = [
        "# SPECTRE real-model validation report",
        "",
        "## Setup",
        "",
        "| item | value |",
        "| --- | --- |",
        "| base model | GPT-2 small (124M, 12 layers, 12 heads, d=768) |",
        "| surgery | all 12 attention blocks -> SPECTRE layers (warm init, D6) |",
        "| trainable | 31,556,016 (SPECTRE only; backbone frozen) |",
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
        "| model | PPL |",
        "| --- | --- |",
        "| GPT-2 small baseline (R4) | {:.4f} |".format(BASELINE_PPL),
        "| SPECTRE fine-tuned (R5) | {:.4f} |".format(spectre_ppl),
        "| ratio | {:.3f} |".format(ratio),
        "",
        "## Training summary (R5)",
        "",
        "| item | value |",
        "| --- | --- |",
        "| best val loss | {:.4f} |".format(ckpt.get("val_loss", float("nan"))),
        "| best step | {} |".format(ckpt.get("step", "?")),
        "| best epoch | {} |".format(ckpt.get("epoch", "?")),
        "| schedule | AdamW 3e-4, warmup 100, cosine to 2e-5, wd 0.05, clip 1.0 |",
        "| effective batch | 32 (micro 8 x grad-accum 4) |",
        "",
        "## Gate adaptivity diagnostics (10 diverse inputs)",
        "",
        "| metric | at init (all-pass) | after training |",
        "| --- | --- | --- |",
        "| mean abs(g) | {:.4f} | {:.4f} |".format(init_stats["mean_abs_g"], trained_stats["mean_abs_g"]),
        "| bin std of abs(g) | {:.4f} | {:.4f} |".format(init_stats["bin_std"], trained_stats["bin_std"]),
        "| pairwise cosine distance | {:.4f} | {:.4f} |".format(init_stats["pairwise_cos_dist"], trained_stats["pairwise_cos_dist"]),
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
    print(f"SPECTRE ppl: {spectre_ppl:.4f} (baseline {BASELINE_PPL:.4f}, ratio {ratio:.3f})")
    print(f"report written to {REPORT_PATH}")
    print(f"**Overall: {verdict}**")


if __name__ == "__main__":
    main()