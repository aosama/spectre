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
from .r10_causal import causal_query_pools
from .paper_spectre import swap_gpt2_attention_official
from .causal_hybrid import swap_gpt2_attention_official_causal
from .surgery import swap_gpt2_attention

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASELINE_PPL = 30.3261  # R4, token-level WikiText-2 test
PASS_RATIO = 1.10
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
CKPT_FROZEN = os.path.join(_CKPT_DIR, "best.pt")
CKPT_COTRAIN = os.path.join(_CKPT_DIR, "best-cotraining.pt")
CKPT_OFFICIAL = os.path.join(_CKPT_DIR, "best-official.pt")
CKPT_OFFICIAL_CAUSAL = os.path.join(_CKPT_DIR, "best-official-causal.pt")
CKPT_R10 = os.path.join(_CKPT_DIR, "best-r10.pt")
CKPT_OFFICIAL_CAUSAL_R10 = os.path.join(_CKPT_DIR, "best-official-causal-r10.pt")
CKPT_R10_CAUSAL_CHUNKS = 4
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
def _gate_magnitude_stats(model, inputs: torch.Tensor, causal_chunks: int = 1) -> dict:
    """Per-gate |g| mean and bin-std, averaged over layers/heads/inputs.

    For the R10 arches the gate is computed per chunk (one descriptor per
    chunk, from strictly earlier queries), so we gather the gate over all
    chunks and average the same statistics.
    """
    all_mean, all_std = [], []
    for block, x in zip(model.transformer.h, _layer_inputs(model, inputs)):
        for head in block.attn.spectre.heads:
            q = x @ head.wq + head.bq  # (B, n, d)
            if causal_chunks == 1:
                g = head.v1_gate(q.mean(dim=1))  # (B, F) legacy descriptor
                mag = g.abs()
                all_mean.append(mag.mean().item())
                all_std.append(mag.std(dim=1).mean().item())
            else:
                pools = causal_query_pools(q, causal_chunks)  # (B, C, d)
                g = torch.stack([head.v1_gate(pools[:, i]) for i in range(causal_chunks)], dim=1)  # (B, C, F)
                mag = g.abs()
                all_mean.append(mag.mean().item())
                all_std.append(mag.std(dim=2).mean().item())
    return {"mean_abs_g": sum(all_mean) / len(all_mean), "bin_std": sum(all_std) / len(all_std)}


@torch.no_grad()
def _gate_pairwise_distance(model, inputs: torch.Tensor, causal_chunks: int = 1) -> float:
    """Mean pairwise cosine distance between gate vectors of different inputs.

    For R10 the gate is one vector per (input, chunk); we flatten that to
    (B*C, 2F) and measure how separable the gate vectors are across inputs.
    """
    dists = []
    for block, x in zip(model.transformer.h, _layer_inputs(model, inputs)):
        for head in block.attn.spectre.heads:
            q = x @ head.wq + head.bq
            if causal_chunks == 1:
                g = head.v1_gate(q.mean(dim=1))  # (B, F)
            else:
                pools = causal_query_pools(q, causal_chunks)  # (B, C, d)
                g = torch.stack([head.v1_gate(pools[:, i]) for i in range(causal_chunks)], dim=1)  # (B, C, F)
            gv = torch.view_as_real(g)
            gv = gv.reshape(g.shape[0], -1) if causal_chunks == 1 else gv.reshape(-1, gv.shape[-1])
            gv = gv / gv.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cos = gv @ gv.t()  # (B*C, B*C)
            n = cos.shape[0]
            off = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
            dists.append((1.0 - off).mean().item())
    return sum(dists) / len(dists)


def _eval_checkpoint(path: str, diag_inputs: torch.Tensor) -> tuple[float, dict, dict]:
    """Load a checkpoint, measure test PPL and (v1 arch only) gate diagnostics."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "v1")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if arch == "official":
        swap_gpt2_attention_official(model)
    elif arch == "official-causal":
        swap_gpt2_attention_official_causal(model)
    elif arch == "official-causal-r10":
        swap_gpt2_attention_official_causal(model, causal_chunks=CKPT_R10_CAUSAL_CHUNKS)
    elif arch == "v1-r10":
        swap_gpt2_attention(model, causal_chunks=CKPT_R10_CAUSAL_CHUNKS)
    else:
        swap_gpt2_attention(model)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    ppl = perplexity(model, loaders(8, "test"), DEVICE)
    if arch in ("official", "official-causal", "official-causal-r10"):
        stats: dict = {}  # v1 gate diagnostics don't apply to the grouped official gate
    else:
        diag_chunks = CKPT_R10_CAUSAL_CHUNKS if arch == "v1-r10" else 1
        stats = _gate_magnitude_stats(model, diag_inputs, diag_chunks)
        stats["pairwise_cos_dist"] = _gate_pairwise_distance(model, diag_inputs, diag_chunks)
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

    official_ppl, official_ckpt, official_stats = None, {}, {}
    if os.path.exists(CKPT_OFFICIAL):
        official_ppl, official_ckpt, official_stats = _eval_checkpoint(CKPT_OFFICIAL, diag_inputs)
    official_ratio = official_ppl / BASELINE_PPL if official_ppl else None

    official_causal_ppl, official_causal_ckpt, official_causal_stats = None, {}, {}
    if os.path.exists(CKPT_OFFICIAL_CAUSAL):
        official_causal_ppl, official_causal_ckpt, official_causal_stats = _eval_checkpoint(CKPT_OFFICIAL_CAUSAL, diag_inputs)
    official_causal_ratio = official_causal_ppl / BASELINE_PPL if official_causal_ppl else None

    r10_ppl, r10_ckpt, r10_stats = None, {}, {}
    if os.path.exists(CKPT_R10):
        r10_ppl, r10_ckpt, r10_stats = _eval_checkpoint(CKPT_R10, diag_inputs)
    r10_ratio = r10_ppl / BASELINE_PPL if r10_ppl else None

    official_causal_r10_ppl, official_causal_r10_ckpt, official_causal_r10_stats = None, {}, {}
    if os.path.exists(CKPT_OFFICIAL_CAUSAL_R10):
        official_causal_r10_ppl, official_causal_r10_ckpt, official_causal_r10_stats = _eval_checkpoint(
            CKPT_OFFICIAL_CAUSAL_R10, diag_inputs
        )
    official_causal_r10_ratio = official_causal_r10_ppl / BASELINE_PPL if official_causal_r10_ppl else None

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
    if official_ppl:
        ppl_rows.append(
            "| SPECTRE, official author math (R8) | {:.4f} | {:.3f} |".format(official_ppl, official_ratio)
        )
    if official_causal_ppl:
        ppl_rows.append(
            "| SPECTRE, official gate + causal mixing (R9) | {:.4f} | {:.3f} |".format(official_causal_ppl, official_causal_ratio)
        )
    if r10_ppl:
        ppl_rows.append(
            "| SPECTRE, v1 gate + strictly causal gate (R10) | {:.4f} | {:.3f} |".format(r10_ppl, r10_ratio)
        )
    if official_causal_r10_ppl:
        ppl_rows.append(
            "| SPECTRE, official gate + strictly causal gate (R10) | {:.4f} | {:.3f} |".format(
                official_causal_r10_ppl, official_causal_r10_ratio
            )
        )

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

    official_summary = ""
    if official_ckpt:
        official_summary = (
            "\n## Official-math summary (R8, vendored author implementation)\n"
            "\n"
            "| item | value |\n"
            "| --- | --- |\n"
            "| best val loss | {:.4f} |\n"
            "| best epoch | {} |\n"
            "| arch | vendored SpectreMultiHead: grouped gate G=4, DCT pooling, "
            "cubic-interp anchors, smooth modReLU, circular mixing n_fft=1024, "
            "wavelet off; block-diagonal warm start |\n".format(
                official_ckpt.get("val_loss", float("nan")), official_ckpt.get("epoch", "?")
            )
        )

    official_causal_summary = ""
    if official_causal_ckpt:
        official_causal_summary = (
            "\n## Causal-official summary (R9, honest hybrid)\n"
            "\n"
            "| item | value |\n"
            "| --- | --- |\n"
            "| best val loss | {:.4f} |\n"
            "| best epoch | {} |\n"
            "| arch | vendored author gate (grouped G=4 anchors, cubic interp, DCT pooling, "
            "smooth modReLU) + strictly causal zero-padded linear convolution (FFT length 2N); "
            "near-identity gate warm start; wavelet off |\n".format(
                official_causal_ckpt.get("val_loss", float("nan")), official_causal_ckpt.get("epoch", "?")
            )
        )

    r10_summary = ""
    if r10_ckpt:
        r10_summary = (
            "\n## Strictly causal gate summary (R10, v1 gate)\n"
            "\n"
            "| item | value |\n"
            "| --- | --- |\n"
            "| best val loss | {:.4f} |\n"
            "| best epoch | {} |\n"
            "| arch | v1 gate with one spectral kernel per {}-token chunk, each computed from "
            "the query mean of strictly earlier positions (chunk 0 gets a zeros descriptor, so its "
            "kernel is the learned cold-start kernel); per-chunk zero-padded linear convolution |\n".format(
                r10_ckpt.get("val_loss", float("nan")), r10_ckpt.get("epoch", "?"), 1024 // CKPT_R10_CAUSAL_CHUNKS
            )
        )

    official_causal_r10_summary = ""
    if official_causal_r10_ckpt:
        official_causal_r10_summary = (
            "\n## Strictly causal gate summary (R10, official gate)\n"
            "\n"
            "| item | value |\n"
            "| --- | --- |\n"
            "| best val loss | {:.4f} |\n"
            "| best epoch | {} |\n"
            "| arch | vendored author gate (grouped G=4 anchors, cubic interp, smooth modReLU) fed "
            "a strictly causal cumulative-mean query descriptor per {}-token chunk, one kernel per "
            "chunk; near-identity gate warm start; wavelet off |\n".format(
                official_causal_r10_ckpt.get("val_loss", float("nan")),
                official_causal_r10_ckpt.get("epoch", "?"),
                1024 // CKPT_R10_CAUSAL_CHUNKS,
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
        official_summary,
        official_causal_summary,
        r10_summary,
        official_causal_r10_summary,
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
    if official_ppl:
        print(f"SPECTRE (official author math, R8) ppl: {official_ppl:.4f} (ratio {official_ratio:.3f})")
    if official_causal_ppl:
        print(f"SPECTRE (official gate + causal mixing, R9) ppl: {official_causal_ppl:.4f} (ratio {official_causal_ratio:.3f})")
    if r10_ppl:
        print(f"SPECTRE (v1 gate + strictly causal gate, R10) ppl: {r10_ppl:.4f} (ratio {r10_ratio:.3f})")
    if official_causal_r10_ppl:
        print(
            f"SPECTRE (official gate + strictly causal gate, R10) ppl: {official_causal_r10_ppl:.4f} "
            f"(ratio {official_causal_r10_ratio:.3f})"
        )
    print(f"report written to {REPORT_PATH}")
    print(f"**Overall: {verdict}**")


if __name__ == "__main__":
    main()