"""Leakage test v3: per-position loss exposes future-token copying.

Why the shuffle test failed (v1 of this file): shuffling the future also
shuffles the targets, so a model that copies exactly one token ahead stays
self-consistent and shows delta 0. The decisive test needs no target trick:

  Test A (position 0): predict token 1 from a single visible token (token 0).
    Any CAUSAL model must pay near-unigram cost here (~6-8 nats) because
    token 0 carries almost no information about token 1. A model scoring
    ~0 at position 0 is reading token 1 from somewhere = future leakage.
  Test B (first 64 positions): honest models are worst exactly here (little
    context); a future-copier aces them.
Test C (revised shuffle): score only predictions of tokens 1..256 after
     shuffling tokens 257..1023. Catches broad-window leakage; blind to pure
     one-step copying (kept for completeness, see comment).
   Test D (future-invariance, R10): replace every token from position 256 on
     with random ones and measure the largest logit change at positions
     0..255. A strictly causal model must score exactly 0.0 here. This is the
     probe that exposes the gate-descriptor leak the mixing tests above miss:
     the convolution was already causal, but the *kernel* was pooled over the
     whole window, so it moved whenever any future token moved.

Run: cd realmodel && uv run python -m spectre_torch.leakage
"""
import os

import torch
from transformers import GPT2LMHeadModel

from .data import blocks
from .official import swap_gpt2_attention_official
from .official_causal import swap_gpt2_attention_official_causal
from .surgery import swap_gpt2_attention

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_OFFICIAL = "checkpoints/best-official.pt"
CKPT_V1 = "checkpoints/best.pt"
CKPT_R10 = "checkpoints/best-r10.pt"
CKPT_OFFICIAL_CAUSAL_R10 = "checkpoints/best-official-causal-r10.pt"
R10_CAUSAL_CHUNKS = 4
N_BLOCKS = 24
SPLIT = 256


@torch.no_grad()
def _nll_per_position(model, ids: torch.Tensor, max_pos: int) -> torch.Tensor:
    """NLL of predicting token t+1, for positions t in [0, max_pos). (max_pos,)"""
    logits = model(ids).logits  # (B, L, V)
    nll = torch.nn.functional.cross_entropy(
        logits[:, :max_pos, :].reshape(-1, logits.shape[-1]).log_softmax(-1),
        ids[:, 1 : max_pos + 1].reshape(-1),
        reduction="none",
    )
    return nll.view(ids.shape[0], max_pos).mean(dim=0)  # average over blocks


@torch.no_grad()
def _future_invariance(model, ids: torch.Tensor, split: int) -> float:
    """Test D: largest logit change at positions < split when the tail is replaced.

    Randomizing tokens from `split` onward leaves every past position's own
    input untouched. A strictly causal model returns bit-identical past
    logits; any nonzero delta is a read of the replaced tail, whichever
    hidden path carried it.
    """
    past_logits = model(ids).logits[:, :split, :]
    perturbed_ids = ids.clone()
    generator = torch.Generator().manual_seed(11)
    tail_length = ids.shape[1] - split
    perturbed_ids[:, split:] = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(ids.shape[0], tail_length),
        generator=generator,
    )
    perturbed_logits = model(perturbed_ids).logits[:, :split, :]
    return (perturbed_logits - past_logits).abs().max().item()


@torch.no_grad()
def _run(name: str, build, ckpt_path: str | None, ids: torch.Tensor, shuffled: torch.Tensor):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    build(model)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    per_pos = _nll_per_position(model, ids.to(DEVICE), 64)
    pos0 = per_pos[0].item()
    first64 = per_pos.mean().item()
    # Test C: predictions of tokens 1..256 with tokens 257+ shuffled.
    loss_orig = _nll_per_position(model, ids.to(DEVICE), 256).mean().item()
    loss_shuf = _nll_per_position(model, shuffled.to(DEVICE), 256).mean().item()
    delta = loss_shuf - loss_orig
    # Test D: past logits must not move when the tail is randomized.
    future_delta = _future_invariance(model, ids.to(DEVICE), SPLIT)

    if future_delta > 1e-4:
        verdict = f"LEAKS (future tail moves past logits by {future_delta:.4f})"
    elif pos0 < 1.0:
        verdict = "LEAKS (reads future)"
    else:
        verdict = "causal-consistent"
    print(f"{name}:")
    print(f"  A. loss at position 0 (no context; honest ~6-8): {pos0:.4f}")
    print(f"  B. mean loss, first 64 positions:                 {first64:.4f}")
    print(f"  C. loss tok 1..256 orig vs future-shuffled:       {loss_orig:.4f} vs {loss_shuf:.4f} (delta {delta:+.4f})")
    print(f"  D. past-logit change when tail randomized:        {future_delta:.6f} (strictly causal: 0)")
    print(f"  -> {verdict}\n")
    del model


def main() -> None:
    test_blocks = blocks("test")[:N_BLOCKS]
    torch.manual_seed(7)
    shuffled = test_blocks.clone()
    perm = torch.randperm(test_blocks.shape[1] - 256)
    shuffled[:, 256:] = shuffled[:, 256:][:, perm]

    print(f"leakage test v3 on {N_BLOCKS} test blocks\n")
    _run("official SPECTRE math (R8 ckpt, 1 epoch)", swap_gpt2_attention_official, CKPT_OFFICIAL, test_blocks, shuffled)
    _run("our causal v1 SPECTRE (R5 ckpt)", swap_gpt2_attention, CKPT_V1, test_blocks, shuffled)
    _run(
        "official gate + causal mixing (R9 ckpt)",
        swap_gpt2_attention_official_causal,
        "checkpoints/best-official-causal.pt",
        test_blocks,
        shuffled,
    )
    if os.path.exists(CKPT_R10):
        _run(
            f"v1 gate + strictly causal gate, C={R10_CAUSAL_CHUNKS} (R10 ckpt)",
            lambda m: swap_gpt2_attention(m, causal_chunks=R10_CAUSAL_CHUNKS),
            CKPT_R10,
            test_blocks,
            shuffled,
        )
    if os.path.exists(CKPT_OFFICIAL_CAUSAL_R10):
        _run(
            f"official gate + strictly causal gate, C={R10_CAUSAL_CHUNKS} (R10 ckpt)",
            lambda m: swap_gpt2_attention_official_causal(m, causal_chunks=R10_CAUSAL_CHUNKS),
            CKPT_OFFICIAL_CAUSAL_R10,
            test_blocks,
            shuffled,
        )
    _run("stock GPT-2 attention (control)", lambda m: None, None, test_blocks, shuffled)


if __name__ == "__main__":
    main()
