"""Leakage test v2: per-position loss exposes future-token copying.

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

Run: cd realmodel && uv run python -m spectre_torch.leakage
"""
import torch
from transformers import GPT2LMHeadModel

from .data import blocks
from .official import swap_gpt2_attention_official
from .surgery import swap_gpt2_attention

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_OFFICIAL = "checkpoints/best-official.pt"
CKPT_V1 = "checkpoints/best.pt"
N_BLOCKS = 24


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

    verdict = "LEAKS (reads future)" if pos0 < 1.0 else "causal-consistent"
    print(f"{name}:")
    print(f"  A. loss at position 0 (no context; honest ~6-8): {pos0:.4f}")
    print(f"  B. mean loss, first 64 positions:                 {first64:.4f}")
    print(f"  C. loss tok 1..256 orig vs future-shuffled:       {loss_orig:.4f} vs {loss_shuf:.4f} (delta {delta:+.4f})")
    print(f"  -> {verdict}\n")
    del model


def main() -> None:
    test_blocks = blocks("test")[:N_BLOCKS]
    torch.manual_seed(7)
    shuffled = test_blocks.clone()
    perm = torch.randperm(test_blocks.shape[1] - 256)
    shuffled[:, 256:] = shuffled[:, 256:][:, perm]

    print(f"leakage test v2 on {N_BLOCKS} test blocks\n")
    _run("official SPECTRE math (R8 ckpt, 1 epoch)", swap_gpt2_attention_official, CKPT_OFFICIAL, test_blocks, shuffled)
    _run("our causal v1 SPECTRE (R5 ckpt)", swap_gpt2_attention, CKPT_V1, test_blocks, shuffled)
    _run("stock GPT-2 attention (control)", lambda m: None, None, test_blocks, shuffled)


if __name__ == "__main__":
    main()
