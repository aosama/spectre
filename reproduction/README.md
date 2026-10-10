# Reproduction — the paper's math, nothing added, nothing removed

This folder is the earnest reproduction of **SPECTRE** (arXiv:2502.18394). It exists to answer one question with zero ambiguity: *what does the paper's own math actually do?*

## The faithfulness contract

1. **Every SPECTRE tensor in the official variant flows through `vendored/spectre.py`** — the paper author's own implementation, kept byte-identical to upstream ([jacobfa/fft](https://github.com/jacobfa/fft) @ `6aa353e`) and refused by a SHA-256 pin if it ever drifts (see `vendored/README.md`).
2. **No math lives here.** `spectre_paper/official.py` is a thin loader around the vendored module plus the GPT-2 transplant of it ("our math matches the author's" holds by construction, not transcription discipline).
3. **Neutral infrastructure only.** `surgery.py` (GPT-2 geometry + transplant helpers), `data.py`, `evaluate.py`, `memlog.py` are variant-agnostic plumbing shared with the research package.
4. **Our modifications live in `../research`** — causal fixes, new gates, experiment drivers. They must never leak into this folder.
5. **Wavelet refinement is off** (`on_rate=0`): the paper's Table 1 reports SPECTRE and SPECTRE+Wavelet separately; we reproduce plain SPECTRE.

## Paper → code map

| Paper concept | Where |
| --- | --- |
| The author's implementation (all of §3) | `vendored/spectre.py` (not ours, pinned) |
| Loading it with the hash pin | `spectre_paper/official.py::load_official_module` |
| GPT-2 transplant of the official layer (warm start) | `spectre_paper/official.py::swap_gpt2_attention_official` |
| GPT-2 geometry (d_model 768, 12 heads, n_fft 1024) | `spectre_paper/surgery.py` |

## Tests

```bash
uv run pytest reproduction/tests -q    # from the repo root
```

`tests/test_official.py` verifies the SHA pin holds, that a tampered vendored copy is refused, and that the official transplant behaves (forward pass, causality probe compatibility).
