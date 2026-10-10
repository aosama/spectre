# SPECTRE Reproduction & Incremental Research

An independent reproduction of **SPECTRE** — "An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts" ([arXiv:2502.18394](https://arxiv.org/abs/2502.18394)) — carried one step beyond the paper: the mechanism is transplanted into a real pretrained language model (GPT-2 small) to measure what it actually costs on real text. On top of the reproduction we run our own incremental research: fixing the paper's causal-mixing flaw, adding a strictly-causal gate, and iterating toward honest attention-parity at O(n log n).

## Where everything is

The repo is organized by **intent** — you should never have to guess which folder has what:

```
spectre/
├── paper/             # the source whitepaper, arXiv:2502.18394v7              (INPUT — not ours)
├── reproduction/      # EARNEST REPRODUCTION of the paper — nothing added, nothing removed
│   ├── vendored/      #   the author's own PyTorch spectre.py, SHA-256-pinned   (not ours)
│   ├── spectre_paper/ #   thin loader over the vendored math + neutral GPT-2 infra
│   └── tests/         #   faithfulness tests (SHA pin, tamper refusal, official behavior)
├── research/          # OUR INCREMENTS — one module per experiment, R-numbered  (ours)
│   ├── spectre_research/  # v1 (R5), causal_hybrid (R9), r10_causal (R10) + drivers
│   └── tests/
├── related-work/      # research context: Caracal / CAT lineage                 (not ours)
├── runs/              # training logs (small, tracked evidence)
├── checkpoints/       # trained checkpoints (gitignored)
├── docs/              # plans, validation reports, deviations log (historical records)
├── AGENTS.md          # project north star
└── README.md          # this file
```

The one dependency rule: **research may import `spectre_paper`; the reproduction never imports research.** The paper's math stays pristine; every experiment lives in `research/`.

## The two sides

| Side | What it is | Provenance |
| --- | --- | --- |
| `reproduction/vendored/spectre.py` | The paper author's own PyTorch implementation, byte-identical to upstream ([jacobfa/fft](https://github.com/jacobfa/fft) @ `6aa353e`), SHA-256-pinned | Not ours |
| `reproduction/spectre_paper/official.py` | Thin loader for the vendored math + GPT-2 transplant of it — "our math matches the author's" holds by construction, not transcription | Ours (loader only) |
| `reproduction/spectre_paper/surgery.py`, `data.py`, `evaluate.py`, `memlog.py` | Neutral GPT-2 geometry, transplant helpers, data/eval/memory infrastructure | Ours (infrastructure only) |
| `research/spectre_research/v1_spectre.py`, `v1_gate.py` | R5: our causal reimplementation of the paper's layer | Ours |
| `research/spectre_research/causal_hybrid.py` | R9: the author's gate + strictly causal mixing | Ours |
| `research/spectre_research/r10_causal.py` | R10: strictly-causal per-chunk gate (fixes the gate-descriptor leak, Issue #3) | Ours |
| `research/spectre_research/transplant.py` | GPT-2 surgery warm-init for the v1 layer | Ours |
| `research/spectre_research/train.py`, `report.py`, `leakage.py` | Experiment drivers over all variants (training, reporting, causality probe) | Ours |

## Running it

One Python environment at the repo root (Python 3.12+, Apple Silicon MPS or CPU; ~2.5 GB RAM):

```bash
uv run pytest -q                                      # all tests (reproduction + research)
uv run python -m spectre_research.train               # fine-tune (~7 min/epoch on M-series)
uv run python -m spectre_research.report              # writes docs/realmodel-report.md
uv run python -m spectre_research.leakage             # causality probe (position-0 loss is decisive)
```

Training variants: `--official` (the paper's own math), `--official-causal` (R9), `--r10` / `--official-causal-r10` (R10 gate), `--r11` (R10 v1 gate initialized from its chunk-0 cold-start gate; no R11 training result yet), `--cotraining` (unfreeze backbone), `--resume` (continue an interrupted run).

### Resumable training

- `--save-every-steps N` — full-state checkpoint (`<ckpt>.latest.pt`) every `N` steps.
- SIGINT / Ctrl-C / SIGTERM — saves and exits immediately.
- `--resume` — restores step, optimizer, and mid-epoch position, so an interrupted run continues with no lost progress.

```bash
uv run python -m spectre_research.train --save-every-steps=25
uv run python -m spectre_research.train --resume
```

## Headline result

Replacing GPT-2's attention with SPECTRE and fine-tuning only the transplanted parts on WikiText-2:

| Model | Test perplexity | Ratio vs baseline |
| --- | --- | --- |
| GPT-2 small baseline | 30.33 | 1.000 |
| SPECTRE, frozen backbone (8 epochs) | 49.13 | 1.620 |
| SPECTRE, co-trained backbone (10 epochs) | 47.36 | 1.562 |

**Verdict: the mechanism learns (perplexity 230 → 47 after surgery) and its gates become spectrally selective and input-discriminative exactly as the paper predicts — but it does not recover baseline quality within a laptop-scale fine-tuning budget.** Full details and diagnostics: [`docs/realmodel-report.md`](docs/realmodel-report.md).

## The causality finding (our main research result so far)

The paper's own math (R8 experiment: the vendored implementation, trained honestly) **collapses to a next-token copier**: the circular mixing reads one token from the future (position-0 loss 0.0028 vs 6.4–7.2 for honest models), reaching test PPL 1.00 — which explains the paper's headline number. The honest paths:

- **R9 causal hybrid** (`research/spectre_research/causal_hybrid.py`): author gate + zero-padded linear convolution — provably leak-free, test PPL 57.94 (1.911×).
- **R10 gate** (`research/spectre_research/r10_causal.py`): strictly-causal per-chunk gate — `causal_chunks=0` reproduces R5/R9 exactly; `≥2` (default 4) closes the gate-descriptor future leak that earlier "causal" labels missed (Issue #3).
- **R11 initialization** (`research/spectre_research/r11_chunk0_init.py`): initializes every R10 chunk from its head's empty-prefix (chunk-0) gate output; gates remain trainable. Run with `--r11`; no training result is measured yet.

Causality is measured directly by the probe in `research/spectre_research/leakage.py`; a naive shuffle-the-future test cannot catch one-step copying — use the position-0 loss.

## Key documents

- [`docs/plan-poc.md`](docs/plan-poc.md) — Phase 1 plan: the paper's math, decision log, and 18 executed tickets
- [`docs/claims-report.md`](docs/claims-report.md) — every reproducible paper claim, verified PASS
- [`docs/plan-realmodel.md`](docs/plan-realmodel.md) — Phase 2 plan: the GPT-2 transplant experiment design
- [`docs/realmodel-report.md`](docs/realmodel-report.md) — the final validation report with the verdict
- [`docs/audit-vs-official.md`](docs/audit-vs-official.md) — our transcription vs the author's code, differences recorded
- [`docs/deviations.md`](docs/deviations.md) — every place reality differed from plan (and from our own experiments), and why
- [`AGENTS.md`](AGENTS.md) — the project's north star (reduce attention's O(n²) to O(n log n))
