# SPECTRE Reproduction & Real-Model Validation

An independent reproduction of **SPECTRE** — "An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts" ([arXiv:2502.18394](https://arxiv.org/abs/2502.18394)) — carried one step beyond the paper: the mechanism is transplanted into a real pretrained language model (GPT-2 small) to measure what it actually costs on real text. Along the way we fixed the paper's causal-mixing flaw and added a strictly-causal gate, so we also carry our own experiments here.

## What is in this repo

The repo contains four different "SPECTREs". We keep them visually distinct — module names in the Python package and folder names below tell you exactly which one you're looking at.

| Layer | What it is | Who wrote it | Where |
| --- | --- | --- | --- |
| Rust crate (`src/`) | Faithful `f64` reproduction of every paper equation, oracle-tested with exact deterministic op counts | Not ours (the paper's math) | root `src/`, `tests/`, `Cargo.toml` |
| Reference copy (`reference/`) | Byte-identical mirror of the committed Rust source + the paper's v7 doc | Not ours | `reference/` |
| Paper PDF | The source paper (v7) | Not ours | `paper/spectre-paper-2502.18394v7.pdf` |
| Vendored author code (`spectre.py`) | The paper author's own `f64` PyTorch implementation, hash-pinned to a SHA-256 | Not ours | `vendored/` |
| PyTorch package (`realmodel/spectre_torch/`) | GPT-2 small with attention surgically replaced by SPECTRE layers | **Ours** | `realmodel/spectre_torch/` |
| Execution plans & reports | The plans, the validation report, the deviations log, the claims audit | Not ours | `docs/` |
| Research context (`from-grok.md`) | Caracal / CAT lineage — we independently reinvented the fix | Not ours | `other_white_papers/` |

The Rust crate is the *oracle*: every PyTorch change in `realmodel/spectre_torch/` must stay numerically faithful to it.

## Where things live — the PyTorch package

All experiments live in `realmodel/spectre_torch/`. Modules are named by **role**, so provenance is obvious from the filename:

| File | Role | Provenance |
| --- | --- | --- |
| `paper_spectre.py` | The paper's own math, loaded from the audited vendored copy | ours? **no** — thin loader over `vendored/spectre.py` |
| `v1_spectre.py` / `v1_gate.py` | Our first faithful transcription of the paper (R5 exact-preservation) | **ours** |
| `causal_hybrid.py` | R9: causal linear-convolution hybrid that fixes the leak | **ours** |
| `r10_causal.py` | R10: strictly-causal per-chunk gate (fixed in this repo) | **ours** |
| `surgery.py` | Replace GPT-2 attention with a SPECTRE head (plumbing) | **ours** |
| `data.py` `evaluate.py` `memlog.py` `xval.py` `train.py` `report.py` `leakage.py` | Data, evaluation, logging, cross-validation, training loop, reporting, causality probe | **ours** |

## Running it

Root Rust crate (no GPU needed):

```bash
cargo test                                            # 74 tests
cargo run --release --features opcount --bin claims_report -- docs/claims-report.md
```

PyTorch real-model validation (Python 3.12, Apple Silicon / CPU; ~2.5 GB RAM):

```bash
cd realmodel
uv run pytest tests/ -q                               # 32 tests
uv run python -m spectre_torch.train                  # fine-tune (~7 min/epoch on M-series)
uv run python -m spectre_torch.report                 # writes docs/realmodel-report.md
```

### Resumable training

`train.py` is written to be babysitable *and* stoppable:

- `--save-every-steps N` — full-state checkpoint (`<ckpt>.latest.pt`) every `N` steps.
- SIGINT / Ctrl-C / SIGTERM — saves and exits immediately.
- `--resume` — restores step, optimizer, and mid-epoch position from the latest checkpoint, so an interrupted run continues where it left off with no lost progress.

```bash
uv run python -m spectre_torch.train --save-every-steps=25   # checkpoint every 25 steps
uv run python -m spectre_torch.train --resume                # continue from the latest checkpoint
```

## Headline result

Replacing GPT-2's attention with SPECTRE and fine-tuning only the transplanted parts on WikiText-2:

| Model | Test perplexity | Ratio vs baseline |
| --- | --- | --- |
| GPT-2 small baseline | 30.33 | 1.000 |
| SPECTRE, frozen backbone (8 epochs) | 49.13 | 1.620 |
| SPECTRE, co-trained backbone (10 epochs) | 47.36 | 1.562 |

**Verdict: the mechanism learns (perplexity 230 → 47 after surgery) and its gates become spectrally selective and input-discriminative exactly as the paper predicts — but it does not recover baseline quality within a laptop-scale fine-tuning budget.** The evidence points to the frozen backbone having been pretrained *with attention*; co-training helps modestly, and the remaining gap likely requires pretraining-scale budgets. Full details and diagnostics: [`docs/realmodel-report.md`](docs/realmodel-report.md).

## The causal-mixing fix

The paper's circular convolution leaks future tokens in an autoregressive setting. This repo fixes it in two versions:

- **R9 causal hybrid** (`causal_hybrid.py`): causal linear convolution (zero-padded FFT) that is mathematically faithful to the paper and provably leak-free.
- **R10 gate** (`r10_causal.py`): a strictly-causal gate with a `causal_chunks` knob — `0` preserves the legacy R5/R9 behavior exactly; `≥2` (default `4`) enables the strictly-causal per-chunk gate.

Causality is tested directly: a causality probe (in `leakage.py`) measures whether a head's gate depends on future tokens; the leak-free variants score **0.000000** where the paper's original mechanism and R5 score large and positive.

## Key documents

- [`docs/plan-poc.md`](docs/plan-poc.md) — Phase 1 plan: the paper's math, decision log, and 18 executed tickets
- [`docs/claims-report.md`](docs/claims-report.md) — every reproducible paper claim, verified PASS
- [`docs/plan-realmodel.md`](docs/plan-realmodel.md) — Phase 2 plan: the GPT-2 transplant experiment design
- [`docs/realmodel-report.md`](docs/realmodel-report.md) — the final validation report with the verdict
- [`docs/deviations.md`](docs/deviations.md) — every place reality differed from plan (and from our own experiments), and why
- [`AGENTS.md`](AGENTS.md) — the project's north star (reduce attention's O(n²) to O(n log n))
