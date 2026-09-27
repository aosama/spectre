# SPECTRE Reproduction & Real-Model Validation

An independent reproduction and validation of **SPECTRE** — "An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts" ([arXiv:2502.18394](https://arxiv.org/abs/2502.18394)) — carried one step beyond the paper: the mechanism is transplanted into a real pretrained language model (GPT-2 small) to measure what it actually costs on real text.

## What is in this repo

| Layer | What it is | Status |
| --- | --- | --- |
| Rust crate (`src/`) | Faithful `f64` reproduction of every paper equation, oracle-tested with exact deterministic op counts | Complete — every reproducible claim PASS |
| `realmodel/` (PyTorch) | GPT-2 small with all 12 attention blocks surgically replaced by SPECTRE layers, cross-validated against the Rust implementation to ~1e-7 relative error | Complete — honest FAIL at fine-tuning budget (see report) |
| `docs/` | The two execution plans, the claims report, the final validation report, and a deviations log | — |

## Headline result

Replacing GPT-2's attention with SPECTRE and fine-tuning only the transplanted parts on WikiText-2:

| Model | Test perplexity | Ratio vs baseline |
| --- | --- | --- |
| GPT-2 small baseline | 30.33 | 1.000 |
| SPECTRE, frozen backbone (8 epochs) | 49.13 | 1.620 |
| SPECTRE, co-trained backbone (10 epochs) | 47.36 | 1.562 |

**Verdict: the mechanism learns (perplexity 230 → 47 after surgery) and its gates become spectrally selective and input-discriminative exactly as the paper predicts — but it does not recover baseline quality within a laptop-scale fine-tuning budget.** The evidence points to the frozen backbone having been pretrained *with attention*; co-training helps modestly, and the remaining gap likely requires pretraining-scale budgets. Full details and diagnostics: [`docs/realmodel-report.md`](docs/realmodel-report.md).

One deliberate scientific deviation from the paper: **causal mixing.** The paper's circular convolution leaks future tokens in an autoregressive setting; we use a causal linear convolution (zero-padded FFT) so the swap is valid for language modeling. See `docs/plan-realmodel.md` §1.1.

## Running it

Rust PoC (no GPU needed):

```bash
cargo test                                            # 74 tests
cargo run --release --features opcount --bin claims_report -- docs/claims-report.md
```

PyTorch real-model validation (Python 3.12, Apple Silicon / CPU; ~2.5 GB RAM):

```bash
cd realmodel
uv run pytest tests/ -q                               # 10 tests
uv run python -m spectre_torch.train                  # fine-tune (~7 min/epoch on M-series)
uv run python -m spectre_torch.report                 # writes docs/realmodel-report.md
```

## Key documents

- [`docs/plan-poc.md`](docs/plan-poc.md) — Phase 1 plan: the paper's math, decision log, and 18 executed tickets
- [`docs/claims-report.md`](docs/claims-report.md) — every reproducible paper claim, verified PASS
- [`docs/plan-realmodel.md`](docs/plan-realmodel.md) — Phase 2 plan: the GPT-2 transplant experiment design
- [`docs/realmodel-report.md`](docs/realmodel-report.md) — the final validation report with the verdict
- [`docs/deviations.md`](docs/deviations.md) — every place reality differed from plan, and why
