# SPECTRE real-model validation report

## Setup

| item | value |
| --- | --- |
| base model | GPT-2 small (124M, 12 layers, 12 heads, d=768) |
| surgery | all 12 attention blocks -> SPECTRE layers (warm init, D6) |
| trainable | 31,556,016 (SPECTRE only; backbone frozen) |
| data | WikiText-2 raw, 1024-token non-overlapping blocks |
| hardware | Apple Silicon, MPS, float32 |

## Cross-validation (R2, vs Rust PoC)

| check | rel diff | verdict |
| --- | --- | --- |
| full layer | 2.630e-07 | PASS |
| rfft | 7.199e-08 | PASS |

## Perplexity (WikiText-2 test, token-level)

| model | PPL |
| --- | --- |
| GPT-2 small baseline (R4) | 30.3261 |
| SPECTRE fine-tuned (R5) | 49.1279 |
| ratio | 1.620 |

## Training summary (R5)

| item | value |
| --- | --- |
| best val loss | 3.8837 |
| best step | 584 |
| best epoch | 7 |
| schedule | AdamW 3e-4, warmup 100, cosine to 2e-5, wd 0.05, clip 1.0 |
| effective batch | 32 (micro 8 x grad-accum 4) |

## Gate adaptivity diagnostics (10 diverse inputs)

| metric | at init (all-pass) | after training |
| --- | --- | --- |
| mean abs(g) | 1.0198 | 0.5908 |
| bin std of abs(g) | 0.1909 | 0.2242 |
| pairwise cosine distance | 0.0002 | 0.0175 |

Soft expectation: after training, higher bin-std (more spectrally
selective) and higher pairwise distance (more input-discriminative).

**Overall: FAIL (ratio 1.620)**
