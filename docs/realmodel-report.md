# SPECTRE real-model validation report

## Setup

| item | value |
| --- | --- |
| base model | GPT-2 small (124M, 12 layers, 12 heads, d=768) |
| surgery | all 12 attention blocks -> SPECTRE layers (warm init, D6) |
| trainable (R5) | 31,556,016 (SPECTRE only; backbone frozen) |
| trainable (R5b) | 127,647,408 (all params; backbone at 30x lower LR) |
| data | WikiText-2 raw, 1024-token non-overlapping blocks |
| hardware | Apple Silicon, MPS, float32 |

## Cross-validation (R2, vs Rust PoC)

| check | rel diff | verdict |
| --- | --- | --- |
| full layer | 2.630e-07 | PASS |
| rfft | 7.199e-08 | PASS |

## Perplexity (WikiText-2 test, token-level)

| model | PPL | ratio vs baseline |
| --- | --- | --- |
| GPT-2 small baseline (R4) | 30.3261 | 1.000 |
| SPECTRE, frozen backbone (R5) | 49.1279 | 1.620 |
| SPECTRE, co-trained backbone (R5b) | 47.3649 | 1.562 |

## Training summary (R5, frozen backbone)

| item | value |
| --- | --- |
| best val loss | 3.8837 |
| best step | 584 |
| best epoch | 7 |
| schedule | AdamW 3e-4, warmup 100, cosine to 2e-5, wd 0.05, clip 1.0 |
| effective batch | 32 (micro 8 x grad-accum 4) |

## Co-training summary (R5b)

| item | value |
| --- | --- |
| best val loss | 3.8449 |
| best epoch | 9 |
| schedule | SPECTRE 3e-4 + backbone 1e-5 (30x lower), cosine, patience 3 |
| trainable | 127,647,408 (all params) |
| wall | 84.6 min, 10/10 epochs, no early stop |

## Gate adaptivity diagnostics (10 diverse inputs)

| metric | at init (all-pass) | frozen (R5) | co-trained (R5b) |
| --- | --- | --- | --- |
| mean abs(g) | 1.0198 | 0.5908 | 0.5593 |
| bin std of abs(g) | 0.1909 | 0.2242 | 0.2248 |
| pairwise cosine distance | 0.0002 | 0.0175 | 0.0196 |

Soft expectation: after training, higher bin-std (more spectrally
selective) and higher pairwise distance (more input-discriminative).

**Overall: FAIL (ratio 1.620)**
