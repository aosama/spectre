# SPECTRE Real-Model Validation — Implementation Plan

> **HISTORICAL PLAN — Phase 2, kept as a record of the GPT-2 experiment (do not re-execute it verbatim).** This plan designed the GPT-2 transplant that is now run from `realmodel/`. Its experiment design is still accurate (causal-convolution deviation, warm init from `c_attn`, frozen backbone). Two paths changed since 2026: the Rust oracle it cross-validates against now lives at `oracle/` (it was at the repo root, byte-identical to the now-removed `reference/spec-code/`), and `realmodel/xval-dump/`'s dependency path is now `../../oracle` (was `../..`) — so run `cargo` from `oracle/`. See the top-level README for the current layout.

> **For agentic workers:** This plan is fully autonomous: never stop to ask a human. Every open question is settled in §3 (Decision Log). §0 tells you what to do when something fails. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the question the PoC could not: *does SPECTRE actually work when swapped into a real pretrained language model?* Take GPT-2 small (124M), replace all 12 attention layers with SPECTRE layers (warm-initialized from the original attention weights), freeze the entire backbone, fine-tune **only** the SPECTRE parameters on WikiText-2, and compare test perplexity against the untouched baseline. Success = SPECTRE perplexity within 10% of baseline.

**Architecture:** A Python/PyTorch package at `realmodel/` (sibling of the Rust crate). The SPECTRE layer is re-implemented in PyTorch with `torch.fft`, cross-validated against the verified Rust PoC (the root crate) on identical weights and inputs. The one deliberate deviation from the paper is **causal mixing** (§1.1): the paper's circular convolution leaks future tokens in an autoregressive LM, and the paper is silent on causality. We use a causal linear convolution (zero-padded FFT of length 2N) so training and inference are both strictly causal on the mixing path.

**Tech stack:** Python 3.12 (via `uv` venv — system Python 3.14 is too new for torch wheels), `torch` (MPS backend), `transformers`, `datasets`, `numpy`. Rust side: one small throwaway crate `realmodel/xval-dump/` that depends on the root `spectre` crate to dump weights/inputs/outputs for cross-validation.

**Spec:** the paper — `docs/spectre-paper-2502.18394v7.pdf` (arXiv:2502.18394v7; its text layer is lossy, so figure captions were verified against high-resolution page renders during development) — the PoC plan at `docs/plan-poc.md`, and the verified Rust implementation at the repo root (`src/`, byte-identical to `reference/spec-code/`).

**Machine (reference):** Apple Silicon, 48 GB RAM, MPS backend. HF CLI installed and authenticated.

---

## 0. Autonomy protocol (read first, obey always)

1. **No questions to humans.** If something is ambiguous, the Decision Log (§3) decides. If the Decision Log is silent, choose the option that keeps the experiment honest (no leakage, no tolerance loosening) and record the choice in `docs/deviations.md`: date, ticket, what, why.
2. **Never modify** the root Rust crate (`src/`, `tests/`, `Cargo.toml`, `Cargo.lock`), `reference/spec-code/`, or `docs/plan-poc.md`. They are the verified oracle. All new code lives under `realmodel/` (plus `docs/realmodel-report.md` and, if needed, `docs/deviations.md`).
3. **Ticket loop:** read the ticket → implement → run the ticket command → confirm the expected result → run the regression gate → commit with the given message.
4. **When a check fails**, follow this ladder in order and stop at the first rung that fixes it:
   1. Read the printed numbers; most failures are shape/index bugs in the new Python code.
   2. Cross-check the failing computation against the Rust PoC's equivalent (run the xval dump, compare).
   3. If a *wall-clock* guard (timebox) trips, do not loosen it silently: stop training, evaluate the best checkpoint so far, and record the measured rate in `docs/realmodel-report.md`.
   4. **Never** loosen the cross-validation tolerance (1e-4), the causal-oracle tolerance (1e-5), or the success threshold (1.10×). A failure there is a real finding — report it honestly.
5. **Commits:** one per ticket, with the message given.
6. **Done** means every box in §6 (Final acceptance) is checked. Then stop.

---

## 1. Background and the one adaptation

### 1.1 The paper's causality gap (why we deviate)

SPECTRE's mixing is a **circular** convolution: `Ṽ = iRFFT(diag(g)·RFFT(V))`, where every output row depends on every input row. Page 5 (§3.2) states it plainly: the operation is "equivalent to a global circular convolution in the time domain, giving every token a full receptive field over the entire sequence," and the gate descriptor is the global mean `q̄ = (1/n)Σq_i`. The paper applies this to autoregressive LLMs (Llama-3.2-1B on PG-19) but **never mentions causality or masking** — the word "causal" does not appear in the paper; the training details (p. 13, App. D) and the cache section (p. 6, a sliding window of the most recent N_max tokens, causal only at inference) are silent on it. Training an LM with a non-causal mixer leaks future tokens into the loss; the learned weights adapt to the leak and the result is not a fair test of SPECTRE as an attention replacement.

**Our adaptation (D3):** replace the circular convolution with a **causal linear convolution**:

$$y[m,c] = \sum_{s=0}^{m} h[s]\, v[m-s, c], \qquad h = \mathcal{R}^{-1}(g)$$

computed with a zero-padded FFT of length `2·n_fft` (both `h` and `v` padded to `2N`, one rfft each, one product, one irfft, take the first `n` rows). Output `m` depends only on inputs `0..m` — strictly causal, still `O(n log n)` (≈2× the FFT cost of the circular version).

**Residual leakage (D4, documented):** the gate descriptor `q̄ = LN(mean(q))` uses the mean over the whole sequence (the paper's formula). Future tokens therefore influence the *kernel choice* indirectly, through a low-dimensional (d=64) summary. A per-position descriptor would make the gate time-varying and destroy the single-FFT-multiply structure (cost returns to O(n²d)), so the global mean is the only descriptor that keeps SPECTRE's complexity. This is a mild, indirect leakage; the dominant information path (the convolution) is strictly causal.

### 1.2 Why warm initialization

The SPECTRE layer discards attention's K projection and its softmax scoring. Starting from random weights, the layer is a random linear map and the frozen backbone (tuned for attention) sees garbage. **Warm init (D6)** slices the original `c_attn` weights: per-head `W_q` and `W_v` are exact column slices of `c_attn.weight`, `W_o` is a copy of `c_proj.weight`, and the gate starts all-pass (`l2.w = 0`, so `g ≡ b2 = (1,0)`; modReLU bias 0). At init the layer is a fixed linear map with the *correct input/output geometry*; fine-tuning only has to learn the content-adaptive mixing. This is what makes a ~1-hour budget feasible.

### 1.3 Why cross-validate against the Rust PoC

The Rust PoC is verified (91 tests, all paper claims PASS). The Python layer re-implements the same math in f32. Before trusting any perplexity number, the Python layer must reproduce the Rust layer's output on identical weights and inputs (all-pass gate, where circular ≡ causal ≡ identity mixing, so the two implementations must agree). Tolerance 1e-4 relative (f32 vs f64).

---

## 2. Global constraints

- All new code under `realmodel/`; report at `docs/realmodel-report.md`; deviations at `docs/deviations.md` (only if needed).
- Python 3.12 in a `uv` venv at `realmodel/.venv` (never system Python).
- `f32` throughout, no AMP (D10). Device: `mps` (fall back to `cpu` only if MPS fails, and record it).
- All randomness seeded: `torch.manual_seed(42)` (the paper's seed), dataset shuffles seeded.
- GPT-2 small config (fixed): `n_layer=12, n_head=12, n_embd=768, d_head=64, n_inner=3072, n_positions=1024`.
- SPECTRE config (fixed): `n_fft=1024, F=513, 2F=1026, gate_hidden=64`, per-head gate, **no Toeplitz, no WRM** (v1).
- Trainable = exactly the SPECTRE parameters (≈31.48M, 25.4% of the model); everything else frozen and verified frozen.
- Total wall-clock budget: **~1 hour**, of which ≤30 min for training (hard stop; evaluate best checkpoint).
- Zero tolerance for silent failures: every ticket command must print its verdict line.

---

## 3. Decision log (settled; do not revisit)

| ID | Decision | Why |
|---|---|---|
| D1 | PyTorch + MPS, Python 3.12 via `uv` | User choice. Fastest path to a trustworthy answer; Rust PoC stays the math oracle. |
| D2 | GPT-2 small (124M) | User choice. Best HF support, well-known baseline perplexity (~6.5 on WikiText-2). |
| D3 | Causal linear convolution (FFT length 2N) instead of the paper's circular convolution | The paper is silent on causality; circular mixing leaks future tokens in an LM. §1.1. |
| D4 | Gate descriptor = global mean of q (paper's formula) | Per-position descriptors break the O(n log n) structure. Mild indirect leakage, documented. |
| D5 | `n_fft=1024`, seq 1024, per-head gate hidden 64, no Toeplitz, no WRM | Matches GPT-2's context; simplest faithful config. WRM/Toeplitz are optional in the paper. |
| D6 | Warm init: `W_q`/`W_v` sliced from `c_attn`, `W_o` copied from `c_proj`, all-pass gate | §1.2. Makes a 1-hour budget feasible; recorded as our protocol, not the paper's. |
| D7 | WikiText-2 for both training and evaluation | Standard, literature-comparable, tiny download. |
| D8 | AdamW lr 3e-4, warmup 100 steps, cosine decay to 2e-5, weight decay 0.05 (0 for biases/LN), grad clip 1.0 | The paper's PG-19 defaults (App. D), warmup scaled to our step count. |
| D9 | Batch 32, ≤10 epochs, early stop after 2 consecutive val-loss increases, 30-min hard stop | Fits the ~1-hour budget; early stop guards the small corpus. |
| D10 | f32, no AMP | MPS + AMP is finicky; 124M fits comfortably in f32. |
| D11 | Cross-validation tolerances: full layer 1e-4 rel, rfft 1e-5 rel, causal oracle 1e-5 rel | f32 vs f64 arithmetic; the causal oracle is f32-vs-f32 direct convolution. |
| D12 | Success: SPECTRE WikiText-2 test PPL ≤ 1.10 × baseline PPL | "Maintaining performance" (paper's claim) with a 10% band for a small fine-tune budget. |

---

## 4. File structure (final state)

```
realmodel/
├── pyproject.toml              uv-managed: torch, transformers, datasets, numpy
├── .venv/                      (gitignored)
├── spectre_torch/
│   ├── __init__.py
│   ├── v1_gate.py              SpectreGate: LN → Linear → GELU(tanh) → Linear(2F) → modReLU → phase
│   ├── v1_spectre.py           SpectreHead + SpectreLayer: causal linear-convolution mixing
│   ├── surgery.py              swap_gpt2_attention(): warm init, freeze, param accounting
│   ├── data.py                 WikiText-2 → 1024-token blocks (train/valid/test)
│   ├── train.py                fine-tune loop: AdamW, schedule, early stop, timebox, checkpoint
│   ├── evaluate.py             WikiText-2 perplexity (batched, mps)
│   ├── xval.py                 cross-validation vs the Rust dump
│   └── report.py               writes docs/realmodel-report.md
├── xval-dump/                  small Rust crate, depends on root spectre crate (path = "../..")
│   ├── Cargo.toml
│   └── src/main.rs             builds all-pass layer (seed 42), dumps weights+input+output JSON
└── tests/
    └── test_v1_spectre.py      rfft roundtrip, all-pass identity, causal oracle, shapes
docs/realmodel-report.md        generated by R6
docs/deviations.md              only if something deviated
```

---

## 5. Tickets

**Command conventions.** Python runs as `cd realmodel && uv run python ...` (uv creates the venv from `pyproject.toml` on first run). "Expected" lines are the verdicts to confirm.

### R0 — Environment, model, data

**Why:** everything downstream needs a working MPS torch, the GPT-2 small weights, and WikiText-2.

**Files:** `realmodel/pyproject.toml`.

- [ ] **1.** `cd realmodel && uv venv --python 3.12 && uv add torch transformers datasets numpy`. If torch has no cp312 wheel for this platform, use the newest Python with a wheel (record in `docs/deviations.md`).
- [ ] **2.** Verify: `uv run python -c "import torch; assert torch.backends.mps.is_available(); x=torch.randn(64,64,device='mps'); print((x@x).sum().item())"` prints a number.
- [ ] **3.** Download: `uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2')"` (caches to `~/.cache/huggingface`). Verify `datasets` loads `wikitext/wikitext-2-raw-v1` and prints train/valid/test token counts (expect ~3.4M / ~0.4M / ~0.6M).
- [ ] Commit `"R0: environment, GPT-2 small, WikiText-2"`.

### R1 — SPECTRE layer in PyTorch (causal variant)

**Why:** the unit under test. Mirrors the Rust PoC's gate/head/layer, with the D3 causal convolution.

**Files:** `realmodel/spectre_torch/{__init__,v1_gate,v1_spectre}.py`, `realmodel/tests/test_v1_spectre.py`.

**Interfaces produced:**
- `SpectreGate(d_head, n_fft, hidden)`: `forward(q_mean: (B,d)) -> (B,F) complex64`. Pipeline: LayerNorm(eps 1e-5) → Linear(d,hidden) → GELU(tanh approx) → Linear(hidden,2F) → interleave `[re0,im0,re1,im1,…]` → modReLU(`ReLU(|g|+b)·g/|g|`, 0 when g=0) → phase shift 0.
- `SpectreHead(d_model, d_head, n_fft, hidden)`: `wq (d_model,d_head)`, `bq`, `wv`, `bv`, `gate`. `forward(x: (B,n,d_model)) -> (B,n,d_head)`.
- `SpectreLayer(d_model, n_heads, n_fft, hidden)`: heads + `wo (d_model,d_model)`, `bo`. `forward(x: (B,n,d_model)) -> (B,n,d_model)`.
- `causal_conv_fft(v: (B,n,d), h: (B,N)) -> (B,n,d)`: zero-pad both to `2N`, `torch.fft.rfft`, multiply, `torch.fft.irfft(n=2N)`, take first `n` rows.

- [ ] **Red:** write `tests/test_v1_spectre.py` first. Run `uv run python -m pytest tests/ -x -q` → expect import/attribute failures.
- [ ] **Green:** implement. Tests (all must pass):
  - `rfft_roundtrip`: `irfft(rfft(x)) == x` for n ∈ {16, 64, 1024} (tol 1e-5).
  - `all_pass_gate_is_identity`: gate with `l2.weight=0` (so g≡(1,0)) → head output equals `v` (tol 1e-5).
  - `causal_conv_matches_direct_oracle`: random `v (4, 64, 8)`, random real `h (4, 64)` → FFT path vs direct `y[m,c]=Σ_{s≤m} h[s]v[m-s,c]` in numpy (tol 1e-5).
  - `causal_no_future_leakage`: change `v[:, m:, :]` → output rows `0..m` unchanged (exact).
  - `shape_and_batch`: `(B,n,d_model) → (B,n,d_model)` for B ∈ {1, 3}, n ∈ {1, 1024}.
- [ ] Regression gate: `uv run python -m pytest tests/ -q` all pass.
- [ ] Commit `"R1: SPECTRE layer in PyTorch (causal variant)"`.

### R2 — Cross-validation against the Rust PoC

**Why:** §1.3. The Python layer must reproduce the verified Rust layer before any model result is trusted.

**Files:** `realmodel/xval-dump/{Cargo.toml,src/main.rs}`, `realmodel/spectre_torch/xval.py`.

- [ ] **1.** `xval-dump` crate: `spectre = { path = "../.." }`. `main.rs`: build `SpectreLayer` with `LayerConfig { d_model: 64, n_heads: 4, n_fft: 64, gate_hidden: 32, toeplitz_r: None, wavelet_levels: None }`, seed 42; set every head's `gate.mlp.l2.w = Mat::zeros(…)` (all-pass, as in the PoC's `all_pass_gate_returns_v`); `x = rand_mat(48, 64, 7)` (n < n_fft: the padded case); `y = layer.forward(&x)`; also `v = rand_vec(64, 11)`, `v_hat = rfft(&v)`. Dump one JSON to `realmodel/xval-dump/out.json`: cfg, per-head `wq/bq/wv/bv`, per-head gate (`ln.gamma/beta`, `l1.w/b`, `l2.w/b`, `modrelu_bias`), `wo.w/b`, `x`, `y`, `rfft_input`, `rfft_output` (re/im pairs).
- [ ] **2.** `xval.py`: load the JSON, build the Python `SpectreLayer` with the same weights (all-pass gate), run `forward(x)`, compare with `y`; compare `torch.fft.rfft(rfft_input)` with `rfft_output`. Print verdict lines:
  - `XVAL layer max rel err: <x> (threshold 1e-4) PASS/FAIL`
  - `XVAL rfft max rel err: <x> (threshold 1e-5) PASS/FAIL`
- [ ] Run `cargo run --release --manifest-path realmodel/xval-dump/Cargo.toml` then `cd realmodel && uv run python -m spectre_torch.xval`. Expected: both PASS.
- [ ] Commit `"R2: cross-validation against the Rust PoC"`.

### R3 — Model surgery (swap, warm init, freeze)

**Why:** the actual drop-in. GPT-2's `c_attn` is `(3·768) × 768` with rows `[Q | K | V]`; per-head `d=64`.

**Files:** `realmodel/spectre_torch/surgery.py`.

**Mapping (exact):** for head `h` (0-based), `d=64`:
- `wq[h] = c_attn.weight[64h : 64(h+1), :]`, `bq[h] = c_attn.bias[64h : 64(h+1)]`
- `wv[h] = c_attn.weight[1536+64h : 1536+64(h+1), :]`, `bv[h] = c_attn.bias[1536+64h : 1536+64(h+1)]`
- `wo = c_proj.weight`, `bo = c_proj.bias` (shared across heads, as in the PoC)
- K rows (768..1535) are discarded (SPECTRE has no key projection, PoC D3).
- Gate: fresh, all-pass init (`l2.weight=0`, `l2.bias` real parts 1 / imag 0, modReLU bias 0).

- [ ] Implement `swap_gpt2_attention(model) -> model` (replaces each `GPT2Attention` with a `SpectreAttention` wrapper that keeps the block's residual/LN wiring), `freeze_backbone(model)` (all non-SPECTRE params `requires_grad_(False)`), and `param_report(model)` (trainable/frozen counts).
- [ ] Tests (add to `tests/test_v1_spectre.py` or `tests/test_surgery.py`):
  - `warm_init_slices_match`: `wq[h]`/`wv[h]`/`wo` equal the original `c_attn`/`c_proj` slices exactly (copied before swap).
  - `forward_runs_on_mps`: batch 2, seq 1024 forward pass completes; output shape `(2, 1024, 50257)`.
  - `frozen_params_unchanged`: hash a sample of frozen params, run one optimizer step on a random loss, re-hash → identical.
  - `param_report`: trainable == 31,482,144 (12 layers × 2,623,512); print it.
- [ ] Commit `"R3: GPT-2 surgery with warm init and frozen backbone"`.

### R4 — Baseline perplexity

**Why:** the comparison anchor.

**Files:** `realmodel/spectre_torch/{data,evaluate}.py`.

- [ ] `data.py`: WikiText-2 → tokenizer → 1024-token blocks (concatenate, drop remainder), train/valid/test.
- [ ] `evaluate.py`: standard batched perplexity (shifted logits, `exp(mean NLL)`), mps, batch 16.
- [ ] Run on the **original** GPT-2 small, WikiText-2 test. Expected: PPL in [28, 33] (token-level; GPT-2 small's published token-level WikiText perplexity is ~29-30 — the ~6.5 figure in the literature is *word-level*). Print `BASELINE ppl: <x>`.
- [ ] Commit `"R4: baseline WikiText-2 perplexity"`.

### R5 — Fine-tune (SPECTRE params only)

**Why:** the experiment itself.

**Files:** `realmodel/spectre_torch/train.py`.

- [ ] Loop per D8/D9: AdamW (lr 3e-4, warmup 100 steps, cosine → 2e-5, wd 0.05 with 0 for biases/LN), grad clip 1.0, batch 32, ≤10 epochs, early stop after 2 consecutive val-loss increases, **hard stop at 30 min** (evaluate best checkpoint so far). Log every 25 steps: step, loss, tokens/sec. Save best checkpoint (by val loss) to `realmodel/checkpoints/best.pt`.
- [ ] Run it. Expected: training loss decreases from its (spiky) init value; tokens/sec printed; wall time ≤ 30 min.
- [ ] Commit `"R5: fine-tune SPECTRE params on WikiText-2"`.

### R6 — Evaluation, diagnostics, report

**Why:** the verdict.

**Files:** `realmodel/spectre_torch/report.py`.

- [ ] Evaluate the best checkpoint on WikiText-2 test → `SPECTRE ppl: <x>`.
- [ ] Gate adaptivity diagnostics (10 diverse 1024-token inputs, one per batch element): at init (all-pass) vs after training, report per gate: mean `|g|`, std of `|g|` across bins, and mean pairwise cosine distance between gates of different inputs. Expectation (soft, not a pass/fail): after training, gates are more spectrally selective (higher bin-std) and more input-discriminative (higher pairwise distance).
- [ ] `report.py` writes `docs/realmodel-report.md`: setup table, cross-validation verdicts (from R2), baseline vs SPECTRE PPL + ratio, training summary (steps, epochs, wall time, tokens/sec, best val loss), gate diagnostics, and the final line:
  - `**Overall: PASS**` if ratio ≤ 1.10, else `**Overall: FAIL (ratio <x>)**` with the honest numbers.
- [ ] Commit `"R6: evaluation and real-model report"`.

### R7 — (Optional) Random-init control

**Why:** shows warm init (D6) is what makes the small budget work.

- [ ] Only if ≥15 min remain in the total budget: repeat R5 with Xavier-random `wq/wv/wo` (no slicing) for 1 epoch, evaluate, add a row to the report table.
- [ ] Commit `"R7: random-init control run"`.

---

## 6. Final acceptance checklist

- [ ] R0–R6 committed in order (R7 optional); `git log --oneline` shows them.
- [ ] Cross-validation: both XVAL lines PASS (layer ≤1e-4 rel, rfft ≤1e-5 rel).
- [ ] Baseline PPL measured and in [28, 33] (token-level; see R4).
- [ ] SPECTRE PPL measured; ratio vs baseline reported; PASS if ≤ 1.10 (otherwise an honest FAIL with numbers — that is a valid, reportable outcome).
- [ ] Frozen params verified unchanged after an optimizer step; trainable count exactly 31,482,144.
- [ ] `docs/realmodel-report.md` exists with all sections and the Overall line.
- [ ] Total wall time ≤ ~1 hour (training ≤ 30 min).
- [ ] Root Rust crate, `reference/spec-code/`, and `docs/plan-poc.md` untouched (`git status` clean for them).
- [ ] `docs/deviations.md` exists if anything deviated; otherwise note "no deviations" in the R6 commit message.

## 7. Out of scope, and the next phase

- **Long-context evaluation** (4k–128k perplexity/latency): the PoC already proved the scaling on CPU; a real-model long-context study needs more compute and is a separate phase.
- **Shared-across-heads gate, Toeplitz, WRM ablations**: the paper's <3% parameter variant and component ablations.
- **More data / more epochs** (C4, PG-19): if R6 lands near the 1.10 boundary, the obvious lever is more fine-tuning data, not a different architecture.
- **Inference-time Prefix-FFT cache in PyTorch**: the PoC proved it in Rust; porting it for generation speed-ups is a separate phase.
