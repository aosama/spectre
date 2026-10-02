# Spectre — Repo Discovery Guide for Agents

A map of non-obvious facts about this repository: a Rust reproduction of the SPECTRE attention mechanism (arXiv:2502.18394), driven by a ticket-based autonomous plan.

## Maintenance Mandate

- Load the repo-discovery-guide skill immediately without delay and follow its instructions to the letter.
- Update this guide in the same change when adding, removing, renaming, or discovering an expensive gotcha.
- If `Last verified` is older than 3 days, treat the guide as suspect and re-verify.
- Keep this guide updated before committing or pushing.
- Last verified: 2026-10-02

## Project Overview

This repository reproduces the SPECTRE attention computations (arXiv:2502.18394, "SPECTRE: An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts") in two layers: a CPU-only Rust library crate at `oracle/` (moved off the root long ago — one module per paper equation, `f64`, rayon-parallel, oracle-verified) and a PyTorch real-model validation under `realmodel/` (GPT-2 small with all 12 attention blocks surgically replaced by SPECTRE layers, fine-tuned on WikiText-2). Phase 1 (`docs/plan-poc.md`, T0–T17) is fully executed and `docs/claims-report.md` shows every reproducible claim PASS. Phase 2 (`docs/plan-realmodel.md`, R0–R6) is executed: cross-validation vs the Rust PoC PASS, final test PPL 49.13 vs baseline 30.33 (ratio 1.620) — honest FAIL at the fine-tuning budget, with gate diagnostics moving in the paper's predicted direction; see `docs/realmodel-report.md`. Beyond the plan, R8/R9 (see commit 89bb9e6) settled the causality question: the paper's own official math (R8, circular mixing) collapses to a next-token copier (position-0 loss 0.0028, test PPL 1.00), while the honest hybrid (R9, author gate + causal conv) lands at 57.94 (1.911) — proving the paper's headline is only reachable via the future-token leak. Evidence: `spectre_torch.leakage` probe, `docs/audit-vs-official.md`, GitHub Issue #2.

## Known Gotchas

- `oracle/` is a **read-only verified reference**. Never edit it — it is the single canonical copy of the verified Rust oracle (moved off the root long ago; the old byte-identical `reference/spec-code/` mirror was removed, so there is no "which one is real?" ambiguity). Edit only `oracle/`.
- The plan is fully autonomous: never stop to ask a human. Ambiguities are settled by the Decision Log (§3); anything else goes in `docs/deviations.md` (date, ticket, what, why).
- Never loosen a numerical-accuracy tolerance (`1e-9`/`1e-10`/`1e-12`), never change an exact op-count equality, never delete or `#[ignore]` a test that is not already ignored.
- Wall-clock tests (`tests/timing.rs`, T15) are `#[ignore]`, run only with `--release --test-threads=1` (~40 s), and are machine-dependent. On failure, that threshold may be widened by at most 25% with the measured numbers recorded in `docs/deviations.md`.
- T13's `global_pool_uses_every_core` fails if `RAYON_NUM_THREADS` is set — `unset RAYON_NUM_THREADS` and retry.
- The `opcount` feature is off by default (zero cost). The `claims_report` binary requires it (`required-features = ["opcount"]`).
- The paper's literal positional phase is one row off from the cache's output (decision D6) — intentional, proven by T12, not a bug.
- Reference numbers in the plan (test counts, timings, speed-ups) were measured on an 8-core Intel i3-N305; wall-clock expectations differ on other machines.
- Running the oracle tests creates `oracle/target/` — a build artifact, never commit it. The root `.gitignore` uses unanchored `target/` to cover it.
- The oracle crate at `oracle/` is the single copy of the verified Rust implementation (the old byte-identical `reference/spec-code` mirror was removed). Edit only `oracle/`.
- The paper PDF's text layer is lossy (garbled glyph IDs in figure captions); during development the paper was verified against page images of `docs/spectre-paper-2502.18394v7.pdf` (arXiv:2502.18394v7) rendered at high resolution.
- `realmodel/` runs via `uv` (`cd realmodel && uv run ...`); the venv is Python 3.12, torch 2.14 with MPS. Tests need `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath = ["."]` — don't run pytest with a clobbered cwd.
- `realmodel/xval-dump/` is a tiny Rust crate that builds the oracle layer with an all-pass gate and dumps JSON so `xval.py` can cross-validate PyTorch against the oracle. Its `Cargo.toml` depends on the oracle via `path = "../../oracle"` (from `realmodel/xval-dump/`, up to the repo root, into `oracle/`) — keep that path in sync if the crate moves again.
- PyTorch batched SPECTRE layer: never use `torch.einsum` with stacked per-head weights — it materializes a (B,n,H,d_model,d) ~77GB broadcast intermediate. Concatenate per-head weights into (d_model, H·d) and use one plain matmul (GPT-2's c_attn pattern).
- The all-pass gate init (l2.weight=0) zeroes the entire gate path — any test using it cannot catch gate-path bugs. Gate-path changes need the non-trivial-gate equivalence test (`test_batched_layer_equals_per_head_loop`).
- Rust `Linear.w` is (in×out) computing wᵀx; PyTorch `nn.Linear.weight` is (out×in). Transpose l1/l2/wo when loading Rust dumps in `xval.py`; raw `wq/wv` params (d_model, d_head) are used as `x @ w` in both — no transpose.
- huggingface_hub 1.x needs namespace/name (`Salesforce/wikitext`), and the split is `validation`, not `valid`.
- macOS: `ps -o rss=` for RSS, `sysctl -n vm.swapusage` for swap; MPS `driver_allocated_memory` >> `current_allocated_memory` is a reserved pool, not a leak. Run one heavy process at a time.
- Training checkpoints (`realmodel/checkpoints/best.pt`) hold model + optimizer state + epoch/step/bad_epochs; `--resume` warm-starts AdamW if optimizer state is absent (pre-resume-format checkpoints).
- The official-math variants live in `spectre_torch/paper_spectre.py` (vendored author code, SHA-256-pinned; editing the vendored file breaks the loader and tests — don't "fix" it, fix the wrapper) and `spectre_torch/causal_hybrid.py` (R9 hybrid: author gate + causal conv; includes corrected `interp_complex_1d_cubic` — the vendored version interleaves real/imag across groups). Checkpoints are arch-tagged; `report.py` rebuilds the right surgery from the tag.
- `realmodel/` commands must run with cwd = `realmodel/` (`cd realmodel && uv run ...`); from the repo root `uv run` resolves to a different interpreter without pytest/torch.
- **Gate-descriptor leak (Issue #3, fixed by R10):** the R5/R9 variants (v1 `v1_spectre.py`, R9 `causal_hybrid.py`) compute the kernel's query descriptor by pooling over the WHOLE sequence — the mixing conv is causal but the kernel depends on future tokens once the gate MLP is non-constant. The committed causality tests pass vacuously (constant gate at init / V=0 in the spike test). Don't trust those tests; use the randomized-gate future-invariance probe.

## Conventions

- `f64` everywhere; `C64` = `num_complex::Complex64` (re-exported as `spectre::C64`). FFT lengths are powers of two and asserted.
- All randomness comes from `ChaCha8Rng::seed_from_u64(seed)`; the paper's seed is 42.
- Zero compiler warnings: `cargo build --all-targets --features opcount`.
- Every parallel loop is over independent outputs only — no parallel float reductions — so results are bitwise identical for any thread count.
- One commit per ticket, with the exact message given in the plan.
- Dependencies are pinned exactly: `num-complex 0.4`, `rand 0.8.5`, `rand_chacha 0.3.1`, `rayon 1.10`, via the copied `Cargo.lock`.

## Structure Map

```
README.md                      Public overview: what/how to run/headline result
AGENTS.md                      Project north star (O(n log n) attention)
oracle/                        THE PAPER'S OWN MATH — Rust oracle (NOT ours). `cargo test` runs here
paper/                         The source paper (v7)
vendored/                      Paper author's PyTorch spectre.py, hash-pinned (NOT ours)
realmodel/                     OUR experiments — PyTorch GPT-2 transplant + R9/R10 fixes
realmodel/spectre_torch/       Role-named modules: paper_spectre / v1_spectre / v1_gate / causal_hybrid / r10_causal / surgery / data / evaluate / train / report / memlog / xval / leakage
realmodel/xval-dump/           Rust crate building the oracle layer (all-pass gate), dumps JSON for xval.py cross-validation
realmodel/checkpoints/         Training checkpoints (gitignored)
docs/                          Plans, validation report, deviations log, claims audit (historical records)
other_white_papers/            Research context: Caracal / CAT lineage
repo-discovery-guide-...       This guide (kept current)
```

## Entry Points

- Oracle crate: `cd oracle && cargo test` (74 tests), `cargo test --release --features opcount --test opcount -- --nocapture` (12), timing suite `cargo test --release --test timing -- --ignored --test-threads=1 --nocapture` (~17 s).
- Regenerate the claims report: `cd oracle && cargo run --release --features opcount --bin claims_report -- ../../docs/claims-report.md` (~10 s).
- PyTorch suite: `cd realmodel && uv run pytest tests/ -q` (32 tests).
- Cross-validation: `cargo run --release --manifest-path realmodel/xval-dump/Cargo.toml` then `cd realmodel && uv run python -m spectre_torch.xval` (both must PASS ≤1e-4/1e-5).
- Training: `cd realmodel && uv run python -m spectre_torch.train [--resume] [--official | --official-causal] [--cotraining] [--budget-minutes=N]`; report: `uv run python -m spectre_torch.report`. `--official` trains the vendored author implementation (R8, non-causal — collapses to a next-token copier); `--official-causal` is the R9 honest hybrid (author gate + causal conv). Leakage probe: `uv run python -m spectre_torch.leakage` (position-0 loss is the decisive number; a naive shuffle-the-future test cannot catch one-step copying).
- If re-executing the plan from scratch: follow the ticket loop in `docs/plan-poc.md` §0 — read ticket → red (tests only) → green (transcribe spec) → ticket command → regression gate (`cargo test`) → commit.

## What to Verify

- rustc ≥ 1.75 (plan verified with rustc 1.91.1; built here with 1.97.1, zero warnings).
- The oracle crate at `oracle/` is the single copy of the verified Rust implementation (the byte-identical `reference/spec-code` mirror was removed — no diff needed). `cd oracle && cargo build --all-targets --features opcount` is clean (zero warnings).
- `git log --oneline` shows 18 ticket commits in order (T0–T17) plus the initial commit, then R0–R6 phase-2 commits, three audit commits, and R8/R9 (`89bb9e6`).
- `docs/claims-report.md` exists, every row PASS, ends with `**Overall: ALL REPRODUCIBLE CLAIMS PASS**`.
- `docs/realmodel-report.md` exists with the final PPL ratio and `**Overall:**` verdict line.
- No test deleted or newly ignored; no accuracy tolerance changed.
