# Spectre — Repo Discovery Guide for Agents

A map of non-obvious facts about this repository: an earnest PyTorch reproduction of the SPECTRE attention mechanism (arXiv:2502.18394) plus our own incremental causal-fix research, organized by intent.

## Maintenance Mandate

- Load the repo-discovery-guide skill immediately without delay and follow its instructions to the letter.
- Update this guide in the same change when adding, removing, renaming, or discovering an expensive gotcha.
- If `Last verified` is older than 3 days, treat the guide as suspect and re-verify.
- Keep this guide updated before committing or pushing.
- Last verified: 2026-10-10

## Project Overview

This repository reproduces the SPECTRE attention computations (arXiv:2502.18394, "SPECTRE: An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts") in PyTorch and then runs our own incremental research on top. `reproduction/` holds the paper's math exactly as published — the author's vendored `spectre.py` (SHA-256-pinned, loaded verbatim by a thin wrapper) plus neutral GPT-2 infrastructure; nothing added, nothing removed. `research/` holds every experiment we add, one module per increment (R5 v1, R9 causal hybrid, R10 strictly-causal gate). The headline finding: the paper's own math collapses to a next-token copier under honest LM training (future-token leak, position-0 loss 0.0028), while our leak-free variants land at test PPL 47.36–57.94 vs baseline 30.33 within laptop-scale fine-tuning budgets. The former Rust oracle was removed at the owner's direction (2026-10-09); its verified history lives in `docs/`.

## Known Gotchas

- **Dependency direction is sacred:** `research` may import `spectre_paper`; the reproduction must never import research. The paper's math stays pristine.
- `reproduction/vendored/spectre.py` is the author's code, byte-identical to upstream and SHA-256-pinned. Editing it breaks the loader and the pin test — never "fix" it, fix the wrapper (`reproduction/spectre_paper/official.py`).
- `official.py` resolves the vendored file relative to its own location (`parents[1] / "vendored"`); keep that pair together if folders move again.
- The Rust `xval` cross-validation is GONE (oracle + dumper deleted). Do not look for it; PyTorch self-consistency is guarded by the unit tests only now.
- The paper's literal positional phase is one row off from a cache's output (decision D6 in `docs/plan-poc.md`) — intentional, proven, not a bug.
- PyTorch batched SPECTRE layer: never use `torch.einsum` with stacked per-head weights — it materializes a (B,n,H,d_model,d) ~77GB broadcast intermediate. Concatenate per-head weights into (d_model, H·d) and use one plain matmul (GPT-2's c_attn pattern).
- The all-pass gate init (l2.weight=0) zeroes the entire gate path — any test using it cannot catch gate-path bugs. Gate-path changes need non-trivial-gate equivalence tests and the randomized-gate future-invariance probe; committed "causality" tests that use constant gates are vacuous.
- PyTorch `nn.Linear.weight` is (out×in). The vendored author code has its own geometry (per-head head_dim→head_dim W_q/W_v seeing only the head's 64-dim slice, bias-free out_proj) — read `official.py`'s warm start before assuming either convention.
- huggingface_hub 1.x needs namespace/name (`Salesforce/wikitext`), and the split is `validation`, not `valid`.
- macOS memory: `ps -o rss=` for RSS, `sysctl -n vm.swapusage` for swap; MPS `driver_allocated_memory` >> `current_allocated_memory` is a reserved pool, not a leak. Run one heavy process at a time.
- Training checkpoints (`checkpoints/*.pt` at the repo root, gitignored) hold model + optimizer state + epoch/step/bad_epochs; `--resume` warm-starts AdamW if optimizer state is absent.
- R11 (`--r11`) is the v1 R10 gate with each head's final gate projection initialized so every chunk begins with the empty-prefix (chunk-0) gate output. This preserves the cold-start kernel across chunks; gates remain trainable. No training result is measured yet.
- Checkpoint paths are computed from module locations (`Path(__file__).resolve().parents[2] / "checkpoints"`) — moving `spectre_research/` deeper than two levels below the root breaks them silently.
- The Python environment lives at the repo root (single `pyproject.toml`); run `uv run ...` from the repo root. Old `cd realmodel && uv run` muscle memory is obsolete — `realmodel/` no longer exists.
- `docs/` is a historical archive (plans, reports, deviations). It still references the deleted Rust oracle and old `realmodel/` paths — that is expected; do not "fix" archived records.

## Conventions

- Every module is named by role/provenance: `official.py` (the paper's math wrapper), `v1_spectre.py` (our R5 layer), `causal_hybrid.py` (R9), `r10_causal.py` (R10), `r11_chunk0_init.py` (R11), `transplant.py` (GPT-2 surgery for v1). New increments keep the R-numbering and get one module + one test file.
- Research increments are registered in two places: `research/spectre_research/__init__.py` and the ledger in `research/README.md`.
- Experiments run on the shared neutral infrastructure (`spectre_paper.data/evaluate/memlog/surgery`); drivers (`train.py`, `report.py`, `leakage.py`) grow a flag per variant.
- Checkpoints are arch-tagged; `report.py` rebuilds the right surgery from the tag.
- The causality probe (`leakage.py`) measures position-0 loss; a naive shuffle-the-future test cannot catch one-step copying. Zero-warning discipline applies to pytest output too.
- Ambiguities and surprises go in `docs/deviations.md` (date, ticket, what, why).
- Dependencies are pinned by `uv.lock`; Python 3.12+, torch with MPS.

## Structure Map

```
paper/                        The source whitepaper (arXiv:2502.18394v7 PDF) — INPUT, untouched
reproduction/                 THE EARNEST REPRODUCTION — nothing added, nothing removed
  vendored/spectre.py         Author's official implementation, SHA-256-pinned (NOT ours)
  spectre_paper/              official.py (vendored loader + official GPT-2 transplant),
                              surgery.py (GPT-2 geometry, neutral), data/evaluate/memlog
  tests/                      Faithfulness tests (SHA pin, tamper refusal, official behavior)
research/                     OUR INCREMENTS (one module per variant)
  spectre_research/           v1_gate / v1_spectre (R5), causal_hybrid (R9), r10_causal (R10),
                              r11_chunk0_init (R11),
                              transplant (v1 GPT-2 surgery), train / report / leakage (drivers)
  tests/                      One test file per research module
related-work/                 Caracal / CAT lineage PDFs + from-grok.md (NOT ours)
runs/                         Training logs (tracked evidence)
checkpoints/                  Training checkpoints (gitignored)
docs/                         Plans, reports, deviations log — HISTORICAL ARCHIVE; references
                              to the removed Rust oracle there are intentional
pyproject.toml / uv.lock      Single root Python environment (packages: spectre_paper,
                              spectre_research)
AGENTS.md                     Project north star (O(n log n) attention)
repo-discovery-guide-...      This guide
```

## Entry Points

- Full test suite: `uv run pytest -q` (37 tests: reproduction + research) from the repo root.
- Reproduction tests only: `uv run pytest reproduction/tests -q`; research only: `uv run pytest research/tests -q`.
- Training: `uv run python -m spectre_research.train [--official | --official-causal | --r10 | --r11 | --official-causal-r10] [--cotraining] [--budget-minutes=N] [--resume] [--save-every-steps=N]`.
- Report: `uv run python -m spectre_research.report` (writes `docs/realmodel-report.md`).
- Causality probe: `uv run python -m spectre_research.leakage` (position-0 loss is the decisive number).
- Regenerating the old claims report is no longer possible (Rust oracle removed); `docs/claims-report.md` is a frozen historical record.

## What To Verify

- `uv run pytest -q` is green (37 tests) with no warnings.
- `uv run python -c "import spectre_paper.official, spectre_research.train"` imports cleanly.
- The vendored SHA pin holds: `uv run pytest reproduction/tests/test_official.py -q` (its first test checks it).
- `git status` shows no Rust anywhere (no `oracle/`, no `xval-dump/`, no `Cargo.*`).
- No file outside `docs/` and this guide references `oracle`, `realmodel`, `spectre_torch` (the old package name), or `xval`.
- `git log` history is intact: 18 ticket commits (T0–T17), R0–R6, audits, R8/R9 (`89bb9e6`), R10, reorg commits.
