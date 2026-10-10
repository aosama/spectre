# Research — our increments on top of the reproduction

Everything experimental lives here, one module per increment, R-numbered. The paper's math stays pristine in `../reproduction`.

**Dependency rule:** research may import `spectre_paper`; the reproduction never imports research.

## Increment ledger

| Increment | Module(s) | What it is | Result |
| --- | --- | --- | --- |
| R5 (v1) | `v1_spectre.py`, `v1_gate.py`, `transplant.py` | Our causal reimplementation of the paper's layer (linear-conv mixing instead of circular) for the GPT-2 transplant | Test PPL 49.13 frozen / **47.36 co-trained** (1.562×) — best honest result |
| R8 | *(no research module — uses `spectre_paper.official`)* | Control run of the paper's own math | Collapses to a next-token copier (position-0 loss 0.0028) — proves the paper's headline is future-token leakage |
| R9 | `causal_hybrid.py` | The author's gate (kept verbatim by subclassing) + strictly causal zero-padded mixing; near-identity warm start | Test PPL 57.94 (1.911×) — leak-free, but the official per-head geometry transplants poorly |
| R10 | `r10_causal.py` | Strictly-causal per-chunk gate (`causal_chunks` knob): chunk *c*'s kernel pools queries from strictly earlier positions only; `0` reproduces R5/R9 exactly | Closes the gate-descriptor future leak (Issue #3) that vacuous constant-gate tests missed |

Shared drivers (not increments themselves): `train.py` (fine-tuning, all variants), `report.py` (evaluation + `docs/realmodel-report.md`), `leakage.py` (the causality probe — position-0 loss is the decisive number).

## Adding the next increment

1. New module named for the increment; keep the R-numbering going (R11, R12, …).
2. Register it in `spectre_research/__init__.py` and in the ledger above.
3. Tests mirror the module: `research/tests/test_<module>.py`.
4. Give the drivers a flag: `train.py` grows `--<arch>`, `report.py` learns the arch tag, `leakage.py` picks up the new checkpoint.
5. Any surprise (a vacuous test, a leak, a plan mismatch) goes to `../docs/deviations.md`.

## Historical context

The R-numbering and the experiment history (R0–R10) are documented in `../docs/plan-realmodel.md`, `../docs/realmodel-report.md`, and `../docs/deviations.md`. Related literature (Caracal, CAT) that independently arrived at causal Fourier mixing: `../related-work/`.
