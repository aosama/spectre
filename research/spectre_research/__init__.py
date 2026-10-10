"""Our incremental SPECTRE research — one variant per module, in R-number order.

  * v1_gate.py / v1_spectre.py — R5: our causal reimplementation of the paper's layer.
  * causal_hybrid.py — R9: the author's gate + strictly causal mixing (leak-free).
  * r10_causal.py — R10: strictly-causal per-chunk gate machinery (Issue #3).
  * transplant.py — GPT-2 surgery warm-init for the v1 layer.
  * train.py / report.py / leakage.py — experiment drivers over all variants.

Everything here depends on the paper's math in reproduction/spectre_paper;
the dependency never points the other way. New increments: add a module named
for the increment, list it here, and log it in research/README.md.
"""
