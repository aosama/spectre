"""SPECTRE in PyTorch, transplanted into GPT-2 small for real-model validation.

Every module is named by role so provenance is obvious from the filename:

  * paper_spectre.py  — the paper's own math, loaded from the audited vendored copy
                         (vendor/spectre.py, hash-pinned). Not our code.
  * v1_spectre.py / v1_gate.py — our first faithful transcription of the paper (R5).
  * causal_hybrid.py — R9: causal linear-convolution hybrid; leak-free.
  * r10_causal.py    — R10: strictly-causal per-chunk gate (causal_chunks knob).

surgery.py, train.py, report.py, evaluate.py, memlog.py, xval.py, data.py and
leakage.py are plumbing: replacing GPT-2 attention, training, reporting, data and
the causality probe.

The Rust crate at the repo root is the oracle: every change here must stay
numerically faithful to it. See docs/plan-realmodel.md for the causal-mixing
deviation and the full experiment history.
"""
