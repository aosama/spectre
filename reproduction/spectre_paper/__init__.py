"""The earnest reproduction of SPECTRE (arXiv:2502.18394) — the paper's own math.

Nothing added, nothing removed: every SPECTRE tensor in the official variant
flows through the vendored author implementation (../vendored/spectre.py,
SHA-256-pinned) loaded verbatim by official.py. The remaining modules are
neutral GPT-2 transplant and experiment infrastructure shared with the
research package. Our modifications live in research/spectre_research and
must never leak into this package.
"""
