# Official SPECTRE implementation (vendored copy)

This directory holds an unmodified copy of `spectre.py`, the official
SPECTRE reference implementation written by one of the paper's authors.

- **Upstream repository:** https://github.com/jacobfa/fft
- **Source file:** https://github.com/jacobfa/fft/blob/main/spectre.py
- **Commit pinned:** `6aa353e1f4e52b36fec51ab0c58e860f394597c8` (2025-08-04, "Update spectre.py")
- **SHA-256 of `spectre.py`:** `ae63a56a3ad549d561dee67eb65f2267cadc9e5052fabcf4b45dc74965dcfbe2`

It is vendored here for two reasons:

1. **Reference for auditing.** Our reproduction lives in
   `realmodel/spectre_torch/` and was written from the paper
   (arXiv:2502.18394). This copy is the ground truth we diff against when
   checking whether our math matches the author's intent. See
   `docs/audit-vs-official.md` for the recorded differences.
2. **Stability.** The upstream repository has no releases or tags; pinning
   a commit (and its hash above) means our audit references can't drift.

The file is kept byte-for-byte identical to upstream — do not edit it.
