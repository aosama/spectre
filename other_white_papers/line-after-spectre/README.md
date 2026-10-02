# Line after SPECTRE

Follow-up and adjacent work to **SPECTRE** (arXiv:2502.18394). These papers attack the same
two weaknesses SPECTRE exposed — non-causal FFT mixing and weak local features — but split into
two camps:

- **O(N log N) Fourier mixers** (Caracal, CAT): replace attention's quadratic term outright, like
  SPECTRE, but fix causality and expressivity. This is the direct successor line.
- **Spectral-filtered attention** (FourierQK, FFT-IA, and one unlocated King Saud paper): keep the
  softmax attention structure and only reshape it in frequency space. Quality tricks, not
  sub-quadratic replacements.

Verified via `arxiv.org/abs` and [exa](https://exa.ai). Metadata is taken from the published
abstracts, not from the original `from-grok.md` summary (which is unverified).

| File | arXiv / source | Author(s) | Venue | What it does |
| --- | --- | --- | --- | --- |
| `caracal-causal-architecture-spectral-mixing-2605.00292.pdf` | arXiv:2605.00292 (v2, May 2026) | Gan, Zhang, Li, Huang, Shi, Ding, Yu | ICML 2026 | O(L log L) Multi-Head Fourier module; frequency-domain causal masking via asymmetric padding + truncation; standard FFT ops only. |
| `cat-circular-convolutional-attention-subquadratic-2504.06704.pdf` | arXiv:2504.06704 (v2, Jan 2026) | Y. Yamada | NeurIPS 2025 poster | Softmixer made circulant so mixing = FFT → Hadamard → iFFT; fewer parameters; ~10% speedup in naive PyTorch. |
| `fourierqk-spectral-qk-projections-2607.07478.pdf` | arXiv:2607.07478 (Jul 2026) | A. Zeris | arXiv preprint (unreviewed) | Spectrally preprocesses learned Q/K projections, then runs normal attention. Bilateral kernel is non-causal; distinct from FNet. |
| `fft-ia-hierarchical-structural-pruning-2511.0076.pdf` | ai.viXra:2511.0076 (Nov 2025) | — | Open archive (not a venue) | "O(N log N) via hierarchical structural pruning and softmax fidelity." Low provenance — treat claims skeptically. |

The King Saud University "spectral attention" paper Grok referenced (FFT the attention map +
learnable per-head masks, ~10.7% lower PPL on WikiText-2, 15.3% on WikiText-103) could **not** be
located on arXiv or via web search. It is likely journal-only (not freely downloadable) or the
summary was conflated. Not included here.

## Why each matters for the SPECTRE line

**Caracal — the strongest successor.** Same shape as this repo's goal (quadratic term → O(L log L)),
but it closes the one leak SPECTRE has: the bilateral FFT couples every position to future tokens,
which is fine for classification and fatal for autoregressive LM. Caracal's
*frequency-domain causal masking* (asymmetric padding + truncation) is exactly the fix this repo's
causal hybrid (`causal_hybrid.py`) and R10 gate (`r10_causal.py`) pursue. ICML 2026 and competitive
with Transformer/SSM baselines, not just a speed win. **Read this one first.**

**CAT — the closest structural cousin.** Where SPECTRE uses a diagonal spectral gate, CAT makes the
mixer circulant so it runs as an FFT → Hadamard product → iFFT. It keeps a softmax-style global
mixer (closer to real attention than SPECTRE's diagonal gate), trades fewer parameters for ~10%
speed. Relevant because it is another working "FFT the mixing" recipe; worth diffing against the
R5 exact-preservation path this repo started from.

**FourierQK — a cautionary complement.** It spectral-preprocesses Q/K and keeps standard attention,
so it is O(N²) in the score, not a replacement. But its abstract is a clear statement of the
boundary: the bilateral FFT kernel is "structurally non-causal," and it calls for a companion
word-scale causal design. That boundary is precisely the R9/R10 contribution here.

**FFT-IA — low-provenance pointer.** On an open preprint archive (ai.viXra, not peer-reviewed), so
verify before trusting any number. The idea — factorize attention at O(N log N) while preserving
softmax fidelity — sits in the same family; useful as a "what not to trust" comparison.

## Grok vs. verified record

| Claim in `from-grok.md` | Verdict |
| --- | --- |
| Caracal = "Multi-Head Fourier module, O(L log L), frequency-domain causal masking" | **Accurate.** Actual title: *"Causal Architecture via Spectral Mixing"* (ICML 2026). |
| CAT = softmax-style global mixer, circulant → FFT, O(N log N), ~10% speedup | **Accurate.** Single author (Yamada), NeurIPS 2025 poster. |
| FourierQK = spectral Q/K preprocessing, runs normal attention, TinyShakespeare | **Accurate.** Explicitly distinct from FNet. |
| FFT-IA = "not a major venue" | **Accurate.** ai.viXra:2511.0076, open archive. |
| Spectral attention (King Saud Univ.) — 10.7% / 15.3% PPL gains | **Unverified.** Not located on arXiv/web. |
