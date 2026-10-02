Yes. The plain SPECTRE head was a bit worse on perplexity, and later work mostly tried to fix the two things that caused that: non-causal FFT mixing, and weak local features.

On their own PG-19 setup (1k context), plain SPECTRE was 39.8 PPL versus 39.4 for standard attention. The wavelet residual module (WRM) in the same paper brought it to 39.0, so the authors already treated local detail as the missing piece. The deeper issue is that a bilateral FFT couples every position to future tokens, which is fine for classification and bad for autoregressive language modeling.

The closest follow-ups:

- **Caracal** (arXiv 2605.00292, ICML 2026) is the direct architectural successor. It replaces attention with a Multi-Head Fourier module, still \(\mathcal{O}(L \log L)\), but adds frequency-domain causal masking by asymmetric padding and truncation. That is the fix for the causality leak. It uses only standard FFT operators, and the authors report it competitive with Transformer and SSM baselines rather than a pure speed-for-PPL trade.

- **CAT: Circular-Convolutional Attention** (arXiv 2504.06704, NeurIPS 2025 poster) keeps a softmax-style global mixer, but makes the mixing circulant so it can be done as an FFT, a Hadamard product, and an inverse FFT. Complexity is \(\mathcal{O}(N \log N)\), with fewer parameters. On their language-modeling setup it improves word perplexity over a Transformer-XL baseline (13.94 for the baseline in their table) and gets about a 10% speedup in plain PyTorch. It is closer to real attention than SPECTRE’s diagonal spectral gate.

Adjacent work improves quality but does not replace the quadratic term:

- **Spectral attention** (Journal of King Saud University, March 2026) FFTs the attention map itself and applies learnable per-head masks. They report about 10.7% lower perplexity on WikiText-2 and 15.3% on WikiText-103 versus standard attention. The score matrix is still built, so this is a quality trick, not an \(\mathcal{O}(N \log N)\) replacement.
- **FourierQK** (arXiv 2607.07478, July 2026) spectrally preprocesses Q and K, then runs normal attention. Large loss drops on TinyShakespeare only; it cites SPECTRE and is explicit that it is a different design.

There is also a less established note, **FFT-IA** (December 2025), which factorizes attention with Cooley–Tukey-style butterfly blocks and claims local softmax is preserved. It has not landed in a major venue.

So the line after SPECTRE splits in two: Caracal and CAT keep the \(\mathcal{O}(N \log N)\) Fourier mixer and attack causality and expressivity, while spectral-filtering papers keep softmax attention and only reshape it in frequency space. None of these is a widely adopted production replacement yet.
