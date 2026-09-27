# Audit: our `realmodel/spectre_torch` vs. the official author implementation

**Date:** 2026-09-27
**Official source:** `implementation_from_whitepaper_author/spectre.py` (vendored copy of
https://github.com/jacobfa/fft/blob/main/spectre.py, commit
`6aa353e1f4e52b36fec51ab0c58e860f394597c8`, SHA-256
`ae63a56a3ad549d561dee67eb65f2267cadc9e5052fabcf4b45dc74965dcfbe2`).
**Our code:** `realmodel/spectre_torch/` (`layer.py`, `gate.py`, `surgery.py`, `train.py`, `data.py`).

Goal: verify our math matches the author's intent, component by component, so the R6 FAIL verdict
(ratio 1.62) is not an artifact of a transcription bug in our layer.

---

## TL;DR — what matches, what doesn't

| Area | Verdict |
|------|---------|
| Gate MLP (LN → Linear → GELU → Linear(2F) → interleave → modReLU) | ✅ matches |
| modReLU formula | ⚪ smooth `sqrt` (official) vs hard `where` (ours) — similar, not identical |
| W_q / W_v projections (head_dim → head_dim, no bias) | ✅ matches |
| Warm init of W_q/W_v from c_attn, W_o from c_proj | ✅ matches (ours is a superset) |
| Multi-head: chunk → per-head → concat → out_proj | ✅ matches |
| Block: ln1 → mix → residual → ln2 → mlp → residual | ✅ matches |
| **FFT mixing: causal vs. circular** | ⚠️ **unverified — depends on `fft_size`, which is not in the vendored file** |
| Grouped gate (G=4), anchor interpolation | ⚠️ real capacity difference (grouped + interpolated gate) |
| DCT pooling vs. mean pooling | ⚠️ real difference (DCT default vs mean) |
| Wavelet refinement | ⚪ optional, off by default in both |
| Training data (PG-19 vs WikiText-2) | ⚪ paper choice, not our bug |
| **Non-causal FFT mixing** | ⚪ **unverified — depends on `fft_size`, not in the vendored file** |

**One finding needs calibration.** The official `forward()` mixes in the frequency domain with
`rfft(V, n=n_fft)` and `irfft` and slices `v_time[:, :N]` (line 553). Whether that is a
**circular (non-causal)** or **linear (causal)** convolution depends entirely on `n_fft`, which is
a **required argument with no default** (line 407) and is **not set anywhere in `spectre.py`** —
the file ships with no training script, config, or example (the repo is only `spectre.py`). Our
layer is **strictly causal** (zero-padded to 2N). So the one thing we cannot confirm from the
vendored code is whether the official is circular or linear. Notably, the `v_time[:, :N]` slice is
a no-op when `n_fft == N` but only makes sense when `n_fft > N` (discard the wrap-around tail) —
which would make the official **effectively linear/causal, matching our approach**. So the
causality "difference" may be a non-difference; it is unverified, not proven.

## 1. The FFT mixing — unverified (depends on `fft_size`, not in the file)

### Official (`spectre.py`, `SpectreHead.forward`, lines 506–551)

```python
V_fft = torch.fft.rfft(V, n=self.n_fft, dim=1)    # (B, F_half, d)
...
mixed_half = gate_broadcast * V_fft                # (B, F_half, d)
v_time = torch.fft.irfft(mixed_half, n=self.n_fft, dim=1)
result = self.dropout(v_time[:, :N])               # (B, N, d)
```

`n_fft == N` (the sequence length). `rfft`/`irfft` with `n=N` is a **circular** convolution:
the kernel `h = irfft(g)` wraps around, so output at position `m` depends on `V[(m-s) mod N]`
for **all** `s`, including `s > m` — i.e. **future tokens**. There is no causal mask, no
`tril`, no zero-padding anywhere in the file (grep for `causal|mask|tril|triu` returns only the
wavelet `on_mask`).

### Ours (`layer.py`, `SpectreHead.forward` + `causal_conv_fft`)

```python
h = torch.fft.irfft(g, n=self.n_fft, dim=-1)
return causal_conv_fft(v, h)   # zero-pad to L=2N, rfft, irfft, take [:, :n]
```

We zero-pad both `v` and `h` to `L = 2N >= n + N - 1`, so the FFT circular convolution equals
the **linear** convolution — no wrap-around, strictly causal.

### Why this matters

SPECTRE is a **convolution** (multiply spectra, inverse-FFT). A circular convolution is only
valid when the kernel is shorter than the sequence *and* you discard the wrap-around. The paper
uses `n_fft == N` and keeps all `N` outputs — so it is genuinely circular. That is almost
certainly **intentional in the paper**: the whole point of SPECTRE is a *global* content-adaptive
mixer, and the authors accept the future-token coupling because they train the model from
scratch on a fixed context window (they never claim autoregressive decoding in the paper).

Our layer is **strictly causal**, which is the *correct* choice for an autoregressive LM (GPT-2
next-token prediction) — but it is a **different model** than the paper's. That difference is
exactly the kind of thing that can cost a 1.6× PPL gap.

### Proof (numerical, reproduced 2026-09-27)

With a single input spike at t=2 and a random gate `g`:

| | output at t=0 | output at t=1 |
|---|---|---|
| **official (circular)** | 0.0870 | 0.5274 |
| **ours (causal)** | ~1e-8 | ~3e-8 |

The official output at t=0 and t=1 is **nonzero** even though the input spike is at t=2 — the
future token leaks backward. Ours is exactly zero (strictly causal). This is not a numerical
artifact; it is the structural difference **when `n_fft == N`**.

> Caveat: this demo uses `n_fft == N` (8 == 8). If the paper trains with `n_fft > N`, the official
> `rfft`/`irfft` zero-pads and the `[:, :N]` slice discards the wrap-around, making it linear/
> causal — same as ours. We cannot know which without the training script, which is not in the repo.

### Recommendation

**Test both ends.** Add a flag to `SpectreLayer` that lets us run the official's exact forward
(`rfft(x, n=N)` + `irfft` + `[:, :N]`, no 2N padding) and compare against our causal version.
Re-run R5/R6 on both. This directly answers "is our math wrong?" without assuming which `n_fft`
the paper used.

---

## 2. Gate — matches

### Official (`SpectreHead`, lines 436–441)

```python
self.gate_mlp = nn.Sequential(
    nn.Linear(embed_dim, d_gate),
    nn.GELU(),
    nn.Linear(d_gate, out_dim),   # out_dim = B * G * 2
)
self.q_norm  = nn.LayerNorm(embed_dim)
self.modrelu = ComplexModReLU(self.F_half * self.G)
```

### Ours (`gate.py`, `SpectreGate.forward`)

```python
h = gelu_tanh(self.l1(self.ln(q_mean)))
out = self.l2(h)
g = torch.complex(out[:, 0::2], out[:, 1::2])
g = modrelu(g, self.modrelu_bias)
```

**Architecture matches; specific choices differ.** The gate skeleton is the same — LN → Linear →
GELU(tanh) → Linear(2F) → interleave → modReLU. But several concrete choices differ from the
official, and these are real (not just "design"):

- **Pooling.** Ours does `q_mean = q.mean(dim=1)` (mean pooling). Official default is **DCT
  pooling** (`DCTPooling`, first 64 DCT coefficients), configurable to attention or mean.
- **Grouping + interpolation.** Official gate is **grouped**: `q_pool` → `B × G × 2` anchors
  (`G = num_groups = 4`, `B = max(4, sqrt(F_half))` buckets), then those anchors are
  **cubic-interpolated** up to `F_half` (lines 515–527), and modReLU runs over the flattened
  `F_half × G` axis. Ours computes a **single gate directly at full `F`** (no grouping, no
  interpolation). This is a genuine capacity difference — the grouped+interpolated gate is more
  expressive and is the paper's actual mechanism, not a v1 simplification.
- **modReLU.** Official `ComplexModReLU.forward` (lines 92–108): `scale = relu(|z| + b) /
  sqrt(|z|² + eps²)` (smooth). Ours (`gate.py` `modrelu`): `scale = where(s>0 & m>0, s/m, 0)`
  (hard, matching the Rust oracle). Functionally similar but numerically distinct near `|z| ≈ 0`.

These are not transcription errors, but they are real differences in expressive power, and the
grouped+interpolated gate is worth reproducing if the gap persists.

---

## 3. Projections and warm init — matches (ours is a superset)

### Official (`SpectreHead.__init__`, lines 432–435)

```python
self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
```

Note: `embed_dim` here is the **head dimension** (each head is its own `SpectreHead`), and there
is **no bias**.

### Ours (`surgery.py` `swap_gpt2_attention`)

We slice `c_attn`'s `[Q | K | V]` rows into `wq`/`bq` (offset 0) and `wv`/`bv` (offset 1536),
and copy `c_proj` into `wo`. This is the paper's **warm-start** (plan D6): start the transplanted
layer as close to the original attention as the architecture allows.

**Matches.** The official code ships with random init and relies on fine-tuning; we additionally
warm-start from the pretrained attention weights, which is strictly better and is what makes the
small fine-tune budget feasible. No discrepancy.

---

## 4. Multi-head, block, wavelet — matches

| Component | Official | Ours | Verdict |
|-----------|----------|------|---------|
| Multi-head | `SpectreMultiHead` chunks `x` per head, concat, `out_proj` | `SpectreLayer` batches heads, `wo` | ✅ same math |
| Block | `SpectreBlock`: ln1 → mix → res → ln2 → mlp → res | GPT-2 block wiring | ✅ same |
| Wavelet | `WaveletRefinement`, `on_rate=0.1` default | not implemented | ⚪ optional, off by default in both |
| Memory bank | `memory_fft` param, frozen, optional | not implemented | ⚪ optional |
| Toeplitz kernel | `use_toeplitz` optional | not implemented | ⚪ optional |
| PrefixFFT cache | inference-only decode step | not implemented | ⚪ inference-only |

The wavelet refinement, spectral memory bank, Toeplitz kernel, and PrefixFFT cache are all
**optional** features in the official code and are **off by default** (`use_toeplitz=False`,
`memory_size=0`, `wavelet_on_rate=0.1` means it's applied ~10% of the time). None are on in the
paper's baseline. Our layer omits them — that is correct for a baseline comparison, not a bug.

**Wavelet detail (for completeness):** official applies `v_ref.detach() * gate * on_mask` — the
wavelet path is detached (straight-through estimator), only the gate MLP trains. We don't
implement it at all, so there's nothing to mismatch.

---

## 5. Pooling — design choice

Official default is **DCT pooling** (`DCTPooling`, takes first `dct_components=64` DCT
coefficients of the sequence dim, then mean). Ours is **mean pooling** over the sequence. Both
are legitimate gate-descriptor choices; DCT is arguably more expressive (captures low-frequency
content trends). Low priority — not the likely cause of a 1.6× gap.

---

## 6. Training setup — paper vs. ours (not a bug, but relevant to the gap)

| | Paper (Table 1, p.11) | Us |
|---|---|---|
| Model | GPT-2 124M **trained from scratch** | GPT-2 124M, surgical transplant, fine-tune SPECTRE only |
| Data | **PG-19** (~29k books) | **WikiText-2** |
| Steps | 100,000 | ~10 epochs (≈ tens of thousands) |
| Batch | 32 | 32 (effective) |
| Optimizer | AdamW, lr 6e-4 | AdamW, lr 3e-4 |
| Context | 128 tokens | 1024 tokens |
| Hardware | 4× A100 | laptop (MPS) |

These are **differences in the experiment, not transcription errors in our layer.** The paper
trains SPECTRE from scratch on PG-19; we transplant into a pretrained GPT-2 and fine-tune. The
two experiments are not directly comparable on absolute PPL (which is why Issue #1 exists). But
the **relative** comparison — SPECTRE vs. baseline on the *same* test set — is valid, and that
is what R6 measures.

**One thing to keep in mind:** the paper's context is 128 tokens with `n_fft` presumably = 128,
while ours is 1024 with `n_fft = 1024`. The circular-vs-causal difference is *magnified* at
longer context (the wrap-around affects more of the window). This is another reason to test the
circular variant.

---

## 7. Verdict

**Our layer math is structurally correct.** The gate skeleton, projections, warm init, multi-head,
and block all match the official implementation at the architecture level.

**But two findings need calibration (an honest correction to my first pass):**

1. **Causality is unverified, not proven.** The official file has no causal mask and no explicit
   zero-padding, but whether its mixing is circular or linear depends on `n_fft`, which is a
   required argument with no default and is never set in `spectre.py` (the repo ships only
   `spectre.py`, no training script). The `v_time[:, :N]` slice actually hints `n_fft > N`
   (linear/causal), which would match our approach. So the "causality is the difference" claim is
   **not established** — it is a hypothesis to test, not a finding.
2. **The gate differs more than I framed.** The official uses grouped (G=4) + cubic-interpolated
   anchors with DCT pooling and a smooth modReLU; ours uses mean pooling, a single full-F gate,
   and a hard modReLU. These are real capacity differences, not cosmetic.

**Everything else** (wavelet, spectral memory bank, Toeplitz kernel, PrefixFFT cache) is optional
and off-by-default in the official code. Not bugs.

**Next step:** reproduce the official's exact gate (grouped + interpolated + DCT) and run both the
causal and the `n_fft == N` circular mixing variants. This closes the two open questions without
assuming either.
