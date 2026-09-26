# SPECTRE Attention PoC in Rust — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan ticket by ticket. Steps use checkbox (`- [ ]`) syntax for tracking. **This plan is fully autonomous: never stop to ask a human.** Every open question is already settled in §3 (Decision Log). §0 tells you what to do when something fails.

**Goal:** Reproduce the computations of SPECTRE (arXiv:2502.18394, "SPECTRE: An FFT-Based Efficient Drop-In Replacement to Self-Attention for Long Contexts"). That means the spectral mixing layer, the content-adaptive gate, the Prefix-FFT cache, persistent memory and the wavelet refinement. The implementation is a CPU Rust library that uses every core through rayon. It must prove three things: that the computations are *correct*, that they have the claimed *complexity order*, and that the gate *actually learns* something a fixed FFT cannot. It ends with a machine-generated report that checks the implementation against the paper's claims.

**Architecture:** A small Rust library crate at the repository root. It has one module per equation of the paper. Every module is checked against an independent, slow oracle, such as an O(n²) DFT or a time-domain convolution. Complexity is checked in two ways. First, exact deterministic operation counts (feature `opcount`). Second, log-log slope fits of op counts and wall-clock time. Parallelism uses rayon over independent outputs only, so results are bitwise identical for any thread count.

**Tech stack:** Rust stable (2021 edition, verified with rustc 1.91.1), `rayon 1.x` (parallelism), `num-complex 0.4` (complex numbers), `rand 0.8.5` + `rand_chacha 0.3.1` (seeded RNG). No BLAS, SIMD, GPU or FFT crate. The goal is reproduction, not optimization.

**Spec:** the paper at `docs/spectre-paper-2502.18394v7.pdf` and the goal in `docs/plan-poc-goal.md`.

**Spec code (verified reference):** `reference/spec-code/` is a complete, compiled and fully passing implementation of this plan: 91 tests plus the claims report, all PASS on an 8-core Intel i3-N305. Every ticket names the exact spec file it reproduces. The file's `#[cfg(test)] mod tests` block holds the ticket's tests, and the code above it is the reference implementation.

---

## 0. Autonomy protocol (read first, obey always)

1. **No questions to humans.** If something is ambiguous, the Decision Log (§3) decides. If the Decision Log is silent, choose the option that keeps the spec-code tests passing unchanged. Record the choice in `docs/deviations.md`: date, ticket, what, why.
2. **Work at the repo root** (the directory containing this `docs/` folder). Do **not** edit `reference/spec-code/`. It is read-only reference material.
3. **Ticket loop**, the same for every ticket:
   1. Read the ticket's *Why* section and its equations.
   2. **Red:** create the target file containing only the ticket's tests. They are the `#[cfg(test)] mod tests { … }` block of the spec file, or the whole file for `tests/*.rs`. Add the `pub mod` line to `src/lib.rs`. Run the ticket command and confirm it **fails to compile** (unresolved names). Integration-test tickets have no red step.
   3. **Green:** write the implementation above the tests. You may transcribe the spec file. The code there is verified, so that is the intended path. Keep every public name and signature listed under *Interfaces*, because later tickets depend on them.
   4. Run the ticket command and confirm the expected number of passing tests.
   5. Run the **regression gate**: `cargo test` (all earlier tickets must still pass, with no warnings).
   6. Commit with the given message.
4. **When a check fails**, follow this ladder in order and stop at the first rung that fixes it:
   1. Diff your file against the spec file: `diff src/X.rs reference/spec-code/src/X.rs`. Most failures are transcription slips.
   2. Rerun the single test with `-- --nocapture <test_name>` and read the printed numbers.
   3. If the diff is empty and it still fails, the cause is toolchain drift, such as a newer crate API. Make the smallest source change that compiles with identical semantics. Record it in `docs/deviations.md`.
   4. **Wall-clock tests only** (ticket T15, and rows C7, C8, P1 of the claims report): rerun up to 3 times, with nothing else heavy running. If a test still fails, you may widen *that* threshold by at most 25%, then record the measured numbers and the new threshold in `docs/deviations.md`.
   5. **Never** loosen a numerical-accuracy tolerance (the `1e-9`/`1e-10`/`1e-12` checks). Never change an exact op-count equality. Never delete a test, or `#[ignore]` a test that is not already ignored. Those tests *are* the reproduction. A failure there means a bug in your code, so go back to rung 1.
5. **Commits:** one per ticket, with the message given. If the repository is not yet a git repo, T0 creates it.
6. **Done** means every box in §7 (Final acceptance) is checked. Then stop.

---

## 1. The mathematics, and why the code is cut this way

### 1.1 Notation

| Symbol | Meaning | Code name |
|---|---|---|
| $n$ | sequence length (tokens) | `x.rows` |
| $D$ | model width | `d_model` |
| $H$, $d$ | number of heads, per-head width $d = D/H$ | `n_heads`, `d_head` |
| $N$ | FFT length (power of two); sequences with $n<N$ are zero-padded | `n_fft` |
| $F = N/2+1$ | number of non-redundant real-FFT bins | `num_bins(N)` |
| $X\in\mathbb R^{n\times D}$ | input tokens | `x: &Mat` |
| $Q, V \in\mathbb R^{n\times d}$ | per-head query and value projections | `project()` |
| $\hat V\in\mathbb C^{F\times d}$ | column-wise RFFT of $V$ | `rfft_cols` |
| $g\in\mathbb C^{F}$ | spectral gate (one complex number per frequency, shared by all $d$ channels of a head) | `SpectralGate::compute` |
| $N_{\max}$, $N_{\text{mem}}$ | decode window length, persistent-memory length ($N = N_{\text{mem}}+N_{\max}$) | `n_max`, `n_mem` |

### 1.2 What is being replaced: attention costs $O(n^2 d)$

For each head, standard attention computes

$$\text{Attn}(Q,K,V)=\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt d}\Big)V .$$

The score matrix alone is $n\times n$, and each entry is a length-$d$ dot product. That gives exactly $n^2 d$ multiply-adds for the scores and $n^2 d$ more for the weighted sum, so **$2n^2d$ per head**. Ticket T9 builds this baseline and T14 asserts that exact count. Without the baseline there would be nothing to compare SPECTRE's $O(n d\log n)$ against.

### 1.3 The real FFT and why only $N/2+1$ bins are kept (paper eq. 1, Appendix B)

$$\hat x_k = (\mathcal R_N x)_k=\sum_{t=0}^{N-1}x_t\,e^{-j2\pi kt/N},\qquad k=0,\dots,N/2 .$$

For real $x$, $\hat x_{N-k}=\overline{\hat x_k}$ (Theorem B.1), so bins $N/2+1\ldots N-1$ are redundant (Corollary B.2). Bins $0$ (DC) and $N/2$ (Nyquist) are real. The inverse rebuilds the full Hermitian spectrum and applies the complex inverse FFT:

$$x_t=\frac1N\sum_{k=0}^{N-1}\hat x_k e^{+j2\pi kt/N}.$$

The radix-2 Cooley–Tukey FFT performs exactly $\tfrac N2\log_2 N$ butterflies. That exact number is the atomic unit of the $O(n\log n)$ claim, which is why T2 builds the complex FFT **alone** and T14 asserts that count **exactly**. The RFFT (T3) is layered on top and tested against an $O(N^2)$ textbook DFT oracle.

### 1.4 The core identity: gating in frequency is global convolution in time

SPECTRE's mixing (paper eqs. 3–4) is

$$\tilde V=\mathcal R_N^{-1}\big(\operatorname{diag}(g)\,\mathcal R_N(V)\big).$$

By the convolution theorem, this is **exactly** a circular convolution of every channel with one real kernel:

$$\tilde V[m,c]=\sum_{s=0}^{N-1}h[s]\,V[(m-s)\bmod N,\,c],\qquad h=\mathcal R_N^{-1}(g).$$

One subtlety: the inverse RFFT discards the imaginary part of the DC and Nyquist bins, as numpy does. The real kernel therefore sees $\operatorname{Re} g_0$ and $\operatorname{Re} g_{N/2}$. Both sides of the identity make the same simplification, so the identity holds to about $10^{-15}$.

**Why this matters for the plan:** the identity gives an FFT-free, $O(N^2 d)$ oracle (`circular_conv_reference`, T7), which checks the entire fast path end to end. It also *is* the paper's "global receptive field" claim: every output row depends on every input row. T7 tests that directly.

### 1.5 The content-adaptive gate (paper §3.2 step 3, "Positional Awareness")

The gate pipeline, in this order (D5):

$$\bar q=\operatorname{LN}\Big(\tfrac1n\textstyle\sum_i q_i\Big),\qquad
u=W_2\,\operatorname{GELU}(W_1\bar q+b_1)+b_2\in\mathbb R^{2F},\qquad g_k=u_{2k}+j\,u_{2k+1}.$$

1. **Toeplitz low-rank update** (optional, bandwidth $2r+1$). This is a zero-padded "same" convolution along the frequency axis:
$$g_k\leftarrow g_k+\sum_{i=-r}^{r}t_{i+r}\,g_{k-i}.$$
2. **modReLU:**
$$g_k\leftarrow\operatorname{ReLU}(|g_k|+b_k)\,\frac{g_k}{|g_k|},\qquad \text{and }0\text{ when }g_k=0.$$
It keeps the phase and shrinks or zeroes the magnitude.
3. **Positional phase:**
$$g_k\leftarrow g_k\,e^{+j2\pi k s/N}.$$
By the DFT shift theorem this rotates the time-domain output: $y[m]=x[(m+s)\bmod N]$. T5 proves this with a test, and the cache (T10) relies on it.

Each step is a separate public function (`modrelu`, `toeplitz_update`, `apply_phase`), so each property can be tested in isolation. The tests cover phase preservation, the zero-kernel identity, bandedness, and the rotation.

### 1.6 Wavelet Refinement Module (paper §3.5)

One orthonormal Haar level maps pairs to

$$a_i=\tfrac{x_{2i}+x_{2i+1}}{\sqrt2},\qquad d_i=\tfrac{x_{2i}-x_{2i+1}}{\sqrt2},$$

and repeats on $a$ for $J$ levels. The layout is $[a_J\,|\,d_J\,|\,\dots\,|\,d_1]$, and the cost is $\sum_{\ell=1}^{J}N/2^\ell = N(1-2^{-J})$ pair operations, which is linear. With per-(band, channel) gates $s\in\mathbb R^{(J+1)\times d}$ from an MLP on $\bar q$:

$$V_{\text{out}}=\tilde V+\mathcal W^{-1}\big(s\odot\mathcal W(\tilde V)\big).$$

The tests are chosen from this equation. Orthogonality means energy is preserved and reconstruction is perfect. Setting $s=1$ gives $2\tilde V$, and $s=0$ gives $\tilde V$. Exact counts of $N(1-2^{-J})$ confirm the $O(nd)$ cost claimed in Table 6.

### 1.7 Prefix–FFT cache (paper §3.3, Algorithm 1, eq. 5)

**The state invariant, held after every call.** `V_buf` is a ring buffer where token $t$ lives in slot $t \bmod N_{\max}$, and

$$\texttt{prefix\_fft}=\mathcal R_N\big(\texttt{V\_buf}\big).$$

**Pre-fill** (§3.3.1): one padded RFFT, $O(N\log N\,d)$.

**Decode update (eq. 5).** Token $t$ enters slot $i=t\bmod N_{\max}$ and evicts $v_{\text{old}}$ from the same slot:

$$\hat V_{k}\leftarrow\hat V_{k}-\mathbb 1_{t\ge N}\,v_{\text{old}}\,e^{-j2\pi k(t-N)/N}+v_t\,e^{-j2\pi kt/N}.$$

Since $(t-N)\equiv t \pmod N$, the two twiddles are **identical**. The update is therefore a single rank-1 correction, $\hat V_k \mathrel{+}= (v_t-v_{\text{old}})\,w^{k i}$ with $w=e^{-j2\pi/N}$, which costs exactly $F\cdot d$ complex multiplies, i.e. $O(Nd)$ per token (D8). T12 re-implements eq. 5 *literally*, with both terms, and checks agreement at every step.

**Decode output.** The descriptor is $\bar q=\operatorname{LN}(\texttt{sum\_q}/N_{\max})$ (D7). The gate gets the phase shift $s=t+1$, where $t+1$ is the number of tokens seen. The output is $\mathcal R^{-1}_N(\operatorname{diag}(g)\,\texttt{prefix\_fft})$. By §1.5, row $m$ of the result holds ring slot $(m+s)\bmod N$. With $s=t+1$ the rows are in **chronological order, newest last**, so the last $L'=\min(t+1,N_{\max})$ rows are the live context.

The paper's Algorithm 1 uses $s=t$, which is off by exactly one row: the newest token lands on row 0. We use $t+1$ (D6), and T12 contains a test that pins down the one-row relationship.

**Oracle.** Decoding must equal "recompute the window from scratch": unroll the last $N_{\max}$ tokens chronologically, then apply the head's `mix` with the same descriptor. T10 checks this through four wrap-arounds of the ring buffer.

**Cost.** Eq. 5 is $O(Nd)$, and the inverse RFFT is $O(Nd\log N)$. The paper states $O(\tfrac N2 d)$ per step, which counts only the update. We verify both parts separately (D14).

### 1.8 Persistent memory (paper §3.4)

The effective sequence is the time-domain concatenation $[M;\texttt{V\_buf}]$ of length $N=N_{\text{mem}}+N_{\max}$. Because the RFFT is linear,

$$\mathcal R_N([M;\texttt{V\_buf}])=\underbrace{\mathcal R_N([M;0])}_{\text{computed once}}+\underbrace{\mathcal R_N([0;\texttt{V\_buf}])}_{\text{eq. 5 at position }N_{\text{mem}}+i}.$$

This is how we implement the paper's "prepend $\hat M$ … no additional FFT" (D10). The memory spectrum must stay bit-for-bit unchanged across the whole session, and T10 asserts that.

### 1.9 How complexity is validated, and why two methods

1. **Exact op counts** (`--features opcount`, T0/T14). Global atomic counters are incremented at the innermost loop level of each primitive: butterflies, real MACs, complex multiplies and Haar pairs. Counts are deterministic, so small components get **exact equality** checks, for example FFT $=\tfrac N2\log_2N$, matmul $=mkn$, attention $=2n^2d$, cache update $=F d$.
2. **Asymptotic slope.** For cost $y(n)$ we fit the least-squares slope $p$ of $\ln y$ against $\ln n$. If $y=c\,n^\alpha$ then $p=\alpha$. For $y=n\log n$,
   $$p=\frac{d\ln(n\log n)}{d\ln n}=1+\frac{1}{\ln n},$$
   which is between 1.07 and 1.15 on $n\in[2^8,2^{16}]$. Lower-order linear terms, such as projections, pull the fitted value toward 1.0. Hence the acceptance bands:
   - SPECTRE: slope in $(0.95,\,1.2)$.
   - Attention: slope in $(1.8,\,2.05)$.
   - Flatness in $t$ for decode.

   A second check is `spread` $=\max(y/n\log n)/\min(y/n\log n)$, which must stay below 2. For a quadratic it grows without bound.
3. **Wall clock** (T15, ignored by default, release mode). The same slope method is applied to median times, with looser bands because real hardware is noisy.

Op counts prove the *algorithm's* order independently of hardware noise. Wall clock proves that the *implementation* realizes it.

### 1.10 Parallelism and determinism

Every parallel loop (rayon) splits **independent outputs**: matrix rows, FFT columns (one FFT per channel), heads, frequency bins, query rows and batch samples. Each output element is produced by one sequential loop, and there are no parallel floating-point reductions. The results are therefore **bitwise identical** for 1, 2, 3 or all threads. T13 asserts that. It also asserts that the global pool size equals `available_parallelism()`, and T15 measures the speed-up.

### 1.11 Proof that it actually works: the learning demo (T16)

**Task.** Each sequence has class $c\in\{0,1\}$, encoded as a ±1 offset on the mean of channel 0, which makes the class visible in $\bar q$. The target is $V$ circularly delayed by $s_c\in\{1,5\}$. A delay by $s$ is exactly the gate $g_k=e^{-j2\pi ks/N}$, so the task is representable. The correct delay depends on content, so no fixed filter can fit both classes.

**Training.** Only the gate MLP's output layer $(W_2,b_2)$ is trained, with modReLU bias 0, no Toeplitz kernel and phase 0. The output is then *linear* in $(W_2,b_2)$, so this is convex least squares. Let $G=\partial L/\partial\tilde V$, $\hat G=\mathcal R(G)$, $z_k=\sum_c\hat V_{kc}\overline{\hat G_{kc}}$, and $w_k=1$ for $k\in\{0,N/2\}$, else $2$ (the Hermitian doubling). Then

$$\frac{\partial L}{\partial\operatorname{Re}g_k}=\frac{w_k}{N}\operatorname{Re}z_k,\qquad \frac{\partial L}{\partial\operatorname{Im}g_k}=-\frac{w_k}{N}\operatorname{Im}z_k,\qquad \frac{\partial L}{\partial W_2[i,j]}=h_i\frac{\partial L}{\partial u_j}.$$

A finite-difference test verifies this gradient. The optimizer is Adam.

**Baseline bound.** A content-independent filter (the FNet-style case, $W_2=0$ and only $b_2$ trained) can fit the DC part, which a delay does not change. On the rest it can do no better than the average of the two delay filters. The error of that average is $\tfrac14|e^{-j\theta_0}-e^{-j\theta_1}|^2$, which averages to $\tfrac12$. So

$$L_{\text{fixed}}^{\star}=\tfrac12\,(1-\text{DC energy share})\approx0.28 .$$

**Measured with the spec code:** adaptive held-out loss is $0.0003$; fixed filter is $0.2925$, against a theoretical optimum of $0.2789$. That is the "it actually works" evidence, and T16 asserts it.

### 1.12 Why the work is split into these tickets

Each module owns **one equation** and is tested against an **independent oracle**. An error is caught in the ticket that introduced it, not three layers later.

```
T0 opcount ─┬─ T1 tensor(+testutil) ── T2 fft ── T3 rfft ─┬─ T5 gate ─┐
            │                                              │           ├─ T7 head ─┬─ T8 layer
            └─ T4 nn ──────────────────────────────────────┴─ T6 wavelet┘          ├─ T10 cache
                                         T9 attention (needs T1, T4)               │
T11 complexity (standalone helpers)                                                │
T12 paper-literal ─ T13 parallel ─ T14 opcount ─ T15 timing ─ T16 learning ─ T17 claims report
```

- **T2 before T3:** a complex FFT bug would otherwise be misdiagnosed as a Hermitian-reconstruction bug.
- **T5 before T7:** the gate's properties (modReLU, Toeplitz, phase) are checked on vectors before being hidden inside matrix code.
- **T7 before T10:** the cache's oracle is the head's own `mix`, so `mix` must already be proven against the convolution oracle.
- **T12–T17 are pure verification tickets.** They add no library logic, except the demo trainer in T16. They cross-check the finished pieces against the paper's literal formulas, the complexity claims, parallelism and learnability.

---

## 2. Global constraints

- Crate name `spectre`, edition 2021, at the repository root. Dependencies exactly: `num-complex = "0.4"`, `rand = "0.8.5"`, `rand_chacha = "0.3.1"`, `rayon = "1.10"`. Pin them by copying `reference/spec-code/Cargo.lock`.
- `f64` everywhere. The complex type is `num_complex::Complex64`, re-exported as `spectre::C64`.
- FFT lengths are powers of two. Any code path taking `n_fft` asserts it.
- All randomness comes from `ChaCha8Rng::seed_from_u64(seed)`. The paper's seed is 42 (Appendix A.5).
- There must be zero compiler warnings (`cargo build --all-targets --features opcount`).
- `[profile.test] opt-level = 2`, so that debug-mode tests run fast.
- Every parallel loop must be over independent outputs (§1.10).
- Wall-clock tests are `#[ignore]` and run only in `--release` with `--test-threads=1`.

## 3. Decision log (settled; do not revisit)

| ID | Decision | Why |
|---|---|---|
| D1 | `f64` throughout; accuracy tolerances $10^{-9}$–$10^{-15}$ | Reproduction of the math, not of fp16 throughput |
| D2 | Own radix-2 FFT; `n_fft` is a power of two; inputs with $n<N$ are zero-padded; `forward` returns the first $n$ rows | A self-built FFT gives exact, countable butterflies. The gate MLP output size $2F$ must be fixed per head, so $N$ is fixed and shorter inputs are padded. |
| D3 | Per head, $W_q,W_v\in\mathbb R^{D\times d}$ (no key projection); layer output $W_o\in\mathbb R^{D\times D}$ applied to the concatenation of heads | Paper eq. 2 has only Q and V. $W_o$ is the standard multi-head output ("concatenated as usual"). |
| D4 | Gate MLP: `LN → Linear(d, hidden) → GELU(tanh approx) → Linear(hidden, 2F)`, outputs interleaved `[re0, im0, re1, im1, …]`; the $b_2$ real parts are initialized to 1 (all-pass, $g\approx1$) | "Two-layer MLP to a complex vector". An all-pass init makes a fresh layer ≈ identity mixing, which is the drop-in property. |
| D5 | Gate order: MLP → Toeplitz → modReLU → phase. Toeplitz kernel initialized to 0; modReLU bias initialized to 0 | Paper order (a)(b)(c), then the positional phase. Zero inits make the optional parts start as the identity. |
| D6 | Positional phase: `forward` uses shift 0; the cache uses shift $=t$ (tokens seen $=$ newest index $+1$) | Chronological output with newest last (§1.7). The paper's literal $t$ is one row off, which T12 proves. |
| D7 | Descriptor: `forward` uses the mean over the $n$ real tokens (§3.2); the cache uses `sum_q / N_max` (Alg. 1), even while the window is filling | Follow each paper formula literally. `prefill` with $L=N_{\max}$ therefore equals `forward` exactly (tested in T10). |
| D8 | Eq. 5 implemented as the single delta $(v_t-v_{\text{old}})\,w^{ki}$ | The twiddles are provably equal (§1.7); T12 checks the literal form. |
| D9 | `prefill` returns the live-context output (the same computation as `decode_step`'s output) as well as filling the state | Needed to measure time-to-first-token. Algorithm 1 returns only state. |
| D10 | Persistent memory = time-domain concatenation $[M;\,\texttt{V\_buf}]$, spectra summed by linearity; no phase rotation in memory mode; output is all $N$ rows in placed order | The only interpretation in which "prepend $\hat M$, no extra FFT" is mathematically exact. A rotation would move the memory block. |
| D11 | WRM = $J$-level Haar; gates $s\in\mathbb R^{(J+1)\times d}$ (per band, per channel, no activation); applied in `forward` only; the learned skip controller is replaced by a config switch | The paper's "channel-wise wavelet level gates". Algorithm 1 has no WRM. The controller needs training. |
| D12 | `forward` is non-causal (encoder-style); causal generation goes through the cache | The paper's layer mixes all tokens; causality comes from the cache window. |
| D13 | Baseline = naive non-causal softmax attention with $W_q,W_k,W_v,W_o$ | Standard SDPA semantics for complexity comparison |
| D14 | Decode cost is reported in two parts: eq. 5 update $O(Nd)$ (the paper's claim) and the full step $O(Nd\log N)$ (including the inverse RFFT) | An honest cross-check of the paper's statement |
| D15 | rayon over independent outputs; no parallel float reductions | Uses all cores and stays deterministic (§1.10) |
| D16 | The learning demo trains only gate $W_2,b_2$ with analytic gradients and Adam | Proves learnability without an autograd framework (§1.11) |
| D17 | Toeplitz cost is $O(F\cdot r)$ per head, because the gate is shared across channels; this is ≤ the paper's $O(nrd)$ | Recorded so that T14's exact count is not "fixed" to match the paper's looser bound |

## 4. File structure (final state)

```
Cargo.toml  Cargo.lock  .gitignore
src/lib.rs            module list + `pub use num_complex::Complex64 as C64`
src/opcount.rs        T0  op counters (feature `opcount`)
src/tensor.rs         T1  Mat / CMat, parallel matmul
src/testutil.rs       T1  seeded random inputs, diff helpers
src/fft.rs            T2  radix-2 FFT + O(n²) DFT oracle
src/rfft.rs           T3  RFFT / iRFFT, column-parallel versions
src/nn.rs             T4  Linear, LayerNorm, GELU, Mlp2, seeded_rng
src/gate.rs           T5  spectral gate, modReLU, Toeplitz, phase
src/wavelet.rs        T6  Haar DWT + WRM
src/head.rs           T7  SPECTRE head + convolution oracle
src/layer.rs          T8  multi-head SPECTRE layer
src/attention.rs      T9  naive attention baseline
src/cache.rs          T10 Prefix-FFT cache + persistent memory
src/complexity.rs     T11 slope / spread / timing helpers
tests/paper_literal.rs T12  literal paper formulas
tests/parallel.rs      T13  all cores + determinism
tests/opcount.rs       T14  exact op counts and complexity order
tests/timing.rs        T15  wall-clock order + speed-up (ignored)
src/train.rs, tests/learning.rs T16  learning demo
src/bin/claims_report.rs        T17  claims verification report
docs/claims-report.md           T17  generated output
docs/deviations.md              any ticket (only if something deviated)
```

## 5. Paper-claims verification matrix

| ID | Paper claim | Source in paper | Verified by |
|---|---|---|---|
| C1 | RFFT keeps ⌊n/2⌋+1 coefficients | eq. 1, Fig. 2 | T3 `keeps_n_over_2_plus_1_bins`; T17 |
| C2 | Hermitian symmetry | Thm. B.1 | T3 `hermitian_symmetry_theorem_b1`; T17 |
| C3 | Lossless half-spectrum reconstruction | Cor. B.2 | T3 `roundtrip_is_lossless`; T17 |
| C4 | Spectral gating = global token mixing | §2, §3.2 | T7 `matches_circular_convolution_oracle`, `global_receptive_field`; T17 |
| C5 | Per-layer $O(nd\log n)$ | Abstract, Table 6 | T14 `spectre_layer_ops_scale_as_n_log_n`; T17 |
| C6 | Attention is $O(n^2d)$ | §1–2 | T14 `attention_is_exactly_quadratic`; T17 |
| C7 | Near-$O(n\log n)$ runtime | Fig. 1, §4.6 | T15 `spectre_wallclock_scales_n_log_n`; T17 |
| C8 | Speed-up over attention grows with $n$ | Fig. 1, Table 7 | T15 `spectre_is_faster_than_attention_at_4k`; T17 |
| C9 | Prefix-FFT decode is exact | §3.3, Alg. 1 | T10 `decode_matches_recomputation_from_scratch`; T12 `eq5_literal_matches_cache_update`; T17 |
| C10 | Constant per-token decode cost | §3.3.2, Table 1 | T14 `decode_step_cost_is_independent_of_t`; T15 `decode_step_time_is_flat_in_t`; T17 |
| C11 | Decode update $O(\tfrac N2 d)$ | §3.3.2, Fig. 6 | T14 `cache_update_is_linear_in_window`, `decode_step_and_prefill_scale_as_n_log_n_in_window`; T17 |
| C12 | Cache memory $O(N_{\max}d)$, constant | §3.3.2 | T10 `footprint_is_constant_across_steps`; T17 |
| C13 | Persistent memory static and exact | §3.4 | T10 `persistent_memory_is_static_and_output_matches_oracle`; T17 |
| C14 | WRM orthogonal and $O(nd)$ | §3.5, Table 6 | T6 `orthogonal_energy_preserved`; T14 `wavelet_is_linear`; T17 |
| C15 | Content-adaptive gate beats fixed spectral filters (FNet) | §1–2 | T16 `tests/learning.rs`; T17 |
| C16 | Real FFT is translation-equivariant | §3.5 "Positional Awareness" | T7 `circular_translation_equivariance` |
| C17 | Positional phase injects absolute position | §3.5, Alg. 1 line 27 | T5 `phase_shift_rotates_time_signal`; T12 `algorithm1_literal_phase_is_our_output_rotated_by_one_row` |
| C18 | Drop-in: output shape equals attention's | §3.6 | T8 `preserves_shape_like_attention` |
| P1 | (PoC requirement) all cores used, deterministic | goal | T13, T15 `uses_all_cores_for_speedup`; T17 |

**Not reproducible by design:** PG-19 and ImageNet accuracy (Tables 2–4) need full training on GPUs. Absolute A100 timings against FlashAttention-2 are GPU-specific; C7 and C8 reproduce the *trend* instead. The "RFFT ≈1.8× faster" claim is not reproduced because this PoC builds the RFFT from a full complex FFT. The "<6% parameters" claim depends on the host model. The learned WRM skip controller needs training. T17's report lists all of these with reasons.

---

## 6. Tickets

**Command conventions.** If `cargo` is not on `PATH`, prepend the rustup toolchain directory (T0 step 1). "Expected N passed" means the `test result:` line for the unit-test binary, filtered by module.

### T0 — Toolchain, scaffold, op counters

**Why:** §1.9. Op counting must exist before any primitive is written, because every primitive reports its cost as it is built. Counters compile to nothing unless `--features opcount` is set, so timing is unaffected.

**Files:** create `Cargo.toml`, `Cargo.lock` (copied), `.gitignore`, `src/lib.rs`, `src/opcount.rs`. Spec: `reference/spec-code/Cargo.toml` **without** its trailing `[[bin]]` section (T17 adds it), and `reference/spec-code/src/opcount.rs`.

**Interfaces produced:** `enum Op { Butterfly, Mac, CMul, Haar }`; `struct Snapshot { butterfly, mac, cmul, haar: u64 }` with `total()`; `fn add(op: Op, n: u64)`; `fn reset()`; `fn snapshot() -> Snapshot`; `fn lock() -> MutexGuard<'static, ()>`; `fn measure<R>(f: impl FnOnce() -> R) -> (R, Snapshot)`.

- [ ] **1. Toolchain.** Run `cargo --version`. If it is not found: `export PATH="$HOME/.cargo/bin:$(ls -d $HOME/.rustup/toolchains/*/bin | head -1):$PATH"` and retry. If there is still no cargo: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && source "$HOME/.cargo/env"`. You need rustc ≥ 1.75.
- [ ] **2. Git.** If `git rev-parse` fails, run `git init`. Write `.gitignore` containing `target/` (unanchored, so it also covers `reference/spec-code/target/`).
- [ ] **3.** Copy `reference/spec-code/Cargo.toml` to `./Cargo.toml` and delete its last three lines (the `[[bin]]` block). Copy `reference/spec-code/Cargo.lock` to `./Cargo.lock`.
- [ ] **4. Red.** Create `src/opcount.rs` containing only the spec file's `#[cfg(test)] mod tests` block. Create `src/lib.rs`:
  ```rust
  //! SPECTRE (arXiv:2502.18394) proof-of-concept.
  pub mod opcount;
  pub use num_complex::Complex64 as C64;
  ```
  Run `cargo test opcount::`. Expected: compile errors (`lock`, `measure`, `add` not found).
- [ ] **5. Green.** Add the implementation above the tests (spec `src/opcount.rs`). Run `cargo test opcount::` and expect `1 passed`. Run `cargo test --features opcount opcount::` and expect `1 passed`.
- [ ] **6.** `git add -A && git commit -m "T0: scaffold crate and operation counters"`

### T1 — Dense matrices and test utilities

**Why:** everything is matrices ($X$, $Q$, $V$, $\hat V$). `matmul` is the $O(mkn)$ primitive behind the projections. It is parallel over output rows, and each row is one sequential loop (§1.10). `testutil` gives seeded inputs so failures are reproducible.

**Files:** `src/tensor.rs`, `src/testutil.rs`; add `pub mod tensor;` and `pub mod testutil;` to `lib.rs`. Spec: the same paths under `reference/spec-code/`.

**Interfaces produced:**
- `Mat { rows, cols, data: Vec<f64> }` (row-major) with `zeros`, `from_vec`, `from_fn`, `random(rows, cols, scale, rng)`, `get/set/row/row_mut/col`, `from_cols(&[Vec<f64>])`, `matmul(&Mat) -> Mat` (counts `m·k·n` Mac), `vec_mul(&[f64]) -> Vec<f64>` (x·M), `slice_rows`, `slice_cols`, `pad_rows`, `hconcat(&[Mat])`, `col_means`, `add`, `max_abs_diff`.
- `CMat { rows, cols, data: Vec<C64> }` with `zeros/get/set/row/from_cols/col/add/max_abs_diff`.
- `testutil`: `PAPER_SEED = 42`, `rng(seed) -> ChaCha8Rng`, `rand_mat(rows, cols, seed)`, `rand_vec(n, seed)`, `rand_cvec(n, seed)`, `max_abs_diff_c`, `max_abs_diff_r`.

- [ ] **Red:** write the tests block of `tensor.rs`, then write `testutil.rs` in full (it is a helper, not logic under test). Run `cargo test tensor::` and expect a compile failure.
- [ ] **Green:** write the implementation. Run `cargo test tensor::` and expect **4 passed** (`matmul_small_known_values`, `matmul_matches_naive_on_random`, `vec_mul_matches_matmul`, `slice_pad_concat_and_means`).
- [ ] Regression gate: `cargo test`.
- [ ] `git commit -am "T1: dense matrices and test utilities"` (use `git add -A` first so new files are included).

### T2 — Complex radix-2 FFT and DFT oracle

**Why:** §1.3. This is the only $O(n\log n)$ primitive, and every complexity claim rests on it. The oracle `naive_dft` is the definition, eq. 1, evaluated directly in $O(n^2)$ with $(kt \bmod n)$ for accurate angles.

**Files:** `src/fft.rs`; add `pub mod fft;`. Spec: `reference/spec-code/src/fft.rs`.

**Interfaces produced:** `is_power_of_two(usize) -> bool`; `fft_in_place(&mut [C64], inverse: bool)` (unnormalized, $\tfrac N2\log_2N$ Butterfly counts, panics with "not a power of two"); `fft(&[C64]) -> Vec<C64>`; `ifft(&[C64]) -> Vec<C64>` (divides by $n$); `naive_dft(&[C64], inverse) -> Vec<C64>`.

**Implementation notes (from the equation):**
- Bit-reversal permutation first.
- Twiddle table $w_j=e^{\mp j2\pi j/N}$ computed directly for $j<N/2$. Do not use repeated multiplication, because error accumulates.
- Stage `len` uses stride $N/\text{len}$.
- Add $N/2$ butterflies to the counter once per stage.

- [ ] Red → green. `cargo test fft::` must show **7 passed**: `power_of_two_detection`, `impulse_transforms_to_all_ones`, `matches_naive_dft_for_all_sizes_up_to_1024` (tolerance $10^{-12}N$), `roundtrip_is_identity`, `parseval_energy_identity`, `linearity`, `rejects_non_power_of_two`.
- [ ] Regression gate; `git add -A && git commit -m "T2: radix-2 FFT with DFT oracle"`.

### T3 — Real FFT and column transforms

**Why:** §1.3, Appendix B. The RFFT is the paper's $\mathcal R_N$. The column versions apply it independently to each of the $d$ channels, **in parallel over columns**. That is the main source of multi-core parallelism for the layer.

**Files:** `src/rfft.rs`; add `pub mod rfft;`. Spec: `reference/spec-code/src/rfft.rs`.

**Interfaces produced:**
- `num_bins(n) -> n/2+1`.
- `rfft(&[f64]) -> Vec<C64>`.
- `irfft(&[C64], n) -> Vec<f64>`. It rebuilds the Hermitian spectrum, forces DC and Nyquist to be real, and normalizes by $1/n$.
- `naive_rfft(&[f64])`.
- `rfft_cols(&Mat, n_fft) -> CMat`, which zero-pads rows to `n_fft`; output shape $F\times$ cols.
- `irfft_cols(&CMat, n_fft) -> Mat` ($n_{fft}\times$ cols).

- [ ] Red → green. `cargo test rfft::` must show **6 passed**, including `hermitian_symmetry_theorem_b1`, `roundtrip_is_lossless` and `irfft_ignores_imag_of_dc_and_nyquist`.
- [ ] Regression gate; commit `"T3: real FFT (paper eq. 1, Appendix B)"`.

### T4 — Neural building blocks

**Why:** the gate (eq. 3a) and the WRM (§3.5) are "two-layer MLPs on LN(q̄)". These are plain, well-known functions. They are tested against reference values so that later gate bugs cannot hide here.

**Files:** `src/nn.rs`; add `pub mod nn;`. Spec: `reference/spec-code/src/nn.rs`.

**Interfaces produced:**
- `seeded_rng(seed) -> ChaCha8Rng`.
- `Linear { w: Mat /*in×out*/, b }`: `new(in, out, rng)` (Xavier-uniform $a=\sqrt{6/(in+out)}$, zero bias), `forward_vec`, `forward`, `in_dim`, `out_dim`.
- `gelu(x)` (tanh approximation).
- `LayerNorm { gamma, beta, eps: 1e-5 }`: `new(dim)`, `forward_vec`.
- `Mlp2 { l1, l2 }`: `new(in, hidden, out, rng)`, `forward_vec`.

- [ ] Red → green. `cargo test nn::` must show **4 passed** (includes $\text{gelu}(1)=0.841192$).
- [ ] Regression gate; commit `"T4: Linear, LayerNorm, GELU, MLP"`.

### T5 — Spectral gate: MLP → Toeplitz → modReLU → phase

**Why:** §1.5. This is the "content-adaptive" part of SPECTRE. Each sub-step is a free function, so its defining property is tested alone:
- modReLU keeps the phase: $\arg$ is unchanged.
- The Toeplitz update is banded: an impulse spreads to at most $2r+1$ bins.
- The phase $e^{+j2\pi ks/N}$ rotates the time signal by $s$.

**Files:** `src/gate.rs`; add `pub mod gate;`. Spec: `reference/spec-code/src/gate.rs`.

**Interfaces produced:**
- `GateConfig { d_head, n_fft, hidden, toeplitz_r: Option<usize> }`.
- `SpectralGate { cfg, ln, mlp, toeplitz: Option<Vec<C64>>, modrelu_bias: Vec<f64> }`, with:
  - `new(cfg, rng)`: sets $b_2$ real parts to 1 (D4); Toeplitz kernel and bias start at 0 (D5).
  - `num_bins()`.
  - `raw(q_mean) -> Vec<C64>`.
  - `compute(q_mean, phase_shift) -> Vec<C64>`.
- Free functions:
  - `modrelu(z, b) -> C64`.
  - `toeplitz_update(g, t) -> Vec<C64>`: counts $F(2r+1)$ CMul, parallel over $k$.
  - `apply_phase(&mut g, shift, n_fft)`: counts $F$ CMul; uses $(k\cdot s) \bmod N$ for the angle.

- [ ] Red → green. `cargo test gate::` must show **6 passed**.
- [ ] Regression gate; commit `"T5: content-adaptive spectral gate (eq. 3)"`.

### T6 — Haar DWT and Wavelet Refinement Module

**Why:** §1.6. The DWT is linear and orthogonal, so its tests are exact identities: reconstruction, energy, and the $s=1\Rightarrow 2\tilde V$ and $s=0\Rightarrow\tilde V$ cases.

**Files:** `src/wavelet.rs`; add `pub mod wavelet;`. Spec: `reference/spec-code/src/wavelet.rs`.

**Interfaces produced:**
- `haar_forward(&[f64], levels) -> Vec<f64>` and `haar_inverse(...)`. The layout is `[a_J | d_J | … | d_1]`; each counts $N(1-2^{-J})$ Haar ops.
- `band_of_index(i, n, levels)`: 0 = approximation, $\ell$ = detail $d_\ell$.
- `Wrm { levels, d_head, ln, mlp }` with `new(d_head, levels, hidden, rng)`, `gates(q_mean) -> Mat` of shape $(J+1)\times d$, and `refine(&Mat, q_mean) -> Mat`.
- `refine_with_gates(&Mat, &Mat, levels) -> Mat` (parallel over columns).

- [ ] Red → green. `cargo test wavelet::` must show **6 passed**.
- [ ] Regression gate; commit `"T6: Haar DWT and wavelet refinement (§3.5)"`.

### T7 — SPECTRE head and its convolution oracle

**Why:** §1.4. This ticket assembles eqs. 2–4 and proves them against $\tilde V=h\circledast V$, which is computed without any FFT. It also verifies three paper claims: translation equivariance (C16), global receptive field (C4) and content adaptivity.

**Files:** `src/head.rs`; add `pub mod head;`. Spec: `reference/spec-code/src/head.rs`.

**Interfaces produced:**
- `HeadConfig { d_model, d_head, n_fft, gate_hidden, toeplitz_r: Option<usize>, wavelet_levels: Option<usize> }`.
- `SpectreHead { cfg, wq, wv, gate, wrm: Option<Wrm> }`, with:
  - `new(cfg, rng)`: asserts a power of two; draws $W_q$, $W_v$, then the gate, then the WRM, in this order from the same rng.
  - `project(&Mat) -> (Q, V)`.
  - `project_token(&[f64]) -> (q, v)`.
  - `mix_spectrum(&CMat, q_mean, phase_shift) -> Mat` ($N\times d$).
  - `mix(&Mat, q_mean, phase_shift) -> Mat`.
  - `forward(&Mat) -> Mat`: requires $1\le n\le N$; uses shift 0; applies the WRM when enabled; returns the first $n$ rows.
- `apply_gate(&CMat, &[C64]) -> CMat`: row $k$ times $g_k$; counts $F\cdot d$ CMul.
- `circular_conv_reference(&Mat, &[f64]) -> Mat`.
- `forward_reference(&SpectreHead, &Mat) -> Mat`: $h=\mathcal R^{-1}(g)$, then convolution, then the WRM.

- [ ] Red → green. `cargo test head::` must show **8 passed**, including `matches_circular_convolution_oracle` (padded and unpadded cases, $<10^{-10}$) and `all_pass_gate_returns_v`.
- [ ] Regression gate; commit `"T7: SPECTRE head (eqs. 2-4) with convolution oracle"`.

### T8 — Multi-head layer (the drop-in)

**Why:** §3.6 "drop-in". The layer maps $n\times D\to n\times D$ exactly like attention. Heads run in parallel. Parameters are drawn sequentially from one seed, so they are independent of the thread count.

**Files:** `src/layer.rs`; add `pub mod layer;`. Spec: `reference/spec-code/src/layer.rs`.

**Interfaces produced:** `LayerConfig { d_model, n_heads, n_fft, gate_hidden, toeplitz_r, wavelet_levels }` with `head_config()`; `SpectreLayer { cfg, heads, wo: Linear }` with `new(cfg, seed)` and `forward(&Mat) -> Mat`.

- [ ] Red → green. `cargo test layer::` must show **3 passed**.
- [ ] Regression gate; commit `"T8: multi-head SPECTRE layer"`.

### T9 — Naive attention baseline

**Why:** §1.2. This is the $O(n^2d)$ reference that SPECTRE must beat asymptotically. The softmax subtracts the row maximum for stability.

**Files:** `src/attention.rs`; add `pub mod attention;`. Spec: `reference/spec-code/src/attention.rs`.

**Interfaces produced:** `NaiveAttention { n_heads, wq, wk, wv, wo }` with `new(d_model, n_heads, seed)` and `forward(&Mat) -> Mat`; `attention_head(q, k, v) -> Mat` (parallel over query rows; counts $2 n_q n_k d$ Mac).

- [ ] Red → green. `cargo test attention::` must show **4 passed**.
- [ ] Regression gate; commit `"T9: naive softmax attention baseline"`.

### T10 — Prefix-FFT cache and persistent memory

**Why:** §1.7–1.8. The tests encode the invariant $\texttt{prefix\_fft}=\mathcal R_N(\texttt{V\_buf})$ after every step, through wrap-around. They also check exact equality with recomputation from scratch, constant memory, and a static memory spectrum.

**Files:** `src/cache.rs`; add `pub mod cache;`. Spec: `reference/spec-code/src/cache.rs`.

**Interfaces produced:** `PrefixFftCache { n_fft, n_mem, n_max, d, prefix_fft: CMat, mem_fft: Option<CMat>, v_buf, q_buf, sum_q, twiddles, t }` with:
- `new(&SpectreHead, memory: Option<&Mat>)`.
- `live_len()`.
- `placed_window() -> Mat`.
- `prefill(&head, &Mat) -> Mat`: requires $1\le L\le N_{\max}$; only valid as the first call (D9).
- `update_spectrum(slot, &delta)`: eq. 5; counts $F\cdot d$ CMul.
- `decode_step(&head, &[f64]) -> Mat`.
- `output(&head) -> Mat`: D6 and D10.
- `footprint_bytes() -> usize`.

- [ ] Red → green. `cargo test cache::` must show **7 passed**. The footprint test pins the exact formula $17\cdot4\cdot16+(2\cdot32\cdot4+4)\cdot8+32\cdot16$ bytes.
- [ ] Regression gate; commit `"T10: Prefix-FFT cache and persistent memory (§3.3-3.4)"`.

### T11 — Complexity helpers

**Why:** §1.9. The slope and spread estimators are themselves tested on exact power laws, so a complexity verdict can never come from a broken estimator.

**Files:** `src/complexity.rs`; add `pub mod complexity;`. Spec: `reference/spec-code/src/complexity.rs`.

**Interfaces produced:** `loglog_slope(xs, ys) -> f64`; `nlog2n(n) -> f64`; `spread(xs, ys, model) -> f64`; `median(&mut [f64]) -> f64`; `time_median(reps, f) -> f64` (one warm-up run, returns seconds).

- [ ] Red → green. `cargo test complexity::` must show **4 passed** (a quadratic gives slope 2, a linear gives 1, and $n\log n$ gives a slope in (1.05, 1.15)).
- [ ] Regression gate; commit `"T11: complexity estimation helpers"`.

### T12 — Cross-verification against the paper's literal formulas

**Why:** D6 and D8 deviate from the paper's *wording* while keeping its *math*. This ticket proves that claim by running the literal formulas side by side with the implementation.

**Files:** `tests/paper_literal.rs` (copy the spec file). No library code.

- [ ] Run `cargo test --release --test paper_literal` and expect **4 passed**:
  - `eq5_literal_matches_cache_update`: eq. 5 with both twiddle terms equals our cache at every step, through wrap-around.
  - `eviction_and_insertion_twiddles_coincide`.
  - `algorithm1_literal_phase_is_our_output_rotated_by_one_row`.
  - `prefill_is_single_padded_rfft`.
- [ ] Commit `"T12: literal paper-formula cross-checks"`.

### T13 — Parallelism: all cores, bitwise determinism

**Why:** §1.10 and the goal's "utilize all CPU cores" requirement.

**Files:** `tests/parallel.rs` (copy the spec file).

- [ ] Run `cargo test --release --test parallel` and expect **4 passed**. If `global_pool_uses_every_core` fails, run `unset RAYON_NUM_THREADS` and retry.
- [ ] Commit `"T13: all-core and determinism tests"`.

### T14 — Complexity order by exact operation counts

**Why:** §1.9. These are hardware-independent proofs of every $O(\cdot)$ statement in the paper, plus exact formulas for each primitive.

**Files:** `tests/opcount.rs` (copy the spec file; it starts with `#![cfg(feature = "opcount")]`).

- [ ] Run `cargo test --release --features opcount --test opcount -- --nocapture` and expect **12 passed**. Reference numbers from the spec run:
  - SPECTRE layer slope 1.011.
  - Attention slope ≈1.9.
  - Decode slope 1.058, prefill slope 1.018.
  - At $n=4096$: SPECTRE 15.0M ops vs attention 1.09G ops (72× fewer).
- [ ] Also run `cargo test --features opcount` (whole suite with counters on). Everything must pass.
- [ ] Commit `"T14: exact op-count complexity validation"`.

### T15 — Wall-clock complexity and speed-up

**Why:** proves that the implementation, not only the algorithm, has the claimed order and uses the cores (§1.9, point 3).

**Files:** `tests/timing.rs` (copy the spec file). All tests are `#[ignore]`.

- [ ] Run `cargo test --release --test timing -- --ignored --test-threads=1 --nocapture` and expect **5 passed**. The run takes about 40 s. Reference results on 8 cores:
  - SPECTRE slope 1.2 (band 0.8–1.4).
  - Attention slope 1.8–1.95 (band 1.6–2.4).
  - 20× faster than attention at $n=4096$.
  - Decode max/min time ratio < 2.
  - Speed-up 3.3× on 8 threads (threshold $0.3\cdot$cores).
- [ ] On failure, apply §0 rung 4 only.
- [ ] Commit `"T15: wall-clock complexity and parallel speed-up"`.

### T16 — Proof that it works: learning demo

**Why:** §1.11. Correct arithmetic does not prove usefulness. This ticket shows that the content-adaptive gate learns a content-dependent global operation to near zero error. A content-independent (FNet-style) spectral filter provably cannot: it plateaus at its computed optimum.

**Files:** `src/train.rs` (add `pub mod train;`), `tests/learning.rs`. Spec: the same paths.

**Interfaces produced:**
- `SHIFTS = [1, 5]`.
- `Sample { x, y }`.
- `delay_rows(&Mat, s)`.
- `make_shift_task(n, d, count, seed)`.
- `task_head(n, d, hidden, seed)` (identity $W_q$ and $W_v$).
- `Grad { w, b }`.
- `loss_and_grad(&head, &Sample)`: asserts no Toeplitz and a zero modReLU bias.
- `batch_loss_and_grad`: parallel per sample, then an in-order sum.
- `normalized_loss`.
- `Adam::new(len, lr)`, `Adam::step`.
- `train_gate(&mut head, data, steps, lr, adaptive) -> Vec<f64>`.

- [ ] Red → green on unit tests: `cargo test train::` must show **4 passed**. The critical one is `analytic_gradient_matches_finite_differences` (tolerance $10^{-6}$). If it fails, the gradient formula in §1.11 is mistranscribed, most likely the $w_k$ weights or the sign of the imaginary part.
- [ ] Run `cargo test --release --test learning -- --nocapture` and expect **2 passed**. The printed numbers should be close to:
  - adaptive test loss 1.19 → 0.0003;
  - fixed gate 0.29, against a theoretical optimum of 0.28.
- [ ] Regression gate; commit `"T16: learning demo proves content-adaptive gating works"`.

### T17 — Claims verification report and final gate

**Why:** §5. One command re-measures every reproducible claim and writes a dated verdict table that people can read.

**Files:** append to `Cargo.toml`:
```toml

[[bin]]
name = "claims_report"
required-features = ["opcount"]
```
Create `src/bin/claims_report.rs` (spec file).

- [ ] Run `cargo run --release --features opcount --bin claims_report -- docs/claims-report.md`. It runs in about 10 s. Every row must be **PASS** and the file must end with `**Overall: ALL REPRODUCIBLE CLAIMS PASS**`. The process exits 1 if any row fails; handle that with §0.
- [ ] Final full gate, all must pass:
  - `cargo build --all-targets --features opcount` (0 warnings);
  - `cargo test`;
  - `cargo test --release --features opcount`;
  - `cargo test --release --test timing -- --ignored --test-threads=1`.
- [ ] `git add -A && git commit -m "T17: claims verification report"`.

---

## 7. Final acceptance checklist

- [ ] T0–T17 committed in order; `git log --oneline` shows 18 ticket commits.
- [ ] `cargo test`: 64 unit tests plus 2 learning, 4 paper-literal and 4 parallel tests pass.
- [ ] `cargo test --release --features opcount --test opcount`: 12 passed.
- [ ] Timing suite: 5 passed (or has a documented, ≤25%-widened threshold in `docs/deviations.md`).
- [ ] `docs/claims-report.md` exists, and every row is PASS.
- [ ] No test deleted or newly ignored; no accuracy tolerance changed (check with `git diff reference/spec-code` and by comparing tolerance literals against the spec files).
- [ ] `docs/deviations.md` exists if anything deviated. Otherwise note "no deviations" in the final commit message.

## 8. Out of scope, and the next phase

- **Performance** (SIMD, split-radix or packed real FFT, f32/fp16, GPU, other platforms). This is explicitly the *next* phase. The op-count and slope harness from T14/T15/T17 is the regression baseline for it.
- **Real pretrained models.** SPECTRE is not training-free: the paper fine-tunes the new gate weights (§3.6). A pretrained checkpoint with SPECTRE swapped in, and no training, would give meaningless output. Validating on a real model is therefore a separate phase that needs autograd (for example `candle` or `burn`) and a small checkpoint. Good candidates under 1B parameters:
  - Pythia-14M / 70M;
  - GPT-2 small (124M);
  - SmolLM2-135M.

  The procedure would be: swap the attention layers for SPECTRE, freeze the rest, fine-tune only the gates, and compare perplexity. This PoC's T16 demo is the minimal, fully self-contained version of that experiment.
