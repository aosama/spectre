//! One SPECTRE mixing head (paper §3.2):
//!   1. Q = X·W_q, V = X·W_v                      (token projection, eq. 2)
//!   2. V̂ = R_{n_fft}(pad(V))                     (spectral transform, eq. 3)
//!   3. g = gate(mean_i q_i)                      (content-adaptive gate)
//!   4. Ṽ = R^{-1}_{n_fft}(diag(g)·V̂)             (inverse transform, eq. 4)
//!   5. optional WRM: Ṽ ← Ṽ + W^{-1}(s ⊙ W(Ṽ))    (§3.5)
//! Sequences shorter than n_fft are zero-padded; the first n rows are returned.

use crate::gate::{GateConfig, SpectralGate};
use crate::opcount::{self, Op};
use crate::rfft::{irfft, irfft_cols, rfft_cols};
use crate::tensor::{CMat, Mat};
use crate::wavelet::Wrm;
use crate::C64;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct HeadConfig {
    pub d_model: usize,
    pub d_head: usize,
    /// FFT length (power of two). Maximum sequence length for `forward`.
    pub n_fft: usize,
    pub gate_hidden: usize,
    pub toeplitz_r: Option<usize>,
    /// Haar levels for the WRM; None disables the WRM.
    pub wavelet_levels: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct SpectreHead {
    pub cfg: HeadConfig,
    /// d_model × d_head
    pub wq: Mat,
    /// d_model × d_head
    pub wv: Mat,
    pub gate: SpectralGate,
    pub wrm: Option<Wrm>,
}

impl SpectreHead {
    pub fn new(cfg: HeadConfig, rng: &mut impl Rng) -> Self {
        assert!(crate::fft::is_power_of_two(cfg.n_fft), "n_fft must be a power of two");
        let a = (6.0 / (cfg.d_model + cfg.d_head) as f64).sqrt();
        let wq = Mat::random(cfg.d_model, cfg.d_head, a, rng);
        let wv = Mat::random(cfg.d_model, cfg.d_head, a, rng);
        let gate = SpectralGate::new(
            GateConfig { d_head: cfg.d_head, n_fft: cfg.n_fft, hidden: cfg.gate_hidden, toeplitz_r: cfg.toeplitz_r },
            rng,
        );
        let wrm = cfg.wavelet_levels.map(|l| Wrm::new(cfg.d_head, l, cfg.gate_hidden, rng));
        SpectreHead { cfg, wq, wv, gate, wrm }
    }

    /// Step 1 for a batch of tokens: (Q, V), each n × d_head.
    pub fn project(&self, x: &Mat) -> (Mat, Mat) {
        (x.matmul(&self.wq), x.matmul(&self.wv))
    }

    /// Step 1 for a single token: (q, v), each of length d_head.
    pub fn project_token(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        (self.wq.vec_mul(x), self.wv.vec_mul(x))
    }

    /// Steps 3–4 on an existing spectrum ((n_fft/2+1) × d_head) → n_fft × d_head.
    pub fn mix_spectrum(&self, spec: &CMat, q_mean: &[f64], phase_shift: usize) -> Mat {
        let g = self.gate.compute(q_mean, phase_shift);
        irfft_cols(&apply_gate(spec, &g), self.cfg.n_fft)
    }

    /// Steps 2–4: V (n × d_head, n ≤ n_fft) → Ṽ (n_fft × d_head).
    pub fn mix(&self, v: &Mat, q_mean: &[f64], phase_shift: usize) -> Mat {
        self.mix_spectrum(&rfft_cols(v, self.cfg.n_fft), q_mean, phase_shift)
    }

    /// Full non-causal head: X (n × d_model) → n × d_head.
    pub fn forward(&self, x: &Mat) -> Mat {
        assert!(x.rows >= 1 && x.rows <= self.cfg.n_fft, "forward: need 1 ≤ n ≤ n_fft");
        let (q, v) = self.project(x);
        let q_mean = q.col_means();
        let mut out = self.mix(&v, &q_mean, 0);
        if let Some(w) = &self.wrm {
            out = w.refine(&out, &q_mean);
        }
        out.slice_rows(0, x.rows)
    }
}

/// diag(g)·spec: row k (frequency bin) of every channel is multiplied by g_k.
/// Parallel over bins; counts F·d `Op::CMul`.
pub fn apply_gate(spec: &CMat, g: &[C64]) -> CMat {
    assert_eq!(spec.rows, g.len(), "apply_gate: gate length must equal number of bins");
    let mut out = spec.clone();
    out.data.par_chunks_mut(spec.cols.max(1)).zip(g.par_iter()).for_each(|(row, gk)| {
        row.iter_mut().for_each(|v| *v *= gk);
    });
    opcount::add(Op::CMul, (spec.rows * spec.cols) as u64);
    out
}

/// O(N²·d) time-domain oracle: out[m][c] = Σ_s h[s]·v[(m − s) mod N][c].
pub fn circular_conv_reference(v: &Mat, h: &[f64]) -> Mat {
    let n = v.rows;
    assert_eq!(h.len(), n, "circular_conv_reference: kernel length must equal rows");
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|m| {
            let mut acc = vec![0.0; v.cols];
            for (s, &hs) in h.iter().enumerate() {
                let src = v.row((m + n - s) % n);
                acc.iter_mut().zip(src).for_each(|(a, x)| *a += hs * x);
            }
            opcount::add(Op::Mac, (n * v.cols) as u64);
            acc
        })
        .collect();
    Mat::from_vec(n, v.cols, rows.concat())
}

/// FFT-free oracle for `SpectreHead::forward`: the gate is a real circular
/// convolution kernel h = R^{-1}(g) (convolution theorem).
pub fn forward_reference(head: &SpectreHead, x: &Mat) -> Mat {
    let n_fft = head.cfg.n_fft;
    let (q, v) = head.project(x);
    let q_mean = q.col_means();
    let h = irfft(&head.gate.compute(&q_mean, 0), n_fft);
    let mut out = circular_conv_reference(&v.pad_rows(n_fft), &h);
    if let Some(w) = &head.wrm {
        out = w.refine(&out, &q_mean);
    }
    out.slice_rows(0, x.rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{rand_cvec, rand_mat, rand_vec, rng};

    fn cfg(n_fft: usize, r: Option<usize>, wl: Option<usize>) -> HeadConfig {
        HeadConfig { d_model: 12, d_head: 4, n_fft, gate_hidden: 16, toeplitz_r: r, wavelet_levels: wl }
    }

    /// A head whose gate has random, non-trivial Toeplitz and modReLU parameters.
    fn busy_head(n_fft: usize, wl: Option<usize>) -> SpectreHead {
        let mut h = SpectreHead::new(cfg(n_fft, Some(2), wl), &mut rng(1));
        h.gate.toeplitz = Some(rand_cvec(5, 2).iter().map(|v| v * 0.3).collect());
        h.gate.modrelu_bias = rand_vec(n_fft / 2 + 1, 3).iter().map(|v| v * 0.1).collect();
        h
    }

    #[test]
    fn output_shapes() {
        let h = SpectreHead::new(cfg(64, None, None), &mut rng(1));
        assert_eq!(h.forward(&rand_mat(64, 12, 2)).rows, 64);
        let y = h.forward(&rand_mat(40, 12, 2));
        assert_eq!((y.rows, y.cols), (40, 4));
    }

    #[test]
    fn apply_gate_scales_each_bin() {
        let mut spec = CMat::zeros(3, 2);
        spec.data = rand_cvec(6, 4);
        let g = rand_cvec(3, 5);
        let out = apply_gate(&spec, &g);
        for k in 0..3 {
            for c in 0..2 {
                assert!((out.get(k, c) - spec.get(k, c) * g[k]).norm() < 1e-15);
            }
        }
    }

    #[test]
    fn matches_circular_convolution_oracle() {
        for (n_fft, n) in [(64, 64), (64, 40), (128, 1)] {
            let h = busy_head(n_fft, None);
            let x = rand_mat(n, 12, 6);
            let d = h.forward(&x).max_abs_diff(&forward_reference(&h, &x));
            assert!(d < 1e-10, "n_fft={n_fft} n={n} diff={d}");
        }
    }

    #[test]
    fn matches_oracle_with_wavelet_refinement() {
        let h = busy_head(64, Some(3));
        let x = rand_mat(64, 12, 7);
        assert!(h.forward(&x).max_abs_diff(&forward_reference(&h, &x)) < 1e-10);
    }

    #[test]
    fn all_pass_gate_returns_v() {
        let mut h = SpectreHead::new(cfg(32, None, None), &mut rng(8));
        h.gate.mlp.l2.w = Mat::zeros(h.gate.mlp.l2.w.rows, h.gate.mlp.l2.w.cols); // g ≡ 1 (bias init)
        let x = rand_mat(32, 12, 9);
        let (_, v) = h.project(&x);
        assert!(h.forward(&x).max_abs_diff(&v) < 1e-12);
    }

    // §3.5: "the real FFT is translation-equivariant". Rolling the tokens rolls the output.
    #[test]
    fn circular_translation_equivariance() {
        let h = busy_head(32, None);
        let x = rand_mat(32, 12, 10);
        let s = 5;
        let rolled = Mat::from_fn(32, 12, |r, c| x.get((r + 32 - s) % 32, c));
        let y = h.forward(&x);
        let yr = h.forward(&rolled);
        let expected = Mat::from_fn(32, 4, |r, c| y.get((r + 32 - s) % 32, c));
        assert!(yr.max_abs_diff(&expected) < 1e-10);
    }

    // Global receptive field: changing token 0 changes every output row.
    #[test]
    fn global_receptive_field() {
        let h = busy_head(32, None);
        let x = rand_mat(32, 12, 11);
        let mut x2 = x.clone();
        x2.row_mut(0).iter_mut().for_each(|v| *v += 1.0);
        let (y, y2) = (h.forward(&x), h.forward(&x2));
        for r in 0..32 {
            let d: f64 = (0..4).map(|c| (y.get(r, c) - y2.get(r, c)).abs()).sum();
            assert!(d > 1e-9, "row {r} unaffected");
        }
    }

    // Content adaptivity: the gate depends on the input's query descriptor.
    #[test]
    fn gate_is_content_adaptive() {
        let h = busy_head(32, None);
        let (q1, _) = h.project(&rand_mat(32, 12, 12));
        let (q2, _) = h.project(&rand_mat(32, 12, 13));
        let g1 = h.gate.compute(&q1.col_means(), 0);
        let g2 = h.gate.compute(&q2.col_means(), 0);
        assert!(crate::testutil::max_abs_diff_c(&g1, &g2) > 1e-3);
    }
}
