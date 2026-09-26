//! Content-adaptive spectral gate (paper §3.2 step 3 and "Positional Awareness"):
//!   q̄ = LN(mean_i q_i) → two-layer MLP → g ∈ C^{n_fft/2+1}
//!   → optional Toeplitz update g ← g + t * g (bandwidth 2r+1)
//!   → modReLU → positional phase g_k ← g_k·e^{+j2πk·shift/n_fft}.

use crate::nn::{LayerNorm, Mlp2};
use crate::opcount::{self, Op};
use crate::rfft::num_bins;
use crate::C64;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct GateConfig {
    /// Per-head channel dimension d (length of the query descriptor).
    pub d_head: usize,
    /// FFT length; the gate has n_fft/2+1 complex entries.
    pub n_fft: usize,
    /// Hidden width of the gate MLP.
    pub hidden: usize,
    /// Toeplitz half-bandwidth r (kernel length 2r+1); None disables it.
    pub toeplitz_r: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct SpectralGate {
    pub cfg: GateConfig,
    pub ln: LayerNorm,
    /// Outputs 2·F reals laid out [re_0, im_0, re_1, im_1, ...].
    pub mlp: Mlp2,
    /// Toeplitz kernel t ∈ C^{2r+1}, index j ↔ offset j − r. Zero at init.
    pub toeplitz: Option<Vec<C64>>,
    /// modReLU bias b ∈ R^F. Zero at init.
    pub modrelu_bias: Vec<f64>,
}

impl SpectralGate {
    pub fn new(cfg: GateConfig, rng: &mut impl Rng) -> Self {
        let f = num_bins(cfg.n_fft);
        let mut mlp = Mlp2::new(cfg.d_head, cfg.hidden, 2 * f, rng);
        // Initialize near the all-pass filter g ≈ 1 so the layer starts close to identity.
        for k in 0..f {
            mlp.l2.b[2 * k] = 1.0;
        }
        SpectralGate {
            cfg,
            ln: LayerNorm::new(cfg.d_head),
            mlp,
            toeplitz: cfg.toeplitz_r.map(|r| vec![C64::new(0.0, 0.0); 2 * r + 1]),
            modrelu_bias: vec![0.0; f],
        }
    }

    pub fn num_bins(&self) -> usize {
        num_bins(self.cfg.n_fft)
    }

    /// Step 3(a): LN of the (already averaged) query descriptor, then the MLP.
    pub fn raw(&self, q_mean: &[f64]) -> Vec<C64> {
        let out = self.mlp.forward_vec(&self.ln.forward_vec(q_mean));
        out.chunks_exact(2).map(|p| C64::new(p[0], p[1])).collect()
    }

    /// Full gate: raw → Toeplitz (if enabled) → modReLU → positional phase.
    pub fn compute(&self, q_mean: &[f64], phase_shift: usize) -> Vec<C64> {
        let mut g = self.raw(q_mean);
        if let Some(t) = &self.toeplitz {
            g = toeplitz_update(&g, t);
        }
        let mut g: Vec<C64> = g.par_iter().zip(&self.modrelu_bias).map(|(z, b)| modrelu(*z, *b)).collect();
        apply_phase(&mut g, phase_shift, self.cfg.n_fft);
        g
    }
}

/// modReLU(z) = ReLU(|z| + b)·z/|z|, and 0 when z = 0.
pub fn modrelu(z: C64, b: f64) -> C64 {
    let m = z.norm();
    if m == 0.0 {
        return C64::new(0.0, 0.0);
    }
    let s = m + b;
    if s > 0.0 {
        z * (s / m)
    } else {
        C64::new(0.0, 0.0)
    }
}

/// Step 3(b): g ← g + (t * g), a "same"-size zero-padded convolution along the
/// frequency axis: (t*g)_k = Σ_{j=-r}^{r} t_{j+r}·g_{k-j}. Parallel over k.
/// Counts F·(2r+1) `Op::CMul`.
pub fn toeplitz_update(g: &[C64], t: &[C64]) -> Vec<C64> {
    assert!(t.len() % 2 == 1, "toeplitz kernel length must be odd (2r+1)");
    let r = (t.len() / 2) as isize;
    let f = g.len() as isize;
    let out = (0..f)
        .into_par_iter()
        .map(|k| {
            let mut acc = g[k as usize];
            for j in -r..=r {
                let idx = k - j;
                if (0..f).contains(&idx) {
                    acc += t[(j + r) as usize] * g[idx as usize];
                }
            }
            acc
        })
        .collect();
    opcount::add(Op::CMul, (g.len() * t.len()) as u64);
    out
}

/// Multiply g_k by e^{+j2πk·shift/n_fft}. By the DFT shift theorem this makes
/// the inverse transform return y[m] = x[(m + shift) mod n_fft].
pub fn apply_phase(g: &mut [C64], shift: usize, n_fft: usize) {
    g.par_iter_mut().enumerate().for_each(|(k, v)| {
        let ang = 2.0 * PI * ((k * shift) % n_fft) as f64 / n_fft as f64;
        *v *= C64::from_polar(1.0, ang);
    });
    opcount::add(Op::CMul, g.len() as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rfft::{irfft, rfft};
    use crate::testutil::{max_abs_diff_c, max_abs_diff_r, rand_cvec, rand_vec, rng};

    fn cfg(r: Option<usize>) -> GateConfig {
        GateConfig { d_head: 8, n_fft: 32, hidden: 16, toeplitz_r: r }
    }

    #[test]
    fn gate_has_n_over_2_plus_1_complex_entries() {
        let g = SpectralGate::new(cfg(None), &mut rng(1));
        assert_eq!(g.compute(&rand_vec(8, 2), 0).len(), 17);
    }

    #[test]
    fn modrelu_properties() {
        let z = C64::new(3.0, 4.0); // |z| = 5
        assert!((modrelu(z, 0.0) - z).norm() < 1e-15);
        assert!((modrelu(z, 1.0) - z * 1.2).norm() < 1e-15);
        assert_eq!(modrelu(z, -5.0), C64::new(0.0, 0.0));
        assert_eq!(modrelu(C64::new(0.0, 0.0), 1.0), C64::new(0.0, 0.0));
        assert!((modrelu(z, 2.0).arg() - z.arg()).abs() < 1e-15, "phase preserved");
    }

    #[test]
    fn toeplitz_zero_kernel_is_identity_and_matches_naive() {
        let g = rand_cvec(17, 3);
        let zero = vec![C64::new(0.0, 0.0); 5];
        assert_eq!(toeplitz_update(&g, &zero), g);
        let t = rand_cvec(5, 4);
        let naive: Vec<C64> = (0..17i32)
            .map(|k| {
                let mut acc = g[k as usize];
                for (j, tj) in t.iter().enumerate() {
                    let idx = k - (j as i32 - 2);
                    if (0..17).contains(&idx) {
                        acc += tj * g[idx as usize];
                    }
                }
                acc
            })
            .collect();
        assert!(max_abs_diff_c(&toeplitz_update(&g, &t), &naive) < 1e-14);
    }

    #[test]
    fn toeplitz_is_banded() {
        // An impulse at bin 8 may only spread to bins 8-r..=8+r.
        let mut g = vec![C64::new(0.0, 0.0); 17];
        g[8] = C64::new(1.0, 0.0);
        let out = toeplitz_update(&g, &rand_cvec(5, 5));
        for (k, v) in out.iter().enumerate() {
            if !(6..=10).contains(&k) {
                assert_eq!(*v, C64::new(0.0, 0.0), "bin {k}");
            }
        }
    }

    #[test]
    fn phase_shift_rotates_time_signal() {
        let n = 32;
        let x = rand_vec(n, 6);
        for shift in [0usize, 1, 5, 31] {
            let mut g = vec![C64::new(1.0, 0.0); n / 2 + 1];
            apply_phase(&mut g, shift, n);
            let spec: Vec<C64> = rfft(&x).iter().zip(&g).map(|(a, b)| a * b).collect();
            let y = irfft(&spec, n);
            let expected: Vec<f64> = (0..n).map(|m| x[(m + shift) % n]).collect();
            assert!(max_abs_diff_r(&y, &expected) < 1e-12, "shift={shift}");
        }
    }

    #[test]
    fn toeplitz_path_changes_gate_only_when_enabled() {
        let q = rand_vec(8, 7);
        let mut with = SpectralGate::new(cfg(Some(2)), &mut rng(8));
        let without = SpectralGate::new(cfg(None), &mut rng(8));
        assert_eq!(with.compute(&q, 0), without.compute(&q, 0), "zero kernel = identity");
        with.toeplitz = Some(rand_cvec(5, 9));
        assert!(max_abs_diff_c(&with.compute(&q, 0), &without.compute(&q, 0)) > 1e-3);
    }
}
