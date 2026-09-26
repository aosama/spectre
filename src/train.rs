//! "It actually works" demo: train the spectral gate on a synthetic task that
//! requires content-adaptive global mixing, with analytic gradients (no
//! autograd). Only the gate MLP's output layer (l2) is trained; with modReLU
//! bias 0, no Toeplitz kernel and phase 0 the output is linear in l2's
//! parameters, so the problem is a convex least-squares fit.
//!
//! Task: each sequence carries a class c ∈ {0, 1} in the mean of channel 0.
//! Target: the value sequence circularly delayed by SHIFTS[c] positions.
//! A content-independent (FNet-style) filter cannot fit both delays at once:
//! its best normalized loss is 0.5. A content-adaptive gate can reach ≈ 0.

use crate::head::{apply_gate, HeadConfig, SpectreHead};
use crate::nn::gelu;
use crate::rfft::{irfft_cols, rfft_cols};
use crate::tensor::Mat;
use crate::testutil::rng;
use crate::C64;
use rand::Rng;
use rayon::prelude::*;

pub const SHIFTS: [usize; 2] = [1, 5];

#[derive(Clone, Debug)]
pub struct Sample {
    pub x: Mat,
    pub y: Mat,
}

/// y[m] = v[(m − s) mod n]: circular delay by s rows.
pub fn delay_rows(v: &Mat, s: usize) -> Mat {
    let n = v.rows;
    Mat::from_fn(n, v.cols, |m, c| v.get((m + n - s % n) % n, c))
}

/// `count` samples of n × d tokens; sample i has class i % 2.
/// Channel 0 is offset by +1 (class 0) or −1 (class 1) so the class is
/// visible in the query descriptor. With identity projections V = X.
pub fn make_shift_task(n: usize, d: usize, count: usize, seed: u64) -> Vec<Sample> {
    let mut r = rng(seed);
    (0..count)
        .map(|i| {
            let class = i % 2;
            let offset = if class == 0 { 1.0 } else { -1.0 };
            let mut x = Mat::zeros(n, d);
            for row in 0..n {
                for c in 0..d {
                    x.set(row, c, r.gen_range(-1.0..=1.0) + if c == 0 { offset } else { 0.0 });
                }
            }
            let y = delay_rows(&x, SHIFTS[class]);
            Sample { x, y }
        })
        .collect()
}

/// A head suited to the task: identity projections, no Toeplitz, no WRM.
pub fn task_head(n: usize, d: usize, hidden: usize, seed: u64) -> SpectreHead {
    let cfg = HeadConfig { d_model: d, d_head: d, n_fft: n, gate_hidden: hidden, toeplitz_r: None, wavelet_levels: None };
    let mut h = SpectreHead::new(cfg, &mut rng(seed));
    let eye = Mat::from_fn(d, d, |r, c| if r == c { 1.0 } else { 0.0 });
    h.wq = eye.clone();
    h.wv = eye;
    h
}

/// Gradients of the loss w.r.t. the gate MLP output layer.
#[derive(Clone, Debug)]
pub struct Grad {
    /// Same layout as `gate.mlp.l2.w.data` (hidden × 2F, row-major).
    pub w: Vec<f64>,
    /// Same layout as `gate.mlp.l2.b` (2F).
    pub b: Vec<f64>,
}

/// Head output for a full-length sample (n = n_fft), phase 0.
pub fn predict(head: &SpectreHead, x: &Mat) -> Mat {
    head.forward(x)
}

/// Per-sample loss ||Ṽ − Y||² / (n·d) and its gradient w.r.t. l2.
///
/// Derivation: Ṽ = R^{-1}(g ⊙ V̂). With G = ∂L/∂Ṽ and Ĝ = R(G), for each bin k
/// let z_k = Σ_c V̂_kc · conj(Ĝ_kc) and w_k = 1 for k ∈ {0, n/2}, else 2. Then
///   ∂L/∂Re(g_k) = (w_k/n)·Re z_k,   ∂L/∂Im(g_k) = −(w_k/n)·Im z_k.
/// g = l2(h) with outputs [re_0, im_0, re_1, ...], so ∂L/∂l2.b = ∂L/∂g and
/// ∂L/∂l2.w[i][j] = h_i · ∂L/∂g_j.
pub fn loss_and_grad(head: &SpectreHead, s: &Sample) -> (f64, Grad) {
    let gate = &head.gate;
    assert!(gate.toeplitz.is_none(), "gradient assumes no Toeplitz kernel");
    assert!(gate.modrelu_bias.iter().all(|b| *b == 0.0), "gradient assumes modReLU bias 0");
    let n = head.cfg.n_fft;
    assert_eq!(s.x.rows, n, "samples must have n_fft rows");
    let (q, v) = head.project(&s.x);
    let q_mean = q.col_means();
    let h: Vec<f64> = gate.mlp.l1.forward_vec(&gate.ln.forward_vec(&q_mean)).into_iter().map(gelu).collect();
    let spec = rfft_cols(&v, n);
    let g = gate.compute(&q_mean, 0);
    let out = irfft_cols(&apply_gate(&spec, &g), n);
    let scale = 1.0 / (n * v.cols) as f64;
    let resid = Mat::from_fn(n, v.cols, |r, c| out.get(r, c) - s.y.get(r, c));
    let loss = resid.data.iter().map(|e| e * e).sum::<f64>() * scale;
    let gmat = Mat::from_vec(n, v.cols, resid.data.iter().map(|e| 2.0 * e * scale).collect());
    let ghat = rfft_cols(&gmat, n);
    let f = spec.rows;
    let mut db = vec![0.0; 2 * f];
    for k in 0..f {
        let z: C64 = (0..v.cols).map(|c| spec.get(k, c) * ghat.get(k, c).conj()).sum();
        let wk = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        db[2 * k] = wk / n as f64 * z.re;
        db[2 * k + 1] = -wk / n as f64 * z.im;
    }
    let mut dw = vec![0.0; h.len() * 2 * f];
    for (i, hi) in h.iter().enumerate() {
        for (j, dbj) in db.iter().enumerate() {
            dw[i * 2 * f + j] = hi * dbj;
        }
    }
    (loss, Grad { w: dw, b: db })
}

/// Mean loss and gradient over a batch: per-sample work in parallel, then a
/// sequential in-order sum (keeps results bitwise deterministic).
pub fn batch_loss_and_grad(head: &SpectreHead, data: &[Sample]) -> (f64, Grad) {
    let parts: Vec<(f64, Grad)> = data.par_iter().map(|s| loss_and_grad(head, s)).collect();
    let inv = 1.0 / data.len() as f64;
    let mut loss = 0.0;
    let mut acc = Grad { w: vec![0.0; parts[0].1.w.len()], b: vec![0.0; parts[0].1.b.len()] };
    for (l, g) in &parts {
        loss += l * inv;
        acc.w.iter_mut().zip(&g.w).for_each(|(a, v)| *a += v * inv);
        acc.b.iter_mut().zip(&g.b).for_each(|(a, v)| *a += v * inv);
    }
    (loss, acc)
}

/// Σ||Ṽ − Y||² / Σ||Y||² over the data set (0 = perfect, 1 = predicting zero).
pub fn normalized_loss(head: &SpectreHead, data: &[Sample]) -> f64 {
    let (num, den) = data
        .iter()
        .map(|s| {
            let out = predict(head, &s.x);
            let e: f64 = out.data.iter().zip(&s.y.data).map(|(a, b)| (a - b).powi(2)).sum();
            let y: f64 = s.y.data.iter().map(|b| b * b).sum();
            (e, y)
        })
        .fold((0.0, 0.0), |(a, b), (e, y)| (a + e, b + y));
    num / den
}

/// Plain Adam optimizer over one flat parameter vector.
#[derive(Clone, Debug)]
pub struct Adam {
    pub lr: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl Adam {
    pub fn new(len: usize, lr: f64) -> Self {
        Adam { lr, m: vec![0.0; len], v: vec![0.0; len], t: 0 }
    }

    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        self.t += 1;
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            params[i] -= self.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + eps);
        }
    }
}

/// Full-batch training of the gate output layer. With `adaptive = false` the
/// weight matrix is zeroed and frozen, so only the bias trains and the gate is
/// the same for every input (content-independent baseline).
/// Returns the batch loss before each step.
pub fn train_gate(head: &mut SpectreHead, data: &[Sample], steps: usize, lr: f64, adaptive: bool) -> Vec<f64> {
    if !adaptive {
        head.gate.mlp.l2.w.data.iter_mut().for_each(|w| *w = 0.0);
    }
    let mut opt_w = Adam::new(head.gate.mlp.l2.w.data.len(), lr);
    let mut opt_b = Adam::new(head.gate.mlp.l2.b.len(), lr);
    let mut history = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (loss, g) = batch_loss_and_grad(head, data);
        history.push(loss);
        if adaptive {
            opt_w.step(&mut head.gate.mlp.l2.w.data, &g.w);
        }
        opt_b.step(&mut head.gate.mlp.l2.b, &g.b);
    }
    history
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_rows_moves_row_zero_to_row_s() {
        let v = Mat::from_vec(4, 1, vec![1., 2., 3., 4.]);
        assert_eq!(delay_rows(&v, 1).data, vec![4., 1., 2., 3.]);
        assert_eq!(delay_rows(&v, 4), v);
    }

    #[test]
    fn exact_delay_gate_exists() {
        // g_k = e^{−j2πks/n} delays by s: the task is representable.
        let n = 16;
        let v = crate::testutil::rand_mat(n, 3, 1);
        let g: Vec<C64> = (0..=n / 2).map(|k| C64::from_polar(1.0, -2.0 * std::f64::consts::PI * (k * 5) as f64 / n as f64)).collect();
        let out = irfft_cols(&apply_gate(&rfft_cols(&v, n), &g), n);
        assert!(out.max_abs_diff(&delay_rows(&v, 5)) < 1e-12);
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let data = make_shift_task(16, 4, 1, 3);
        let mut head = task_head(16, 4, 8, 4);
        let (_, g) = loss_and_grad(&head, &data[0]);
        let eps = 1e-6;
        let fd = |head: &mut SpectreHead, pick: &dyn Fn(&mut SpectreHead) -> &mut f64| {
            *pick(head) += eps;
            let lp = loss_and_grad(head, &data[0]).0;
            *pick(head) -= 2.0 * eps;
            let lm = loss_and_grad(head, &data[0]).0;
            *pick(head) += eps;
            (lp - lm) / (2.0 * eps)
        };
        for j in [0usize, 1, 2, 3, 7, 16, 17] {
            let num = fd(&mut head, &|h: &mut SpectreHead| &mut h.gate.mlp.l2.b[j]);
            assert!((num - g.b[j]).abs() < 1e-6, "b[{j}]: fd {num} vs analytic {}", g.b[j]);
        }
        for idx in [0usize, 5, 40, 77] {
            let num = fd(&mut head, &|h: &mut SpectreHead| &mut h.gate.mlp.l2.w.data[idx]);
            assert!((num - g.w[idx]).abs() < 1e-6, "w[{idx}]: fd {num} vs analytic {}", g.w[idx]);
        }
    }

    #[test]
    fn adam_minimizes_a_quadratic() {
        let mut p = vec![5.0, -3.0];
        let mut opt = Adam::new(2, 0.1);
        for _ in 0..500 {
            let g: Vec<f64> = p.iter().map(|x| 2.0 * x).collect();
            opt.step(&mut p, &g);
        }
        assert!(p.iter().all(|x| x.abs() < 1e-2));
    }
}
