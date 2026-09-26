//! Small neural building blocks: Linear, LayerNorm, GELU, two-layer MLP.
//! Weights are randomly initialized from a seeded RNG (no training in this PoC).

use crate::tensor::Mat;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Deterministic RNG used for all parameter initialization.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// y = x·W + b with W stored (in × out).
#[derive(Clone, Debug)]
pub struct Linear {
    pub w: Mat,
    pub b: Vec<f64>,
}

impl Linear {
    /// Xavier/Glorot-uniform weights, zero bias.
    pub fn new(inp: usize, out: usize, rng: &mut impl Rng) -> Self {
        let a = (6.0 / (inp + out) as f64).sqrt();
        Linear { w: Mat::random(inp, out, a, rng), b: vec![0.0; out] }
    }

    pub fn in_dim(&self) -> usize {
        self.w.rows
    }

    pub fn out_dim(&self) -> usize {
        self.w.cols
    }

    /// Single vector; parallel over output units.
    pub fn forward_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.w.vec_mul(x);
        y.iter_mut().zip(&self.b).for_each(|(v, b)| *v += b);
        y
    }

    /// Batch of row vectors (n × in) → (n × out); parallel over rows.
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut y = x.matmul(&self.w);
        for r in 0..y.rows {
            y.row_mut(r).iter_mut().zip(&self.b).for_each(|(v, b)| *v += b);
        }
        y
    }
}

/// GELU, tanh approximation (Hendrycks & Gimpel).
pub fn gelu(x: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// LayerNorm over a single vector with learnable affine (γ=1, β=0 at init).
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        LayerNorm { gamma: vec![1.0; dim], beta: vec![0.0; dim], eps: 1e-5 }
    }

    pub fn forward_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.gamma.len(), "LayerNorm: dim mismatch");
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + self.eps).sqrt();
        x.iter().zip(self.gamma.iter().zip(&self.beta)).map(|(v, (g, b))| (v - mean) * inv * g + b).collect()
    }
}

/// Two-layer MLP: l2(gelu(l1(x))).
#[derive(Clone, Debug)]
pub struct Mlp2 {
    pub l1: Linear,
    pub l2: Linear,
}

impl Mlp2 {
    pub fn new(inp: usize, hidden: usize, out: usize, rng: &mut impl Rng) -> Self {
        Mlp2 { l1: Linear::new(inp, hidden, rng), l2: Linear::new(hidden, out, rng) }
    }

    pub fn forward_vec(&self, x: &[f64]) -> Vec<f64> {
        let h: Vec<f64> = self.l1.forward_vec(x).into_iter().map(gelu).collect();
        self.l2.forward_vec(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{rand_mat, rand_vec, rng};

    #[test]
    fn gelu_reference_values() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841192).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158808).abs() < 1e-5);
        assert!((gelu(10.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn layernorm_output_has_zero_mean_unit_variance() {
        let y = LayerNorm::new(32).forward_vec(&rand_vec(32, 1));
        let mean = y.iter().sum::<f64>() / 32.0;
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 32.0;
        assert!(mean.abs() < 1e-12);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn linear_batch_matches_vector_form() {
        let mut r = rng(2);
        let mut l = Linear::new(6, 4, &mut r);
        l.b = vec![0.1, 0.2, 0.3, 0.4];
        let x = rand_mat(3, 6, 3);
        let y = l.forward(&x);
        for i in 0..3 {
            let yv = l.forward_vec(x.row(i));
            for j in 0..4 {
                assert!((y.get(i, j) - yv[j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn mlp_shapes_and_determinism() {
        let a = Mlp2::new(8, 16, 10, &mut rng(5));
        let b = Mlp2::new(8, 16, 10, &mut rng(5));
        let x = rand_vec(8, 6);
        assert_eq!(a.forward_vec(&x).len(), 10);
        assert_eq!(a.forward_vec(&x), b.forward_vec(&x));
    }
}
