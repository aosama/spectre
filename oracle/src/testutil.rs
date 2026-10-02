//! Deterministic random inputs and comparison helpers shared by tests.
//! Seed 42 is the paper's reproducibility seed (Appendix A.5).

use crate::tensor::Mat;
use crate::C64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const PAPER_SEED: u64 = 42;

pub fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Uniform [-1, 1] matrix from a seed.
pub fn rand_mat(rows: usize, cols: usize, seed: u64) -> Mat {
    Mat::random(rows, cols, 1.0, &mut rng(seed))
}

pub fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut r = rng(seed);
    (0..n).map(|_| r.gen_range(-1.0..=1.0)).collect()
}

pub fn rand_cvec(n: usize, seed: u64) -> Vec<C64> {
    let mut r = rng(seed);
    (0..n).map(|_| C64::new(r.gen_range(-1.0..=1.0), r.gen_range(-1.0..=1.0))).collect()
}

pub fn max_abs_diff_c(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff_c: length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

pub fn max_abs_diff_r(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_abs_diff_r: length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}
