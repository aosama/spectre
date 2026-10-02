//! Optional Wavelet Refinement Module (paper §3.5): orthogonal multi-level Haar
//! DWT along the sequence axis, per-(level, channel) real gates s produced by a
//! two-layer MLP from q̄, inverse DWT. Output V_out = Ṽ + W^{-1}(s ⊙ W(Ṽ)).
//! Coefficient layout after J levels: [a_J | d_J | d_{J-1} | ... | d_1].

use crate::nn::{LayerNorm, Mlp2};
use crate::opcount::{self, Op};
use crate::tensor::Mat;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

/// J-level orthonormal Haar DWT. `x.len()` must be divisible by 2^levels.
pub fn haar_forward(x: &[f64], levels: usize) -> Vec<f64> {
    let n = x.len();
    assert!(n % (1 << levels) == 0, "haar_forward: length {n} not divisible by 2^{levels}");
    let mut out = x.to_vec();
    let mut tmp = vec![0.0; n];
    let mut m = n;
    for _ in 0..levels {
        let h = m / 2;
        for i in 0..h {
            let (a, b) = (out[2 * i], out[2 * i + 1]);
            tmp[i] = (a + b) * FRAC_1_SQRT_2;
            tmp[h + i] = (a - b) * FRAC_1_SQRT_2;
        }
        out[..m].copy_from_slice(&tmp[..m]);
        opcount::add(Op::Haar, h as u64);
        m = h;
    }
    out
}

/// Inverse of `haar_forward`.
pub fn haar_inverse(c: &[f64], levels: usize) -> Vec<f64> {
    let n = c.len();
    assert!(n % (1 << levels) == 0, "haar_inverse: length {n} not divisible by 2^{levels}");
    let mut out = c.to_vec();
    let mut tmp = vec![0.0; n];
    let mut m = n >> levels;
    for _ in 0..levels {
        for i in 0..m {
            let (a, d) = (out[i], out[m + i]);
            tmp[2 * i] = (a + d) * FRAC_1_SQRT_2;
            tmp[2 * i + 1] = (a - d) * FRAC_1_SQRT_2;
        }
        out[..2 * m].copy_from_slice(&tmp[..2 * m]);
        opcount::add(Op::Haar, m as u64);
        m *= 2;
    }
    out
}

/// Band of coefficient index i: 0 = approximation a_J, ℓ ∈ 1..=J = detail d_ℓ
/// (d_1 is the finest scale, occupying [n/2, n)).
pub fn band_of_index(i: usize, n: usize, levels: usize) -> usize {
    if i < n >> levels {
        return 0;
    }
    (1..=levels).find(|&l| i >= n >> l && i < n >> (l - 1)).expect("index out of range")
}

#[derive(Clone, Debug)]
pub struct Wrm {
    pub levels: usize,
    pub d_head: usize,
    pub ln: LayerNorm,
    /// Outputs (levels+1)·d_head reals; gate for (band b, channel c) is at b·d_head + c.
    pub mlp: Mlp2,
}

impl Wrm {
    pub fn new(d_head: usize, levels: usize, hidden: usize, rng: &mut impl Rng) -> Self {
        Wrm { levels, d_head, ln: LayerNorm::new(d_head), mlp: Mlp2::new(d_head, hidden, (levels + 1) * d_head, rng) }
    }

    /// s as a (levels+1) × d_head matrix.
    pub fn gates(&self, q_mean: &[f64]) -> Mat {
        Mat::from_vec(self.levels + 1, self.d_head, self.mlp.forward_vec(&self.ln.forward_vec(q_mean)))
    }

    /// Ṽ + W^{-1}(s ⊙ W(Ṽ)), column-parallel.
    pub fn refine(&self, v_tilde: &Mat, q_mean: &[f64]) -> Mat {
        let s = self.gates(q_mean);
        refine_with_gates(v_tilde, &s, self.levels)
    }
}

/// WRM core with explicit gates s ((levels+1) × cols). Parallel over columns.
pub fn refine_with_gates(v_tilde: &Mat, s: &Mat, levels: usize) -> Mat {
    assert_eq!((s.rows, s.cols), (levels + 1, v_tilde.cols), "refine_with_gates: gate shape");
    let n = v_tilde.rows;
    let cols: Vec<Vec<f64>> = (0..v_tilde.cols)
        .into_par_iter()
        .map(|c| {
            let x = v_tilde.col(c);
            let mut w = haar_forward(&x, levels);
            for (i, wi) in w.iter_mut().enumerate() {
                *wi *= s.get(band_of_index(i, n, levels), c);
            }
            let r = haar_inverse(&w, levels);
            x.iter().zip(r).map(|(a, b)| a + b).collect()
        })
        .collect();
    Mat::from_cols(&cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{max_abs_diff_r, rand_mat, rand_vec, rng};

    #[test]
    fn single_level_haar_known_values() {
        let c = haar_forward(&[1.0, 3.0, 5.0, 7.0], 1);
        let r = FRAC_1_SQRT_2;
        assert!(max_abs_diff_r(&c, &[4.0 * r, 12.0 * r, -2.0 * r, -2.0 * r]) < 1e-15);
    }

    #[test]
    fn perfect_reconstruction_all_levels() {
        let x = rand_vec(64, 1);
        for levels in 0..=6 {
            assert!(max_abs_diff_r(&haar_inverse(&haar_forward(&x, levels), levels), &x) < 1e-13);
        }
    }

    #[test]
    fn orthogonal_energy_preserved() {
        let x = rand_vec(128, 2);
        let c = haar_forward(&x, 4);
        let e = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>();
        assert!((e(&x) - e(&c)).abs() < 1e-11);
    }

    #[test]
    fn band_layout() {
        // n = 16, J = 2: a_2 = [0,4), d_2 = [4,8), d_1 = [8,16)
        let bands: Vec<usize> = (0..16).map(|i| band_of_index(i, 16, 2)).collect();
        assert_eq!(bands, vec![0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(band_of_index(3, 16, 2), 0);
        assert_eq!(band_of_index(4, 16, 2), 2);
        assert_eq!(band_of_index(15, 16, 2), 1);
    }

    #[test]
    fn gates_of_one_double_and_zero_keep() {
        let v = rand_mat(32, 3, 3);
        let ones = Mat::from_fn(4, 3, |_, _| 1.0);
        let zeros = Mat::zeros(4, 3);
        assert!(refine_with_gates(&v, &ones, 3).max_abs_diff(&v.add(&v)) < 1e-13);
        assert!(refine_with_gates(&v, &zeros, 3).max_abs_diff(&v) < 1e-15);
    }

    #[test]
    fn wrm_gate_shape() {
        let w = Wrm::new(4, 3, 8, &mut rng(4));
        let s = w.gates(&rand_vec(4, 5));
        assert_eq!((s.rows, s.cols), (4, 4));
        assert_eq!(w.refine(&rand_mat(16, 4, 6), &rand_vec(4, 5)).rows, 16);
    }
}
