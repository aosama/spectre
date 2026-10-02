//! Naive softmax multi-head self-attention (non-causal), the O(n²·d) baseline
//! SPECTRE replaces (paper §2). Used for complexity comparisons only.

use crate::nn::{seeded_rng as rng, Linear};
use crate::opcount::{self, Op};
use crate::tensor::Mat;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct NaiveAttention {
    pub n_heads: usize,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
}

impl NaiveAttention {
    pub fn new(d_model: usize, n_heads: usize, seed: u64) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        let mut r = rng(seed);
        NaiveAttention {
            n_heads,
            wq: Linear::new(d_model, d_model, &mut r),
            wk: Linear::new(d_model, d_model, &mut r),
            wv: Linear::new(d_model, d_model, &mut r),
            wo: Linear::new(d_model, d_model, &mut r),
        }
    }

    /// X (n × d_model) → n × d_model.
    pub fn forward(&self, x: &Mat) -> Mat {
        let (q, k, v) = (self.wq.forward(x), self.wk.forward(x), self.wv.forward(x));
        let dh = x.cols / self.n_heads;
        let outs: Vec<Mat> = (0..self.n_heads)
            .into_par_iter()
            .map(|h| {
                let s = |m: &Mat| m.slice_cols(h * dh, (h + 1) * dh);
                attention_head(&s(&q), &s(&k), &s(&v))
            })
            .collect();
        self.wo.forward(&Mat::hconcat(&outs))
    }
}

/// softmax(Q·Kᵀ/√d)·V for one head. Parallel over query rows.
/// Counts 2·n_q·n_k·d `Op::Mac` (scores + weighted sum).
pub fn attention_head(q: &Mat, k: &Mat, v: &Mat) -> Mat {
    let (nq, nk, d) = (q.rows, k.rows, q.cols);
    let scale = 1.0 / (d as f64).sqrt();
    let rows: Vec<Vec<f64>> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let qi = q.row(i);
            let scores: Vec<f64> =
                (0..nk).map(|j| qi.iter().zip(k.row(j)).map(|(a, b)| a * b).sum::<f64>() * scale).collect();
            let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = w.iter().sum();
            let mut out = vec![0.0; v.cols];
            for (j, wj) in w.iter().enumerate() {
                out.iter_mut().zip(v.row(j)).for_each(|(o, vv)| *o += wj / z * vv);
            }
            opcount::add(Op::Mac, (2 * nk * d) as u64);
            out
        })
        .collect();
    Mat::from_vec(nq, v.cols, rows.concat())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::rand_mat;

    #[test]
    fn zero_queries_give_uniform_average_of_values() {
        let q = Mat::zeros(3, 4);
        let k = rand_mat(5, 4, 1);
        let v = rand_mat(5, 2, 2);
        let out = attention_head(&q, &k, &v);
        let mean = v.col_means();
        for i in 0..3 {
            assert!((out.get(i, 0) - mean[0]).abs() < 1e-12 && (out.get(i, 1) - mean[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn single_key_returns_its_value() {
        let out = attention_head(&rand_mat(4, 3, 3), &rand_mat(1, 3, 4), &Mat::from_vec(1, 2, vec![7.0, -1.0]));
        for i in 0..4 {
            assert_eq!(out.row(i), &[7.0, -1.0]);
        }
    }

    #[test]
    fn softmax_is_stable_for_huge_scores() {
        let q = Mat::from_vec(1, 1, vec![1e6]);
        let k = Mat::from_vec(2, 1, vec![1.0, 2.0]);
        let v = Mat::from_vec(2, 1, vec![10.0, 20.0]);
        assert_eq!(attention_head(&q, &k, &v).data, vec![20.0]);
    }

    #[test]
    fn multihead_shape() {
        let a = NaiveAttention::new(16, 4, 42);
        let y = a.forward(&rand_mat(10, 16, 5));
        assert_eq!((y.rows, y.cols), (10, 16));
    }
}
