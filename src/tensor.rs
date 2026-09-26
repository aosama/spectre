//! Minimal dense row-major real (`Mat`) and complex (`CMat`) matrices.
//! Parallelism: every operation that produces many independent outputs splits
//! them across rayon threads; each output element is computed by exactly one
//! sequential loop, so results are bitwise identical for any thread count.

use crate::opcount::{self, Op};
use crate::C64;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Mat::from_vec: wrong data length");
        Mat { rows, cols, data }
    }

    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Mat { rows, cols, data }
    }

    /// Uniform random entries in [-scale, scale].
    pub fn random(rows: usize, cols: usize, scale: f64, rng: &mut impl Rng) -> Self {
        let data = (0..rows * cols).map(|_| rng.gen_range(-scale..=scale)).collect();
        Mat { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn col(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Build a matrix from columns (all of equal length). Used after
    /// column-parallel computations.
    pub fn from_cols(cols: &[Vec<f64>]) -> Self {
        let ncols = cols.len();
        let nrows = if ncols == 0 { 0 } else { cols[0].len() };
        let mut m = Mat::zeros(nrows, ncols);
        for (c, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), nrows, "Mat::from_cols: ragged columns");
            for (r, &v) in col.iter().enumerate() {
                m.set(r, c, v);
            }
        }
        m
    }

    /// Dense product `self (m×k) · other (k×n)`, parallel over output rows.
    /// Counts m·k·n `Op::Mac`.
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul: inner dimensions differ");
        let (k, n) = (self.cols, other.cols);
        let mut out = Mat::zeros(self.rows, n);
        out.data.par_chunks_mut(n.max(1)).enumerate().for_each(|(i, out_row)| {
            let a_row = &self.data[i * k..(i + 1) * k];
            for (p, &a) in a_row.iter().enumerate() {
                let b_row = &other.data[p * n..(p + 1) * n];
                for (o, &b) in out_row.iter_mut().zip(b_row) {
                    *o += a * b;
                }
            }
            opcount::add(Op::Mac, (k * n) as u64);
        });
        out
    }

    /// Row-vector times matrix: `x (1×k) · self (k×n)`, parallel over output columns.
    pub fn vec_mul(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.rows, "vec_mul: length mismatch");
        let out: Vec<f64> = (0..self.cols)
            .into_par_iter()
            .map(|j| {
                let mut s = 0.0;
                for (i, &xi) in x.iter().enumerate() {
                    s += xi * self.data[i * self.cols + j];
                }
                s
            })
            .collect();
        opcount::add(Op::Mac, (self.rows * self.cols) as u64);
        out
    }

    /// Rows `[start, end)` as a new matrix.
    pub fn slice_rows(&self, start: usize, end: usize) -> Mat {
        assert!(start <= end && end <= self.rows, "slice_rows: out of range");
        Mat::from_vec(end - start, self.cols, self.data[start * self.cols..end * self.cols].to_vec())
    }

    /// Columns `[start, end)` as a new matrix.
    pub fn slice_cols(&self, start: usize, end: usize) -> Mat {
        assert!(start <= end && end <= self.cols, "slice_cols: out of range");
        Mat::from_fn(self.rows, end - start, |r, c| self.get(r, start + c))
    }

    /// Copy of `self` with zero rows appended so it has `rows` rows.
    pub fn pad_rows(&self, rows: usize) -> Mat {
        assert!(rows >= self.rows, "pad_rows: cannot shrink");
        let mut data = self.data.clone();
        data.resize(rows * self.cols, 0.0);
        Mat::from_vec(rows, self.cols, data)
    }

    /// Horizontal concatenation `[a | b | ...]` (all with equal row count).
    pub fn hconcat(parts: &[Mat]) -> Mat {
        let rows = parts[0].rows;
        let cols: usize = parts.iter().map(|p| p.cols).sum();
        let mut out = Mat::zeros(rows, cols);
        for r in 0..rows {
            let mut off = 0;
            for p in parts {
                assert_eq!(p.rows, rows, "hconcat: row count mismatch");
                out.data[r * cols + off..r * cols + off + p.cols].copy_from_slice(p.row(r));
                off += p.cols;
            }
        }
        out
    }

    /// Mean of each column (length `cols`).
    pub fn col_means(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.cols];
        for r in 0..self.rows {
            for (acc, &v) in s.iter_mut().zip(self.row(r)) {
                *acc += v;
            }
        }
        s.iter().map(|v| v / self.rows as f64).collect()
    }

    pub fn add(&self, other: &Mat) -> Mat {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "add: shape mismatch");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Mat::from_vec(self.rows, self.cols, data)
    }

    pub fn max_abs_diff(&self, other: &Mat) -> f64 {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "max_abs_diff: shape mismatch");
        self.data.iter().zip(&other.data).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<C64>,
}

impl CMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        CMat { rows, cols, data: vec![C64::new(0.0, 0.0); rows * cols] }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> C64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: C64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn row(&self, r: usize) -> &[C64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn from_cols(cols: &[Vec<C64>]) -> Self {
        let ncols = cols.len();
        let nrows = if ncols == 0 { 0 } else { cols[0].len() };
        let mut m = CMat::zeros(nrows, ncols);
        for (c, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), nrows, "CMat::from_cols: ragged columns");
            for (r, &v) in col.iter().enumerate() {
                m.set(r, c, v);
            }
        }
        m
    }

    pub fn col(&self, c: usize) -> Vec<C64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    pub fn add(&self, other: &CMat) -> CMat {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "CMat::add: shape mismatch");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        CMat { rows: self.rows, cols: self.cols, data }
    }

    pub fn max_abs_diff(&self, other: &CMat) -> f64 {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "CMat::max_abs_diff: shape mismatch");
        self.data.iter().zip(&other.data).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::rand_mat;

    fn naive_matmul(a: &Mat, b: &Mat) -> Mat {
        Mat::from_fn(a.rows, b.cols, |i, j| (0..a.cols).map(|p| a.get(i, p) * b.get(p, j)).sum())
    }

    #[test]
    fn matmul_small_known_values() {
        let a = Mat::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let b = Mat::from_vec(3, 2, vec![7., 8., 9., 10., 11., 12.]);
        let c = a.matmul(&b);
        assert_eq!(c, Mat::from_vec(2, 2, vec![58., 64., 139., 154.]));
    }

    #[test]
    fn matmul_matches_naive_on_random() {
        let a = rand_mat(37, 19, 1);
        let b = rand_mat(19, 23, 2);
        assert!(a.matmul(&b).max_abs_diff(&naive_matmul(&a, &b)) < 1e-12);
    }

    #[test]
    fn vec_mul_matches_matmul() {
        let w = rand_mat(11, 7, 3);
        let x = rand_mat(1, 11, 4);
        let v = w.vec_mul(&x.data);
        let m = x.matmul(&w);
        for (a, b) in v.iter().zip(&m.data) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn slice_pad_concat_and_means() {
        let m = Mat::from_vec(3, 2, vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(m.slice_rows(1, 3), Mat::from_vec(2, 2, vec![3., 4., 5., 6.]));
        assert_eq!(m.pad_rows(4).row(3), &[0., 0.]);
        let h = Mat::hconcat(&[m.clone(), m.slice_rows(0, 3)]);
        assert_eq!(h.row(1), &[3., 4., 3., 4.]);
        assert_eq!(m.col_means(), vec![3., 4.]);
        assert_eq!(m.slice_cols(1, 2), Mat::from_vec(3, 1, vec![2., 4., 6.]));
        assert_eq!(Mat::from_cols(&[m.col(0), m.col(1)]), m);
    }
}
