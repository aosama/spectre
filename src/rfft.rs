//! Real FFT R_n and inverse R_n^{-1} (paper eq. 1, Appendix B): keeps the
//! n/2+1 non-redundant bins of a real signal. Implemented via the complex FFT
//! (no packing trick: reproduction, not optimization). Column-wise variants
//! transform each channel of an (n × d) matrix independently, in parallel.

use crate::fft::{fft_in_place, naive_dft};
use crate::tensor::{CMat, Mat};
use crate::C64;
use rayon::prelude::*;

/// Number of non-redundant bins for a length-n real signal: ⌊n/2⌋ + 1.
pub fn num_bins(n: usize) -> usize {
    n / 2 + 1
}

/// R_n(x): bins k = 0..=n/2 of the DFT of real x. n must be a power of two.
pub fn rfft(x: &[f64]) -> Vec<C64> {
    let mut buf: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft_in_place(&mut buf, false);
    buf.truncate(num_bins(x.len()));
    buf
}

/// R_n^{-1}: rebuilds the Hermitian spectrum (X_{n-k} = conj X_k) and returns
/// the real signal, normalized by 1/n. As in numpy.fft.irfft, the imaginary
/// parts of the DC bin and the Nyquist bin are ignored.
pub fn irfft(spec: &[C64], n: usize) -> Vec<f64> {
    assert_eq!(spec.len(), num_bins(n), "irfft: expected n/2+1 bins");
    let mut full = vec![C64::new(0.0, 0.0); n];
    full[0] = C64::new(spec[0].re, 0.0);
    for k in 1..n / 2 {
        full[k] = spec[k];
        full[n - k] = spec[k].conj();
    }
    if n > 1 {
        full[n / 2] = C64::new(spec[n / 2].re, 0.0);
    }
    fft_in_place(&mut full, true);
    let s = 1.0 / n as f64;
    full.iter().map(|v| v.re * s).collect()
}

/// O(n²) oracle for `rfft` (any n).
pub fn naive_rfft(x: &[f64]) -> Vec<C64> {
    let c: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    let mut out = naive_dft(&c, false);
    out.truncate(num_bins(x.len()));
    out
}

/// RFFT of every column of `x` after zero-padding the rows to `n_fft`.
/// Returns (n_fft/2+1) × x.cols. Parallel over columns.
pub fn rfft_cols(x: &Mat, n_fft: usize) -> CMat {
    assert!(x.rows <= n_fft, "rfft_cols: {} rows exceed n_fft {}", x.rows, n_fft);
    let cols: Vec<Vec<C64>> = (0..x.cols)
        .into_par_iter()
        .map(|c| {
            let mut col = x.col(c);
            col.resize(n_fft, 0.0);
            rfft(&col)
        })
        .collect();
    CMat::from_cols(&cols)
}

/// Inverse RFFT of every column of `spec` ((n_fft/2+1) × d) → n_fft × d. Parallel over columns.
pub fn irfft_cols(spec: &CMat, n_fft: usize) -> Mat {
    let cols: Vec<Vec<f64>> = (0..spec.cols).into_par_iter().map(|c| irfft(&spec.col(c), n_fft)).collect();
    Mat::from_cols(&cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::fft;
    use crate::testutil::{max_abs_diff_c, max_abs_diff_r, rand_mat, rand_vec};

    #[test]
    fn keeps_n_over_2_plus_1_bins() {
        assert_eq!(num_bins(8), 5);
        assert_eq!(rfft(&rand_vec(8, 1)).len(), 5);
    }

    #[test]
    fn matches_naive_rfft() {
        for p in 1..=10 {
            let x = rand_vec(1 << p, p as u64);
            assert!(max_abs_diff_c(&rfft(&x), &naive_rfft(&x)) < 1e-11, "n=2^{p}");
        }
    }

    // Appendix B, Theorem B.1: X_{n-k} = conj(X_k); DC and Nyquist are real.
    #[test]
    fn hermitian_symmetry_theorem_b1() {
        let x = rand_vec(64, 3);
        let c: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
        let full = fft(&c);
        for k in 1..64 {
            assert!((full[64 - k] - full[k].conj()).norm() < 1e-12);
        }
        assert!(full[0].im.abs() < 1e-12 && full[32].im.abs() < 1e-12);
    }

    // Corollary B.2: the half spectrum is lossless.
    #[test]
    fn roundtrip_is_lossless() {
        for p in 1..=12 {
            let n = 1 << p;
            let x = rand_vec(n, 50 + p as u64);
            assert!(max_abs_diff_r(&irfft(&rfft(&x), n), &x) < 1e-12, "n={n}");
        }
    }

    #[test]
    fn irfft_ignores_imag_of_dc_and_nyquist() {
        let x = rand_vec(16, 4);
        let mut s = rfft(&x);
        s[0].im = 5.0;
        s[8].im = -3.0;
        assert!(max_abs_diff_r(&irfft(&s, 16), &x) < 1e-12);
    }

    #[test]
    fn column_variants_match_per_column_and_pad() {
        let m = rand_mat(12, 5, 6); // 12 rows padded to 16
        let s = rfft_cols(&m, 16);
        assert_eq!((s.rows, s.cols), (9, 5));
        for c in 0..5 {
            let mut col = m.col(c);
            col.resize(16, 0.0);
            assert!(max_abs_diff_c(&s.col(c), &rfft(&col)) < 1e-15);
        }
        let back = irfft_cols(&s, 16);
        assert_eq!((back.rows, back.cols), (16, 5));
        assert!(back.slice_rows(0, 12).max_abs_diff(&m) < 1e-12);
        assert!(back.slice_rows(12, 16).data.iter().all(|v| v.abs() < 1e-12));
    }
}
