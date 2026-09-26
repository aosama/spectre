//! Complex radix-2 Cooley–Tukey FFT (paper §2, "Real FFT") and an O(n²) DFT
//! oracle. Sign convention (paper eq. 1): forward uses e^{-j2πkt/n}; inverse
//! uses e^{+j2πkt/n}. Neither direction normalizes; `ifft` divides by n.
//! A single FFT is sequential; parallelism comes from running many
//! independent FFTs (one per channel / head) on rayon threads.

use crate::opcount::{self, Op};
use crate::C64;
use std::f64::consts::PI;

pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && n & (n - 1) == 0
}

/// In-place iterative radix-2 FFT. `inverse = true` flips the twiddle sign.
/// Performs exactly (n/2)·log2(n) butterflies (counted as `Op::Butterfly`).
pub fn fft_in_place(buf: &mut [C64], inverse: bool) {
    let n = buf.len();
    assert!(is_power_of_two(n), "fft_in_place: length {n} is not a power of two");
    if n == 1 {
        return;
    }
    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            buf.swap(i, j);
        }
    }
    // Twiddle table w[j] = e^{sign·j2πj/n}, j < n/2 (direct evaluation keeps accuracy).
    let sign = if inverse { 1.0 } else { -1.0 };
    let tw: Vec<C64> = (0..n / 2).map(|j| C64::from_polar(1.0, sign * 2.0 * PI * j as f64 / n as f64)).collect();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len;
        for start in (0..n).step_by(len) {
            for j in 0..half {
                let u = buf[start + j];
                let v = buf[start + j + half] * tw[j * stride];
                buf[start + j] = u + v;
                buf[start + j + half] = u - v;
            }
        }
        opcount::add(Op::Butterfly, (n / 2) as u64);
        len <<= 1;
    }
}

/// Forward FFT (unnormalized).
pub fn fft(x: &[C64]) -> Vec<C64> {
    let mut b = x.to_vec();
    fft_in_place(&mut b, false);
    b
}

/// Inverse FFT, normalized by 1/n so that `ifft(fft(x)) == x`.
pub fn ifft(x: &[C64]) -> Vec<C64> {
    let mut b = x.to_vec();
    fft_in_place(&mut b, true);
    let s = 1.0 / b.len() as f64;
    b.iter_mut().for_each(|v| *v *= s);
    b
}

/// O(n²) textbook DFT; any length. Test oracle only. Unnormalized in both directions.
pub fn naive_dft(x: &[C64], inverse: bool) -> Vec<C64> {
    let n = x.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    (0..n)
        .map(|k| {
            (0..n)
                .map(|t| x[t] * C64::from_polar(1.0, sign * 2.0 * PI * ((k * t) % n) as f64 / n as f64))
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{max_abs_diff_c, rand_cvec};

    #[test]
    fn power_of_two_detection() {
        assert!(is_power_of_two(1) && is_power_of_two(2) && is_power_of_two(1024));
        assert!(!is_power_of_two(0) && !is_power_of_two(3) && !is_power_of_two(1000));
    }

    #[test]
    fn impulse_transforms_to_all_ones() {
        let mut x = vec![C64::new(0.0, 0.0); 8];
        x[0] = C64::new(1.0, 0.0);
        for v in fft(&x) {
            assert!((v - C64::new(1.0, 0.0)).norm() < 1e-15);
        }
    }

    #[test]
    fn matches_naive_dft_for_all_sizes_up_to_1024() {
        for p in 0..=10 {
            let n = 1usize << p;
            let x = rand_cvec(n, 100 + p as u64);
            let tol = 1e-12 * n as f64;
            assert!(max_abs_diff_c(&fft(&x), &naive_dft(&x, false)) < tol, "forward n={n}");
            let mut inv = x.clone();
            fft_in_place(&mut inv, true);
            assert!(max_abs_diff_c(&inv, &naive_dft(&x, true)) < tol, "inverse n={n}");
        }
    }

    #[test]
    fn roundtrip_is_identity() {
        let x = rand_cvec(4096, 7);
        assert!(max_abs_diff_c(&ifft(&fft(&x)), &x) < 1e-12);
    }

    #[test]
    fn parseval_energy_identity() {
        let x = rand_cvec(512, 8);
        let ex: f64 = x.iter().map(|v| v.norm_sqr()).sum();
        let ef: f64 = fft(&x).iter().map(|v| v.norm_sqr()).sum::<f64>() / 512.0;
        assert!((ex - ef).abs() < 1e-9 * ex);
    }

    #[test]
    fn linearity() {
        let a = rand_cvec(64, 9);
        let b = rand_cvec(64, 10);
        let s: Vec<C64> = a.iter().zip(&b).map(|(x, y)| x * 2.0 + y).collect();
        let lhs = fft(&s);
        let rhs: Vec<C64> = fft(&a).iter().zip(fft(&b)).map(|(x, y)| x * 2.0 + y).collect();
        assert!(max_abs_diff_c(&lhs, &rhs) < 1e-12);
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn rejects_non_power_of_two() {
        fft(&vec![C64::new(0.0, 0.0); 12]);
    }
}
