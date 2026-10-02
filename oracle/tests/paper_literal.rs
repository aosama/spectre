//! Cross-verification against the paper's literal formulas, including the two
//! places where this implementation deliberately deviates (see Decision Log).

use spectre::cache::PrefixFftCache;
use spectre::head::{HeadConfig, SpectreHead};
use spectre::rfft::{num_bins, rfft_cols};
use spectre::tensor::CMat;
use spectre::testutil::{rand_mat, rng};
use spectre::C64;
use std::f64::consts::PI;

fn head(n_fft: usize) -> SpectreHead {
    let cfg = HeadConfig { d_model: 12, d_head: 4, n_fft, gate_hidden: 16, toeplitz_r: None, wavelet_levels: None };
    SpectreHead::new(cfg, &mut rng(1))
}

/// Eq. (5) exactly as printed: subtract v_old·e^{−j2πk(t−N)/N}·1{t≥N}, add v_t·e^{−j2πkt/N}.
fn eq5_literal(prefix: &mut CMat, t: usize, n: usize, v_old: &[f64], v_t: &[f64]) {
    for k in 0..num_bins(n) {
        for c in 0..v_t.len() {
            let mut val = prefix.get(k, c);
            if t >= n {
                val -= v_old[c] * C64::from_polar(1.0, -2.0 * PI * (k as f64) * ((t - n) as f64) / n as f64);
            }
            val += v_t[c] * C64::from_polar(1.0, -2.0 * PI * (k as f64) * (t as f64) / n as f64);
            prefix.set(k, c, val);
        }
    }
}

#[test]
fn eq5_literal_matches_cache_update() {
    let n = 16;
    let h = head(n);
    let xs = rand_mat(3 * n, 12, 2);
    let (_, all_v) = h.project(&xs);
    let mut c = PrefixFftCache::new(&h, None);
    c.prefill(&h, &xs.slice_rows(0, 1));
    let mut literal = c.prefix_fft.clone();
    for t in 1..xs.rows {
        let v_old = if t >= n { all_v.row(t - n).to_vec() } else { vec![0.0; 4] };
        eq5_literal(&mut literal, t, n, &v_old, all_v.row(t));
        c.decode_step(&h, xs.row(t));
        assert!(c.prefix_fft.max_abs_diff(&literal) < 1e-9, "t={t}");
    }
}

#[test]
fn eviction_and_insertion_twiddles_coincide() {
    // e^{−j2πk(t−N)/N} = e^{−j2πkt/N}: justifies the single-delta update.
    let n = 64usize;
    for k in 0..=n / 2 {
        for t in [n, n + 3, 5 * n + 7] {
            let a = C64::from_polar(1.0, -2.0 * PI * (k * (t - n)) as f64 / n as f64);
            let b = C64::from_polar(1.0, -2.0 * PI * (k * t) as f64 / n as f64);
            assert!((a - b).norm() < 1e-9);
        }
    }
}

/// Algorithm 1 applies the phase e^{j2πkt/N} with t = index of the newest token.
/// That rotates by one row less than needed for chronological output: with it,
/// the newest token lands on row 0 instead of row N−1. We use shift t+1
/// (Decision D6). This test pins down the exact relationship.
#[test]
fn algorithm1_literal_phase_is_our_output_rotated_by_one_row() {
    let n = 16;
    let h = head(n);
    let xs = rand_mat(40, 12, 3);
    let mut c = PrefixFftCache::new(&h, None);
    c.prefill(&h, &xs.slice_rows(0, 10));
    for t in 10..40 {
        c.decode_step(&h, xs.row(t));
    }
    let t_newest = c.t - 1;
    let q_mean: Vec<f64> = c.sum_q.iter().map(|s| s / n as f64).collect();
    let ours = h.mix_spectrum(&c.prefix_fft, &q_mean, (t_newest + 1) % n);
    let literal = h.mix_spectrum(&c.prefix_fft, &q_mean, t_newest % n);
    for m in 0..n {
        for col in 0..4 {
            assert!((literal.get(m, col) - ours.get((m + n - 1) % n, col)).abs() < 1e-10);
        }
    }
    // Our last row is the filtered newest token's slot; theirs is row 0.
    let window = c.placed_window();
    let plain = h.mix(&window, &q_mean, 0); // buffer order, no rotation
    let newest_slot = t_newest % n;
    for col in 0..4 {
        assert!((ours.get(n - 1, col) - plain.get(newest_slot, col)).abs() < 1e-10);
        assert!((literal.get(0, col) - plain.get(newest_slot, col)).abs() < 1e-10);
    }
}

/// §3.3.1: pre-fill cost is one padded N_max-point RFFT: spectrum equals R_N(pad(V)).
#[test]
fn prefill_is_single_padded_rfft() {
    let h = head(32);
    let x = rand_mat(9, 12, 4);
    let mut c = PrefixFftCache::new(&h, None);
    c.prefill(&h, &x);
    let (_, v) = h.project(&x);
    assert!(c.prefix_fft.max_abs_diff(&rfft_cols(&v, 32)) < 1e-12);
}
