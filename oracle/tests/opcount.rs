//! Deterministic complexity validation by counting primitive operations.
//! Run with: cargo test --release --features opcount --test opcount
#![cfg(feature = "opcount")]

use spectre::attention::{attention_head, NaiveAttention};
use spectre::cache::PrefixFftCache;
use spectre::complexity::{loglog_slope, nlog2n, spread};
use spectre::fft::fft;
use spectre::gate::toeplitz_update;
use spectre::head::{apply_gate, HeadConfig, SpectreHead};
use spectre::layer::{LayerConfig, SpectreLayer};
use spectre::opcount::{self, measure};
use spectre::rfft::{irfft_cols, rfft_cols};
use spectre::tensor::CMat;
use spectre::testutil::{rand_cvec, rand_mat, rand_vec, rng};
use spectre::wavelet::{haar_forward, haar_inverse};

const D_MODEL: usize = 32;
const N_HEADS: usize = 4;

fn layer(n_fft: usize) -> SpectreLayer {
    SpectreLayer::new(
        LayerConfig { d_model: D_MODEL, n_heads: N_HEADS, n_fft, gate_hidden: 32, toeplitz_r: Some(2), wavelet_levels: Some(3) },
        42,
    )
}

fn head(n_fft: usize) -> SpectreHead {
    let cfg = HeadConfig { d_model: D_MODEL, d_head: 8, n_fft, gate_hidden: 32, toeplitz_r: Some(2), wavelet_levels: None };
    SpectreHead::new(cfg, &mut rng(1))
}

fn pow2s(a: u32, b: u32) -> Vec<usize> {
    (a..=b).map(|p| 1usize << p).collect()
}

fn as_f64(v: &[usize]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

#[test]
fn fft_does_exactly_half_n_log_n_butterflies() {
    let _g = opcount::lock();
    for p in 0..=14u32 {
        let n = 1usize << p;
        let x = rand_cvec(n, 1);
        let (_, s) = measure(|| fft(&x));
        assert_eq!(s.butterfly, (n / 2) as u64 * p as u64, "n={n}");
        assert_eq!(s.total(), s.butterfly, "fft must count nothing else");
    }
}

#[test]
fn column_rfft_and_irfft_cost_d_times_single() {
    let _g = opcount::lock();
    let (n, d) = (1024usize, 8usize);
    let x = rand_mat(n, d, 2);
    let (spec, s) = measure(|| rfft_cols(&x, n));
    assert_eq!(s.butterfly, (d * n / 2 * 10) as u64);
    let (_, s) = measure(|| irfft_cols(&spec, n));
    assert_eq!(s.butterfly, (d * n / 2 * 10) as u64);
}

#[test]
fn matmul_costs_m_k_n_macs() {
    let _g = opcount::lock();
    let (a, b) = (rand_mat(13, 7, 3), rand_mat(7, 5, 4));
    let (_, s) = measure(|| a.matmul(&b));
    assert_eq!(s.mac, 13 * 7 * 5);
}

#[test]
fn gating_is_linear_toeplitz_is_n_times_r() {
    let _g = opcount::lock();
    let mut spec = CMat::zeros(513, 8);
    spec.data = rand_cvec(513 * 8, 5);
    let (_, s) = measure(|| apply_gate(&spec, &rand_cvec(513, 6)));
    assert_eq!(s.cmul, 513 * 8);
    for r in [0usize, 1, 4] {
        let (_, s) = measure(|| toeplitz_update(&rand_cvec(513, 7), &rand_cvec(2 * r + 1, 8)));
        assert_eq!(s.cmul, (513 * (2 * r + 1)) as u64, "r={r}");
    }
}

#[test]
fn wavelet_is_linear() {
    let _g = opcount::lock();
    let levels = 3;
    for n in pow2s(6, 14) {
        let x = rand_vec(n, 9);
        let (c, s) = measure(|| haar_forward(&x, levels));
        let expected = (n - (n >> levels)) as u64; // n/2 + n/4 + n/8
        assert_eq!(s.haar, expected, "forward n={n}");
        let (_, s) = measure(|| haar_inverse(&c, levels));
        assert_eq!(s.haar, expected, "inverse n={n}");
    }
}

#[test]
fn attention_is_exactly_quadratic() {
    let _g = opcount::lock();
    let d = 8;
    for n in pow2s(4, 9) {
        let (q, k, v) = (rand_mat(n, d, 1), rand_mat(n, d, 2), rand_mat(n, d, 3));
        let (_, s) = measure(|| attention_head(&q, &k, &v));
        assert_eq!(s.mac, (2 * n * n * d) as u64, "n={n}");
    }
}

/// Paper Table 6: total per-layer cost O(n·d·log n).
#[test]
fn spectre_layer_ops_scale_as_n_log_n() {
    let _g = opcount::lock();
    let ns = pow2s(8, 14);
    let ops: Vec<f64> = ns
        .iter()
        .map(|&n| {
            let l = layer(n);
            let x = rand_mat(n, D_MODEL, 10);
            measure(|| l.forward(&x)).1.total() as f64
        })
        .collect();
    let xs = as_f64(&ns);
    let slope = loglog_slope(&xs, &ops);
    let sp = spread(&xs, &ops, nlog2n);
    println!("spectre ops {ops:?} slope {slope:.3} spread {sp:.3}");
    assert!(slope > 0.95 && slope < 1.2, "slope {slope} is not n log n");
    assert!(sp < 2.0, "ops/(n log n) varies by {sp}x");
    for w in ops.windows(2) {
        assert!(w[1] / w[0] < 2.3, "doubling n multiplied ops by {}", w[1] / w[0]);
    }
}

#[test]
fn naive_attention_layer_ops_scale_as_n_squared() {
    let _g = opcount::lock();
    let ns = pow2s(9, 12);
    let att = NaiveAttention::new(D_MODEL, N_HEADS, 42);
    let ops: Vec<f64> = ns
        .iter()
        .map(|&n| {
            let x = rand_mat(n, D_MODEL, 11);
            measure(|| att.forward(&x)).1.total() as f64
        })
        .collect();
    let slope = loglog_slope(&as_f64(&ns), &ops);
    println!("attention ops {ops:?} slope {slope:.3}");
    assert!(slope > 1.8 && slope < 2.05, "slope {slope} is not n^2");
}

#[test]
fn spectre_uses_fewer_ops_than_attention_for_long_sequences() {
    let _g = opcount::lock();
    let n = 4096;
    let x = rand_mat(n, D_MODEL, 12);
    let (_, s_spec) = measure(|| layer(n).forward(&x));
    let (_, s_att) = measure(|| NaiveAttention::new(D_MODEL, N_HEADS, 42).forward(&x));
    println!("n={n}: spectre {} ops, attention {} ops", s_spec.total(), s_att.total());
    assert!(s_spec.total() * 10 < s_att.total());
}

/// §3.3.2: each decode step costs the same regardless of how many tokens were seen.
#[test]
fn decode_step_cost_is_independent_of_t() {
    let _g = opcount::lock();
    let n = 64;
    let h = head(n);
    let mut c = PrefixFftCache::new(&h, None);
    c.prefill(&h, &rand_mat(1, D_MODEL, 13));
    let xs = rand_mat(4 * n, D_MODEL, 14);
    let counts: Vec<_> = (0..xs.rows).map(|t| measure(|| c.decode_step(&h, xs.row(t))).1).collect();
    assert!(counts.iter().all(|s| *s == counts[0]), "per-step cost varies with t");
}

/// Eq. (5): the cache update touches each of the (N/2+1)·d coefficients once: O(N·d).
#[test]
fn cache_update_is_linear_in_window() {
    let _g = opcount::lock();
    for n in pow2s(6, 14) {
        let h = head(n);
        let mut c = PrefixFftCache::new(&h, None);
        let delta = rand_vec(8, 15);
        let (_, s) = measure(|| c.update_spectrum(3, &delta));
        assert_eq!(s.cmul, ((n / 2 + 1) * 8) as u64, "n={n}");
        assert_eq!(s.butterfly, 0, "update must not run an FFT");
    }
}

/// Full decode step (update + gate + inverse RFFT) and prefill are O(N log N) in N_max.
#[test]
fn decode_step_and_prefill_scale_as_n_log_n_in_window() {
    let _g = opcount::lock();
    let ns = pow2s(8, 14);
    let mut step_ops = vec![];
    let mut prefill_ops = vec![];
    for &n in &ns {
        let h = head(n);
        let mut c = PrefixFftCache::new(&h, None);
        let prompt = rand_mat(n, D_MODEL, 16);
        prefill_ops.push(measure(|| c.prefill(&h, &prompt)).1.total() as f64);
        step_ops.push(measure(|| c.decode_step(&h, &rand_vec(D_MODEL, 17))).1.total() as f64);
    }
    let xs = as_f64(&ns);
    let (s1, s2) = (loglog_slope(&xs, &step_ops), loglog_slope(&xs, &prefill_ops));
    println!("decode slope {s1:.3}, prefill slope {s2:.3}");
    assert!(s1 > 0.95 && s1 < 1.2, "decode slope {s1}");
    assert!(s2 > 0.95 && s2 < 1.2, "prefill slope {s2}");
}
