//! Wall-clock complexity and parallel-speedup validation. Slow and
//! machine-dependent, so ignored by default. Run with:
//!   cargo test --release --test timing -- --ignored --test-threads=1

use rayon::ThreadPoolBuilder;
use spectre::attention::NaiveAttention;
use spectre::cache::PrefixFftCache;
use spectre::complexity::{loglog_slope, time_median};
use spectre::layer::{LayerConfig, SpectreLayer};
use spectre::testutil::{rand_mat, rand_vec};

const D_MODEL: usize = 64;
const N_HEADS: usize = 4;

fn layer(n_fft: usize) -> SpectreLayer {
    SpectreLayer::new(
        LayerConfig { d_model: D_MODEL, n_heads: N_HEADS, n_fft, gate_hidden: 64, toeplitz_r: Some(2), wavelet_levels: None },
        42,
    )
}

fn spectre_time(n: usize) -> f64 {
    let l = layer(n);
    let x = rand_mat(n, D_MODEL, 1);
    time_median(5, || {
        std::hint::black_box(l.forward(&x));
    })
}

fn attention_time(n: usize) -> f64 {
    let a = NaiveAttention::new(D_MODEL, N_HEADS, 42);
    let x = rand_mat(n, D_MODEL, 2);
    time_median(3, || {
        std::hint::black_box(a.forward(&x));
    })
}

#[test]
#[ignore]
fn spectre_wallclock_scales_n_log_n() {
    let ns: Vec<f64> = (12..=16).map(|p| (1u64 << p) as f64).collect();
    let ts: Vec<f64> = ns.iter().map(|&n| spectre_time(n as usize)).collect();
    let s = loglog_slope(&ns, &ts);
    println!("spectre times {ts:?} slope {s:.3}");
    assert!(s > 0.8 && s < 1.4, "slope {s}");
}

#[test]
#[ignore]
fn attention_wallclock_scales_n_squared() {
    let ns: Vec<f64> = (10..=12).map(|p| (1u64 << p) as f64).collect();
    let ts: Vec<f64> = ns.iter().map(|&n| attention_time(n as usize)).collect();
    let s = loglog_slope(&ns, &ts);
    println!("attention times {ts:?} slope {s:.3}");
    assert!(s > 1.6 && s < 2.4, "slope {s}");
}

#[test]
#[ignore]
fn spectre_is_faster_than_attention_at_4k() {
    let (ts, ta) = (spectre_time(4096), attention_time(4096));
    println!("n=4096 spectre {ts:.4}s attention {ta:.4}s speedup {:.1}x", ta / ts);
    assert!(ts < ta);
}

#[test]
#[ignore]
fn decode_step_time_is_flat_in_t() {
    let n = 4096;
    let l = layer(n);
    let h = &l.heads[0];
    let mut c = PrefixFftCache::new(h, None);
    c.prefill(h, &rand_mat(16, D_MODEL, 3));
    let x = rand_vec(D_MODEL, 4);
    let mut per_phase = vec![];
    for _ in 0..4 {
        // advance ~N tokens, then time 51 steps
        for _ in 0..n {
            c.decode_step(h, &x);
        }
        per_phase.push(time_median(51, || {
            std::hint::black_box(c.decode_step(h, &x));
        }));
    }
    let (mn, mx) = per_phase.iter().fold((f64::MAX, 0f64), |(a, b), &t| (a.min(t), b.max(t)));
    println!("decode step times at t≈N,2N,3N,4N: {per_phase:?}");
    assert!(mx / mn < 2.0, "decode time grows with t: {per_phase:?}");
}

#[test]
#[ignore]
fn uses_all_cores_for_speedup() {
    let p = std::thread::available_parallelism().unwrap().get();
    if p < 2 {
        return;
    }
    let n = 1 << 15;
    let l = layer(n);
    let x = rand_mat(n, D_MODEL, 5);
    let run = |threads: usize| {
        ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
            time_median(3, || {
                std::hint::black_box(l.forward(&x));
            })
        })
    };
    let (t1, tp) = (run(1), run(p));
    let speedup = t1 / tp;
    println!("1 thread {t1:.3}s, {p} threads {tp:.3}s, speedup {speedup:.2}x");
    assert!(speedup > 0.3 * p as f64, "speedup {speedup:.2} too low for {p} cores");
}
