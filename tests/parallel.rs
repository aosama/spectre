//! Parallelism: all cores are used by default, and results are bitwise
//! identical for any number of threads (no order-dependent reductions).

use rayon::ThreadPoolBuilder;
use spectre::attention::NaiveAttention;
use spectre::cache::PrefixFftCache;
use spectre::layer::{LayerConfig, SpectreLayer};
use spectre::tensor::Mat;
use spectre::testutil::rand_mat;

fn with_threads<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

fn thread_counts() -> Vec<usize> {
    let p = std::thread::available_parallelism().unwrap().get();
    let mut v = vec![1, 2, 3, p];
    v.dedup();
    v
}

fn layer() -> SpectreLayer {
    SpectreLayer::new(
        LayerConfig { d_model: 32, n_heads: 4, n_fft: 256, gate_hidden: 16, toeplitz_r: Some(2), wavelet_levels: Some(3) },
        42,
    )
}

#[test]
fn global_pool_uses_every_core() {
    // Fails if RAYON_NUM_THREADS is set to something else; unset it.
    assert_eq!(rayon::current_num_threads(), std::thread::available_parallelism().unwrap().get());
}

#[test]
fn spectre_layer_is_deterministic_across_thread_counts() {
    let l = layer();
    let x = rand_mat(200, 32, 1);
    let reference = with_threads(1, || l.forward(&x));
    for t in thread_counts() {
        assert_eq!(with_threads(t, || l.forward(&x)), reference, "threads={t}");
    }
}

#[test]
fn attention_is_deterministic_across_thread_counts() {
    let a = NaiveAttention::new(32, 4, 42);
    let x = rand_mat(100, 32, 2);
    let reference = with_threads(1, || a.forward(&x));
    for t in thread_counts() {
        assert_eq!(with_threads(t, || a.forward(&x)), reference, "threads={t}");
    }
}

#[test]
fn decode_is_deterministic_across_thread_counts() {
    let l = layer();
    let h = &l.heads[0];
    let xs = rand_mat(600, 32, 3);
    let run = || {
        let mut c = PrefixFftCache::new(h, None);
        let mut outs: Vec<Mat> = vec![c.prefill(h, &xs.slice_rows(0, 10))];
        for t in 10..xs.rows {
            outs.push(c.decode_step(h, xs.row(t)));
        }
        outs
    };
    let reference = with_threads(1, run);
    for t in thread_counts() {
        assert_eq!(with_threads(t, run), reference, "threads={t}");
    }
}
