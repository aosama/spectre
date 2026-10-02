//! Measures every reproducible claim of arXiv:2502.18394 on this machine and
//! writes a Markdown verdict table.
//! Run: cargo run --release --features opcount --bin claims_report -- docs/claims-report.md

use spectre::attention::NaiveAttention;
use spectre::cache::PrefixFftCache;
use spectre::complexity::{loglog_slope, time_median};
use spectre::head::{forward_reference, HeadConfig, SpectreHead};
use spectre::layer::{LayerConfig, SpectreLayer};
use spectre::opcount::measure;
use spectre::rfft::{irfft, num_bins, rfft};
use spectre::rfft::rfft_cols;
use spectre::testutil::{max_abs_diff_r, rand_mat, rand_vec, rng};
use spectre::train::{make_shift_task, normalized_loss, task_head, train_gate};
use std::fmt::Write as _;

struct Row {
    id: &'static str,
    claim: &'static str,
    source: &'static str,
    measured: String,
    pass: bool,
}

const D_MODEL: usize = 64;
const N_HEADS: usize = 4;

fn layer(n: usize) -> SpectreLayer {
    SpectreLayer::new(
        LayerConfig { d_model: D_MODEL, n_heads: N_HEADS, n_fft: n, gate_hidden: 64, toeplitz_r: Some(2), wavelet_levels: Some(3) },
        42,
    )
}

fn head(n: usize) -> SpectreHead {
    let cfg = HeadConfig { d_model: D_MODEL, d_head: 16, n_fft: n, gate_hidden: 64, toeplitz_r: Some(2), wavelet_levels: None };
    SpectreHead::new(cfg, &mut rng(1))
}

fn pow2(a: u32, b: u32) -> Vec<usize> {
    (a..=b).map(|p| 1usize << p).collect()
}

fn f(v: &[usize]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "docs/claims-report.md".into());
    let mut rows: Vec<Row> = vec![];

    // C1–C3: RFFT facts (Fig. 2, Appendix B).
    let x = rand_vec(4096, 1);
    let spec = rfft(&x);
    rows.push(Row { id: "C1", claim: "RFFT keeps only ⌊n/2⌋+1 coefficients", source: "§2, Fig. 2, eq. 1", measured: format!("n=4096 → {} bins", spec.len()), pass: spec.len() == num_bins(4096) });
    let full = spectre::fft::fft(&x.iter().map(|&v| spectre::C64::new(v, 0.0)).collect::<Vec<_>>());
    let herm = (1..4096).map(|k| (full[4096 - k] - full[k].conj()).norm()).fold(0.0, f64::max);
    rows.push(Row { id: "C2", claim: "Hermitian symmetry X_{n−k} = conj(X_k)", source: "Thm. B.1", measured: format!("max violation {herm:.2e}"), pass: herm < 1e-9 });
    let rt = max_abs_diff_r(&irfft(&spec, 4096), &x);
    rows.push(Row { id: "C3", claim: "Half spectrum reconstructs the signal losslessly", source: "Cor. B.2", measured: format!("round-trip error {rt:.2e}"), pass: rt < 1e-10 });

    // C4: layer = R^{-1}(diag(g) R(V)) == circular convolution (method correctness).
    let h = head(256);
    let xm = rand_mat(256, D_MODEL, 2);
    let e = h.forward(&xm).max_abs_diff(&forward_reference(&h, &xm));
    rows.push(Row { id: "C4", claim: "Spectral gating = global (circular) token mixing via FFT", source: "§2 'Spectral shortcut', §3.2", measured: format!("FFT path vs O(n²) conv oracle: {e:.2e}"), pass: e < 1e-9 });

    // C5: per-layer O(n d log n) (op counts).
    let ns = pow2(8, 14);
    let ops: Vec<f64> = ns.iter().map(|&n| { let l = layer(n); let x = rand_mat(n, D_MODEL, 3); measure(|| l.forward(&x)).1.total() as f64 }).collect();
    let s_ops = loglog_slope(&f(&ns), &ops);
    rows.push(Row { id: "C5", claim: "Per-layer cost O(n·d·log n)", source: "Abstract, §3, Table 6", measured: format!("op-count log-log slope {s_ops:.3} over n=2^8..2^14 (n log n ⇒ ≈1.0–1.15)"), pass: s_ops > 0.95 && s_ops < 1.2 });
    let ns_a = pow2(9, 12);
    let att = NaiveAttention::new(D_MODEL, N_HEADS, 42);
    let ops_a: Vec<f64> = ns_a.iter().map(|&n| { let x = rand_mat(n, D_MODEL, 4); measure(|| att.forward(&x)).1.total() as f64 }).collect();
    let s_att = loglog_slope(&f(&ns_a), &ops_a);
    rows.push(Row { id: "C6", claim: "Self-attention baseline is O(n²·d)", source: "§1, §2", measured: format!("op-count slope {s_att:.3}"), pass: s_att > 1.8 && s_att < 2.05 });

    // C7–C8: wall-clock scaling and crossover (CPU analogue of Fig. 1 / Table 7).
    let ns_t = pow2(12, 15);
    let ts: Vec<f64> = ns_t.iter().map(|&n| { let l = layer(n); let x = rand_mat(n, D_MODEL, 5); time_median(3, || { std::hint::black_box(l.forward(&x)); }) }).collect();
    let s_t = loglog_slope(&f(&ns_t), &ts);
    rows.push(Row { id: "C7", claim: "Near-O(n log n) runtime in practice", source: "Fig. 1, §4.6(i)", measured: format!("wall-clock slope {s_t:.3} over n=2^12..2^15"), pass: s_t > 0.8 && s_t < 1.4 });
    let mut ratios = vec![];
    for n in [512usize, 1024, 2048, 4096] {
        let l = layer(n);
        let x = rand_mat(n, D_MODEL, 6);
        let t_s = time_median(3, || { std::hint::black_box(l.forward(&x)); });
        let t_a = time_median(3, || { std::hint::black_box(att.forward(&x)); });
        ratios.push((n, t_a / t_s));
    }
    let grows = ratios.last().unwrap().1 > 2.0 * ratios[0].1;
    rows.push(Row { id: "C8", claim: "Speed-up over attention grows with context length", source: "Fig. 1, Table 7", measured: format!("attention/SPECTRE time: {}", ratios.iter().map(|(n, r)| format!("n={n}: {r:.1}×")).collect::<Vec<_>>().join(", ")), pass: grows && ratios.last().unwrap().1 > 1.0 });

    // C9–C12: Prefix-FFT cache.
    let n = 64;
    let hc = head(n);
    let xs = rand_mat(4 * n, D_MODEL, 7);
    let (all_q, all_v) = hc.project(&xs);
    let mut c = PrefixFftCache::new(&hc, None);
    c.prefill(&hc, &xs.slice_rows(0, 1));
    let mut max_err: f64 = 0.0;
    let mut max_drift: f64 = 0.0;
    let mut step_costs = vec![];
    let fp0 = c.footprint_bytes();
    for t in 1..xs.rows {
        let (y, s) = measure(|| c.decode_step(&hc, xs.row(t)));
        step_costs.push(s);
        let seen = t + 1;
        let live = seen.min(n);
        let window = spectre::tensor::Mat::from_fn(n, 16, |m, col| { let i = seen as isize - n as isize + m as isize; if i >= 0 { all_v.get(i as usize, col) } else { 0.0 } });
        let mut qs = vec![0.0; 16];
        for i in seen - live..seen { qs.iter_mut().zip(all_q.row(i)).for_each(|(a, b)| *a += b); }
        let qm: Vec<f64> = qs.iter().map(|v| v / n as f64).collect();
        max_err = max_err.max(y.max_abs_diff(&hc.mix(&window, &qm, 0).slice_rows(n - live, n)));
        max_drift = max_drift.max(c.prefix_fft.max_abs_diff(&rfft_cols(&c.placed_window(), n)));
    }
    rows.push(Row { id: "C9", claim: "Prefix-FFT decode equals recomputing the window from scratch", source: "§3.3, Alg. 1", measured: format!("max error {max_err:.2e}, spectrum drift {max_drift:.2e} over {} steps (4 wrap-arounds)", xs.rows - 1), pass: max_err < 1e-9 && max_drift < 1e-9 });
    let constant = step_costs.iter().all(|s| *s == step_costs[0]);
    rows.push(Row { id: "C10", claim: "Constant per-token decode cost (independent of t)", source: "§3.3.2, Table 1 TPOT", measured: format!("identical op count every step: {constant} ({} ops/step)", step_costs[0].total()), pass: constant });
    let ns_c = pow2(8, 14);
    let mut upd = vec![];
    let mut stp = vec![];
    for &nn in &ns_c {
        let hh = head(nn);
        let mut cc = PrefixFftCache::new(&hh, None);
        cc.prefill(&hh, &rand_mat(1, D_MODEL, 8));
        let d = rand_vec(16, 9);
        upd.push(measure(|| cc.update_spectrum(0, &d)).1.total() as f64);
        stp.push(measure(|| cc.decode_step(&hh, &rand_vec(D_MODEL, 10))).1.total() as f64);
    }
    let (s_u, s_s) = (loglog_slope(&f(&ns_c), &upd), loglog_slope(&f(&ns_c), &stp));
    rows.push(Row { id: "C11", claim: "Decode step costs O(N_max/2 · d)", source: "§3.3.2, Fig. 6", measured: format!("cache update (eq. 5) slope {s_u:.3} = O(N·d) ✔; full step incl. inverse RFFT slope {s_s:.3} = O(N·d·log N)"), pass: (s_u - 1.0).abs() < 0.02 && s_s < 1.2 });
    rows.push(Row { id: "C12", claim: "Cache memory O(N_max·d), constant during decode", source: "§3.3.2", measured: format!("{fp0} bytes before and {} bytes after {} steps", c.footprint_bytes(), xs.rows - 1), pass: fp0 == c.footprint_bytes() });

    // C13: persistent memory is static and exact.
    let mem = rand_mat(8, 16, 11);
    let mut cm = PrefixFftCache::new(&hc, Some(&mem));
    let m0 = cm.mem_fft.clone().unwrap();
    cm.prefill(&hc, &xs.slice_rows(0, 5));
    let mut e_mem: f64 = 0.0;
    for t in 5..xs.rows {
        let y = cm.decode_step(&hc, xs.row(t));
        let placed = spectre::tensor::Mat::from_fn(n, 16, |r, col| if r < 8 { mem.get(r, col) } else { cm.v_buf.get(r - 8, col) });
        let qm: Vec<f64> = cm.sum_q.iter().map(|s| s / cm.n_max as f64).collect();
        e_mem = e_mem.max(y.max_abs_diff(&hc.mix(&placed, &qm, 0)));
    }
    let static_mem = cm.mem_fft.as_ref().unwrap() == &m0;
    rows.push(Row { id: "C13", claim: "Persistent memory: FFT computed once, never changes, exact", source: "§3.4", measured: format!("memory spectrum unchanged: {static_mem}; max error {e_mem:.2e}"), pass: static_mem && e_mem < 1e-9 });

    // C14: WRM is O(n·d) and orthogonal.
    let ns_w = pow2(8, 14);
    let wops: Vec<f64> = ns_w.iter().map(|&nn| { let x = rand_vec(nn, 12); measure(|| spectre::wavelet::haar_forward(&x, 3)).1.total() as f64 }).collect();
    let s_w = loglog_slope(&f(&ns_w), &wops);
    let xw = rand_vec(1024, 13);
    let cw = spectre::wavelet::haar_forward(&xw, 5);
    let e_rel = (xw.iter().map(|v| v * v).sum::<f64>() - cw.iter().map(|v| v * v).sum::<f64>()).abs();
    rows.push(Row { id: "C14", claim: "Wavelet refinement is orthogonal and O(n·d)", source: "§3.5, Table 6", measured: format!("DWT op slope {s_w:.3}; energy change {e_rel:.2e}"), pass: (s_w - 1.0).abs() < 0.02 && e_rel < 1e-9 });

    // C15: content-adaptive gating works (learning demo) and beats a fixed filter.
    let tr = make_shift_task(32, 4, 64, 1);
    let te = make_shift_task(32, 4, 64, 2);
    let mut ha = task_head(32, 4, 16, 3);
    train_gate(&mut ha, &tr, 400, 0.02, true);
    let la = normalized_loss(&ha, &te);
    let mut hf = task_head(32, 4, 16, 3);
    train_gate(&mut hf, &tr, 400, 0.02, false);
    let lf = normalized_loss(&hf, &te);
    rows.push(Row { id: "C15", claim: "Content-adaptive gate adds expressivity beyond fixed spectral filters (FNet)", source: "§1, §2 'Spectral token mixers'", measured: format!("held-out loss: adaptive {la:.4} vs fixed {lf:.4}"), pass: la < 0.05 && lf > 0.2 });

    // C16: parallel speed-up (requirement of this PoC, not a paper claim).
    let p = std::thread::available_parallelism().unwrap().get();
    let l = layer(1 << 14);
    let xp = rand_mat(1 << 14, D_MODEL, 14);
    let t1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| time_median(3, || { std::hint::black_box(l.forward(&xp)); }));
    let tp = time_median(3, || { std::hint::black_box(l.forward(&xp)); });
    rows.push(Row { id: "P1", claim: "(PoC requirement) Uses all CPU cores", source: "plan", measured: format!("{p} threads: speed-up {:.2}× over 1 thread", t1 / tp), pass: p < 2 || t1 / tp > 0.3 * p as f64 });

    let mut md = String::new();
    writeln!(md, "# SPECTRE claims verification report\n").unwrap();
    writeln!(md, "Generated by `cargo run --release --features opcount --bin claims_report`. Machine: {p} logical cores.\n").unwrap();
    writeln!(md, "| ID | Claim | Paper source | Measured | Verdict |\n|---|---|---|---|---|").unwrap();
    for r in &rows {
        writeln!(md, "| {} | {} | {} | {} | {} |", r.id, r.claim, r.source, r.measured, if r.pass { "PASS" } else { "FAIL" }).unwrap();
    }
    writeln!(md, "\n## Not reproducible in this PoC (by design)\n").unwrap();
    for (c, why) in [
        ("PG-19 perplexity, ImageNet-1k accuracy (Tables 2–4)", "requires full-model training on large datasets and GPUs"),
        ("Absolute GPU latencies/throughput vs FlashAttention-2 (Tables 1, 7; 7× at 128k)", "GPU-specific; C7/C8 reproduce the scaling trend on CPU instead"),
        ("RFFT ≈1.8× faster than complex FFT", "this PoC computes the RFFT via a full complex FFT (reproduction, not optimization)"),
        ("<6% extra parameters", "depends on the host model; SPECTRE's gate MLP size here scales with n_fft·gate_hidden"),
        ("Learned WRM skip controller (~90% skipped)", "requires training; WRM is a config switch here"),
    ] {
        writeln!(md, "- **{c}**: {why}.").unwrap();
    }
    let all = rows.iter().all(|r| r.pass);
    writeln!(md, "\n**Overall: {}**", if all { "ALL REPRODUCIBLE CLAIMS PASS" } else { "SOME CLAIMS FAILED" }).unwrap();
    print!("{md}");
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&out_path, &md).expect("write report");
    if !all {
        std::process::exit(1);
    }
}
