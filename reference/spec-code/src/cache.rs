//! Prefix–FFT cache (paper §3.3, Algorithm 1) with optional persistent memory
//! (§3.4). Invariant maintained after every call:
//!     prefix_fft == R_{n_fft}(window placed at rows [n_mem, n_fft), zeros above)
//! where the window is the ring buffer V_buf (slot = t mod n_max).
//! With memory M (n_mem rows), the effective sequence is [M ; V_buf] and its
//! spectrum is mem_fft + prefix_fft (linearity; mem_fft is computed once).

use crate::head::SpectreHead;
use crate::opcount::{self, Op};
use crate::rfft::{num_bins, rfft_cols};
use crate::tensor::{CMat, Mat};
use crate::C64;
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct PrefixFftCache {
    pub n_fft: usize,
    pub n_mem: usize,
    /// Sliding-window capacity N_max = n_fft − n_mem.
    pub n_max: usize,
    pub d: usize,
    /// (n_fft/2+1) × d spectrum of the placed window.
    pub prefix_fft: CMat,
    /// Spectrum of [M ; 0], fixed for the whole session. None without memory.
    pub mem_fft: Option<CMat>,
    pub v_buf: Mat,
    pub q_buf: Mat,
    pub sum_q: Vec<f64>,
    /// Pre-cached twiddles w[m] = e^{−j2πm/n_fft}, m < n_fft.
    pub twiddles: Vec<C64>,
    /// Number of tokens consumed so far.
    pub t: usize,
}

impl PrefixFftCache {
    /// `memory`, if given, is M ∈ R^{n_mem × d_head} (value space).
    pub fn new(head: &SpectreHead, memory: Option<&Mat>) -> Self {
        let n_fft = head.cfg.n_fft;
        let d = head.cfg.d_head;
        let n_mem = memory.map_or(0, |m| m.rows);
        assert!(n_mem < n_fft, "memory must leave room for the window");
        if let Some(m) = memory {
            assert_eq!(m.cols, d, "memory width must equal d_head");
        }
        let n_max = n_fft - n_mem;
        PrefixFftCache {
            n_fft,
            n_mem,
            n_max,
            d,
            prefix_fft: CMat::zeros(num_bins(n_fft), d),
            mem_fft: memory.map(|m| rfft_cols(m, n_fft)),
            v_buf: Mat::zeros(n_max, d),
            q_buf: Mat::zeros(n_max, d),
            sum_q: vec![0.0; d],
            twiddles: (0..n_fft).map(|m| C64::from_polar(1.0, -2.0 * PI * m as f64 / n_fft as f64)).collect(),
            t: 0,
        }
    }

    /// Number of live context rows L' = min(t, N_max).
    pub fn live_len(&self) -> usize {
        self.t.min(self.n_max)
    }

    /// The window placed in an n_fft-row matrix: rows [n_mem, n_fft) = V_buf.
    pub fn placed_window(&self) -> Mat {
        Mat::from_fn(self.n_fft, self.d, |r, c| if r >= self.n_mem { self.v_buf.get(r - self.n_mem, c) } else { 0.0 })
    }

    /// §3.3.1 Pre-fill: one padded n_fft-point RFFT over the prompt (1 ≤ L ≤ N_max).
    /// Returns the live context output (same as `output`).
    pub fn prefill(&mut self, head: &SpectreHead, x: &Mat) -> Mat {
        assert_eq!(self.t, 0, "prefill must be the first call");
        assert!(x.rows >= 1 && x.rows <= self.n_max, "prefill: need 1 ≤ L ≤ N_max");
        let (q, v) = head.project(x);
        for i in 0..x.rows {
            self.v_buf.row_mut(i).copy_from_slice(v.row(i));
            self.q_buf.row_mut(i).copy_from_slice(q.row(i));
            self.sum_q.iter_mut().zip(q.row(i)).for_each(|(s, qi)| *s += qi);
        }
        self.prefix_fft = rfft_cols(&self.placed_window(), self.n_fft);
        self.t = x.rows;
        self.output(head)
    }

    /// Eq. (5) for one ring slot: prefix_fft[k] += delta · e^{−j2πk·pos/n_fft},
    /// pos = n_mem + slot. With delta = v_t − v_old this is exactly "evict v_old,
    /// add v_t", because (t − N_max) ≡ t (mod N_max) gives both terms the same
    /// twiddle. Parallel over bins; counts F·d `Op::CMul`.
    pub fn update_spectrum(&mut self, slot: usize, delta: &[f64]) {
        let pos = self.n_mem + slot;
        let n_fft = self.n_fft;
        let tw = &self.twiddles;
        self.prefix_fft.data.par_chunks_mut(self.d).enumerate().for_each(|(k, row)| {
            let w = tw[(k * pos) % n_fft];
            row.iter_mut().zip(delta).for_each(|(p, dv)| *p += w * *dv);
        });
        opcount::add(Op::CMul, (self.prefix_fft.rows * self.d) as u64);
    }

    /// §3.3.2 Decode one token x_t (length d_model). Returns the live context output.
    pub fn decode_step(&mut self, head: &SpectreHead, x_t: &[f64]) -> Mat {
        let t = self.t;
        let slot = t % self.n_max;
        let (q_t, v_t) = head.project_token(x_t);
        // (a) evict & update FFT cache (v_old = 0 while the window is filling).
        let (v_old, q_old) = if t >= self.n_max {
            (self.v_buf.row(slot).to_vec(), self.q_buf.row(slot).to_vec())
        } else {
            (vec![0.0; self.d], vec![0.0; self.d])
        };
        let delta: Vec<f64> = v_t.iter().zip(&v_old).map(|(a, b)| a - b).collect();
        self.update_spectrum(slot, &delta);
        // (b) refresh ring buffers and running descriptor.
        self.v_buf.row_mut(slot).copy_from_slice(&v_t);
        self.q_buf.row_mut(slot).copy_from_slice(&q_t);
        for ((s, qn), qo) in self.sum_q.iter_mut().zip(&q_t).zip(&q_old) {
            *s += qn - qo;
        }
        self.t = t + 1;
        // (c)–(e)
        self.output(head)
    }

    /// Steps (c)–(e): gate from LN(sum_q / N_max), positional phase, inverse RFFT.
    /// Without memory: returns the last L' rows (chronological order, newest last);
    /// the phase shift t makes row m hold ring slot (m + t) mod N_max.
    /// With memory: returns all n_fft rows in placed order [M ; V_buf] and no
    /// phase rotation is applied (a rotation would move the memory block).
    pub fn output(&self, head: &SpectreHead) -> Mat {
        let q_mean: Vec<f64> = self.sum_q.iter().map(|s| s / self.n_max as f64).collect();
        match &self.mem_fft {
            None => {
                let mixed = head.mix_spectrum(&self.prefix_fft, &q_mean, self.t % self.n_fft);
                mixed.slice_rows(self.n_max - self.live_len(), self.n_max)
            }
            Some(m) => head.mix_spectrum(&m.add(&self.prefix_fft), &q_mean, 0),
        }
    }

    /// Bytes held by the cache; must not grow with t (O(n_fft·d)).
    pub fn footprint_bytes(&self) -> usize {
        let c = std::mem::size_of::<C64>();
        let f = std::mem::size_of::<f64>();
        self.prefix_fft.data.len() * c
            + self.mem_fft.as_ref().map_or(0, |m| m.data.len() * c)
            + (self.v_buf.data.len() + self.q_buf.data.len() + self.sum_q.len()) * f
            + self.twiddles.len() * c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::head::HeadConfig;
    use crate::testutil::{rand_cvec, rand_mat, rand_vec, rng};

    const D_MODEL: usize = 12;

    fn head(n_fft: usize) -> SpectreHead {
        let cfg = HeadConfig { d_model: D_MODEL, d_head: 4, n_fft, gate_hidden: 16, toeplitz_r: Some(2), wavelet_levels: None };
        let mut h = SpectreHead::new(cfg, &mut rng(1));
        h.gate.toeplitz = Some(rand_cvec(5, 2).iter().map(|v| v * 0.3).collect());
        h
    }

    /// Chronological window of the last N_max tokens (zero rows before token 0),
    /// computed directly from the projected values of all tokens seen.
    fn unrolled_window(all_v: &Mat, t: usize, n_max: usize) -> Mat {
        Mat::from_fn(n_max, all_v.cols, |m, c| {
            let idx = t as isize - n_max as isize + m as isize;
            if idx >= 0 { all_v.get(idx as usize, c) } else { 0.0 }
        })
    }

    #[test]
    fn prefill_spectrum_equals_padded_rfft() {
        let h = head(32);
        let x = rand_mat(20, D_MODEL, 3);
        let mut c = PrefixFftCache::new(&h, None);
        c.prefill(&h, &x);
        let (_, v) = h.project(&x);
        assert!(c.prefix_fft.max_abs_diff(&rfft_cols(&v, 32)) < 1e-12);
        assert_eq!(c.live_len(), 20);
    }

    #[test]
    fn full_window_prefill_equals_head_forward() {
        let h = head(32);
        let x = rand_mat(32, D_MODEL, 4);
        let mut c = PrefixFftCache::new(&h, None);
        let y = c.prefill(&h, &x);
        assert!(y.max_abs_diff(&h.forward(&x)) < 1e-10);
    }

    #[test]
    fn spectrum_invariant_holds_through_wraparound() {
        let h = head(16);
        let xs = rand_mat(16 * 5, D_MODEL, 5);
        let mut c = PrefixFftCache::new(&h, None);
        c.prefill(&h, &xs.slice_rows(0, 3));
        for t in 3..xs.rows {
            c.decode_step(&h, xs.row(t));
            let d = c.prefix_fft.max_abs_diff(&rfft_cols(&c.placed_window(), 16));
            assert!(d < 1e-10, "t={t} drift={d}");
        }
    }

    #[test]
    fn decode_matches_recomputation_from_scratch() {
        let n = 16;
        let h = head(n);
        let xs = rand_mat(n * 3 + 5, D_MODEL, 6);
        let (all_q, all_v) = h.project(&xs);
        let mut c = PrefixFftCache::new(&h, None);
        c.prefill(&h, &xs.slice_rows(0, 5));
        for t in 5..xs.rows {
            let y = c.decode_step(&h, xs.row(t));
            let seen = t + 1;
            let live = seen.min(n);
            let mut q_sum = vec![0.0; 4];
            for i in seen - live..seen {
                q_sum.iter_mut().zip(all_q.row(i)).for_each(|(s, q)| *s += q);
            }
            assert!(crate::testutil::max_abs_diff_r(&q_sum, &c.sum_q) < 1e-10, "sum_q at t={t}");
            let q_mean: Vec<f64> = q_sum.iter().map(|s| s / n as f64).collect();
            let expected = h.mix(&unrolled_window(&all_v, seen, n), &q_mean, 0).slice_rows(n - live, n);
            assert_eq!(y.rows, live);
            assert!(y.max_abs_diff(&expected) < 1e-10, "t={t}");
        }
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let h = head(8);
        let xs = rand_mat(11, D_MODEL, 7);
        let mut c = PrefixFftCache::new(&h, None);
        c.prefill(&h, &xs.slice_rows(0, 8));
        for t in 8..11 {
            c.decode_step(&h, xs.row(t));
        }
        let (_, v) = h.project(&xs);
        for t in 3..11 {
            assert_eq!(c.v_buf.row(t % 8), v.row(t), "slot {}", t % 8);
        }
    }

    #[test]
    fn footprint_is_constant_across_steps() {
        let h = head(32);
        let mut c = PrefixFftCache::new(&h, None);
        c.prefill(&h, &rand_mat(4, D_MODEL, 8));
        let before = c.footprint_bytes();
        for i in 0..100 {
            c.decode_step(&h, &rand_vec(D_MODEL, 100 + i));
        }
        assert_eq!(c.footprint_bytes(), before);
        // (n/2+1)·d complex + 2·n·d + d reals + n complex twiddles
        assert_eq!(before, 17 * 4 * 16 + (2 * 32 * 4 + 4) * 8 + 32 * 16);
    }

    #[test]
    fn persistent_memory_is_static_and_output_matches_oracle() {
        let h = head(32);
        let mem = rand_mat(8, 4, 9);
        let mut c = PrefixFftCache::new(&h, Some(&mem));
        assert_eq!(c.n_max, 24);
        let mem_fft0 = c.mem_fft.clone().unwrap();
        let xs = rand_mat(60, D_MODEL, 10);
        c.prefill(&h, &xs.slice_rows(0, 10));
        for t in 10..60 {
            let y = c.decode_step(&h, xs.row(t));
            let placed = Mat::from_fn(32, 4, |r, col| if r < 8 { mem.get(r, col) } else { c.v_buf.get(r - 8, col) });
            let q_mean: Vec<f64> = c.sum_q.iter().map(|s| s / 24.0).collect();
            assert!(y.max_abs_diff(&h.mix(&placed, &q_mean, 0)) < 1e-10, "t={t}");
        }
        assert_eq!(c.mem_fft.unwrap(), mem_fft0, "memory spectrum must never change");
    }
}
