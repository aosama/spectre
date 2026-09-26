//! Multi-head SPECTRE layer: a drop-in replacement for multi-head attention.
//! Heads run in parallel; outputs are concatenated and projected by W_o.

use crate::head::{HeadConfig, SpectreHead};
use crate::nn::Linear;
use crate::tensor::Mat;
use crate::nn::seeded_rng as rng;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct LayerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_fft: usize,
    pub gate_hidden: usize,
    pub toeplitz_r: Option<usize>,
    pub wavelet_levels: Option<usize>,
}

impl LayerConfig {
    pub fn head_config(&self) -> HeadConfig {
        assert!(self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads");
        HeadConfig {
            d_model: self.d_model,
            d_head: self.d_model / self.n_heads,
            n_fft: self.n_fft,
            gate_hidden: self.gate_hidden,
            toeplitz_r: self.toeplitz_r,
            wavelet_levels: self.wavelet_levels,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpectreLayer {
    pub cfg: LayerConfig,
    pub heads: Vec<SpectreHead>,
    /// Output projection d_model × d_model.
    pub wo: Linear,
}

impl SpectreLayer {
    /// Parameters are drawn sequentially from one seeded RNG, so they do not
    /// depend on the thread count.
    pub fn new(cfg: LayerConfig, seed: u64) -> Self {
        let mut r = rng(seed);
        let hc = cfg.head_config();
        let heads = (0..cfg.n_heads).map(|_| SpectreHead::new(hc, &mut r)).collect();
        let wo = Linear::new(cfg.d_model, cfg.d_model, &mut r);
        SpectreLayer { cfg, heads, wo }
    }

    /// X (n × d_model) → n × d_model. Parallel over heads (and inside each head).
    pub fn forward(&self, x: &Mat) -> Mat {
        let outs: Vec<Mat> = self.heads.par_iter().map(|h| h.forward(x)).collect();
        self.wo.forward(&Mat::hconcat(&outs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::rand_mat;

    fn cfg() -> LayerConfig {
        LayerConfig { d_model: 16, n_heads: 4, n_fft: 32, gate_hidden: 8, toeplitz_r: Some(1), wavelet_levels: Some(2) }
    }

    #[test]
    fn preserves_shape_like_attention() {
        let l = SpectreLayer::new(cfg(), 42);
        let y = l.forward(&rand_mat(20, 16, 1));
        assert_eq!((y.rows, y.cols), (20, 16));
    }

    #[test]
    fn equals_manual_concat_of_heads() {
        let l = SpectreLayer::new(cfg(), 42);
        let x = rand_mat(32, 16, 2);
        let parts: Vec<Mat> = l.heads.iter().map(|h| h.forward(&x)).collect();
        let manual = l.wo.forward(&Mat::hconcat(&parts));
        assert_eq!(l.forward(&x), manual);
    }

    #[test]
    fn same_seed_same_layer() {
        let x = rand_mat(8, 16, 3);
        assert_eq!(SpectreLayer::new(cfg(), 7).forward(&x), SpectreLayer::new(cfg(), 7).forward(&x));
    }
}
