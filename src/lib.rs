//! SPECTRE (arXiv:2502.18394) proof-of-concept.
pub mod attention;
pub mod cache;
pub mod fft;
pub mod gate;
pub mod head;
pub mod layer;
pub mod nn;
pub mod opcount;
pub mod rfft;
pub mod tensor;
pub mod testutil;
pub mod wavelet;
pub use num_complex::Complex64 as C64;
