//! SPECTRE (arXiv:2502.18394) proof-of-concept.
pub mod fft;
pub mod gate;
pub mod nn;
pub mod opcount;
pub mod rfft;
pub mod tensor;
pub mod testutil;
pub use num_complex::Complex64 as C64;
