//! Global operation counters used to validate asymptotic complexity
//! deterministically. Every function in `add` is a no-op unless the crate is
//! built with `--features opcount`.

use std::sync::{Mutex, MutexGuard};

/// Kinds of counted primitive operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// One radix-2 FFT butterfly (1 complex multiply + 2 complex adds).
    Butterfly = 0,
    /// One real multiply-accumulate (matmul, attention, time-domain oracles).
    Mac = 1,
    /// One complex multiply in element-wise spectral code (gating, cache update, Toeplitz).
    CMul = 2,
    /// One Haar wavelet pair operation (2 outputs from 2 inputs).
    Haar = 3,
}

/// A copy of all counters at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub butterfly: u64,
    pub mac: u64,
    pub cmul: u64,
    pub haar: u64,
}

impl Snapshot {
    pub fn total(&self) -> u64 {
        self.butterfly + self.mac + self.cmul + self.haar
    }
}

#[cfg(feature = "opcount")]
mod imp {
    use std::sync::atomic::AtomicU64;
    pub static COUNTERS: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
}

/// Add `n` operations of kind `op`. Compiles to nothing without the feature.
#[inline(always)]
pub fn add(op: Op, n: u64) {
    #[cfg(feature = "opcount")]
    imp::COUNTERS[op as usize].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "opcount"))]
    let _ = (op, n);
}

/// Reset all counters to zero.
pub fn reset() {
    #[cfg(feature = "opcount")]
    for c in imp::COUNTERS.iter() {
        c.store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Read all counters.
pub fn snapshot() -> Snapshot {
    #[cfg(feature = "opcount")]
    {
        let g = |i: usize| imp::COUNTERS[i].load(std::sync::atomic::Ordering::SeqCst);
        Snapshot { butterfly: g(0), mac: g(1), cmul: g(2), haar: g(3) }
    }
    #[cfg(not(feature = "opcount"))]
    Snapshot::default()
}

static LOCK: Mutex<()> = Mutex::new(());

/// Tests that read counters must hold this lock so concurrently running tests
/// in the same binary do not pollute each other's counts.
pub fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Reset counters, run `f`, and return the counts it produced.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, Snapshot) {
    reset();
    let r = f();
    (r, snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_with_feature() {
        let _g = lock();
        let (_, s) = measure(|| add(Op::Mac, 5));
        if cfg!(feature = "opcount") {
            // Other unit tests may run concurrently and add more, never less.
            assert!(s.mac >= 5);
        } else {
            assert_eq!(s, Snapshot::default());
        }
    }
}
