//! Helpers for empirical complexity validation: log-log slope fitting,
//! normalized-cost spread, and robust wall-clock timing.

use std::time::Instant;

/// Least-squares slope of ln(y) against ln(x). y ∝ x^p gives p.
pub fn loglog_slope(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    assert!(xs.len() >= 2, "need at least two points");
    let lx: Vec<f64> = xs.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|v| v.ln()).collect();
    let n = lx.len() as f64;
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
    let den: f64 = lx.iter().map(|a| (a - mx).powi(2)).sum();
    num / den
}

pub fn nlog2n(n: f64) -> f64 {
    n * n.log2()
}

/// max(y/model(x)) / min(y/model(x)). Close to 1 when y ∝ model.
pub fn spread(xs: &[f64], ys: &[f64], model: impl Fn(f64) -> f64) -> f64 {
    let r: Vec<f64> = xs.iter().zip(ys).map(|(x, y)| y / model(*x)).collect();
    r.iter().cloned().fold(f64::MIN, f64::max) / r.iter().cloned().fold(f64::MAX, f64::min)
}

pub fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median wall-clock seconds of `reps` runs of `f`, after one warm-up run.
pub fn time_median(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    median(&mut ts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow2s(a: u32, b: u32) -> Vec<f64> {
        (a..=b).map(|p| (1u64 << p) as f64).collect()
    }

    #[test]
    fn slope_recovers_exact_power_laws() {
        let xs = pow2s(4, 12);
        let quad: Vec<f64> = xs.iter().map(|x| 3.0 * x * x).collect();
        let lin: Vec<f64> = xs.iter().map(|x| 7.0 * x).collect();
        assert!((loglog_slope(&xs, &quad) - 2.0).abs() < 1e-12);
        assert!((loglog_slope(&xs, &lin) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nlogn_slope_is_slightly_above_one() {
        let xs = pow2s(8, 16);
        let ys: Vec<f64> = xs.iter().map(|&x| nlog2n(x)).collect();
        let s = loglog_slope(&xs, &ys);
        assert!(s > 1.05 && s < 1.15, "slope {s}");
    }

    #[test]
    fn spread_is_one_for_exact_model() {
        let xs = pow2s(8, 14);
        let ys: Vec<f64> = xs.iter().map(|&x| 5.0 * nlog2n(x)).collect();
        assert!((spread(&xs, &ys, nlog2n) - 1.0).abs() < 1e-12);
        let quad: Vec<f64> = xs.iter().map(|x| x * x).collect();
        assert!(spread(&xs, &quad, nlog2n) > 10.0);
    }

    #[test]
    fn median_of_odd_list() {
        assert_eq!(median(&mut [5.0, 1.0, 3.0]), 3.0);
    }
}
