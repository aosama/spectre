//! Proof that SPECTRE "actually works": its content-adaptive gate learns a
//! task that no fixed (FNet-style) spectral filter can solve.
//! Run with: cargo test --release --test learning -- --nocapture

use spectre::train::{make_shift_task, normalized_loss, task_head, train_gate};

const N: usize = 32;
const D: usize = 4;
const HIDDEN: usize = 16;
const STEPS: usize = 400;
const LR: f64 = 0.02;

#[test]
fn adaptive_gate_learns_content_dependent_delay_and_generalizes() {
    let train = make_shift_task(N, D, 64, 1);
    let test = make_shift_task(N, D, 64, 2);
    let mut head = task_head(N, D, HIDDEN, 3);
    let before = normalized_loss(&head, &test);
    let hist = train_gate(&mut head, &train, STEPS, LR, true);
    let after_train = normalized_loss(&head, &train);
    let after_test = normalized_loss(&head, &test);
    println!("adaptive: test loss {before:.4} -> {after_test:.4} (train {after_train:.4}); batch loss {:.4} -> {:.4}", hist[0], hist[STEPS - 1]);
    assert!(hist[STEPS - 1] < hist[0] * 0.1, "training did not reduce loss");
    assert!(after_train < 0.05, "train loss {after_train}");
    assert!(after_test < 0.05, "held-out loss {after_test}");
}

/// Best possible loss of any content-independent filter on this task: the DC
/// (per-channel mean) part of Y is unchanged by any delay, so it can be fit;
/// on the remaining energy, the optimal fixed filter is the average of the two
/// delay filters, whose error is half of that energy. Bound = 0.5·(1 − DC share).
fn fixed_filter_optimum(data: &[spectre::train::Sample]) -> f64 {
    let (mut total, mut dc) = (0.0, 0.0);
    for s in data {
        total += s.y.data.iter().map(|v| v * v).sum::<f64>();
        dc += s.y.col_means().iter().map(|m| m * m * s.y.rows as f64).sum::<f64>();
    }
    0.5 * (1.0 - dc / total)
}

#[test]
fn fixed_gate_cannot_solve_the_task() {
    let train = make_shift_task(N, D, 64, 1);
    let test = make_shift_task(N, D, 64, 2);
    let mut head = task_head(N, D, HIDDEN, 3);
    train_gate(&mut head, &train, STEPS, LR, false);
    let l = normalized_loss(&head, &test);
    let bound = fixed_filter_optimum(&test);
    println!("fixed gate: test loss {l:.4}, theoretical optimum {bound:.4}");
    assert!(l > bound - 0.03, "fixed filter beat its theoretical optimum: {l} < {bound}");
    assert!(l < bound + 0.05, "fixed-gate training did not converge near its optimum: {l} vs {bound}");
    assert!(l > 0.2, "task too easy for a fixed filter: {l}");
}
