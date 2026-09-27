//! R2 cross-validation dump: build the verified Rust SPECTRE layer with an
//! all-pass gate (g == (1,0)), run a forward pass, and dump weights + I/O as
//! JSON so the PyTorch mirror (realmodel/spectre_torch/xval.py) can be checked
//! against it. The all-pass gate makes the mixing an identity, so the Rust
//! (circular) and Python (causal) paths must agree exactly.

use serde::Serialize;
use spectre::layer::{LayerConfig, SpectreLayer};
use spectre::rfft::rfft;
use spectre::tensor::Mat;
use spectre::testutil::{rand_mat, rand_vec};

#[derive(Serialize)]
struct CfgDump {
    d_model: usize,
    n_heads: usize,
    n_fft: usize,
    gate_hidden: usize,
}

#[derive(Serialize)]
struct GateDump {
    ln_gamma: Vec<f64>,
    ln_beta: Vec<f64>,
    l1_w: Vec<Vec<f64>>,
    l1_b: Vec<f64>,
    l2_w: Vec<Vec<f64>>,
    l2_b: Vec<f64>,
    modrelu_bias: Vec<f64>,
}

#[derive(Serialize)]
struct HeadDump {
    wq: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    gate: GateDump,
}

#[derive(Serialize)]
struct Dump {
    cfg: CfgDump,
    heads: Vec<HeadDump>,
    wo_w: Vec<Vec<f64>>,
    wo_b: Vec<f64>,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rfft_input: Vec<f64>,
    rfft_output: Vec<[f64; 2]>,
}

fn mat_to_vec(m: &Mat) -> Vec<Vec<f64>> {
    (0..m.rows).map(|r| m.row(r).to_vec()).collect()
}

fn main() {
    let cfg = LayerConfig {
        d_model: 64,
        n_heads: 4,
        n_fft: 64,
        gate_hidden: 32,
        toeplitz_r: None,
        wavelet_levels: None,
    };
    let mut layer = SpectreLayer::new(cfg, 42);
    // All-pass gate: zero the l2 weights so g == (1,0) from the bias init —
    // the same trick as the PoC's all_pass_gate_returns_v test.
    for head in layer.heads.iter_mut() {
        head.gate.mlp.l2.w = Mat::zeros(head.gate.mlp.l2.w.rows, head.gate.mlp.l2.w.cols);
    }

    let x = rand_mat(48, 64, 7); // n=48 < n_fft=64: exercises the padded case
    let y = layer.forward(&x);
    let v = rand_vec(64, 11);
    let v_hat = rfft(&v);

    let heads: Vec<HeadDump> = layer
        .heads
        .iter()
        .map(|h| {
            let g = &h.gate;
            HeadDump {
                wq: mat_to_vec(&h.wq),
                wv: mat_to_vec(&h.wv),
                gate: GateDump {
                    ln_gamma: g.ln.gamma.clone(),
                    ln_beta: g.ln.beta.clone(),
                    l1_w: mat_to_vec(&g.mlp.l1.w),
                    l1_b: g.mlp.l1.b.clone(),
                    l2_w: mat_to_vec(&g.mlp.l2.w),
                    l2_b: g.mlp.l2.b.clone(),
                    modrelu_bias: g.modrelu_bias.clone(),
                },
            }
        })
        .collect();

    let dump = Dump {
        cfg: CfgDump {
            d_model: cfg.d_model,
            n_heads: cfg.n_heads,
            n_fft: cfg.n_fft,
            gate_hidden: cfg.gate_hidden,
        },
        heads,
        wo_w: mat_to_vec(&layer.wo.w),
        wo_b: layer.wo.b.clone(),
        x: mat_to_vec(&x),
        y: mat_to_vec(&y),
        rfft_input: v,
        rfft_output: v_hat.iter().map(|c| [c.re, c.im]).collect(),
    };

    let json = serde_json::to_string_pretty(&dump).unwrap();
    let out_path = format!("{}/out.json", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&out_path, &json).unwrap();
    println!("xval-dump: wrote {} ({} bytes)", out_path, json.len());
}
