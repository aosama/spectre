"""R1 tests: the causal SPECTRE layer must be a faithful, strictly-causal
mirror of the verified Rust PoC (circular variant) with the D3 adaptation.

Run: cd realmodel && uv run python -m pytest tests/ -q
"""
import numpy as np
import torch

from spectre_torch.v1_spectre import SpectreHead, SpectreLayer, causal_conv_fft


def test_rfft_roundtrip():
    for n in (16, 64, 1024):
        x = torch.randn(n)
        y = torch.fft.irfft(torch.fft.rfft(x, n=n), n=n)
        assert (y - x).abs().max() < 1e-5, f"rfft roundtrip failed for n={n}"


def test_all_pass_gate_is_identity():
    torch.manual_seed(0)
    d_model, d_head, n_fft, hidden = 32, 8, 64, 16
    head = SpectreHead(d_model, d_head, n_fft, hidden)
    with torch.no_grad():
        head.v1_gate.l2.weight.zero_()  # g == (1, 0) for every bin
    x = torch.randn(2, 40, d_model)
    v = x @ head.wv + head.bv
    out = head(x)
    assert (out - v).abs().max() < 1e-5, "all-pass gate must return v unchanged"


def test_causal_conv_matches_direct_oracle():
    torch.manual_seed(1)
    B, n, d, N = 4, 64, 8, 64
    v = torch.randn(B, n, d)
    h = torch.randn(B, N)
    out = causal_conv_fft(v, h)
    v_np, h_np = v.numpy(), h.numpy()
    y = np.zeros((B, n, d), dtype=np.float64)
    for b in range(B):
        for m in range(n):
            idx = m - np.arange(m + 1)  # [m, m-1, ..., 0]
            y[b, m, :] = (h_np[b, : m + 1, None] * v_np[b, idx, :]).sum(axis=0)
    assert np.abs(out.numpy() - y).max() < 1e-5, "FFT path must equal direct causal convolution"


def test_causal_no_future_leakage():
    torch.manual_seed(2)
    B, n, d, N = 2, 64, 8, 64
    v = torch.randn(B, n, d)
    h = torch.randn(B, N)
    m = 32
    y1 = causal_conv_fft(v, h)
    v2 = v.clone()
    v2[:, m:, :] += torch.randn(B, n - m, d)  # perturb only future tokens
    y2 = causal_conv_fft(v2, h)
    leak = (y1[:, :m, :] - y2[:, :m, :]).abs().max()
    assert leak < 1e-4, f"future tokens leaked into past outputs (max diff {leak.item()})"


def test_shape_and_batch():
    torch.manual_seed(3)
    d_model, n_heads, n_fft, hidden = 64, 4, 1024, 32
    layer = SpectreLayer(d_model, n_heads, n_fft, hidden)
    for B in (1, 3):
        for n in (1, 1024):
            x = torch.randn(B, n, d_model)
            y = layer(x)
            assert y.shape == (B, n, d_model), f"shape mismatch for B={B}, n={n}"


def test_batched_layer_equals_per_head_loop():
    """The batched SpectreLayer.forward must equal the per-head reference
    (head.forward + concat + wo) with a NON-trivial gate — the all-pass gate
    zeroes the gate path and would hide gate-pipeline bugs."""
    torch.manual_seed(4)
    d_model, n_heads, n_fft, hidden = 32, 4, 64, 16
    layer = SpectreLayer(d_model, n_heads, n_fft, hidden)
    # Make the gate non-trivial: random l2 weights and a nonzero modReLU bias.
    with torch.no_grad():
        for head in layer.heads:
            head.v1_gate.l2.weight.normal_(0, 0.5)
            head.v1_gate.l2.bias.normal_(0, 0.1)
            head.v1_gate.modrelu_bias.normal_(0, 0.05)
    x = torch.randn(3, 48, d_model)
    outs = [head(x) for head in layer.heads]
    expected = layer.wo(torch.cat(outs, dim=-1))
    actual = layer(x)
    assert (actual - expected).abs().max() < 1e-5, (
        f"batched path diverges from per-head loop: {(actual - expected).abs().max()}"
    )
