"""R9 tests: official gate + causal mixing must be strictly causal and near-identity at init.

Run: cd realmodel && uv run python -m pytest tests/ -q
"""
import torch
from transformers import GPT2LMHeadModel

from spectre_torch.official import official
from spectre_torch.official_causal import (
    CausalSpectreHead,
    CausalSpectreMultiHead,
    make_gate_near_identity,
    swap_gpt2_attention_official_causal,
)
from spectre_torch.surgery import freeze_backbone

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _head(d=16, n_fft=32, groups=4):
    return CausalSpectreHead(d, fft_size=n_fft, d_gate=32, num_groups=groups)


def test_causal_head_no_future_leak():
    """Spike at t=2 must not reach outputs at t<2 (contrast: official circular head leaks)."""
    head = _head()
    x = torch.zeros(1, 32, 16)
    x[0, 2, :] = 1.0
    y = head(x)
    assert y[0, :2, :].abs().max().item() < 1e-5


def test_causal_multihead_future_input_invariance():
    """Changing future inputs must not change past outputs (the clean causality probe)."""
    torch.manual_seed(0)
    layer = CausalSpectreMultiHead(embed_dim=32, num_heads=4, n_fft=64).to(DEVICE).eval()
    x = torch.randn(2, 64, 32, device=DEVICE)
    with torch.no_grad():
        y_orig = layer(x)
        x_mod = x.clone()
        x_mod[:, 40:, :] = torch.randn(2, 24, 32, device=DEVICE)
        y_mod = layer(x_mod)
    assert torch.allclose(y_orig[:, :40], y_mod[:, :40], atol=1e-4), (
        "past outputs changed when only future inputs changed"
    )


def test_near_identity_gate_init():
    """Warm gate -> kernel 0.9*delta -> head(x) == 0.9 * (x @ W_v.T)."""
    torch.manual_seed(1)
    head = _head()
    make_gate_near_identity(head)
    head.eval()
    x = torch.randn(3, 32, 16)
    with torch.no_grad():
        y = head(x)
        expected = 0.9 * (x @ head.W_v.weight.t())
    assert torch.allclose(y, expected, atol=1e-4)


def test_head_shapes_and_short_sequences():
    layer = CausalSpectreMultiHead(embed_dim=32, num_heads=4, n_fft=64)
    x = torch.randn(2, 64, 32)
    assert layer(x).shape == x.shape
    x_short = torch.randn(2, 40, 32)  # N < n_fft
    assert layer(x_short).shape == x_short.shape


def test_r9_surgery_warm_start_and_freeze():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    attn = model.transformer.h[0].attn
    expected_q0 = attn.c_attn.weight.t()[0:64, 0:64].clone()
    expected_v0 = attn.c_attn.weight.t()[1536:1600, 0:64].clone()
    swap_gpt2_attention_official_causal(model)
    layer = model.transformer.h[0].attn.spectre
    assert isinstance(layer, CausalSpectreMultiHead)
    assert torch.equal(layer.heads[0].W_q.weight, expected_q0)
    assert torch.equal(layer.heads[0].W_v.weight, expected_v0)
    freeze_backbone(model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert ".attn.spectre." in name, f"trainable param outside SPECTRE: {name}"


def test_r9_model_forward_smoke_and_causality():
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention_official_causal(model)
    model.to(DEVICE).eval()
    ids = torch.randint(0, 50256, (1, 32), device=DEVICE)
    with torch.no_grad():
        logits_orig = model(ids).logits
        ids_mod = ids.clone()
        ids_mod[:, 20:] = torch.randint(0, 50256, (1, 12), device=DEVICE)
        logits_mod = model(ids_mod).logits
    assert logits_orig.shape == (1, 32, 50257)
    assert torch.isfinite(logits_orig).all()
    # predictions for tokens 1..20 depend only on tokens 0..19
    assert torch.allclose(logits_orig[:, :19], logits_mod[:, :19], atol=1e-3), (
        "full-model future leakage in R9"
    )