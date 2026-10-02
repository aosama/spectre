"""R8 tests: the official author math, loaded from the vendored copy.

Run: cd realmodel && uv run python -m pytest tests/ -q
"""
import hashlib

import torch
from transformers import GPT2LMHeadModel

from spectre_torch.paper_spectre import (
    PINNED_SHA256,
    VENDORED_PATH,
    load_official_module,
    official,
    swap_gpt2_attention_official,
)
from spectre_torch.surgery import D_HEAD, freeze_backbone, param_report

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def test_vendored_copy_matches_sha_pin():
    """The vendored file must stay byte-identical to the audited upstream copy."""
    digest = hashlib.sha256(VENDORED_PATH.read_bytes()).hexdigest()
    assert digest == PINNED_SHA256


def test_loader_rejects_tampered_copy(tmp_path, monkeypatch):
    bad = tmp_path / "spectre.py"
    bad.write_text("# tampered\n")
    monkeypatch.setattr("spectre_torch.paper_spectre.VENDORED_PATH", bad)
    import spectre_torch.paper_spectre as official_mod

    original = official_mod._OFFICIAL_MODULE
    official_mod._OFFICIAL_MODULE = None
    try:
        try:
            load_official_module()
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "loader must refuse a file that fails the hash pin"
    finally:
        official_mod._OFFICIAL_MODULE = original


def test_official_forward_shapes():
    mod = load_official_module()
    layer = mod.SpectreMultiHead(
        embed_dim=32, num_heads=4, n_fft=64, d_gate=64, num_groups=4, wavelet_on_rate=0.0
    ).to(DEVICE)
    x = torch.randn(2, 64, 32, device=DEVICE)
    y = layer(x)
    assert y.shape == x.shape
    x_short = torch.randn(2, 40, 32, device=DEVICE)  # N < n_fft exercises the [:, :N] slice
    assert layer(x_short).shape == x_short.shape


def test_official_mixing_is_circular_when_n_fft_equals_n():
    """Documents the audited official semantics: with n_fft == N and no causal
    mask, a future token (t=2) contributes to earlier positions (t<2)."""
    mod = load_official_module()
    head = mod.SpectreHead(embed_dim=8, fft_size=16, d_gate=16, num_groups=4, pooling_type="mean")
    x = torch.zeros(1, 16, 8)
    x[0, 2, :] = 1.0  # single future spike
    y = head(x)
    leak = y[0, 0, :].abs().sum().item() + y[0, 1, :].abs().sum().item()
    assert leak > 0.0, "official head with n_fft == N must show wrap-around (circular) leakage"


def test_official_surgery_warm_start_and_freeze():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    attn = model.transformer.h[0].attn
    expected_q0 = attn.c_attn.weight.t()[0:64, 0:64].clone()  # block-diagonal slice, head 0
    expected_v0 = attn.c_attn.weight.t()[1536:1600, 0:64].clone()
    expected_o = attn.c_proj.weight.t().clone()
    swap_gpt2_attention_official(model)
    layer = model.transformer.h[0].attn.spectre
    mod = official()  # cached module: same class objects the surgery used
    assert isinstance(layer, mod.SpectreMultiHead)
    assert layer.heads[0].W_q.weight.shape == (D_HEAD, D_HEAD)
    assert torch.equal(layer.heads[0].W_q.weight, expected_q0)
    assert torch.equal(layer.heads[0].W_v.weight, expected_v0)
    assert torch.equal(layer.out_proj.weight, expected_o)
    assert layer.heads[0].W_q.bias is None  # official projections are bias-free
    assert layer.wavelet_refinement.on_rate == 0.0
    freeze_backbone(model)
    counts = param_report(model)
    assert counts["trainable"] > 0 and counts["frozen"] > 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert ".attn.spectre." in name, f"trainable param outside SPECTRE: {name}"


def test_official_model_forward_smoke():
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    swap_gpt2_attention_official(model)
    model.to(DEVICE).eval()
    ids = torch.randint(0, 50256, (1, 16), device=DEVICE)
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (1, 16, 50257)
    assert torch.isfinite(logits).all()
