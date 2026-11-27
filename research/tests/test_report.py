"""Tests for checkpoint reconstruction and evaluation reporting."""
from types import SimpleNamespace

import torch

from spectre_research import report


def test_r11_checkpoint_rebuilds_chunk0_initialized_causal_model(monkeypatch):
    checkpoint = {"arch": "v1-r11", "model": {}, "val_loss": 1.0}
    model = SimpleNamespace(
        transformer=SimpleNamespace(h=[]),
        load_state_dict=lambda _state: None,
        to=lambda _device: model,
        eval=lambda: model,
    )
    swap_arguments = []
    diagnostic_chunks = []

    class StubGPT2:
        @staticmethod
        def from_pretrained(_model_name):
            return model

    monkeypatch.setattr(report.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(report, "GPT2LMHeadModel", StubGPT2)
    monkeypatch.setattr(
        report,
        "swap_gpt2_attention",
        lambda rebuilt_model, **kwargs: swap_arguments.append((rebuilt_model, kwargs)),
    )
    monkeypatch.setattr(report, "perplexity", lambda *_args: 42.0)
    monkeypatch.setattr(
        report,
        "_gate_magnitude_stats",
        lambda _model, _inputs, causal_chunks: diagnostic_chunks.append(causal_chunks) or {},
    )
    monkeypatch.setattr(
        report,
        "_gate_pairwise_distance",
        lambda _model, _inputs, causal_chunks: diagnostic_chunks.append(causal_chunks) or 0.0,
    )

    ppl, loaded_checkpoint, stats = report._eval_checkpoint("unused.pt", torch.zeros(1, 4))

    assert ppl == 42.0
    assert loaded_checkpoint is checkpoint
    assert swap_arguments == [
        (model, {"causal_chunks": report.CKPT_R10_CAUSAL_CHUNKS, "chunk0_init": True})
    ]
    assert diagnostic_chunks == [report.CKPT_R10_CAUSAL_CHUNKS, report.CKPT_R10_CAUSAL_CHUNKS]
    assert stats == {"pairwise_cos_dist": 0.0}


def _fake_model_with_spectre_layer(causal_chunks: int, n_fft: int = 64):
    from spectre_research.v1_spectre import SpectreLayer

    torch.manual_seed(40)
    layer = SpectreLayer(32, 4, n_fft, 16, causal_chunks=causal_chunks)
    with torch.no_grad():
        for head in layer.heads:
            head.v1_gate.l2.weight.normal_(0, 0.5)
    return SimpleNamespace(transformer=SimpleNamespace(h=[SimpleNamespace(attn=SimpleNamespace(spectre=layer))]))


def test_should_compute_gate_diagnostics_for_every_chunk_count(monkeypatch):
    # n=48 gives K=3 chunks, n=64 gives K=4 and n=100 gives K=7 (more than causal_chunks).
    model = _fake_model_with_spectre_layer(causal_chunks=4)
    for token_count in (48, 64, 100):
        layer_input = torch.randn(2, token_count, 32)
        monkeypatch.setattr(report, "_layer_inputs", lambda _model, _inputs, x=layer_input: [x])
        gate_stats = report._gate_magnitude_stats(model, layer_input, causal_chunks=4)
        pairwise_distance = report._gate_pairwise_distance(model, layer_input, causal_chunks=4)
        assert gate_stats["mean_abs_g"] > 0.0 and gate_stats["bin_std"] >= 0.0, token_count
        assert 0.0 <= pairwise_distance <= 2.0, token_count


def test_should_reject_diagnostics_chunk_count_that_does_not_divide_n_fft(monkeypatch):
    import pytest

    model = _fake_model_with_spectre_layer(causal_chunks=4)
    layer_input = torch.randn(2, 48, 32)
    monkeypatch.setattr(report, "_layer_inputs", lambda _model, _inputs: [layer_input])
    with pytest.raises(ValueError, match="divisible"):
        report._gate_magnitude_stats(model, layer_input, causal_chunks=3)
