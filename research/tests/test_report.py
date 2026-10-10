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
