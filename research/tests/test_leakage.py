"""Tests for the leakage probe's prefix-consistency Test E (Issue #6)."""
from types import SimpleNamespace

import pytest
import torch

from spectre_research import leakage

VOCAB_SIZE = 7


class _CausalStubModel:
    """Logits at position t depend only on tokens <= t (running token histogram)."""

    def __call__(self, ids):
        one_hot_tokens = torch.nn.functional.one_hot(ids, VOCAB_SIZE).float()
        return SimpleNamespace(logits=one_hot_tokens.cumsum(dim=1))


class _LengthRelativeStubModel:
    """Reads no future token, yet logits shift with the total length: the old R10 chunk-boundary defect."""

    def __call__(self, ids):
        return SimpleNamespace(logits=_CausalStubModel()(ids).logits + float(ids.shape[1]))


class _FutureReadingStubModel:
    """Position t reads token t+1: the next-token copier. Must also fail Test E."""

    def __call__(self, ids):
        shifted_ids = torch.roll(ids, shifts=-1, dims=1)
        return SimpleNamespace(logits=torch.nn.functional.one_hot(shifted_ids, VOCAB_SIZE).float())


def _make_ids(token_count: int = 40) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (2, token_count))


def test_should_report_zero_for_a_prefix_consistent_model():
    assert leakage._prefix_consistency(_CausalStubModel(), _make_ids(), prefix_lengths=(5, 13, 39)) == 0.0


def test_should_detect_a_model_whose_logits_depend_on_total_length():
    change = leakage._prefix_consistency(_LengthRelativeStubModel(), _make_ids(), prefix_lengths=(5, 13, 39))
    assert change == pytest.approx(35.0)  # 40 tokens vs the shortest prefix of 5


def test_should_detect_a_future_reading_model():
    assert leakage._prefix_consistency(_FutureReadingStubModel(), _make_ids(), prefix_lengths=(5, 13)) > 0.0


def test_should_cover_cuts_that_are_not_chunk_boundaries():
    chunk_length = 1024 // leakage.R10_CAUSAL_CHUNKS
    assert all(prefix_length % chunk_length != 0 for prefix_length in leakage.PREFIX_LENGTHS)


def test_should_report_zero_for_a_full_length_prefix():
    ids = _make_ids()
    assert leakage._prefix_consistency(_LengthRelativeStubModel(), ids, prefix_lengths=(ids.shape[1],)) == 0.0
