"""Tests for the training heartbeat and the gradient-accumulation step (Issue #7)."""
import math
from types import SimpleNamespace

import pytest
import torch

from spectre_research import train

VOCAB_SIZE = 11
SEQUENCE_LENGTH = 6


class _UniformLogitsModel(torch.nn.Module):
    """Zero logits give cross-entropy ln(VOCAB_SIZE) for any batch: a known constant loss."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        logits = torch.zeros(*batch.shape, VOCAB_SIZE) + self.scale
        return SimpleNamespace(logits=logits)


class _SilentMemLogger:
    def log(self, phase, force=False):
        pass


class _FakeClock:
    def __init__(self, now: float = 1_000.0, seconds_per_call: float = 0.0):
        self.now = now
        self.seconds_per_call = seconds_per_call

    def __call__(self) -> float:
        current = self.now
        self.now += self.seconds_per_call
        return current


def _make_progress(clock=None, **overrides) -> train._ProgressContext:
    return train._ProgressContext(
        start=0.0, total_steps=730, mem=_SilentMemLogger(), clock=clock or _FakeClock(), **overrides
    )


def _make_blocks(block_count: int) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (block_count, SEQUENCE_LENGTH))


@pytest.fixture(autouse=True)
def _run_on_cpu(monkeypatch):
    monkeypatch.setattr(train, "DEVICE", "cpu")


def _heartbeat_lines(captured_output: str) -> list[str]:
    return [line for line in captured_output.splitlines() if line.startswith("[heartbeat]")]


def test_should_report_full_batch_loss_in_every_heartbeat(capsys):
    model = _UniformLogitsModel()
    blocks = _make_blocks(train.EFFECTIVE_BATCH)
    perm = torch.arange(train.EFFECTIVE_BATCH)
    progress = _make_progress(clock=_FakeClock(seconds_per_call=6.0))  # >= 5 s apart: every micro-batch prints
    expected_loss = math.log(VOCAB_SIZE)

    first_step_loss, _ = train._accumulate_step(model, blocks, perm, 0, progress)
    progress.last_step_loss = first_step_loss
    train._accumulate_step(model, blocks, perm, 0, progress)

    assert first_step_loss == pytest.approx(expected_loss, rel=1e-5)
    heartbeat_lines = _heartbeat_lines(capsys.readouterr().out)
    assert len(heartbeat_lines) == 2 * train.ACCUM_STEPS
    # Before any step completes there is no loss to show; afterwards it is always the full-batch loss.
    assert all("last-step loss n/a" in line for line in heartbeat_lines[: train.ACCUM_STEPS])
    for line in heartbeat_lines[train.ACCUM_STEPS :]:
        printed_loss = float(line.split("last-step loss ")[1].split()[0])
        assert printed_loss == pytest.approx(expected_loss, abs=1e-4)


def test_should_show_micro_batch_progress_in_heartbeat(capsys):
    train._heartbeat(_make_progress(), 2)
    assert f"micro 2/{train.ACCUM_STEPS}" in capsys.readouterr().out


def test_should_rate_limit_heartbeat_to_five_seconds(capsys):
    clock = _FakeClock(now=100.0)
    progress = _make_progress(clock=clock)
    train._heartbeat(progress, 1)
    clock.now = 104.9
    train._heartbeat(progress, 2)  # < 5 s later: silent
    clock.now = 105.0
    train._heartbeat(progress, 3)  # exactly 5 s after the last print: prints
    assert len(_heartbeat_lines(capsys.readouterr().out)) == 2


def test_should_print_first_heartbeat_immediately(capsys):
    train._heartbeat(_make_progress(clock=_FakeClock(now=0.5)), 1)
    assert len(_heartbeat_lines(capsys.readouterr().out)) == 1


def test_should_behave_the_same_when_accumulation_is_one_micro_batch(capsys, monkeypatch):
    monkeypatch.setattr(train, "ACCUM_STEPS", 1)
    monkeypatch.setattr(train, "EFFECTIVE_BATCH", train.MICRO_BATCH)
    progress = _make_progress()
    blocks = _make_blocks(train.MICRO_BATCH)
    full_batch_loss, tokens = train._accumulate_step(
        _UniformLogitsModel(), blocks, torch.arange(train.MICRO_BATCH), 0, progress
    )
    assert full_batch_loss == pytest.approx(math.log(VOCAB_SIZE), rel=1e-5)
    assert tokens == train.MICRO_BATCH * SEQUENCE_LENGTH
    assert "micro 1/1" in capsys.readouterr().out


def test_should_accumulate_gradients_over_all_micro_batches():
    model = _UniformLogitsModel()
    blocks = _make_blocks(train.EFFECTIVE_BATCH)
    train._accumulate_step(model, blocks, torch.arange(train.EFFECTIVE_BATCH), 0, _make_progress(last_heartbeat_at=float("inf")))
    assert model.scale.grad is not None  # backward ran for the micro-batches


def test_should_return_the_full_batch_loss_not_a_partial_sum():
    # With ACCUM_STEPS micro-batches of loss L each, the sum must be L, not L/ACCUM_STEPS or a prefix.
    model = _UniformLogitsModel()
    blocks = _make_blocks(train.EFFECTIVE_BATCH)
    full_batch_loss, _ = train._accumulate_step(
        model, blocks, torch.arange(train.EFFECTIVE_BATCH), 0, _make_progress(last_heartbeat_at=float("inf"))
    )
    assert full_batch_loss == pytest.approx(math.log(VOCAB_SIZE), rel=1e-5)


def _eta_text(printed_line: str) -> str:
    return printed_line.split("ETA ")[1].strip()


def test_should_estimate_eta_from_this_runs_steps_after_resume(capsys):
    # 10 steps done in this run (500 -> 510) in 100 s: 10 s/step, 220 steps left = 2200 s = 36m40s.
    progress = _make_progress(clock=_FakeClock(now=100.0), first_step=500)
    progress.step = 510
    train._heartbeat(progress, 1)
    assert _eta_text(_heartbeat_lines(capsys.readouterr().out)[0]) == "36m40s"


def test_should_not_divide_by_zero_before_the_first_step_completes(capsys):
    train._heartbeat(_make_progress(clock=_FakeClock(now=50.0), first_step=500), 1)
    assert len(_heartbeat_lines(capsys.readouterr().out)) == 1
