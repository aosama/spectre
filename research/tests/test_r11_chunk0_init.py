"""Tests for the R11 chunk-0-derived gate initialization."""
import torch

from spectre_research.r11_chunk0_init import initialize_gate_from_empty_prefix
from spectre_research.v1_gate import SpectreGate


def test_empty_prefix_initialization_shares_chunk_zero_gate_and_remains_trainable():
    torch.manual_seed(11)
    gate = SpectreGate(d_head=8, n_fft=64, hidden=16)
    empty_prefix = torch.zeros(1, 8)
    later_chunk_descriptors = torch.randn(3, 8)

    with torch.no_grad():
        chunk_zero_gate = gate(empty_prefix)

    initialize_gate_from_empty_prefix(gate)

    with torch.no_grad():
        gates = gate(later_chunk_descriptors)

    expected_gates = chunk_zero_gate.expand_as(gates)
    assert torch.allclose(gates, expected_gates, atol=1e-6)

    gate(later_chunk_descriptors).abs().square().sum().backward()
    assert gate.l2.weight.grad is not None
    assert gate.l2.weight.grad.abs().sum() > 0
