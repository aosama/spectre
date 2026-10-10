"""R11: initialize every causal chunk from its head's empty-prefix gate output."""
import torch

from .v1_gate import SpectreGate, gelu_tanh


def initialize_gate_from_empty_prefix(gate: SpectreGate) -> None:
    """Copy the gate output for chunk 0's empty prefix to every chunk initially."""
    empty_prefix = gate.ln.weight.new_zeros((1, gate.ln.weight.numel()))
    with torch.no_grad():
        chunk_zero_logits = gate.l2(gelu_tanh(gate.l1(gate.ln(empty_prefix))))
        gate.l2.weight.zero_()
        gate.l2.bias.copy_(chunk_zero_logits.squeeze(0))
