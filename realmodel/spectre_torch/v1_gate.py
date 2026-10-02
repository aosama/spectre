"""Content-adaptive spectral gate (paper §3.2 step 3, "Positional Awareness").

    q̄ = LN(mean_i q_i) → Linear → GELU(tanh) → Linear(2F) → interleave
        → modReLU → positional phase (shift 0 in v1)

Mirrors the verified Rust PoC (src/gate.rs, src/nn.rs). F = n_fft//2 + 1.
"""
import math

import torch
import torch.nn as nn


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU, tanh approximation (Hendrycks & Gimpel). Matches Rust nn::gelu."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x**3)))


def modrelu(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """modReLU(z) = ReLU(|z| + b)·z/|z|, and 0 when z = 0.

    g: (..., F) complex, b: (F,) real. Matches Rust gate::modrelu.
    """
    m = g.abs()
    s = m + b
    safe_m = torch.where(m > 0, m, torch.ones_like(m))
    scale = torch.where((s > 0) & (m > 0), s / safe_m, torch.zeros_like(m))
    return g * scale


class SpectreGate(nn.Module):
    """LN → Linear → GELU(tanh) → Linear(2F) → interleave → modReLU → phase 0.

    forward(q_mean: (B, d_head)) -> (B, F) complex64.
    """

    def __init__(self, d_head: int, n_fft: int, hidden: int):
        super().__init__()
        self.F = n_fft // 2 + 1
        self.ln = nn.LayerNorm(d_head, eps=1e-5)
        self.l1 = nn.Linear(d_head, hidden)
        self.l2 = nn.Linear(hidden, 2 * self.F)
        self.modrelu_bias = nn.Parameter(torch.zeros(self.F))
        # All-pass init: g ≈ 1 (real bias parts 1, imag 0) so the layer starts
        # close to identity — the paper's warm-start, and what makes the small
        # fine-tune budget feasible (plan D6).
        with torch.no_grad():
            bias = torch.zeros(2 * self.F)
            bias[0::2] = 1.0
            self.l2.bias.copy_(bias)

    def forward(self, q_mean: torch.Tensor) -> torch.Tensor:
        h = gelu_tanh(self.l1(self.ln(q_mean)))
        out = self.l2(h)  # (B, 2F)
        g = torch.complex(out[:, 0::2], out[:, 1::2])  # (B, F)
        g = modrelu(g, self.modrelu_bias)
        # Positional phase shift is 0 in v1 (no-op); kept explicit for parity
        # with the Rust gate's apply_phase.
        return g
