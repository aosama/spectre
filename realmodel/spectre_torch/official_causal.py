"""R9: the official author gate + strictly causal mixing (the honest hybrid).

docs/audit-vs-official.md established: the official mixer's gate machinery
(grouped anchors, cubic interpolation, DCT pooling, smooth modReLU) is more
expressive than our v1 gate, but the official mixing is circular and collapses
to a one-step next-token copier under LM fine-tuning (R8, leakage-proven).
This module keeps the author's gate math verbatim — by subclassing the
vendored classes — and replaces ONLY the mixing step with a zero-padded
linear convolution (kernel length N, FFT length 2N), which is strictly
causal. This is the same causality construction later literature (Caracal)
adds to Fourier mixers, applied to SPECTRE's own gate.

The transplant also warm-starts the gate to near-identity (all anchors 1+0j
so the interpolated gate is the constant 0.9 after the official modReLU's
-0.1 bias, i.e. kernel = 0.9*delta): the swapped model starts close to the
original, exactly the "minimal fine-tuning" regime the paper describes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .official import SpectreAttentionOfficial, official
from .surgery import D_HEAD

N_FFT = 1024
D_GATE = 256
NUM_GROUPS = 4
WAVELET_ON_RATE = 0.0

_official = official()


def interp_complex_1d_cubic(x: torch.Tensor, size: int) -> torch.Tensor:
    """Corrected cubic complex interpolation.

    The vendored interp_complex_1d stacks real/imag on dim=1 giving
    (B, 2, G, K) and then reshapes to (B*G, 2, 1, K), which interleaves
    real/imag across groups (constant anchors 1+0j come back as 1+1j).
    Same grid_sample math, correct axis order.
    """
    B, G, K = x.shape
    real_imag = torch.stack([x.real, x.imag], dim=2).reshape(B * G, 2, 1, K)
    grid_x = torch.linspace(-1, 1, size, device=x.device)
    grid = grid_x.view(1, 1, size, 1).expand(B * G, 1, size, 1)
    grid_2d = torch.cat([grid, torch.zeros_like(grid)], dim=-1)
    interp = F.grid_sample(real_imag, grid_2d, mode="bicubic", padding_mode="border", align_corners=True)
    return torch.complex(interp[:, 0, 0, :], interp[:, 1, 0, :]).view(B, G, size)


class CausalSpectreHead(_official.SpectreHead):
    """Official gate, causal mixing. forward signature matches the vendored head."""

    def forward(
        self,
        x: torch.Tensor,
        pos_phase: torch.Tensor | None = None,
        return_q_pool: bool = False,
        memory_fft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_fft is not None:
            raise NotImplementedError("spectral memory is not part of the R9 transplant")
        B, N, d = x.shape
        G, dg = self.G, self.d // self.G
        assert d == self.d and d % G == 0

        # --- official gate path, verbatim (vendored lines 503-535) ---
        Q = self.W_q(x)
        V = self.W_v(x)
        q_pool = self.q_norm(self.pooling(Q))
        gate_rs = self.gate_mlp(q_pool).view(B, self.G, self.B, 2)
        gate_anchor = torch.view_as_complex(gate_rs)
        gate_half = interp_complex_1d_cubic(gate_anchor, size=self.F_half)
        gate_half = self.modrelu(gate_half.reshape(B, -1)).view_as(gate_half)
        if pos_phase is not None:
            gate_half = gate_half * pos_phase.unsqueeze(1 if pos_phase.dim() == 2 else 0)

        # --- causal mixing: per-group kernel, zero-padded linear convolution ---
        kernels = torch.fft.irfft(gate_half, n=self.n_fft, dim=-1)  # (B, G, N)
        v_grouped = V.view(B, N, G, dg).permute(2, 0, 3, 1).reshape(G * B, dg, N)
        kern_flat = kernels.permute(1, 0, 2).reshape(G * B, self.n_fft)

        L = 2 * self.n_fft
        v_pad = torch.zeros(G * B, dg, L, device=V.device, dtype=V.dtype)
        v_pad[..., :N] = v_grouped
        k_pad = torch.zeros(G * B, L, device=V.device, dtype=V.dtype)
        k_pad[..., : self.n_fft] = kern_flat
        v_hat = torch.fft.rfft(v_pad, n=L, dim=-1)
        k_hat = torch.fft.rfft(k_pad, n=L, dim=-1)
        mixed = torch.fft.irfft(v_hat * k_hat.unsqueeze(1), n=L, dim=-1)[..., :N]

        result = self.dropout(mixed.reshape(G, B, dg, N).permute(1, 3, 0, 2).reshape(B, N, d))
        if return_q_pool:
            return result, q_pool
        return result


def make_gate_near_identity(head: nn.Module) -> None:
    """All anchors 1+0j -> interpolated gate constant 1 -> modReLU(|1|-0.1) = 0.9
    -> kernel 0.9*delta, so the head starts as 0.9 * (x @ W_v.T)."""
    last_linear = head.gate_mlp[-1]
    with torch.no_grad():
        last_linear.weight.zero_()
        bias = torch.zeros_like(last_linear.bias)
        bias[0::2] = 1.0  # real part of every anchor; imag stays 0
        last_linear.bias.copy_(bias)


class CausalSpectreMultiHead(_official.SpectreMultiHead):
    """Vendored multihead wrapper with causal heads and the wavelet disabled."""

    def __init__(self, embed_dim: int, num_heads: int, n_fft: int, d_gate: int = D_GATE,
                 num_groups: int = NUM_GROUPS, wavelet_on_rate: float = WAVELET_ON_RATE):
        super().__init__(
            embed_dim, num_heads, n_fft,
            d_gate=d_gate, num_groups=num_groups, wavelet_on_rate=wavelet_on_rate,
        )
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            CausalSpectreHead(head_dim, fft_size=n_fft, d_gate=d_gate, num_groups=num_groups)
            for _ in range(num_heads)
        )
        for head in self.heads:
            make_gate_near_identity(head)


def swap_gpt2_attention_official_causal(model):
    """R9 surgery: official-geometry warm start + near-identity causal gates."""
    for block in model.transformer.h:
        attn = block.attn
        d_model = attn.c_attn.weight.shape[0]
        layer = CausalSpectreMultiHead(embed_dim=d_model, num_heads=attn.num_heads, n_fft=N_FFT)
        with torch.no_grad():
            w_t = attn.c_attn.weight.t()  # (3*d, d): rows [Q | K | V]
            for h, head in enumerate(layer.heads):
                rows_q = slice(D_HEAD * h, D_HEAD * (h + 1))
                rows_v = slice(2 * d_model + D_HEAD * h, 2 * d_model + D_HEAD * (h + 1))
                cols = slice(D_HEAD * h, D_HEAD * (h + 1))
                head.W_q.weight.copy_(w_t[rows_q, cols])
                head.W_v.weight.copy_(w_t[rows_v, cols])
            layer.out_proj.weight.copy_(attn.c_proj.weight.t())
        block.attn = SpectreAttentionOfficial(layer)
    return model
