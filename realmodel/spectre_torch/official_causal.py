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

causal_chunks (R10, Issue #3): 0 keeps the R9 gate — one kernel from the
whole-window DCT pool. >=2 splits the window into chunks and computes chunk
c's gate from the query mean of strictly earlier positions (the same causal
cumulative descriptor the vendored decode_step uses, sum_q / N), so the
kernel can no longer depend on future tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunked_causal import causal_query_pools, chunked_causal_conv
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

    def __init__(
        self,
        head_dim: int,
        fft_size: int,
        d_gate: int = D_GATE,
        num_groups: int = NUM_GROUPS,
        causal_chunks: int = 0,
    ):
        super().__init__(head_dim, fft_size=fft_size, d_gate=d_gate, num_groups=num_groups)
        if causal_chunks < 0 or causal_chunks == 1:
            raise ValueError("causal_chunks must be 0 (legacy) or >= 2")
        self.causal_chunks = causal_chunks

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
        n_chunks = 1 if self.causal_chunks == 0 else self.causal_chunks

        # --- official gate path, verbatim (vendored lines 503-535) ---
        Q = self.W_q(x)
        V = self.W_v(x)
        if n_chunks == 1:
            q_pool = self.q_norm(self.pooling(Q))  # (B, d) R9 legacy: DCT over the whole window
        else:
            q_pool = self.q_norm(causal_query_pools(Q, n_chunks))  # (B, C, d) R10
        pool_2d = q_pool.reshape(B * n_chunks, d)

        gate_rs = self.gate_mlp(pool_2d).view(B * n_chunks, self.G, self.B, 2)
        gate_anchor = torch.view_as_complex(gate_rs)
        gate_half = interp_complex_1d_cubic(gate_anchor, size=self.F_half)
        gate_half = self.modrelu(gate_half.reshape(B * n_chunks, -1)).view_as(gate_half)
        if pos_phase is not None:
            gate_half = gate_half * pos_phase.unsqueeze(1 if pos_phase.dim() == 2 else 0)

        # --- causal mixing: per-chunk kernel, zero-padded linear convolution ---
        kernels = torch.fft.irfft(gate_half, n=self.n_fft, dim=-1)  # (B*C, G, n_fft)
        kernels = (
            kernels.view(B, n_chunks, G, self.n_fft)
            .permute(2, 0, 1, 3)
            .reshape(G * B, n_chunks, self.n_fft)
        )

        v_grouped = V.view(B, N, G, dg).permute(2, 0, 3, 1).reshape(G * B, dg, N).transpose(1, 2)
        mixed = chunked_causal_conv(v_grouped, kernels, n_chunks)  # (G*B, N, dg)
        mixed = mixed.transpose(1, 2)

        result = self.dropout(mixed.reshape(G, B, dg, N).permute(1, 3, 0, 2).reshape(B, N, d))
        if return_q_pool:
            # The vendored wrapper concatenates pools to (B, embed_dim) for the
            # wavelet path: hand back one (B, d) pool — the last chunk's, the
            # richest causal descriptor.
            return result, (q_pool if n_chunks == 1 else q_pool[:, -1])
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
                 num_groups: int = NUM_GROUPS, wavelet_on_rate: float = WAVELET_ON_RATE,
                 causal_chunks: int = 0):
        super().__init__(
            embed_dim, num_heads, n_fft,
            d_gate=d_gate, num_groups=num_groups, wavelet_on_rate=wavelet_on_rate,
        )
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            CausalSpectreHead(
                head_dim, fft_size=n_fft, d_gate=d_gate, num_groups=num_groups,
                causal_chunks=causal_chunks,
            )
            for _ in range(num_heads)
        )
        for head in self.heads:
            make_gate_near_identity(head)


def swap_gpt2_attention_official_causal(model, causal_chunks: int = 0):
    """R9 surgery: official-geometry warm start + near-identity causal gates.

    causal_chunks >= 2 (R10) additionally makes the gate strictly causal:
    one kernel per chunk from strictly earlier queries.
    """
    for block in model.transformer.h:
        attn = block.attn
        d_model = attn.c_attn.weight.shape[0]
        layer = CausalSpectreMultiHead(embed_dim=d_model, num_heads=attn.num_heads, n_fft=N_FFT,
                                       causal_chunks=causal_chunks)
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
