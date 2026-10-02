"""Multi-head SPECTRE layer with causal linear-convolution mixing (plan D3).

Mirrors the verified Rust PoC (src/head.rs, src/layer.rs) but replaces the
paper's circular convolution with a strictly-causal linear convolution so the
layer can be used in an autoregressive LM without leaking future tokens. See
docs/plan-realmodel.md §1.1 for the rationale.

causal_chunks (R10, Issue #3): 0 keeps the legacy gate — a single kernel from
the whole-window query mean. >=2 splits the window into chunks; chunk c's
kernel comes from the query mean of strictly earlier positions, so the kernel
itself can no longer depend on future tokens (the gap that made the R5
causality tests vacuous).
"""
import torch
import torch.nn as nn

from .chunked_causal import causal_query_pools, chunked_causal_conv
from .gate import SpectreGate, gelu_tanh


def causal_conv_fft(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal linear convolution via FFT.

    v: (B, n, d) signal, h: (B, N) kernel (N = n_fft). Returns (B, n, d) with
    out[b, m, c] = sum_{s=0}^{m} h[b, s] · v[b, m-s, c].

    Both are zero-padded to 2N (>= n + N - 1 since n <= N), so the circular
    convolution in FFT space equals the linear convolution — no wrap-around.
    """
    B, n, d = v.shape
    N = h.shape[1]
    L = 2 * N
    v_pad = torch.zeros(B, L, d, device=v.device, dtype=v.dtype)
    v_pad[:, :n, :] = v
    h_pad = torch.zeros(B, L, device=h.device, dtype=h.dtype)
    h_pad[:, :N] = h
    v_hat = torch.fft.rfft(v_pad, n=L, dim=1)
    h_hat = torch.fft.rfft(h_pad, n=L, dim=1)
    y = torch.fft.irfft(v_hat * h_hat.unsqueeze(-1), n=L, dim=1)
    return y[:, :n, :]


class SpectreHead(nn.Module):
    """One SPECTRE mixing head (paper §3.2), causal variant.

    1. Q = X·W_q + b_q,  V = X·W_v + b_v          (token projection, eq. 2)
    2. g = gate(pool_c q) per chunk c             (R10: strictly earlier positions only)
    3. h = iRFFT(g)                               (time-domain kernel, n_fft)
    4. out = chunked_causal_conv(V, h)            (strictly causal, O(n log n))
    """

    def __init__(self, d_model: int, d_head: int, n_fft: int, hidden: int, causal_chunks: int = 0):
        super().__init__()
        if causal_chunks < 0 or causal_chunks == 1:
            raise ValueError("causal_chunks must be 0 (legacy) or >= 2")
        self.d_model = d_model
        self.d_head = d_head
        self.n_fft = n_fft
        self.causal_chunks = causal_chunks
        self.wq = nn.Parameter(torch.empty(d_model, d_head))
        self.bq = nn.Parameter(torch.zeros(d_head))
        self.wv = nn.Parameter(torch.empty(d_model, d_head))
        self.bv = nn.Parameter(torch.zeros(d_head))
        self.gate = SpectreGate(d_head, n_fft, hidden)
        # Xavier/Glorot-uniform, matching Rust nn::Linear::new.
        a = (6.0 / (d_model + d_head)) ** 0.5
        nn.init.uniform_(self.wq, -a, a)
        nn.init.uniform_(self.wv, -a, a)

    @property
    def n_chunks(self) -> int:
        return 1 if self.causal_chunks == 0 else self.causal_chunks

    def _kernels(self, q: torch.Tensor) -> torch.Tensor:
        """(B, C, n_fft) per-chunk kernels: causal pools for R10, full-window mean for legacy."""
        c = self.n_chunks
        if c == 1:
            g = self.gate(q.mean(dim=1)).unsqueeze(1)  # (B, 1, F) legacy: pool over the whole window
        else:
            pools = causal_query_pools(q, c)  # (B, C, d_head)
            g = torch.stack([self.gate(pools[:, i]) for i in range(c)], dim=1)  # (B, C, F)
        return torch.fft.irfft(g, n=self.n_fft, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d_model)
        q = x @ self.wq + self.bq  # (B, n, d_head)
        v = x @ self.wv + self.bv  # (B, n, d_head)
        kernels = self._kernels(q)  # (B, C, n_fft)
        return chunked_causal_conv(v, kernels, self.n_chunks)  # (B, n, d_head)


class SpectreLayer(nn.Module):
    """Multi-head SPECTRE layer: heads in parallel, concat, W_o projection.

    The forward path batches all heads into single ops (one matmul, one gate
    batch, one FFT set) instead of looping per head: 12 heads x 12 layers of
    small separate ops make MPS autograd the bottleneck, not the math.
    Parameters and per-head modules are unchanged.
    """

    def __init__(self, d_model: int, n_heads: int, n_fft: int, hidden: int, causal_chunks: int = 0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_fft = n_fft
        self.causal_chunks = causal_chunks
        d_head = d_model // n_heads
        self.d_head = d_head
        self.heads = nn.ModuleList(
            [SpectreHead(d_model, d_head, n_fft, hidden, causal_chunks=causal_chunks) for _ in range(n_heads)]
        )
        self.wo = nn.Linear(d_model, d_model)

    @property
    def n_chunks(self) -> int:
        return 1 if self.causal_chunks == 0 else self.causal_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d_model)
        B, n, _ = x.shape
        H, d, N = self.n_heads, self.d_head, self.n_fft
        C = self.n_chunks

        # One plain matmul with concatenated per-head weights (GPT-2's own
        # c_attn pattern): (B, n, d_model) @ (d_model, H*d) -> (B, n, H*d).
        # A broadcast einsum here materializes a (B, n, H, d_model, d)
        # intermediate (~77GB at B=32) — that was the RAM blowup.
        wq_all = torch.cat([h.wq for h in self.heads], dim=1)  # (d_model, H*d)
        bq_all = torch.cat([h.bq for h in self.heads])  # (H*d)
        wv_all = torch.cat([h.wv for h in self.heads], dim=1)
        bv_all = torch.cat([h.bv for h in self.heads])

        q = (x @ wq_all + bq_all).view(B, n, H, d)
        v = (x @ wv_all + bv_all).view(B, n, H, d)

        # Batched gate: per-head LN/MLP weights stacked, applied over (H, B*C, d)
        gates = [h.gate for h in self.heads]
        ln_w = torch.stack([g.ln.weight for g in gates])  # (H, d)
        ln_b = torch.stack([g.ln.bias for g in gates])
        l1_w = torch.stack([g.l1.weight for g in gates])  # (H, hidden, d)
        l1_b = torch.stack([g.l1.bias for g in gates])  # (H, hidden)
        l2_w = torch.stack([g.l2.weight for g in gates])  # (H, 2F, hidden)
        l2_b = torch.stack([g.l2.bias for g in gates])  # (H, 2F)
        mrb = torch.stack([g.modrelu_bias for g in gates])  # (H, F)

        if C == 1:
            q_h = q.mean(dim=1).permute(1, 0, 2)  # (H, B, d) legacy full-window pool
        else:
            pools = causal_query_pools(q.reshape(B, n, H * d), C)  # (B, C, H*d)
            q_h = pools.view(B, C, H, d).permute(2, 0, 1, 3).reshape(H, B * C, d)
        # LayerNorm per head (the gate's LN), batched over heads.
        ln_mean = q_h.mean(dim=-1, keepdim=True)
        ln_var = q_h.var(dim=-1, unbiased=False, keepdim=True)
        q_ln = (q_h - ln_mean) / torch.sqrt(ln_var + 1e-5) * ln_w.unsqueeze(1) + ln_b.unsqueeze(1)
        h_hid = torch.baddbmm(l1_b.unsqueeze(1), q_ln, l1_w.permute(0, 2, 1))  # (H, B*C, hidden)
        h_hid = gelu_tanh(h_hid)
        out2 = torch.baddbmm(l2_b.unsqueeze(1), h_hid, l2_w.permute(0, 2, 1))  # (H, B*C, 2F)
        g = torch.complex(out2[..., 0::2], out2[..., 1::2])  # (H, B*C, F)
        m = g.abs()
        s = m + mrb.unsqueeze(1)  # (H, B*C, F)
        safe_m = torch.where(m > 0, m, torch.ones_like(m))
        scale = torch.where((s > 0) & (m > 0), s / safe_m, torch.zeros_like(m))
        g = g * scale
        # g: (H, B*C, F) complex

        h_kern = torch.fft.irfft(g, n=N, dim=-1)  # (H, B*C, N) real kernels

        # Chunked causal conv, batched over (H*B) and channels d:
        v_h = v.permute(2, 0, 1, 3).reshape(H * B, n, d)  # (H*B, n, d)
        k_h = h_kern.reshape(H, B, C, N).reshape(H * B, C, N)  # (H*B, C, N)
        y = chunked_causal_conv(v_h, k_h, C)  # (H*B, n, d)
        y = y.reshape(H, B, n, d).permute(1, 2, 0, 3).reshape(B, n, H * d)
        return self.wo(y)  # (B, n, d_model)
