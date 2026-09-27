"""Multi-head SPECTRE layer with causal linear-convolution mixing (plan D3).

Mirrors the verified Rust PoC (src/head.rs, src/layer.rs) but replaces the
paper's circular convolution with a strictly-causal linear convolution so the
layer can be used in an autoregressive LM without leaking future tokens. See
docs/plan-realmodel.md §1.1 for the rationale.
"""
import torch
import torch.nn as nn

from .gate import SpectreGate


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
    2. g = gate(mean_i q_i)                       (content-adaptive, F complex)
    3. h = iRFFT(g)                               (time-domain kernel, n_fft)
    4. out = causal_conv(V, h)                    (strictly causal, O(n log n))
    """

    def __init__(self, d_model: int, d_head: int, n_fft: int, hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_fft = n_fft
        self.wq = nn.Parameter(torch.empty(d_model, d_head))
        self.bq = nn.Parameter(torch.zeros(d_head))
        self.wv = nn.Parameter(torch.empty(d_model, d_head))
        self.bv = nn.Parameter(torch.zeros(d_head))
        self.gate = SpectreGate(d_head, n_fft, hidden)
        # Xavier/Glorot-uniform, matching Rust nn::Linear::new.
        a = (6.0 / (d_model + d_head)) ** 0.5
        nn.init.uniform_(self.wq, -a, a)
        nn.init.uniform_(self.wv, -a, a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d_model)
        q = x @ self.wq + self.bq  # (B, n, d_head)
        v = x @ self.wv + self.bv  # (B, n, d_head)
        q_mean = q.mean(dim=1)  # (B, d_head)
        g = self.gate(q_mean)  # (B, F) complex
        h = torch.fft.irfft(g, n=self.n_fft, dim=-1)  # (B, n_fft) real
        return causal_conv_fft(v, h)  # (B, n, d_head)


class SpectreLayer(nn.Module):
    """Multi-head SPECTRE layer: heads in parallel, concat, W_o projection."""

    def __init__(self, d_model: int, n_heads: int, n_fft: int, hidden: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_fft = n_fft
        d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [SpectreHead(d_model, d_head, n_fft, hidden) for _ in range(n_heads)]
        )
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d_model)
        outs = [head(x) for head in self.heads]  # each (B, n, d_head)
        concat = torch.cat(outs, dim=-1)  # (B, n, d_model)
        return self.wo(concat)  # (B, n, d_model)
