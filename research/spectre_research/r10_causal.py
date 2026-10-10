"""Strictly causal chunked spectral mixing machinery (R10, Issue #3).

A window is split into chunks; each chunk gets one kernel whose query
descriptor is pooled over strictly earlier positions only. The per-chunk
mixing is a zero-padded linear convolution with an exact raw-tail carry, so
a chunk's output is a pure function of its own kernel and of signal values
at or before the chunk.
"""
import torch


def chunk_starts(n: int, n_chunks: int) -> list[int]:
    """Left boundaries of the n_chunks chunks covering [0, n)."""
    if n_chunks < 1 or n < n_chunks:
        raise ValueError(f"need 1 <= n_chunks <= n, got n_chunks={n_chunks}, n={n}")
    return [n * c // n_chunks for c in range(n_chunks)]


def causal_query_pools(q: torch.Tensor, n_chunks: int) -> torch.Tensor:
    """Per-chunk causal query pools.

    q: (B, n, d). Returns (B, n_chunks, d) where pool_c is the mean of
    q[:, :start_c, :] — strictly earlier positions — and pool_0 is zero
    (empty prefix: the gate then emits its learned cold-start kernel).
    """
    B, n, d = q.shape
    starts = chunk_starts(n, n_chunks)
    pools = torch.zeros(B, n_chunks, d, dtype=q.dtype, device=q.device)
    if n_chunks > 1:
        cumsum = q.cumsum(dim=1)
        for c in range(1, n_chunks):
            pools[:, c] = cumsum[:, starts[c] - 1] / starts[c]
    return pools


def _next_pow2(x: int) -> int:
    return 1 << max(x - 1, 0).bit_length()


def chunked_causal_conv(v: torch.Tensor, kernels: torch.Tensor, n_chunks: int) -> torch.Tensor:
    """Per-chunk causal convolution with exact carry.

    v: (S, n, d) signal; kernels: (S, n_chunks, N) — kernel c has length N
    and applies to chunk c. y[m] = sum_{s <= m} kernels[c(m)][m - s] * v[s],
    computed with one bounded FFT per chunk (carry = the last N-1 raw signal
    samples before the chunk), so chunk c's output never depends on signal
    values after the chunk starts.
    """
    S, n, d = v.shape
    N = kernels.shape[2]
    starts = chunk_starts(n, n_chunks)
    out = torch.empty_like(v)
    for c in range(n_chunks):
        s0 = starts[c]
        s1 = starts[c + 1] if c + 1 < n_chunks else n
        a = max(0, s0 - (N - 1))
        m = s1 - a
        L = _next_pow2(m + N - 1)
        v_pad = torch.zeros(S, L, d, device=v.device, dtype=v.dtype)
        v_pad[:, :m] = v[:, a:s1]
        h_pad = torch.zeros(S, L, device=v.device, dtype=v.dtype)
        h_pad[:, :N] = kernels[:, c]
        y = torch.fft.irfft(
            torch.fft.rfft(v_pad, n=L, dim=1) * torch.fft.rfft(h_pad, n=L, dim=1).unsqueeze(-1),
            n=L, dim=1,
        )
        out[:, s0:s1] = y[:, m - (s1 - s0):m]
    return out
