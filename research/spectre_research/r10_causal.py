"""Strictly causal chunked spectral mixing machinery (R10, Issue #3).

A window is split into chunks of a fixed absolute length (n_fft / causal_chunks,
Issue #6); each chunk gets one kernel whose query descriptor is pooled over
strictly earlier positions only. The per-chunk mixing is a zero-padded linear
convolution with an exact raw-tail carry, so a chunk's output is a pure
function of its own kernel and of signal values at or before the chunk.
"""
import torch


def chunk_length_for(n_fft: int, causal_chunks: int) -> int:
    """Absolute chunk length n_fft / causal_chunks (Issue #6).

    Boundaries sit at multiples of this length, so they do not move when more
    tokens are appended: chunk c always starts at c * chunk_length. That keeps
    layer(x[:, :k]) equal to layer(x)[:, :k] and lets a decoder place
    boundaries without knowing the final length. At n == n_fft this is exactly
    the old n*c/C layout, so trained R10/R11 checkpoints stay valid.
    """
    if causal_chunks < 2:
        raise ValueError(f"causal_chunks must be >= 2 for chunked layouts, got {causal_chunks}")
    if n_fft % causal_chunks != 0:
        raise ValueError(f"n_fft ({n_fft}) must be divisible by causal_chunks ({causal_chunks})")
    return n_fft // causal_chunks


def effective_chunk_length(n: int, n_fft: int, causal_chunks: int) -> int:
    """Chunk length for a length-n input. Legacy (causal_chunks == 0) is ONE chunk
    spanning the whole window, so it is length-relative by definition; chunked
    layouts (>= 2) use the absolute length. Legacy is chosen by causal_chunks,
    never by "K == 1": an R10 input shorter than a chunk also has K == 1 but
    must still take the zero pool (cold-start kernel), not the window mean."""
    return n if causal_chunks == 0 else chunk_length_for(n_fft, causal_chunks)


def chunk_starts(n: int, chunk_length: int) -> list[int]:
    """Left boundaries of the chunks covering [0, n): 0, L, 2L, ... (last chunk may be partial).

    A sequence shorter than one chunk has a single chunk (the cold-start one);
    a sequence longer than n_fft simply gets more than causal_chunks chunks.
    """
    if n < 1 or chunk_length < 1:
        raise ValueError(f"need n >= 1 and chunk_length >= 1, got n={n}, chunk_length={chunk_length}")
    return list(range(0, n, chunk_length))


def causal_query_pools(q: torch.Tensor, chunk_length: int) -> torch.Tensor:
    """Per-chunk causal query pools.

    q: (B, n, d). Returns (B, K, d), K = len(chunk_starts(n, chunk_length)),
    where pool_c is the mean of q[:, :start_c, :] — strictly earlier positions —
    and pool_0 is zero (empty prefix: the gate then emits its learned
    cold-start kernel).
    """
    B, n, d = q.shape
    starts = chunk_starts(n, chunk_length)
    pools = torch.zeros(B, len(starts), d, dtype=q.dtype, device=q.device)
    if len(starts) > 1:
        cumulative_queries = q.cumsum(dim=1)
        for chunk_index in range(1, len(starts)):
            pools[:, chunk_index] = cumulative_queries[:, starts[chunk_index] - 1] / starts[chunk_index]
    return pools


def _next_pow2(x: int) -> int:
    return 1 << max(x - 1, 0).bit_length()


def chunked_causal_conv(v: torch.Tensor, kernels: torch.Tensor, chunk_length: int) -> torch.Tensor:
    """Per-chunk causal convolution with exact carry.

    v: (S, n, d) signal; kernels: (S, K, N) with K = len(chunk_starts(n,
    chunk_length)) — kernel c has length N and applies to chunk c.
    y[m] = sum_{s <= m} kernels[c(m)][m - s] * v[s], computed with one bounded
    FFT per chunk (carry = the last N-1 raw signal samples before the chunk),
    so chunk c's output never depends on signal values after the chunk starts.
    """
    S, n, d = v.shape
    N = kernels.shape[2]
    starts = chunk_starts(n, chunk_length)
    if kernels.shape[1] != len(starts):
        # A silent mismatch would apply the wrong kernel to a chunk.
        raise ValueError(
            f"kernels has {kernels.shape[1]} chunks but n={n}, chunk_length={chunk_length} gives {len(starts)}"
        )
    out = torch.empty_like(v)
    for c, s0 in enumerate(starts):
        s1 = starts[c + 1] if c + 1 < len(starts) else n
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
