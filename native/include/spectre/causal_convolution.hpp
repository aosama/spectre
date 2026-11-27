#pragma once

#include <mlx/mlx.h>

#include "spectre/chunk_layout.hpp"

namespace spectre {

// Per-chunk causal linear convolution with exact carry (Issue #5, T1.2).
//
//   y[m] = sum_{s = 0}^{min(m, N-1)} kernels[c(m)][s] * values[m - s]
//
// where c(m) is the chunk containing position m and N is the kernel length.
// values (S, n, d), kernels (S, K, N) with K = layout.chunk_count(); returns (S, n, d).
//
// Chunk c reads the signal from max(0, chunk_start - (N - 1)) to its end, so
// its output depends only on its own kernel and on values at or before the
// chunk. All chunks are evaluated with ONE batched rfft/irfft (segments are
// left-padded with zeros to a common length) instead of a loop of per-chunk
// FFTs, which keeps the graph small and the GPU busy. Cost is
// O((n / chunk_length) * (chunk_length + N) log(chunk_length + N)).
//
// Throws std::invalid_argument for non-3-D or non-floating inputs, a batch
// mismatch, a kernel count different from the layout's chunk count, or a
// sequence length different from the layout's.
mlx::core::array chunked_causal_convolution(const mlx::core::array& values,
                                            const mlx::core::array& kernels,
                                            const ChunkLayout& layout,
                                            mlx::core::StreamOrDevice stream = {});

}  // namespace spectre
