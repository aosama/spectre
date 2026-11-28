#pragma once

#include <mlx/mlx.h>

#include "spectre/chunk_layout.hpp"

namespace spectre {

// Per-chunk gate descriptors from queries shaped (S, n, d); returns (S, C, d).
//
// CausalPrefixMean (decision D8): chunk c's descriptor is the mean of ALL
// queries at positions strictly before the chunk, and exactly zero for chunk 0
// (empty prefix, so the gate emits its learned cold-start kernel). A decoder
// reproduces this with only a running sum of the queries.
//
// WholeWindowLegacy: a single descriptor, the mean over the whole window (C = 1).
// That descriptor reads future tokens (Issue #3); it exists only to reproduce R5.
//
// Intended for float32 queries: MLX has no float64 on Metal, and lower precision
// would make the running sum drift for long sequences.
//
// Throws std::invalid_argument when queries is not 3-D floating point or its
// length does not match layout.sequence_length.
mlx::core::array causal_query_pools(const mlx::core::array& queries,
                                    const ChunkLayout& layout,
                                    mlx::core::StreamOrDevice stream = {});

}  // namespace spectre
