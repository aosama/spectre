#pragma once

#include <vector>

// Plain C++ double-precision reference implementations used as test oracles
// (Issue #5, trap 12). They deliberately include no MLX: an oracle that shared
// MLX's code paths could repeat an MLX-usage mistake instead of catching it.
//
// Tensors are row-major flat vectors; a queries tensor is [sequence][position][feature].
namespace spectre::oracle {

// Mean of all queries at positions strictly before each chunk start, per
// sequence: result is [sequence][chunk][feature]. A chunk starting at position
// 0 has an empty prefix and a zero pool.
std::vector<double> naive_causal_query_pools(const std::vector<double>& queries, int sequence_count,
                                             int sequence_length, int feature_width,
                                             const std::vector<int>& chunk_start_positions);

// Mean over the whole window per sequence: result is [sequence][1][feature].
std::vector<double> naive_whole_window_query_pool(const std::vector<double>& queries,
                                                  int sequence_count, int sequence_length,
                                                  int feature_width);

// Direct O(n * N) evaluation of y[m] = sum_{s <= min(m, N-1)} h_{chunk(m)}[s] * v[m - s].
// values [sequence][position][feature]; kernels [sequence][chunk][tap]; returns
// [sequence][position][feature]. chunk(m) is the last chunk whose start is <= m.
std::vector<double> naive_chunked_causal_convolution(const std::vector<double>& values,
                                                     const std::vector<double>& kernels,
                                                     int sequence_count, int sequence_length,
                                                     int feature_width, int kernel_length,
                                                     const std::vector<int>& chunk_start_positions);

}  // namespace spectre::oracle
