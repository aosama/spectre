#include "spectre/causal_query_pool.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace spectre {

namespace {

void validate_queries(const mx::array& queries, const ChunkLayout& layout) {
  if (queries.ndim() != 3) {
    throw std::invalid_argument("queries must be shaped (S, n, d), got " +
                                std::to_string(queries.ndim()) + " dimensions");
  }
  if (!mx::issubdtype(queries.dtype(), mx::floating)) {
    throw std::invalid_argument("queries must be floating point");
  }
  if (queries.shape(1) != layout.sequence_length) {
    throw std::invalid_argument("queries length " + std::to_string(queries.shape(1)) +
                                " does not match layout.sequence_length " +
                                std::to_string(layout.sequence_length));
  }
}

// Pools for chunks 1..C-1: the prefix sum just before each chunk's first
// position, divided by the number of positions in that prefix.
mx::array pools_after_the_first_chunk(const mx::array& queries, const ChunkLayout& layout,
                                      mx::StreamOrDevice stream) {
  const int later_chunk_count = layout.chunk_count() - 1;
  std::vector<int> last_prefix_positions;
  std::vector<float> prefix_lengths;
  for (int chunk_index = 1; chunk_index < layout.chunk_count(); ++chunk_index) {
    const int chunk_start = layout.chunk_start_positions[chunk_index];
    last_prefix_positions.push_back(chunk_start - 1);
    prefix_lengths.push_back(static_cast<float>(chunk_start));
  }
  // One cumsum plus one gather is O(n d) and adds a fixed number of graph
  // nodes. Slicing and averaging each prefix separately would add several
  // nodes per chunk and re-read the early positions once per chunk.
  const mx::array running_sums = mx::cumsum(queries, 1, false, true, stream);
  const mx::array gather_indices(last_prefix_positions.begin(), mx::Shape{later_chunk_count},
                                 mx::int32);
  const mx::array prefix_sums = mx::take(running_sums, gather_indices, 1, stream);
  const mx::array prefix_length_divisors(prefix_lengths.begin(), mx::Shape{1, later_chunk_count, 1},
                                         mx::float32);
  return mx::divide(prefix_sums, mx::astype(prefix_length_divisors, queries.dtype(), stream),
                    stream);
}

}  // namespace

mx::array causal_query_pools(const mx::array& queries, const ChunkLayout& layout,
                             mx::StreamOrDevice stream) {
  validate_queries(queries, layout);
  if (layout.gate_pooling == GatePooling::WholeWindowLegacy) {
    return mx::mean(queries, 1, true, stream);
  }
  const mx::array empty_prefix_pool =
      mx::zeros({queries.shape(0), 1, queries.shape(2)}, queries.dtype(), stream);
  if (layout.chunk_count() == 1) {
    return empty_prefix_pool;
  }
  return mx::concatenate(
      {empty_prefix_pool, pools_after_the_first_chunk(queries, layout, stream)}, 1, stream);
}

}  // namespace spectre
