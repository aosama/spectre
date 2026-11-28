#pragma once

#include <vector>

#include "spectre/config_types.hpp"

namespace spectre {

// Where the gate's chunks sit inside a sequence (Issue #5, decision D7).
//
// Chunk c covers positions [c * chunk_length, (c + 1) * chunk_length) with
// chunk_length = n_fft / causal_chunk_count. Boundaries are absolute, so they
// do not move when more tokens are appended: that keeps model(x[:k]) equal to
// model(x)[:k] and lets a decoder place boundaries without knowing the final
// length (Issue #6).
struct ChunkLayout {
  int sequence_length = 0;
  // For WholeWindowLegacy the single chunk spans the whole sequence.
  int chunk_length = 0;
  GatePooling gate_pooling = GatePooling::CausalPrefixMean;
  std::vector<int> chunk_start_positions;

  int chunk_count() const { return static_cast<int>(chunk_start_positions.size()); }

  // Exclusive end of the chunk; the last chunk may be shorter than chunk_length.
  int chunk_end_position(int chunk_index) const;
};

// causal_chunk_count == 0 selects the legacy whole-window layout (one chunk);
// otherwise it must be >= 2 and divide n_fft. A sequence shorter than one chunk
// has a single cold-start chunk; one longer than n_fft has more than
// causal_chunk_count chunks. Throws std::invalid_argument for invalid input.
ChunkLayout make_chunk_layout(int sequence_length, int n_fft, int causal_chunk_count);

}  // namespace spectre
