#include "spectre/chunk_layout.hpp"

#include <stdexcept>
#include <string>

namespace spectre {

namespace {

void validate_layout_arguments(int sequence_length, int n_fft, int causal_chunk_count) {
  if (sequence_length < 1) {
    throw std::invalid_argument("sequence_length must be >= 1, got " +
                                std::to_string(sequence_length));
  }
  if (n_fft < 1) {
    throw std::invalid_argument("n_fft must be >= 1, got " + std::to_string(n_fft));
  }
  if (causal_chunk_count == 0) {
    return;  // legacy whole-window layout
  }
  // A single chunk would pool nothing before it, so the gate could never see context.
  if (causal_chunk_count < 2) {
    throw std::invalid_argument("causal_chunk_count must be 0 (legacy) or >= 2, got " +
                                std::to_string(causal_chunk_count));
  }
  if (n_fft % causal_chunk_count != 0) {
    throw std::invalid_argument("n_fft (" + std::to_string(n_fft) +
                                ") must be divisible by causal_chunk_count (" +
                                std::to_string(causal_chunk_count) + ")");
  }
}

}  // namespace

int ChunkLayout::chunk_end_position(int chunk_index) const {
  if (chunk_index < 0 || chunk_index >= chunk_count()) {
    throw std::invalid_argument("chunk_index " + std::to_string(chunk_index) +
                                " out of range for " + std::to_string(chunk_count()) + " chunks");
  }
  const bool is_last_chunk = chunk_index + 1 == chunk_count();
  return is_last_chunk ? sequence_length : chunk_start_positions[chunk_index + 1];
}

ChunkLayout make_chunk_layout(int sequence_length, int n_fft, int causal_chunk_count) {
  validate_layout_arguments(sequence_length, n_fft, causal_chunk_count);

  ChunkLayout layout;
  layout.sequence_length = sequence_length;
  if (causal_chunk_count == 0) {
    layout.gate_pooling = GatePooling::WholeWindowLegacy;
    layout.chunk_length = sequence_length;
    layout.chunk_start_positions = {0};
    return layout;
  }
  layout.gate_pooling = GatePooling::CausalPrefixMean;
  layout.chunk_length = n_fft / causal_chunk_count;
  for (int start_position = 0; start_position < sequence_length;
       start_position += layout.chunk_length) {
    layout.chunk_start_positions.push_back(start_position);
  }
  return layout;
}

}  // namespace spectre
