#include "spectre/config_types.hpp"

#include <stdexcept>
#include <string>

namespace spectre {

namespace {

bool is_power_of_two(int candidate) {
  return candidate > 0 && (candidate & (candidate - 1)) == 0;
}

void require_positive(const char* field_name, int field_value) {
  if (field_value <= 0) {
    throw std::invalid_argument(std::string(field_name) + " must be positive, got " +
                                std::to_string(field_value));
  }
}

void validate_causal_chunking(const SpectreLayerConfig& layer_config) {
  const int chunk_count = layer_config.causal_chunk_count;
  if (layer_config.gate_pooling == GatePooling::WholeWindowLegacy) {
    // Legacy has no chunks; a non-zero count would silently be ignored.
    if (chunk_count != 0) {
      throw std::invalid_argument(
          "causal_chunk_count must be 0 when gate_pooling is whole_window_legacy, got " +
          std::to_string(chunk_count));
    }
    return;
  }
  // One chunk would pool nothing before it, so the gate could never see context.
  if (chunk_count < 2) {
    throw std::invalid_argument("causal_chunk_count must be >= 2 when gate_pooling is " +
                                std::string(kCausalPrefixMeanName) + ", got " +
                                std::to_string(chunk_count));
  }
  if (layer_config.n_fft % chunk_count != 0) {
    throw std::invalid_argument("n_fft (" + std::to_string(layer_config.n_fft) +
                                ") must be divisible by causal_chunk_count (" +
                                std::to_string(chunk_count) + ")");
  }
}

}  // namespace

std::string_view gate_pooling_name(GatePooling gate_pooling) {
  switch (gate_pooling) {
    case GatePooling::CausalPrefixMean:
      return kCausalPrefixMeanName;
    case GatePooling::WholeWindowLegacy:
      return kWholeWindowLegacyName;
  }
  throw std::invalid_argument("unknown GatePooling enumerator");
}

GatePooling parse_gate_pooling(std::string_view gate_pooling_text) {
  if (gate_pooling_text == kCausalPrefixMeanName) {
    return GatePooling::CausalPrefixMean;
  }
  if (gate_pooling_text == kWholeWindowLegacyName) {
    return GatePooling::WholeWindowLegacy;
  }
  throw std::invalid_argument("unknown gate_pooling '" + std::string(gate_pooling_text) +
                              "'; expected " + std::string(kCausalPrefixMeanName) + " or " +
                              std::string(kWholeWindowLegacyName));
}

void validate_spectre_layer_config(const SpectreLayerConfig& layer_config) {
  require_positive("d_model", layer_config.d_model);
  require_positive("head_count", layer_config.head_count);
  require_positive("gate_hidden_width", layer_config.gate_hidden_width);
  if (layer_config.d_model % layer_config.head_count != 0) {
    throw std::invalid_argument("d_model (" + std::to_string(layer_config.d_model) +
                                ") must be divisible by head_count (" +
                                std::to_string(layer_config.head_count) + ")");
  }
  // FFT lengths stay powers of two: MLX's Metal FFT is fastest there and falls
  // back to slower Bluestein for other sizes (Issue #5, trap 13).
  if (!is_power_of_two(layer_config.n_fft) || layer_config.n_fft < 2) {
    throw std::invalid_argument("n_fft must be a power of two >= 2, got " +
                                std::to_string(layer_config.n_fft));
  }
  validate_causal_chunking(layer_config);
}

}  // namespace spectre
