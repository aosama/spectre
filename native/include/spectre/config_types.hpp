#pragma once

#include <string_view>

namespace spectre {

// How the spectral gate's query descriptor is pooled (Issue #5, D8 and D11).
enum class GatePooling {
  // Chunk c pools the mean of all queries strictly before chunk c (R10/R11).
  CausalPrefixMean,
  // R5/R9 behavior: one descriptor from the whole window, which reads future
  // tokens (Issue #3). Kept only so historical numbers can be reproduced, so
  // it must be requested by name and is never the default.
  WholeWindowLegacy,
};

struct SpectreLayerConfig {
  int d_model = 0;
  int head_count = 0;
  int n_fft = 0;
  int gate_hidden_width = 0;
  // 0 is legal only with WholeWindowLegacy; CausalPrefixMean needs >= 2 and a
  // count that divides n_fft exactly (D7 absolute chunk boundaries).
  int causal_chunk_count = 0;
  GatePooling gate_pooling = GatePooling::CausalPrefixMean;
};

constexpr std::string_view kCausalPrefixMeanName = "causal_prefix_mean";
constexpr std::string_view kWholeWindowLegacyName = "whole_window_legacy";

std::string_view gate_pooling_name(GatePooling gate_pooling);

// Throws std::invalid_argument naming the offending value for any unknown name.
GatePooling parse_gate_pooling(std::string_view gate_pooling_text);

// Throws std::invalid_argument describing the first violated rule.
void validate_spectre_layer_config(const SpectreLayerConfig& layer_config);

}  // namespace spectre
