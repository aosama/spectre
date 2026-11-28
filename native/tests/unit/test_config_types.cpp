#include "spectre/config_types.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

spectre::SpectreLayerConfig valid_r10_config() {
  spectre::SpectreLayerConfig layer_config;
  layer_config.d_model = 768;
  layer_config.head_count = 12;
  layer_config.n_fft = 1024;
  layer_config.gate_hidden_width = 256;
  layer_config.causal_chunk_count = 4;
  layer_config.gate_pooling = spectre::GatePooling::CausalPrefixMean;
  return layer_config;
}

spectre::SpectreLayerConfig valid_legacy_config() {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.causal_chunk_count = 0;
  layer_config.gate_pooling = spectre::GatePooling::WholeWindowLegacy;
  return layer_config;
}

}  // namespace

TEST(ConfigTypes, should_default_to_the_causal_gate_never_the_legacy_one) {
  const spectre::SpectreLayerConfig layer_config;
  EXPECT_EQ(layer_config.gate_pooling, spectre::GatePooling::CausalPrefixMean);
}

TEST(ConfigTypes, should_accept_a_valid_r10_config) {
  EXPECT_NO_THROW(spectre::validate_spectre_layer_config(valid_r10_config()));
}

TEST(ConfigTypes, should_accept_a_valid_legacy_config_with_zero_chunks) {
  EXPECT_NO_THROW(spectre::validate_spectre_layer_config(valid_legacy_config()));
}

TEST(ConfigTypes, should_accept_the_smallest_boundary_values) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.d_model = 2;
  layer_config.head_count = 2;
  layer_config.n_fft = 2;
  layer_config.gate_hidden_width = 1;
  layer_config.causal_chunk_count = 2;
  EXPECT_NO_THROW(spectre::validate_spectre_layer_config(layer_config));
}

TEST(ConfigTypes, should_accept_one_chunk_per_token_boundary) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.causal_chunk_count = layer_config.n_fft;
  EXPECT_NO_THROW(spectre::validate_spectre_layer_config(layer_config));
}

TEST(ConfigTypes, should_reject_a_single_causal_chunk) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.causal_chunk_count = 1;
  EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_zero_chunks_for_the_causal_gate) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.causal_chunk_count = 0;
  EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_a_chunk_count_that_does_not_divide_n_fft) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.causal_chunk_count = 3;
  EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_a_nonzero_chunk_count_for_the_legacy_gate) {
  spectre::SpectreLayerConfig layer_config = valid_legacy_config();
  layer_config.causal_chunk_count = 4;
  EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_an_n_fft_that_is_not_a_power_of_two) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.n_fft = 1000;
  EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_an_n_fft_of_one_or_less) {
  spectre::SpectreLayerConfig layer_config = valid_legacy_config();
  for (const int tiny_n_fft : {1, 0, -8}) {
    layer_config.n_fft = tiny_n_fft;
    EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument)
        << "n_fft=" << tiny_n_fft;
  }
}

TEST(ConfigTypes, should_reject_a_d_model_not_divisible_by_head_count) {
  spectre::SpectreLayerConfig layer_config = valid_r10_config();
  layer_config.head_count = 7;
  EXPECT_THROW(spectre::validate_spectre_layer_config(layer_config), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_non_positive_sizes) {
  spectre::SpectreLayerConfig zero_d_model = valid_r10_config();
  zero_d_model.d_model = 0;
  EXPECT_THROW(spectre::validate_spectre_layer_config(zero_d_model), std::invalid_argument);

  spectre::SpectreLayerConfig negative_heads = valid_r10_config();
  negative_heads.head_count = -4;
  EXPECT_THROW(spectre::validate_spectre_layer_config(negative_heads), std::invalid_argument);

  spectre::SpectreLayerConfig zero_hidden = valid_r10_config();
  zero_hidden.gate_hidden_width = 0;
  EXPECT_THROW(spectre::validate_spectre_layer_config(zero_hidden), std::invalid_argument);
}

TEST(ConfigTypes, should_round_trip_gate_pooling_names) {
  for (const spectre::GatePooling gate_pooling :
       {spectre::GatePooling::CausalPrefixMean, spectre::GatePooling::WholeWindowLegacy}) {
    EXPECT_EQ(spectre::parse_gate_pooling(spectre::gate_pooling_name(gate_pooling)), gate_pooling);
  }
}

TEST(ConfigTypes, should_reject_an_unknown_gate_pooling_name) {
  EXPECT_THROW(spectre::parse_gate_pooling("whole-window"), std::invalid_argument);
  EXPECT_THROW(spectre::parse_gate_pooling(""), std::invalid_argument);
}

TEST(ConfigTypes, should_reject_naming_an_out_of_range_gate_pooling) {
  const auto corrupted_gate_pooling = static_cast<spectre::GatePooling>(99);
  EXPECT_THROW(spectre::gate_pooling_name(corrupted_gate_pooling), std::invalid_argument);
}
