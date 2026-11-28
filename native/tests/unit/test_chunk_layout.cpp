#include "spectre/chunk_layout.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

using Starts = std::vector<int>;

// n_fft = 64 with 4 chunks gives the 16-token chunks used throughout these tests.
spectre::ChunkLayout causal_layout(int sequence_length) {
  return spectre::make_chunk_layout(sequence_length, 64, 4);
}

}  // namespace

TEST(ChunkLayout, should_place_boundaries_at_multiples_of_chunk_length) {
  const spectre::ChunkLayout layout = causal_layout(64);
  EXPECT_EQ(layout.chunk_length, 16);
  EXPECT_EQ(layout.chunk_start_positions, (Starts{0, 16, 32, 48}));
  EXPECT_EQ(layout.gate_pooling, spectre::GatePooling::CausalPrefixMean);
}

TEST(ChunkLayout, should_match_the_gpt2_training_layout_at_full_length) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(1024, 1024, 4);
  EXPECT_EQ(layout.chunk_start_positions, (Starts{0, 256, 512, 768}));
}

TEST(ChunkLayout, should_keep_one_cold_start_chunk_for_a_single_token) {
  const spectre::ChunkLayout layout = causal_layout(1);
  EXPECT_EQ(layout.chunk_start_positions, (Starts{0}));
  EXPECT_EQ(layout.chunk_end_position(0), 1);
}

TEST(ChunkLayout, should_keep_one_chunk_when_shorter_than_a_chunk) {
  EXPECT_EQ(causal_layout(2).chunk_start_positions, (Starts{0}));  // n < causal_chunk_count is legal
  EXPECT_EQ(causal_layout(15).chunk_start_positions, (Starts{0}));
  EXPECT_EQ(causal_layout(16).chunk_start_positions, (Starts{0}));
}

TEST(ChunkLayout, should_open_a_new_chunk_one_token_past_a_boundary) {
  const spectre::ChunkLayout layout = causal_layout(17);
  EXPECT_EQ(layout.chunk_start_positions, (Starts{0, 16}));
  EXPECT_EQ(layout.chunk_end_position(0), 16);
  EXPECT_EQ(layout.chunk_end_position(1), 17);  // partial last chunk
}

TEST(ChunkLayout, should_give_a_partial_last_chunk_when_not_a_multiple_of_chunk_length) {
  const spectre::ChunkLayout layout = causal_layout(50);
  EXPECT_EQ(layout.chunk_start_positions, (Starts{0, 16, 32, 48}));
  EXPECT_EQ(layout.chunk_end_position(3), 50);
}

TEST(ChunkLayout, should_use_more_than_causal_chunk_count_chunks_beyond_n_fft) {
  const spectre::ChunkLayout layout = causal_layout(100);
  EXPECT_EQ(layout.chunk_count(), 7);
  EXPECT_EQ(layout.chunk_start_positions.back(), 96);
}

TEST(ChunkLayout, should_not_move_boundaries_when_tokens_are_appended) {
  for (const int sequence_length : {1, 15, 16, 17, 300, 1000}) {
    const Starts shorter = causal_layout(sequence_length).chunk_start_positions;
    const Starts longer = causal_layout(sequence_length + 37).chunk_start_positions;
    ASSERT_GE(longer.size(), shorter.size());
    EXPECT_TRUE(std::equal(shorter.begin(), shorter.end(), longer.begin()))
        << "boundaries moved when extending n=" << sequence_length;
  }
}

TEST(ChunkLayout, should_allow_one_chunk_per_token) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(5, 8, 8);
  EXPECT_EQ(layout.chunk_length, 1);
  EXPECT_EQ(layout.chunk_start_positions, (Starts{0, 1, 2, 3, 4}));
}

TEST(ChunkLayout, should_use_one_window_wide_chunk_for_the_legacy_layout) {
  for (const int sequence_length : {1, 48, 64, 100}) {
    const spectre::ChunkLayout layout = spectre::make_chunk_layout(sequence_length, 64, 0);
    EXPECT_EQ(layout.gate_pooling, spectre::GatePooling::WholeWindowLegacy);
    EXPECT_EQ(layout.chunk_start_positions, (Starts{0}));
    EXPECT_EQ(layout.chunk_length, sequence_length);
    EXPECT_EQ(layout.chunk_end_position(0), sequence_length);
  }
}

TEST(ChunkLayout, should_reject_a_non_positive_sequence_length) {
  EXPECT_THROW(spectre::make_chunk_layout(0, 64, 4), std::invalid_argument);
  EXPECT_THROW(spectre::make_chunk_layout(-5, 64, 4), std::invalid_argument);
}

TEST(ChunkLayout, should_reject_a_non_positive_n_fft) {
  EXPECT_THROW(spectre::make_chunk_layout(10, 0, 4), std::invalid_argument);
  EXPECT_THROW(spectre::make_chunk_layout(10, -64, 0), std::invalid_argument);
}

TEST(ChunkLayout, should_reject_a_single_or_negative_chunk_count) {
  EXPECT_THROW(spectre::make_chunk_layout(10, 64, 1), std::invalid_argument);
  EXPECT_THROW(spectre::make_chunk_layout(10, 64, -2), std::invalid_argument);
}

TEST(ChunkLayout, should_reject_a_chunk_count_that_does_not_divide_n_fft) {
  EXPECT_THROW(spectre::make_chunk_layout(10, 64, 3), std::invalid_argument);
  EXPECT_THROW(spectre::make_chunk_layout(10, 1024, 6), std::invalid_argument);
}

TEST(ChunkLayout, should_reject_a_chunk_index_out_of_range) {
  const spectre::ChunkLayout layout = causal_layout(40);
  EXPECT_THROW(layout.chunk_end_position(-1), std::invalid_argument);
  EXPECT_THROW(layout.chunk_end_position(layout.chunk_count()), std::invalid_argument);
}
