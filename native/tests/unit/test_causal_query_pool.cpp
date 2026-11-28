#include "spectre/causal_query_pool.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mlx_test_support.hpp"
#include "naive_reference.hpp"

namespace mx = mlx::core;
namespace support = spectre::testing;

namespace {

constexpr int kSequenceCount = 2;
constexpr int kFeatureWidth = 8;
// Float32 accumulation error over <= 100 unit-normal values stays orders of magnitude below
// this, while an off-by-one prefix index would differ by ~0.1 or more.
constexpr double kPoolTolerance = 1e-5;

class CausalQueryPool : public support::DeviceTest {};

struct PoolCase {
  std::vector<float> host_queries;
  mx::array device_queries;
};

PoolCase make_queries(int sequence_length, unsigned seed, const mx::Device& device) {
  std::vector<float> host_queries = support::random_float_values(
      static_cast<std::size_t>(kSequenceCount) * sequence_length * kFeatureWidth, seed);
  mx::array device_queries = support::to_device_array(
      host_queries, mx::Shape{kSequenceCount, sequence_length, kFeatureWidth}, device);
  return PoolCase{std::move(host_queries), std::move(device_queries)};
}

// Reads pools laid out (sequence, chunk, feature) at one coordinate.
float pool_entry_at(const std::vector<float>& host_pools, int chunk_count, int sequence_index,
                    int chunk_index, int feature_index) {
  const std::size_t flat_index =
      (static_cast<std::size_t>(sequence_index) * chunk_count + chunk_index) * kFeatureWidth +
      feature_index;
  return host_pools[flat_index];
}

double largest_difference(const std::vector<float>& actual, const std::vector<double>& expected) {
  EXPECT_EQ(actual.size(), expected.size());
  double largest = 0.0;
  for (std::size_t index = 0; index < std::min(actual.size(), expected.size()); ++index) {
    largest = std::max(largest, std::abs(actual[index] - expected[index]));
  }
  return largest;
}

}  // namespace

SPECTRE_INSTANTIATE_DEVICE_TESTS(CausalQueryPool);

TEST_P(CausalQueryPool, should_match_the_naive_prefix_mean_for_every_sequence_length) {
  for (const int sequence_length : {1, 15, 16, 17, 48, 64, 100}) {
    const spectre::ChunkLayout layout = spectre::make_chunk_layout(sequence_length, 64, 4);
    const PoolCase pool_case = make_queries(sequence_length, 100 + sequence_length, GetParam());

    const mx::array pools = spectre::causal_query_pools(pool_case.device_queries, layout, GetParam());

    ASSERT_EQ(pools.shape(), (mx::Shape{kSequenceCount, layout.chunk_count(), kFeatureWidth}));
    const std::vector<double> expected = spectre::oracle::naive_causal_query_pools(
        support::widen_to_double(pool_case.host_queries), kSequenceCount, sequence_length,
        kFeatureWidth, layout.chunk_start_positions);
    EXPECT_LT(largest_difference(support::to_host_floats(pools), expected), kPoolTolerance)
        << "sequence_length=" << sequence_length;
  }
}

TEST_P(CausalQueryPool, should_give_chunk_zero_an_exactly_zero_pool) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(48, 64, 4);
  const PoolCase pool_case = make_queries(48, 7, GetParam());
  const mx::array pools = spectre::causal_query_pools(pool_case.device_queries, layout, GetParam());
  const std::vector<float> host_pools = support::to_host_floats(pools);
  for (int sequence_index = 0; sequence_index < kSequenceCount; ++sequence_index) {
    for (int feature_index = 0; feature_index < kFeatureWidth; ++feature_index) {
      EXPECT_EQ(pool_entry_at(host_pools, layout.chunk_count(), sequence_index, 0, feature_index),
                0.0f);
    }
  }
}

TEST_P(CausalQueryPool, should_give_a_single_token_sequence_only_the_zero_pool) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(1, 64, 4);
  const PoolCase pool_case = make_queries(1, 8, GetParam());
  const mx::array pools = spectre::causal_query_pools(pool_case.device_queries, layout, GetParam());
  ASSERT_EQ(pools.shape(), (mx::Shape{kSequenceCount, 1, kFeatureWidth}));
  for (const float pool_entry : support::to_host_floats(pools)) {
    EXPECT_EQ(pool_entry, 0.0f);
  }
}

TEST_P(CausalQueryPool, should_keep_a_prefixs_pools_equal_to_the_full_sequences_pools) {
  const PoolCase full_case = make_queries(100, 9, GetParam());
  const spectre::ChunkLayout full_layout = spectre::make_chunk_layout(100, 64, 4);
  const std::vector<float> full_pools = support::to_host_floats(
      spectre::causal_query_pools(full_case.device_queries, full_layout, GetParam()));

  for (const int prefix_length : {1, 15, 16, 17, 48, 99}) {
    const spectre::ChunkLayout prefix_layout = spectre::make_chunk_layout(prefix_length, 64, 4);
    const mx::array prefix_queries = mx::slice(full_case.device_queries, {0, 0, 0},
                                               {kSequenceCount, prefix_length, kFeatureWidth}, GetParam());
    const std::vector<float> prefix_pools =
        support::to_host_floats(spectre::causal_query_pools(prefix_queries, prefix_layout, GetParam()));
    for (int sequence_index = 0; sequence_index < kSequenceCount; ++sequence_index) {
      for (int chunk_index = 0; chunk_index < prefix_layout.chunk_count(); ++chunk_index) {
        for (int feature_index = 0; feature_index < kFeatureWidth; ++feature_index) {
          const float prefix_entry = pool_entry_at(prefix_pools, prefix_layout.chunk_count(),
                                                   sequence_index, chunk_index, feature_index);
          const float full_entry = pool_entry_at(full_pools, full_layout.chunk_count(),
                                                 sequence_index, chunk_index, feature_index);
          EXPECT_NEAR(prefix_entry, full_entry, kPoolTolerance)
              << "prefix_length=" << prefix_length;
        }
      }
    }
  }
}

TEST_P(CausalQueryPool, should_return_the_whole_window_mean_as_one_chunk_for_the_legacy_layout) {
  for (const int sequence_length : {1, 48, 64}) {
    const spectre::ChunkLayout layout = spectre::make_chunk_layout(sequence_length, 64, 0);
    const PoolCase pool_case = make_queries(sequence_length, 300 + sequence_length, GetParam());
    const mx::array pools = spectre::causal_query_pools(pool_case.device_queries, layout, GetParam());
    ASSERT_EQ(pools.shape(), (mx::Shape{kSequenceCount, 1, kFeatureWidth}));
    const std::vector<double> expected = spectre::oracle::naive_whole_window_query_pool(
        support::widen_to_double(pool_case.host_queries), kSequenceCount, sequence_length, kFeatureWidth);
    EXPECT_LT(largest_difference(support::to_host_floats(pools), expected), kPoolTolerance);
  }
}

TEST_P(CausalQueryPool, should_reject_queries_that_are_not_three_dimensional) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(16, 64, 4);
  const mx::array two_dimensional_queries = mx::zeros({16, kFeatureWidth}, mx::float32, GetParam());
  EXPECT_THROW(spectre::causal_query_pools(two_dimensional_queries, layout, GetParam()),
               std::invalid_argument);
}

TEST_P(CausalQueryPool, should_reject_queries_whose_length_differs_from_the_layout) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(16, 64, 4);
  const mx::array wrong_length_queries =
      mx::zeros({kSequenceCount, 17, kFeatureWidth}, mx::float32, GetParam());
  EXPECT_THROW(spectre::causal_query_pools(wrong_length_queries, layout, GetParam()),
               std::invalid_argument);
}

TEST_P(CausalQueryPool, should_reject_integer_queries) {
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(16, 64, 4);
  const mx::array integer_queries =
      mx::zeros({kSequenceCount, 16, kFeatureWidth}, mx::int32, GetParam());
  EXPECT_THROW(spectre::causal_query_pools(integer_queries, layout, GetParam()),
               std::invalid_argument);
}
