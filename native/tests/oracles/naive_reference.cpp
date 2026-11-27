#include "naive_reference.hpp"

#include <cstddef>

namespace spectre::oracle {

namespace {

std::size_t flat_index(int sequence_index, int position, int feature_index, int sequence_length,
                       int feature_width) {
  return (static_cast<std::size_t>(sequence_index) * sequence_length + position) * feature_width +
         feature_index;
}

}  // namespace

std::vector<double> naive_causal_query_pools(const std::vector<double>& queries, int sequence_count,
                                             int sequence_length, int feature_width,
                                             const std::vector<int>& chunk_start_positions) {
  const int chunk_count = static_cast<int>(chunk_start_positions.size());
  std::vector<double> pools(static_cast<std::size_t>(sequence_count) * chunk_count * feature_width,
                            0.0);
  for (int sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
    for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int prefix_length = chunk_start_positions[chunk_index];
      for (int feature_index = 0; feature_index < feature_width; ++feature_index) {
        double prefix_sum = 0.0;
        for (int position = 0; position < prefix_length; ++position) {
          prefix_sum +=
              queries[flat_index(sequence_index, position, feature_index, sequence_length, feature_width)];
        }
        const std::size_t pool_index =
            (static_cast<std::size_t>(sequence_index) * chunk_count + chunk_index) * feature_width +
            feature_index;
        pools[pool_index] = prefix_length == 0 ? 0.0 : prefix_sum / prefix_length;
      }
    }
  }
  return pools;
}

std::vector<double> naive_whole_window_query_pool(const std::vector<double>& queries,
                                                  int sequence_count, int sequence_length,
                                                  int feature_width) {
  std::vector<double> pools(static_cast<std::size_t>(sequence_count) * feature_width, 0.0);
  for (int sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
    for (int feature_index = 0; feature_index < feature_width; ++feature_index) {
      double window_sum = 0.0;
      for (int position = 0; position < sequence_length; ++position) {
        window_sum +=
            queries[flat_index(sequence_index, position, feature_index, sequence_length, feature_width)];
      }
      pools[static_cast<std::size_t>(sequence_index) * feature_width + feature_index] =
          window_sum / sequence_length;
    }
  }
  return pools;
}

std::vector<double> naive_chunked_causal_convolution(const std::vector<double>& values,
                                                     const std::vector<double>& kernels,
                                                     int sequence_count, int sequence_length,
                                                     int feature_width, int kernel_length,
                                                     const std::vector<int>& chunk_start_positions) {
  const int chunk_count = static_cast<int>(chunk_start_positions.size());
  std::vector<double> output(static_cast<std::size_t>(sequence_count) * sequence_length * feature_width,
                             0.0);
  int chunk_index = 0;
  for (int position = 0; position < sequence_length; ++position) {
    while (chunk_index + 1 < chunk_count && chunk_start_positions[chunk_index + 1] <= position) {
      ++chunk_index;
    }
    const int reachable_taps = position + 1 < kernel_length ? position + 1 : kernel_length;
    for (int sequence_index = 0; sequence_index < sequence_count; ++sequence_index) {
      for (int tap = 0; tap < reachable_taps; ++tap) {
        const double tap_weight =
            kernels[(static_cast<std::size_t>(sequence_index) * chunk_count + chunk_index) * kernel_length + tap];
        for (int feature_index = 0; feature_index < feature_width; ++feature_index) {
          output[flat_index(sequence_index, position, feature_index, sequence_length, feature_width)] +=
              tap_weight * values[flat_index(sequence_index, position - tap, feature_index, sequence_length,
                                             feature_width)];
        }
      }
    }
  }
  return output;
}

}  // namespace spectre::oracle
