#include "spectre/causal_convolution.hpp"

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

constexpr int kBatchSize = 2;
constexpr int kFeatureWidth = 3;
constexpr int kKernelLength = 64;  // n_fft = 64, so chunk_count 4 gives 16-token chunks
// Default forward tolerances from Issue #5; float32 FFT error here is ~1e-5.
constexpr double kForwardAtol = 1e-4;
constexpr double kForwardRtol = 1e-4;

class CausalConvolution : public support::DeviceTest {};

// The ticket's "C = 1" is the whole-window layout, which is the legacy layout (count 0);
// a causal layout needs at least two chunks.
spectre::ChunkLayout layout_for(int sequence_length, int causal_chunk_count) {
  const int layout_chunk_count = causal_chunk_count == 1 ? 0 : causal_chunk_count;
  return spectre::make_chunk_layout(sequence_length, kKernelLength, layout_chunk_count);
}

struct ConvolutionInputs {
  std::vector<float> host_values;
  std::vector<float> host_kernels;
  mx::array device_values;
  mx::array device_kernels;
};

ConvolutionInputs make_inputs(int sequence_length, int chunk_count, unsigned seed,
                              const mx::Device& device) {
  std::vector<float> host_values = support::random_float_values(
      static_cast<std::size_t>(kBatchSize) * sequence_length * kFeatureWidth, seed);
  std::vector<float> host_kernels = support::random_float_values(
      static_cast<std::size_t>(kBatchSize) * chunk_count * kKernelLength, seed + 1000);
  mx::array device_values = support::to_device_array(
      host_values, mx::Shape{kBatchSize, sequence_length, kFeatureWidth}, device);
  mx::array device_kernels =
      support::to_device_array(host_kernels, mx::Shape{kBatchSize, chunk_count, kKernelLength}, device);
  return ConvolutionInputs{std::move(host_values), std::move(host_kernels), std::move(device_values),
                           std::move(device_kernels)};
}

std::vector<float> convolve_to_host(const ConvolutionInputs& inputs, const spectre::ChunkLayout& layout,
                                    const mx::Device& device) {
  return support::to_host_floats(
      spectre::chunked_causal_convolution(inputs.device_values, inputs.device_kernels, layout, device));
}

}  // namespace

SPECTRE_INSTANTIATE_DEVICE_TESTS(CausalConvolution);

TEST_P(CausalConvolution, should_return_the_values_for_an_impulse_kernel) {
  for (const int sequence_length : {1, 37, 64, 128}) {
    const spectre::ChunkLayout layout = layout_for(sequence_length, 4);
    ConvolutionInputs inputs = make_inputs(sequence_length, layout.chunk_count(), 1, GetParam());
    std::fill(inputs.host_kernels.begin(), inputs.host_kernels.end(), 0.0f);
    for (int tap_zero_index = 0; tap_zero_index < kBatchSize * layout.chunk_count(); ++tap_zero_index) {
      inputs.host_kernels[static_cast<std::size_t>(tap_zero_index) * kKernelLength] = 1.0f;
    }
    inputs.device_kernels = support::to_device_array(
        inputs.host_kernels, mx::Shape{kBatchSize, layout.chunk_count(), kKernelLength}, GetParam());

    EXPECT_TRUE(support::is_close(convolve_to_host(inputs, layout, GetParam()),
                                  support::widen_to_double(inputs.host_values), 1e-5, 1e-5))
        << "sequence_length=" << sequence_length;
  }
}

TEST_P(CausalConvolution, should_match_the_naive_oracle_for_every_layout_and_length) {
  for (const int causal_chunk_count : {1, 2, 4}) {
    for (const int sequence_length : {1, 37, 64, 128}) {  // 128 > n_fft: more chunks than causal_chunk_count
      const spectre::ChunkLayout layout = layout_for(sequence_length, causal_chunk_count);
      const ConvolutionInputs inputs = make_inputs(sequence_length, layout.chunk_count(), 2, GetParam());
      const std::vector<double> expected = spectre::oracle::naive_chunked_causal_convolution(
          support::widen_to_double(inputs.host_values), support::widen_to_double(inputs.host_kernels),
          kBatchSize, sequence_length, kFeatureWidth, kKernelLength, layout.chunk_start_positions);
      EXPECT_TRUE(support::is_close(convolve_to_host(inputs, layout, GetParam()), expected, kForwardAtol,
                                    kForwardRtol))
          << "causal_chunk_count=" << causal_chunk_count << " sequence_length=" << sequence_length;
    }
  }
}

TEST_P(CausalConvolution, should_match_the_naive_oracle_for_a_single_tap_kernel) {
  // Kernel length 1: no carry at all, so each chunk just scales its own positions.
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(40, 64, 4);
  std::vector<float> host_values = support::random_float_values(kBatchSize * 40 * kFeatureWidth, 3);
  std::vector<float> host_kernels = support::random_float_values(kBatchSize * layout.chunk_count() * 1, 4);
  const mx::array values = support::to_device_array(host_values, mx::Shape{kBatchSize, 40, kFeatureWidth}, GetParam());
  const mx::array kernels =
      support::to_device_array(host_kernels, mx::Shape{kBatchSize, layout.chunk_count(), 1}, GetParam());
  const std::vector<double> expected = spectre::oracle::naive_chunked_causal_convolution(
      support::widen_to_double(host_values), support::widen_to_double(host_kernels), kBatchSize, 40,
      kFeatureWidth, 1, layout.chunk_start_positions);
  EXPECT_TRUE(support::is_close(
      support::to_host_floats(spectre::chunked_causal_convolution(values, kernels, layout, GetParam())),
      expected, 1e-5, 1e-5));
}

TEST_P(CausalConvolution, should_not_change_outputs_before_the_perturbed_chunk_at_all) {
  // Issue #5 trap 3: chunks before the perturbed one never share an FFT with it, so their
  // outputs must be bit-for-bit equal; only the perturbed chunk itself may move by FFT rounding.
  constexpr int kSequenceLength = 128;
  constexpr int kFirstChangedPosition = 70;  // inside the chunk covering [64, 80)
  constexpr int kPerturbedChunkStart = 64;
  const spectre::ChunkLayout layout = layout_for(kSequenceLength, 4);
  const ConvolutionInputs original = make_inputs(kSequenceLength, layout.chunk_count(), 5, GetParam());
  ConvolutionInputs perturbed = original;
  const std::vector<float> replacement =
      support::random_float_values(original.host_values.size(), 999);
  for (int sequence_index = 0; sequence_index < kBatchSize; ++sequence_index) {
    for (int position = kFirstChangedPosition; position < kSequenceLength; ++position) {
      for (int feature_index = 0; feature_index < kFeatureWidth; ++feature_index) {
        const std::size_t flat_index =
            (static_cast<std::size_t>(sequence_index) * kSequenceLength + position) * kFeatureWidth + feature_index;
        perturbed.host_values[flat_index] = replacement[flat_index];
      }
    }
  }
  perturbed.device_values = support::to_device_array(
      perturbed.host_values, mx::Shape{kBatchSize, kSequenceLength, kFeatureWidth}, GetParam());

  const std::vector<float> original_output = convolve_to_host(original, layout, GetParam());
  const std::vector<float> perturbed_output = convolve_to_host(perturbed, layout, GetParam());

  for (int sequence_index = 0; sequence_index < kBatchSize; ++sequence_index) {
    for (int position = 0; position < kFirstChangedPosition; ++position) {
      for (int feature_index = 0; feature_index < kFeatureWidth; ++feature_index) {
        const std::size_t flat_index =
            (static_cast<std::size_t>(sequence_index) * kSequenceLength + position) * kFeatureWidth + feature_index;
        if (position < kPerturbedChunkStart) {
          EXPECT_EQ(original_output[flat_index], perturbed_output[flat_index])
              << "position " << position << " moved although it precedes the perturbed chunk";
        } else {
          EXPECT_NEAR(original_output[flat_index], perturbed_output[flat_index], 1e-5)
              << "position " << position;
        }
      }
    }
  }
}

TEST_P(CausalConvolution, should_be_prefix_consistent_for_every_prefix_length) {
  constexpr int kSequenceLength = 70;
  const spectre::ChunkLayout full_layout = layout_for(kSequenceLength, 4);
  const ConvolutionInputs full = make_inputs(kSequenceLength, full_layout.chunk_count(), 6, GetParam());
  const std::vector<float> full_output = convolve_to_host(full, full_layout, GetParam());

  for (int prefix_length = 1; prefix_length <= kSequenceLength; ++prefix_length) {
    const spectre::ChunkLayout prefix_layout = layout_for(prefix_length, 4);
    const mx::array prefix_values = mx::slice(full.device_values, {0, 0, 0},
                                              {kBatchSize, prefix_length, kFeatureWidth}, GetParam());
    const mx::array prefix_kernels = mx::slice(full.device_kernels, {0, 0, 0},
                                               {kBatchSize, prefix_layout.chunk_count(), kKernelLength}, GetParam());
    const std::vector<float> prefix_output = support::to_host_floats(
        spectre::chunked_causal_convolution(prefix_values, prefix_kernels, prefix_layout, GetParam()));

    // The padded window and FFT length can differ with n, so rounding differs: compare with a tolerance.
    std::vector<double> expected_prefix;
    for (int sequence_index = 0; sequence_index < kBatchSize; ++sequence_index) {
      for (int position = 0; position < prefix_length; ++position) {
        for (int feature_index = 0; feature_index < kFeatureWidth; ++feature_index) {
          expected_prefix.push_back(full_output[(static_cast<std::size_t>(sequence_index) * kSequenceLength + position) * kFeatureWidth + feature_index]);
        }
      }
    }
    EXPECT_TRUE(support::is_close(prefix_output, expected_prefix, kForwardAtol, kForwardRtol))
        << "prefix_length=" << prefix_length;
  }
}

TEST_P(CausalConvolution, should_reject_a_kernel_count_that_differs_from_the_chunk_count) {
  const spectre::ChunkLayout layout = layout_for(50, 4);  // 4 chunks
  const ConvolutionInputs inputs = make_inputs(50, 2, 7, GetParam());  // 2 kernels
  EXPECT_THROW(spectre::chunked_causal_convolution(inputs.device_values, inputs.device_kernels, layout, GetParam()),
               std::invalid_argument);
}

TEST_P(CausalConvolution, should_reject_values_whose_length_differs_from_the_layout) {
  const spectre::ChunkLayout layout = layout_for(50, 4);
  const ConvolutionInputs inputs = make_inputs(51, layout.chunk_count(), 8, GetParam());
  EXPECT_THROW(spectre::chunked_causal_convolution(inputs.device_values, inputs.device_kernels, layout, GetParam()),
               std::invalid_argument);
}

TEST_P(CausalConvolution, should_reject_mismatched_batch_sizes) {
  const spectre::ChunkLayout layout = layout_for(32, 4);
  const mx::array values = mx::zeros({2, 32, kFeatureWidth}, mx::float32, GetParam());
  const mx::array kernels = mx::zeros({3, layout.chunk_count(), kKernelLength}, mx::float32, GetParam());
  EXPECT_THROW(spectre::chunked_causal_convolution(values, kernels, layout, GetParam()), std::invalid_argument);
}

TEST_P(CausalConvolution, should_reject_inputs_that_are_not_three_dimensional) {
  const spectre::ChunkLayout layout = layout_for(32, 4);
  const mx::array flat_values = mx::zeros({32, kFeatureWidth}, mx::float32, GetParam());
  const mx::array kernels = mx::zeros({1, layout.chunk_count(), kKernelLength}, mx::float32, GetParam());
  EXPECT_THROW(spectre::chunked_causal_convolution(flat_values, kernels, layout, GetParam()), std::invalid_argument);
}

TEST_P(CausalConvolution, should_match_the_naive_oracle_when_the_kernel_is_shorter_than_a_chunk) {
  // Carry (N - 1 = 7) is smaller than the 16-token chunk: windows barely overlap the previous chunk.
  constexpr int kShortKernelLength = 8;
  constexpr int kSequenceLength = 50;
  const spectre::ChunkLayout layout = spectre::make_chunk_layout(kSequenceLength, 64, 4);
  const std::vector<float> host_values =
      support::random_float_values(kBatchSize * kSequenceLength * kFeatureWidth, 11);
  const std::vector<float> host_kernels =
      support::random_float_values(kBatchSize * layout.chunk_count() * kShortKernelLength, 12);
  const mx::array values = support::to_device_array(
      host_values, mx::Shape{kBatchSize, kSequenceLength, kFeatureWidth}, GetParam());
  const mx::array kernels = support::to_device_array(
      host_kernels, mx::Shape{kBatchSize, layout.chunk_count(), kShortKernelLength}, GetParam());
  const std::vector<double> expected = spectre::oracle::naive_chunked_causal_convolution(
      support::widen_to_double(host_values), support::widen_to_double(host_kernels), kBatchSize,
      kSequenceLength, kFeatureWidth, kShortKernelLength, layout.chunk_start_positions);
  EXPECT_TRUE(support::is_close(
      support::to_host_floats(spectre::chunked_causal_convolution(values, kernels, layout, GetParam())),
      expected, kForwardAtol, kForwardRtol));
}

TEST_P(CausalConvolution, should_match_the_naive_oracle_for_a_sequence_many_chunks_long) {
  // 300 tokens = 19 chunks of 16: each window's carry spans four earlier chunks.
  const spectre::ChunkLayout layout = layout_for(300, 4);
  const ConvolutionInputs inputs = make_inputs(300, layout.chunk_count(), 13, GetParam());
  const std::vector<double> expected = spectre::oracle::naive_chunked_causal_convolution(
      support::widen_to_double(inputs.host_values), support::widen_to_double(inputs.host_kernels),
      kBatchSize, 300, kFeatureWidth, kKernelLength, layout.chunk_start_positions);
  EXPECT_TRUE(support::is_close(convolve_to_host(inputs, layout, GetParam()), expected, kForwardAtol,
                                kForwardRtol));
}

TEST_P(CausalConvolution, should_reject_half_precision_inputs) {
  const spectre::ChunkLayout layout = layout_for(32, 4);
  const mx::array half_values = mx::zeros({1, 32, kFeatureWidth}, mx::float16, GetParam());
  const mx::array kernels = mx::zeros({1, layout.chunk_count(), kKernelLength}, mx::float32, GetParam());
  EXPECT_THROW(spectre::chunked_causal_convolution(half_values, kernels, layout, GetParam()), std::invalid_argument);
}

TEST_P(CausalConvolution, should_reject_integer_inputs) {
  const spectre::ChunkLayout layout = layout_for(32, 4);
  const mx::array integer_values = mx::zeros({1, 32, kFeatureWidth}, mx::int32, GetParam());
  const mx::array kernels = mx::zeros({1, layout.chunk_count(), kKernelLength}, mx::float32, GetParam());
  EXPECT_THROW(spectre::chunked_causal_convolution(integer_values, kernels, layout, GetParam()), std::invalid_argument);
}

TEST_P(CausalConvolution, should_reject_an_empty_kernel) {
  const spectre::ChunkLayout layout = layout_for(32, 4);
  const mx::array values = mx::zeros({1, 32, kFeatureWidth}, mx::float32, GetParam());
  const mx::array empty_kernels = mx::zeros({1, layout.chunk_count(), 0}, mx::float32, GetParam());
  EXPECT_THROW(spectre::chunked_causal_convolution(values, empty_kernels, layout, GetParam()), std::invalid_argument);
}
