#include "spectre/causal_convolution.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace spectre {

namespace {

// Where each chunk's signal window sits, and the common padded sizes.
struct ConvolutionGeometry {
  int chunk_count = 0;
  // Every chunk's window is right-aligned inside this length; shorter windows get zeros in front.
  int common_window_length = 0;
  // Power-of-two FFT length >= common_window_length + kernel_length - 1, so the
  // circular convolution equals the linear one (no wrap-around). A power of two
  // also keeps MLX's Metal FFT off the slower Bluestein path (Issue #5, trap 13).
  int fft_length = 0;
  // For each (chunk, window slot): which signal position to read, and whether
  // that slot holds real signal (1) or the zero padding before the carry (0).
  std::vector<int> window_source_positions;
  std::vector<float> window_slot_is_signal;
  // For each output position: which row of the flattened (chunk, fft slot) result holds it.
  std::vector<int> output_rows;
};

int next_power_of_two(int minimum_length) {
  int candidate_length = 1;
  while (candidate_length < minimum_length) {
    candidate_length *= 2;
  }
  return candidate_length;
}

void validate_inputs(const mx::array& values, const mx::array& kernels, const ChunkLayout& layout) {
  if (values.ndim() != 3 || kernels.ndim() != 3) {
    throw std::invalid_argument("values must be (S, n, d) and kernels (S, K, N)");
  }
  // MLX's real FFT takes float32; other dtypes would fail deep inside MLX with an unclear error.
  if (values.dtype() != mx::float32 || kernels.dtype() != mx::float32) {
    throw std::invalid_argument("values and kernels must be float32");
  }
  if (values.shape(0) != kernels.shape(0)) {
    throw std::invalid_argument("values and kernels must share the batch size, got " +
                                std::to_string(values.shape(0)) + " and " +
                                std::to_string(kernels.shape(0)));
  }
  if (values.shape(1) != layout.sequence_length) {
    throw std::invalid_argument("values length " + std::to_string(values.shape(1)) +
                                " does not match layout.sequence_length " +
                                std::to_string(layout.sequence_length));
  }
  // A silent mismatch would apply the wrong kernel to a chunk.
  if (kernels.shape(1) != layout.chunk_count()) {
    throw std::invalid_argument("kernels has " + std::to_string(kernels.shape(1)) +
                                " chunks but the layout has " + std::to_string(layout.chunk_count()));
  }
  if (kernels.shape(2) < 1) {
    throw std::invalid_argument("kernel length must be >= 1");
  }
}

struct ChunkWindows {
  std::vector<int> starts;
  std::vector<int> ends;  // exclusive
  int longest_length = 0;
};

// Chunk c's window: its own positions plus the previous N-1 raw samples (the carry).
ChunkWindows compute_chunk_windows(const ChunkLayout& layout, int kernel_length) {
  ChunkWindows windows;
  for (int chunk_index = 0; chunk_index < layout.chunk_count(); ++chunk_index) {
    const int chunk_start = layout.chunk_start_positions[chunk_index];
    windows.starts.push_back(std::max(0, chunk_start - (kernel_length - 1)));
    windows.ends.push_back(layout.chunk_end_position(chunk_index));
    windows.longest_length =
        std::max(windows.longest_length, windows.ends.back() - windows.starts.back());
  }
  return windows;
}

void fill_window_gather_plan(const ChunkWindows& windows, ConvolutionGeometry& geometry) {
  for (int chunk_index = 0; chunk_index < geometry.chunk_count; ++chunk_index) {
    for (int slot = 0; slot < geometry.common_window_length; ++slot) {
      const int source_position = windows.ends[chunk_index] - geometry.common_window_length + slot;
      const bool is_signal = source_position >= windows.starts[chunk_index];
      geometry.window_source_positions.push_back(is_signal ? source_position : 0);
      geometry.window_slot_is_signal.push_back(is_signal ? 1.0f : 0.0f);
    }
  }
}

void fill_output_rows(const ChunkLayout& layout, const ChunkWindows& windows,
                      ConvolutionGeometry& geometry) {
  for (int chunk_index = 0; chunk_index < geometry.chunk_count; ++chunk_index) {
    for (int position = layout.chunk_start_positions[chunk_index]; position < windows.ends[chunk_index];
         ++position) {
      const int slot = geometry.common_window_length - (windows.ends[chunk_index] - position);
      geometry.output_rows.push_back(chunk_index * geometry.fft_length + slot);
    }
  }
}

ConvolutionGeometry compute_geometry(const ChunkLayout& layout, int kernel_length) {
  const ChunkWindows windows = compute_chunk_windows(layout, kernel_length);
  ConvolutionGeometry geometry;
  geometry.chunk_count = layout.chunk_count();
  geometry.common_window_length = windows.longest_length;
  geometry.fft_length = next_power_of_two(windows.longest_length + kernel_length - 1);
  fill_window_gather_plan(windows, geometry);
  fill_output_rows(layout, windows, geometry);
  return geometry;
}

// (S, K, M, d): each chunk's right-aligned signal window, zero where there is no carry.
mx::array gather_chunk_windows(const mx::array& values, const ConvolutionGeometry& geometry,
                               mx::StreamOrDevice stream) {
  const int batch_size = values.shape(0);
  const int feature_width = values.shape(2);
  const int slot_count = geometry.chunk_count * geometry.common_window_length;
  const mx::array source_positions(geometry.window_source_positions.begin(), mx::Shape{slot_count},
                                   mx::int32);
  const mx::array gathered = mx::take(values, source_positions, 1, stream);
  const mx::array windows =
      mx::reshape(gathered, {batch_size, geometry.chunk_count, geometry.common_window_length, feature_width},
                  stream);
  const mx::array slot_is_signal(geometry.window_slot_is_signal.begin(),
                                 mx::Shape{1, geometry.chunk_count, geometry.common_window_length, 1},
                                 mx::float32);
  return mx::multiply(windows, mx::astype(slot_is_signal, values.dtype(), stream), stream);
}

}  // namespace

mx::array chunked_causal_convolution(const mx::array& values, const mx::array& kernels,
                                     const ChunkLayout& layout, mx::StreamOrDevice stream) {
  validate_inputs(values, kernels, layout);
  const ConvolutionGeometry geometry = compute_geometry(layout, kernels.shape(2));
  const int batch_size = values.shape(0);
  const int feature_width = values.shape(2);

  const mx::array windows = gather_chunk_windows(values, geometry, stream);  // (S, K, M, d)
  // Explicit n = fft_length: the length cannot be inferred from the bin count (trap 7).
  const mx::array window_spectra =
      mx::fft::rfft(windows, geometry.fft_length, 2, mx::fft::FFTNorm::Backward, stream);  // (S, K, F, d)
  const mx::array kernel_spectra = mx::expand_dims(
      mx::fft::rfft(kernels, geometry.fft_length, 2, mx::fft::FFTNorm::Backward, stream), 3, stream);  // (S, K, F, 1)
  const mx::array convolved_windows = mx::fft::irfft(mx::multiply(window_spectra, kernel_spectra, stream),
                                             geometry.fft_length, 2, mx::fft::FFTNorm::Backward,
                                             stream);  // (S, K, L, d)

  const mx::array flattened_rows = mx::reshape(
      convolved_windows, {batch_size, geometry.chunk_count * geometry.fft_length, feature_width}, stream);
  const mx::array output_rows(geometry.output_rows.begin(), mx::Shape{layout.sequence_length}, mx::int32);
  return mx::take(flattened_rows, output_rows, 1, stream);
}

}  // namespace spectre
