// Deliberately has no production counterpart: it proves the MLX install links and
// runs on both devices before any SPECTRE code depends on it. The FFT round trip
// is the primitive every later ticket builds on.
#include <gtest/gtest.h>

#include <stdexcept>

#include "mlx_test_support.hpp"

namespace mx = mlx::core;

namespace {

class MlxToolchain : public spectre::testing::DeviceTest {};

}  // namespace

SPECTRE_INSTANTIATE_DEVICE_TESTS(MlxToolchain);

TEST_P(MlxToolchain, should_round_trip_a_real_signal_through_rfft_and_irfft) {
  const mx::Device device = GetParam();
  const mx::array signal = mx::array({1.f, -2.f, 3.f, 0.5f, 7.f, -1.f, 2.f, 4.f});
  const mx::array spectrum = mx::fft::rfft(signal, 8, -1, mx::fft::FFTNorm::Backward, device);
  const mx::array recovered_signal =
      mx::fft::irfft(spectrum, 8, -1, mx::fft::FFTNorm::Backward, device);
  mx::array is_recovered = mx::allclose(signal, recovered_signal, 1e-5, 1e-5, false, device);
  mx::eval(is_recovered);  // MLX is lazy: item() on an unevaluated const array throws
  EXPECT_TRUE(is_recovered.item<bool>());
}

TEST_P(MlxToolchain, should_round_trip_the_smallest_even_length) {
  const mx::Device device = GetParam();
  const mx::array two_sample_signal = mx::array({3.f, -5.f});
  const mx::array spectrum =
      mx::fft::rfft(two_sample_signal, 2, -1, mx::fft::FFTNorm::Backward, device);
  EXPECT_EQ(spectrum.shape(), (mx::Shape{2}));  // n/2 + 1 frequency entries
  const mx::array recovered_signal =
      mx::fft::irfft(spectrum, 2, -1, mx::fft::FFTNorm::Backward, device);
  EXPECT_TRUE(mx::allclose(two_sample_signal, recovered_signal, 1e-5, 1e-5, false, device).item<bool>());
}

TEST_P(MlxToolchain, should_surface_invalid_shapes_as_standard_exceptions) {
  const mx::array six_entries = mx::zeros({6}, mx::float32, GetParam());
  EXPECT_THROW(mx::reshape(six_entries, {4}, GetParam()), std::invalid_argument);
}
