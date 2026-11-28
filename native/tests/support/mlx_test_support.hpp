#pragma once

// Shared plumbing for tests that touch MLX: run each test on both devices and
// move data between plain C++ vectors and MLX arrays.
#include <gtest/gtest.h>
#include <mlx/mlx.h>

#include <random>
#include <string>
#include <vector>

namespace spectre::testing {

namespace mx = mlx::core;

// Parameterized-test base: GetParam() is the device the test must run on.
class DeviceTest : public ::testing::TestWithParam<mx::Device> {};

inline std::string device_test_suffix(const ::testing::TestParamInfo<mx::Device>& device_info) {
  return device_info.param.type == mx::Device::cpu ? "cpu" : "gpu";
}

// Test names end in /cpu or /gpu so CI can drop the GPU variants with one filter.
#define SPECTRE_INSTANTIATE_DEVICE_TESTS(suite_name)                                           \
  INSTANTIATE_TEST_SUITE_P(Devices, suite_name,                                                \
                           ::testing::Values(mlx::core::Device(mlx::core::Device::cpu),        \
                                             mlx::core::Device(mlx::core::Device::gpu)),       \
                           ::spectre::testing::device_test_suffix)

// Deterministic standard-normal values; float so MLX and the double oracle see identical inputs.
inline std::vector<float> random_float_values(std::size_t value_count, unsigned seed) {
  std::mt19937 random_engine(seed);
  std::normal_distribution<float> standard_normal(0.0f, 1.0f);
  std::vector<float> values(value_count);
  for (float& value : values) {
    value = standard_normal(random_engine);
  }
  return values;
}

inline mx::array to_device_array(const std::vector<float>& values, const mx::Shape& shape,
                                 const mx::Device& device) {
  return mx::copy(mx::array(values.begin(), shape, mx::float32), device);
}

// Evaluates (MLX is lazy) and copies the array back as float32 host values.
inline std::vector<float> to_host_floats(const mx::array& device_array) {
  mx::array float_array = mx::astype(device_array, mx::float32);
  mx::eval(float_array);
  return std::vector<float>(float_array.data<float>(), float_array.data<float>() + float_array.size());
}

inline std::vector<double> widen_to_double(const std::vector<float>& values) {
  return std::vector<double>(values.begin(), values.end());
}

}  // namespace spectre::testing
