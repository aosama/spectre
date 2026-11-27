#include "spectre/yaml_config_loader.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

const char* const kValidR10Yaml = R"(
spectre_layer:
  d_model: 768
  head_count: 12
  n_fft: 1024
  gate_hidden_width: 256
  causal_chunk_count: 4
)";

// Replaces one line of the valid document so each negative test changes exactly
// one thing and a failure points at that thing.
std::string valid_yaml_with_line_replaced(const std::string& original_line,
                                          const std::string& replacement_line) {
  std::string yaml_text = kValidR10Yaml;
  const std::size_t line_start = yaml_text.find(original_line);
  EXPECT_NE(line_start, std::string::npos) << "test setup: line not found: " << original_line;
  yaml_text.replace(line_start, original_line.size(), replacement_line);
  return yaml_text;
}

// Removes the file on scope exit so a failing assertion cannot leave it behind.
class ScopedConfigFile {
 public:
  ScopedConfigFile(const std::string& file_name, const std::string& file_text)
      : config_path_(std::filesystem::temp_directory_path() / file_name) {
    std::ofstream(config_path_) << file_text;
  }
  ~ScopedConfigFile() { std::filesystem::remove(config_path_); }
  const std::filesystem::path& path() const { return config_path_; }

 private:
  std::filesystem::path config_path_;
};

}  // namespace

TEST(YamlConfigLoader, should_parse_a_valid_r10_config_with_the_causal_default) {
  const spectre::SpectreLayerConfig layer_config = spectre::parse_spectre_layer_config(kValidR10Yaml);
  EXPECT_EQ(layer_config.d_model, 768);
  EXPECT_EQ(layer_config.head_count, 12);
  EXPECT_EQ(layer_config.n_fft, 1024);
  EXPECT_EQ(layer_config.gate_hidden_width, 256);
  EXPECT_EQ(layer_config.causal_chunk_count, 4);
  EXPECT_EQ(layer_config.gate_pooling, spectre::GatePooling::CausalPrefixMean);
}

TEST(YamlConfigLoader, should_parse_an_explicit_legacy_config) {
  const std::string legacy_yaml = valid_yaml_with_line_replaced(
      "causal_chunk_count: 4", "causal_chunk_count: 0\n  gate_pooling: whole_window_legacy");
  const spectre::SpectreLayerConfig layer_config = spectre::parse_spectre_layer_config(legacy_yaml);
  EXPECT_EQ(layer_config.gate_pooling, spectre::GatePooling::WholeWindowLegacy);
  EXPECT_EQ(layer_config.causal_chunk_count, 0);
}

TEST(YamlConfigLoader, should_ignore_sections_owned_by_other_loaders) {
  const std::string yaml_with_extra_section = std::string(kValidR10Yaml) + "train:\n  epochs: 3\n";
  EXPECT_NO_THROW(spectre::parse_spectre_layer_config(yaml_with_extra_section));
}

TEST(YamlConfigLoader, should_load_a_config_from_a_file) {
  const ScopedConfigFile config_file("spectre_loader_test_valid.yaml", kValidR10Yaml);
  EXPECT_EQ(spectre::load_spectre_layer_config(config_file.path()).n_fft, 1024);
}

TEST(YamlConfigLoader, should_reject_an_unknown_key) {
  const std::string yaml_with_typo = valid_yaml_with_line_replaced("n_fft: 1024", "n_ftt: 1024");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_with_typo), std::invalid_argument);
}

TEST(YamlConfigLoader, should_name_the_unknown_key_in_the_error_message) {
  const std::string yaml_with_extra_key = std::string(kValidR10Yaml) + "  dropout: 0.1\n";
  try {
    spectre::parse_spectre_layer_config(yaml_with_extra_key);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& rejection) {
    EXPECT_NE(std::string(rejection.what()).find("dropout"), std::string::npos);
  }
}

TEST(YamlConfigLoader, should_reject_a_missing_required_key) {
  const std::string yaml_without_n_fft = valid_yaml_with_line_replaced("  n_fft: 1024\n", "");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_without_n_fft), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_missing_layer_section) {
  EXPECT_THROW(spectre::parse_spectre_layer_config("train:\n  epochs: 3\n"), std::invalid_argument);
  EXPECT_THROW(spectre::parse_spectre_layer_config(""), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_non_integer_size) {
  for (const char* bad_size : {"abc", "64.5", "\"\""}) {
    const std::string yaml_text =
        valid_yaml_with_line_replaced("d_model: 768", std::string("d_model: ") + bad_size);
    EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument)
        << "d_model=" << bad_size;
  }
}

TEST(YamlConfigLoader, should_reject_malformed_yaml) {
  EXPECT_THROW(spectre::parse_spectre_layer_config("spectre_layer: [unclosed"),
               std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_an_unknown_gate_pooling_name) {
  const std::string yaml_text =
      valid_yaml_with_line_replaced("causal_chunk_count: 4", "causal_chunk_count: 4\n  gate_pooling: legacy");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_n_fft_that_is_not_a_power_of_two) {
  const std::string yaml_text = valid_yaml_with_line_replaced("n_fft: 1024", "n_fft: 1000");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_single_causal_chunk) {
  const std::string yaml_text = valid_yaml_with_line_replaced("causal_chunk_count: 4", "causal_chunk_count: 1");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_chunk_count_that_does_not_divide_n_fft) {
  const std::string yaml_text = valid_yaml_with_line_replaced("causal_chunk_count: 4", "causal_chunk_count: 3");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument);
}

TEST(YamlConfigLoader, should_throw_a_runtime_error_for_a_missing_file) {
  const std::filesystem::path absent_path = std::filesystem::temp_directory_path() / "spectre_no_such_config.yaml";
  EXPECT_THROW(spectre::load_spectre_layer_config(absent_path), std::runtime_error);
}

TEST(YamlConfigLoader, should_accept_an_explicit_causal_prefix_mean_pooling) {
  const std::string yaml_text = valid_yaml_with_line_replaced(
      "causal_chunk_count: 4", "causal_chunk_count: 4\n  gate_pooling: causal_prefix_mean");
  EXPECT_EQ(spectre::parse_spectre_layer_config(yaml_text).gate_pooling,
            spectre::GatePooling::CausalPrefixMean);
}

TEST(YamlConfigLoader, should_reject_documents_that_are_not_mappings) {
  for (const char* non_mapping_document : {"42", "just text", "- a\n- b", "null"}) {
    EXPECT_THROW(spectre::parse_spectre_layer_config(non_mapping_document), std::invalid_argument)
        << "document=" << non_mapping_document;
  }
}

TEST(YamlConfigLoader, should_reject_a_layer_section_that_is_not_a_mapping) {
  EXPECT_THROW(spectre::parse_spectre_layer_config("spectre_layer: 5"), std::invalid_argument);
  EXPECT_THROW(spectre::parse_spectre_layer_config("spectre_layer: [1, 2]"), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_non_scalar_key) {
  const std::string yaml_with_list_key = std::string(kValidR10Yaml) + "  ? [a, b]\n  : 1\n";
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_with_list_key), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_list_value_for_a_size) {
  const std::string yaml_text = valid_yaml_with_line_replaced("d_model: 768", "d_model: [1, 2]");
  EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument);
}

TEST(YamlConfigLoader, should_reject_a_null_or_list_gate_pooling) {
  for (const char* bad_pooling : {"", "[causal_prefix_mean]"}) {
    const std::string yaml_text = valid_yaml_with_line_replaced(
        "causal_chunk_count: 4", std::string("causal_chunk_count: 4\n  gate_pooling: ") + bad_pooling);
    EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument)
        << "gate_pooling=" << bad_pooling;
  }
}

TEST(YamlConfigLoader, should_reject_negative_and_overflowing_sizes) {
  for (const char* bad_size : {"-768", "0", "99999999999"}) {
    const std::string yaml_text =
        valid_yaml_with_line_replaced("d_model: 768", std::string("d_model: ") + bad_size);
    EXPECT_THROW(spectre::parse_spectre_layer_config(yaml_text), std::invalid_argument)
        << "d_model=" << bad_size;
  }
}

TEST(YamlConfigLoader, should_throw_a_runtime_error_when_the_path_is_a_directory) {
  EXPECT_THROW(spectre::load_spectre_layer_config(std::filesystem::temp_directory_path()),
               std::runtime_error);
}
