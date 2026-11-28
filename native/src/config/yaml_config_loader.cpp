#include "spectre/yaml_config_loader.hpp"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <yaml-cpp/yaml.h>

namespace spectre {

namespace {

constexpr std::string_view kLayerSectionName = "spectre_layer";

constexpr std::array<std::string_view, 6> kLayerKeys = {
    "d_model", "head_count", "n_fft", "gate_hidden_width", "causal_chunk_count", "gate_pooling"};

bool is_known_layer_key(const std::string& key_name) {
  for (const std::string_view known_key : kLayerKeys) {
    if (key_name == known_key) {
      return true;
    }
  }
  return false;
}

// Typos in a config silently falling back to defaults would be a worse failure
// than a refused run, hence the strict key check.
void reject_unknown_layer_keys(const YAML::Node& layer_section) {
  for (const auto& key_and_node : layer_section) {
    if (!key_and_node.first.IsScalar()) {
      throw std::invalid_argument("keys in the " + std::string(kLayerSectionName) +
                                  " section must be plain names, not lists or maps");
    }
    const std::string key_name = key_and_node.first.Scalar();
    if (!is_known_layer_key(key_name)) {
      throw std::invalid_argument("unknown key '" + key_name + "' in " +
                                  std::string(kLayerSectionName) + " section");
    }
  }
}

YAML::Node require_key(const YAML::Node& layer_section, const char* key_name) {
  const YAML::Node key_node = layer_section[key_name];
  if (!key_node) {
    throw std::invalid_argument(std::string("missing required key '") + key_name + "' in " +
                                std::string(kLayerSectionName) + " section");
  }
  return key_node;
}

int read_integer_key(const YAML::Node& layer_section, const char* key_name) {
  const YAML::Node key_node = require_key(layer_section, key_name);
  // A list or map value has an empty Scalar(), which would make the message
  // below say "got ''"; naming the node kind is more useful.
  const std::string offending_text = key_node.IsScalar() ? key_node.Scalar() : "a list or map";
  try {
    return key_node.as<int>();
  } catch (const YAML::Exception&) {
    throw std::invalid_argument(std::string("key '") + key_name +
                                "' must be an integer, got '" + offending_text + "'");
  }
}

GatePooling read_gate_pooling_key(const YAML::Node& layer_section) {
  // Absent means the causal default; legacy must always be spelled out (D11).
  const YAML::Node pooling_node = layer_section["gate_pooling"];
  if (!pooling_node) {
    return GatePooling::CausalPrefixMean;
  }
  if (!pooling_node.IsScalar()) {
    throw std::invalid_argument("key 'gate_pooling' must be a name, not a list or map");
  }
  return parse_gate_pooling(pooling_node.Scalar());
}

YAML::Node parse_yaml_document(const std::string& yaml_text) {
  try {
    return YAML::Load(yaml_text);
  } catch (const YAML::Exception& parse_error) {
    throw std::invalid_argument(std::string("invalid YAML: ") + parse_error.what());
  }
}

YAML::Node load_layer_section(const std::string& yaml_text) {
  const YAML::Node document = parse_yaml_document(yaml_text);
  // A bare scalar or list document would make operator[] throw a yaml-cpp
  // exception, which would escape the invalid_argument contract.
  if (!document.IsMap()) {
    throw std::invalid_argument("the config must be a YAML mapping with a '" +
                                std::string(kLayerSectionName) + "' section");
  }
  const YAML::Node layer_section = document[std::string(kLayerSectionName)];
  if (!layer_section || !layer_section.IsMap()) {
    throw std::invalid_argument("missing '" + std::string(kLayerSectionName) +
                                "' mapping at the top level of the config");
  }
  return layer_section;
}

}  // namespace

SpectreLayerConfig parse_spectre_layer_config(const std::string& yaml_text) {
  const YAML::Node layer_section = load_layer_section(yaml_text);
  reject_unknown_layer_keys(layer_section);

  SpectreLayerConfig layer_config;
  layer_config.d_model = read_integer_key(layer_section, "d_model");
  layer_config.head_count = read_integer_key(layer_section, "head_count");
  layer_config.n_fft = read_integer_key(layer_section, "n_fft");
  layer_config.gate_hidden_width = read_integer_key(layer_section, "gate_hidden_width");
  layer_config.causal_chunk_count = read_integer_key(layer_section, "causal_chunk_count");
  layer_config.gate_pooling = read_gate_pooling_key(layer_section);

  validate_spectre_layer_config(layer_config);
  return layer_config;
}

SpectreLayerConfig load_spectre_layer_config(const std::filesystem::path& config_path) {
  // A directory opens successfully as an ifstream and reads as empty, which
  // would surface later as a misleading "missing section" error.
  std::ifstream config_stream(config_path);
  if (!std::filesystem::is_regular_file(config_path) || !config_stream) {
    throw std::runtime_error("cannot read config file: " + config_path.string());
  }
  std::ostringstream config_text;
  config_text << config_stream.rdbuf();
  return parse_spectre_layer_config(config_text.str());
}

}  // namespace spectre
