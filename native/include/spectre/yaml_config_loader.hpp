#pragma once

#include <filesystem>
#include <string>

#include "spectre/config_types.hpp"

namespace spectre {

// Parses the `spectre_layer:` section of a YAML document. Unknown keys, missing
// required keys, wrong value types and rule violations throw
// std::invalid_argument. Other top-level sections belong to other loaders and
// are ignored here, so one `--config` file can hold every section an app needs.
SpectreLayerConfig parse_spectre_layer_config(const std::string& yaml_text);

// Reads the file and delegates to parse_spectre_layer_config. Throws
// std::runtime_error if the file cannot be read.
SpectreLayerConfig load_spectre_layer_config(const std::filesystem::path& config_path);

}  // namespace spectre
