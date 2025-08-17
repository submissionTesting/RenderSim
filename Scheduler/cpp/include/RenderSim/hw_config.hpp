#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace rendersim {

struct HWUnit {
    std::string id;
    std::string type; // module_type in JSON
};

struct HWConfig {
    std::string accelerator_name;
    std::vector<HWUnit> units;

    // Helper: units grouped by type
    std::unordered_map<std::string, std::vector<HWUnit>> units_by_type() const;
};

/**
 * Parse a RenderSim hardware configuration JSON file and return HWConfig.
 * Throws std::runtime_error on failure.
 */
HWConfig load_hw_config_from_json(const std::string &json_path);

} // namespace rendersim 