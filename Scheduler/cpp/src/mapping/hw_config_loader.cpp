#include "RenderSim/hw_config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace rendersim {

using json = nlohmann::json;

std::unordered_map<std::string, std::vector<HWUnit>> HWConfig::units_by_type() const {
    std::unordered_map<std::string, std::vector<HWUnit>> result;
    for (const auto &u : units) {
        result[u.type].push_back(u);
    }
    return result;
}

HWConfig load_hw_config_from_json(const std::string &json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) throw std::runtime_error("Failed to open HW config: " + json_path);
    std::stringstream buffer; buffer << ifs.rdbuf();
    json root = json::parse(buffer.str());

    HWConfig cfg;
    cfg.accelerator_name = root.value("accelerator_name", "UNKNOWN");

    if (!root.contains("hardware_modules")) {
        throw std::runtime_error("HW config missing 'hardware_modules'");
    }

    const auto &mods = root.at("hardware_modules");
    for (auto it = mods.begin(); it != mods.end(); ++it) {
        const std::string id = it.key();
        const json &mod_json = it.value();
        HWUnit unit;
        unit.id = id;
        unit.type = mod_json.value("module_type", "GENERIC");
        int count = mod_json.value("count", 1);
        // If count >1 replicate ids as id#idx to mimic multiple instances
        for (int i = 0; i < count; ++i) {
            HWUnit inst = unit;
            if (count > 1) inst.id = id + "_" + std::to_string(i);
            cfg.units.push_back(std::move(inst));
        }
    }

    if (cfg.units.empty()) {
        throw std::runtime_error("HW config has no hardware units");
    }

    return cfg;
}

} // namespace rendersim 