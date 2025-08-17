#include "RenderSim/mapping_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

// Header-only nlohmann/json is preferred for ease of integration.
// A system-wide installation works too – CMake will link it if available.
#include <nlohmann/json.hpp>

namespace rendersim {

using json = nlohmann::json;

static TensorDesc parse_tensor_desc(const json &j) {
    TensorDesc td;
    if (!j.is_object() || !j.contains("shape")) {
        throw std::runtime_error("Tensor descriptor missing required fields");
    }
    td.shape = j.at("shape").get<std::vector<int32_t>>();
    td.dtype = j.value("dtype", "float32");
    return td;
}

MappedIR load_mapped_ir_from_json(const std::string &json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open mapped IR JSON file: " + json_path);
    }

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    json root = json::parse(buffer.str());

    // Handle wrapper {"mapped_ir": ...} formats (may be string-encoded)
    json mapped_json;
    if (root.contains("mapped_ir")) {
        const auto &mir = root.at("mapped_ir");
        if (mir.is_string()) {
            mapped_json = json::parse(mir.get<std::string>());
        } else {
            mapped_json = mir;
        }
    } else {
        mapped_json = root;  // Assume IR is at the top level
    }

    // Basic validation
    if (!mapped_json.contains("nodes") || !mapped_json.contains("edges")) {
        throw std::runtime_error("Mapped IR JSON is missing 'nodes' or 'edges' fields");
    }

    MappedIR result;

    // Parse nodes
    const auto &nodes_obj = mapped_json.at("nodes");
    if (!nodes_obj.is_object()) {
        throw std::runtime_error("'nodes' must be a JSON object");
    }

    for (auto it = nodes_obj.begin(); it != nodes_obj.end(); ++it) {
        const std::string node_id = it.key();
        const json &node_json = it.value();

        MappedIRNode mapped_node;

        // --------------------------
        // OperatorNode fields
        // --------------------------
        const json &op_node_json = node_json.at("op_node");
        mapped_node.op_node.id = op_node_json.at("id").get<std::string>();
        mapped_node.op_node.op_type = op_node_json.at("op_type").get<std::string>();
        mapped_node.op_node.call_count = op_node_json.value("call_count", 1);

        // Inputs / outputs
        for (const auto &t_j : op_node_json.at("inputs")) {
            mapped_node.op_node.inputs.push_back(parse_tensor_desc(t_j));
        }
        for (const auto &t_j : op_node_json.at("outputs")) {
            mapped_node.op_node.outputs.push_back(parse_tensor_desc(t_j));
        }

        // --------------------------
        // Mapping information
        // --------------------------
        mapped_node.hw_unit = node_json.at("hw_unit").get<std::string>();

        if (node_json.contains("attrs")) {
            for (const auto &attr_it : node_json.at("attrs").items()) {
                mapped_node.attrs[attr_it.key()] = attr_it.value().dump();
            }
        }

        result.nodes[node_id] = std::move(mapped_node);
    }

    // Parse edges (list of 2-element arrays)
    for (const auto &edge_j : mapped_json.at("edges")) {
        if (edge_j.is_array() && edge_j.size() == 2) {
            result.edges.emplace_back(
                edge_j.at(0).get<std::string>(),
                edge_j.at(1).get<std::string>()
            );
        }
    }

    return result;
}

} // namespace rendersim 