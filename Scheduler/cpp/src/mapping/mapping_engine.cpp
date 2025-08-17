#include "RenderSim/mapping_engine.hpp"
#include <stdexcept>

#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <string>

namespace rendersim {

static std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return std::toupper(c);});
    return s;
}

static const std::unordered_map<std::string, std::vector<std::string>> kFallbacks = {
    {"SAMPLING", {"VOLUME_RENDERING", "FIELD_COMPUTATION"}},
    {"BLENDING", {"VOLUME_RENDERING", "BLENDING"}},
    {"RAY_TRACING", {"VOLUME_RENDERING", "FIELD_COMPUTATION"}},
    {"HASH_ENCODE", {"HASH_ENCODE", "POSITIONAL_ENCODE", "FIELD_COMPUTATION"}},
    {"POSITIONAL_ENCODE", {"POSITIONAL_ENCODE", "HASH_ENCODE", "FIELD_COMPUTATION"}},
    {"MLP", {"MLP", "FIELD_COMPUTATION"}},
    {"POSITIONAL_ENCODING", {"POSITIONAL_ENCODE", "FIELD_COMPUTATION"}},
    {"MLP_COMPUTATION", {"MLP", "FIELD_COMPUTATION"}},
    {"RGB_VOLUME_RENDERING", {"VOLUME_RENDERING", "BLENDING"}},
    {"VOLUME_RENDERING", {"VOLUME_RENDERING", "BLENDING"}},
    {"UNKNOWN", {"FIELD_COMPUTATION", "VOLUME_RENDERING", "POSITIONAL_ENCODE"}}
};

static int64_t product(const std::vector<int32_t>& v) {
    if (v.empty()) return 0;
    int64_t p = 1;
    for (auto x : v) p *= std::max<int32_t>(1, x);
    return p;
}

MappedIR map_operator_graph(const OperatorGraph &graph, const HWConfig &cfg) {
    MappedIR ir;

    auto units_by_type = cfg.units_by_type();

    // Build mapped nodes
    for (const auto &node : graph.nodes) {
        std::string op_type = to_upper(node.op_type);
        const std::vector<HWUnit> *candidates = nullptr;

        // direct mapping
        auto it = units_by_type.find(op_type);
        if (it != units_by_type.end() && !it->second.empty()) {
            candidates = &it->second;
        }
        // fallbacks
        if (!candidates) {
            auto fb_it = kFallbacks.find(op_type);
            if (fb_it != kFallbacks.end()) {
                for (const auto &fb : fb_it->second) {
                    auto fit = units_by_type.find(fb);
                    if (fit != units_by_type.end() && !fit->second.empty()) {
                        candidates = &fit->second;
                        break;
                    }
                }
            }
        }
        // generic fallback
        if (!candidates) {
            for (const std::string &generic : {"GENERIC", "FIELD_COMPUTATION", "ENCODING"}) {
                auto git = units_by_type.find(generic);
                if (git != units_by_type.end() && !git->second.empty()) {
                    candidates = &git->second;
                    break;
                }
            }
        }
        if (!candidates) {
            // last resort: first unit overall
            if (!cfg.units.empty()) {
                candidates = &cfg.units;
            } else {
                throw std::runtime_error("No hardware units available for operator mapping");
            }
        }

        const HWUnit &selected = (*candidates)[0]; // simple selection

        MappedIRNode mnode;
        mnode.op_node = node;
        mnode.hw_unit = selected.id;

        // Attach approximate workload and byte counts for scheduling
        int64_t in_elems_sum = 0, out_elems_sum = 0;
        for (const auto& t : node.inputs)  in_elems_sum  += product(t.shape);
        for (const auto& t : node.outputs) out_elems_sum += product(t.shape);
        int64_t bytes = (in_elems_sum + out_elems_sum) * 4; // assume fp32
        mnode.attrs["work_elems"] = std::to_string(std::max<int64_t>(1, in_elems_sum));
        mnode.attrs["out_elems"]  = std::to_string(std::max<int64_t>(1, out_elems_sum));
        mnode.attrs["bytes"]      = std::to_string(std::max<int64_t>(1, bytes));
        // also propagate op_type for optimizer heuristics
        mnode.attrs["op_type"]    = op_type;
        // (Optional) If extended metadata is added to OperatorNode in the future,
        // those hints (e.g., flop_count) can be propagated here into attrs.

        ir.nodes[node.id] = std::move(mnode);
    }

    // Preserve edges from OperatorGraph by converting index pairs to node IDs
    for (const auto &e : graph.edges) {
        int src_idx = e.first;
        int dst_idx = e.second;
        if (src_idx >= 0 && src_idx < static_cast<int>(graph.nodes.size()) &&
            dst_idx >= 0 && dst_idx < static_cast<int>(graph.nodes.size())) {
            const auto &src_node = graph.nodes[src_idx];
            const auto &dst_node = graph.nodes[dst_idx];
            ir.edges.emplace_back(src_node.id, dst_node.id);
        }
    }

    return ir;
}

} // namespace rendersim 