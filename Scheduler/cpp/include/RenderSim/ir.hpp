#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace rendersim {

struct TensorDesc {
    std::vector<int32_t> shape;
    std::string dtype{"float32"};
};

struct OperatorNode {
    std::string id;
    std::string op_type;
    std::vector<TensorDesc> inputs;
    std::vector<TensorDesc> outputs;
    int32_t call_count{1};
};

struct OperatorGraph {
    std::vector<OperatorNode> nodes;
    std::vector<std::pair<int32_t,int32_t>> edges; // indices into nodes
};

} // namespace rendersim 