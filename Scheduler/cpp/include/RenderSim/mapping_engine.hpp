#pragma once

#include "RenderSim/ir.hpp"
#include "RenderSim/hw_config.hpp"
#include "RenderSim/operator_scheduler.hpp" // for MappedIR, MappedIRNode

namespace rendersim {

/** Greedy taxonomy-based mapping identical to Python implementation. */
MappedIR map_operator_graph(const OperatorGraph &graph, const HWConfig &cfg);

} // namespace rendersim 