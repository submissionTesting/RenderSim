#pragma once

#include <string>

#include "RenderSim/operator_scheduler.hpp"  // For MappedIR definition

namespace rendersim {

/** Load a mapped IR from JSON; throws on failure. */
MappedIR load_mapped_ir_from_json(const std::string &json_path);

} // namespace rendersim 