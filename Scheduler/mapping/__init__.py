# mapping subpackage

"""MappingEngine: assigns OperatorGraph nodes to hardware instances.
Implementation with fallback mappings for neural rendering operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from pathlib import Path

from ..IR import OperatorGraph, MappedIR, MappedIRNode
from .hw_config import HWConfig, load_hw_config

__all__ = ["MappingEngine"]


@dataclass
class MappingEngine:
    hw_config: HWConfig

    @classmethod
    def from_json(cls, cfg_path: str | Path) -> "MappingEngine":
        return cls(hw_config=load_hw_config(cfg_path))

    def run(self, graph: OperatorGraph) -> MappedIR:
        """Greedy mapping with fallback logic for neural rendering operators."""
        ir = MappedIR()
        type_to_units = self.hw_config.units_by_type()
        
        # Define fallback mappings for common neural rendering operators
        # ICARUS-compatible fallback mappings for neural rendering
        fallback_mappings = {
            "SAMPLING": ["VOLUME_RENDERING", "FIELD_COMPUTATION"],
            "BLENDING": ["VOLUME_RENDERING", "BLENDING"],
            "RAY_TRACING": ["VOLUME_RENDERING", "FIELD_COMPUTATION"],
            "HASH_ENCODE": ["HASH_ENCODE", "POSITIONAL_ENCODE", "FIELD_COMPUTATION"],
            "POSITIONAL_ENCODE": ["POSITIONAL_ENCODE", "HASH_ENCODE", "FIELD_COMPUTATION"],
            "MLP": ["MLP", "FIELD_COMPUTATION"],
            "POSITIONAL_ENCODING": ["POSITIONAL_ENCODE", "FIELD_COMPUTATION"],
            "MLP_COMPUTATION": ["MLP", "FIELD_COMPUTATION"],
            "RGB_VOLUME_RENDERING": ["VOLUME_RENDERING", "BLENDING"],
            "VOLUME_RENDERING": ["VOLUME_RENDERING", "BLENDING"],
            "unknown": ["FIELD_COMPUTATION", "VOLUME_RENDERING", "POSITIONAL_ENCODE"]
        }
        
        for nid, node in graph.nodes.items():
            op_type = node.op_type.upper()
            hw_units = None
            
            # Try direct mapping first
            if op_type in type_to_units and type_to_units[op_type]:
                hw_units = type_to_units[op_type]
            
            # Try fallback mappings
            if not hw_units and op_type in fallback_mappings:
                for fallback_type in fallback_mappings[op_type]:
                    if fallback_type in type_to_units and type_to_units[fallback_type]:
                        hw_units = type_to_units[fallback_type]
                        break
            
            # Try generic fallback
            if not hw_units:
                for generic_type in ["GENERIC", "FIELD_COMPUTATION", "ENCODING"]:
                    if generic_type in type_to_units and type_to_units[generic_type]:
                        hw_units = type_to_units[generic_type]
                        break
            
            # If still no hardware found, use the first available unit
            if not hw_units:
                all_units = [unit for units_list in type_to_units.values() for unit in units_list]
                if all_units:
                    hw_units = [all_units[0]]
                else:
                    raise RuntimeError(f"No hardware units available for mapping operator {node.op_type}")
            
            # Use load balancing - select unit with least assigned operators
            selected_unit = hw_units[0]  # Simple: just use first available
            
            ir.nodes[nid] = MappedIRNode(op_node=node, hw_unit=selected_unit.id)
        
        ir.edges = list(graph.edges)
        return ir 