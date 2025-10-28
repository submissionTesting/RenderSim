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
        """Greedy mapping with fallback logic for neural rendering operators, supporting training."""
        ir = MappedIR()
        type_to_units = self.hw_config.units_by_type()
        
        # Define fallback mappings for common neural rendering operators
        # Including training-specific operators for GSArch, GBU, and Instant3D
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
            
            # GSArch-specific mappings
            "TILEMERGING": ["TILEMERGING", "BLENDING", "FIELD_COMPUTATION"],
            "FEATURECOMPUTE": ["FEATURECOMPUTE", "FIELD_COMPUTATION"],
            "GRADIENTCOMPUTE": ["GRADIENTCOMPUTE", "GRADIENT_ACCUMULATION", "FIELD_COMPUTATION"],
            "GRADIENTPRUNING": ["GRADIENTPRUNING", "OPTIMIZATION"],
            "REARRANGEMENT": ["REARRANGEMENT", "OPTIMIZATION"],
            
            # GBU-specific mappings
            "ROWPROCESSING": ["ROWPROCESSING", "FIELD_COMPUTATION"],
            "ROWGENERATION": ["ROWGENERATION", "ENCODING"],
            "DECOMPBINNING": ["DECOMPBINNING", "OPTIMIZATION"],
            
            # Instant3D-specific mappings
            "FRM": ["FRM", "HASH_ENCODE", "ENCODING"],
            "BUM": ["BUM", "GRADIENT_ACCUMULATION", "OPTIMIZATION"],
            
            # Generic backward pass mappings (handle operators with (B) suffix)
            "MLP (B)": ["MLP", "FIELD_COMPUTATION"],
            "HASH_ENCODE (B)": ["HASH_ENCODE", "BUM", "ENCODING"],
            "HASHENCODING (B)": ["HASH_ENCODE", "BUM", "ENCODING"],
            "BLENDING (B)": ["BLENDING", "GRADIENTCOMPUTE", "VOLUME_RENDERING"],
            "GAUSSIANALPHABLEND (B)": ["GRADIENTCOMPUTE", "BLENDING"],
            "RGBRENDERER (B)": ["VOLUME_RENDERING", "BLENDING"],
            "DENSITYRENDERER (B)": ["VOLUME_RENDERING", "BLENDING"],
            
            "unknown": ["FIELD_COMPUTATION", "VOLUME_RENDERING", "POSITIONAL_ENCODE"]
        }
        
        for nid, node in graph.nodes.items():
            # Handle backward operators - check if op_type ends with (B)
            op_type = node.op_type.upper()
            is_backward = op_type.endswith(" (B)")
            
            # For backward operators, also try mapping without the (B) suffix
            if is_backward:
                base_op_type = op_type[:-4]  # Remove " (B)" suffix
            else:
                base_op_type = op_type
            
            hw_units = None
            
            # Try direct mapping first
            if op_type in type_to_units and type_to_units[op_type]:
                hw_units = type_to_units[op_type]
            
            # For backward operators, also try base op type
            if not hw_units and is_backward:
                if base_op_type in type_to_units and type_to_units[base_op_type]:
                    hw_units = type_to_units[base_op_type]
            
            # Try fallback mappings
            if not hw_units and op_type in fallback_mappings:
                for fallback_type in fallback_mappings[op_type]:
                    if fallback_type in type_to_units and type_to_units[fallback_type]:
                        hw_units = type_to_units[fallback_type]
                        break
            
            # For backward operators, also try fallback with base op type
            if not hw_units and is_backward and base_op_type in fallback_mappings:
                for fallback_type in fallback_mappings[base_op_type]:
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