"""
Equation-based Operator Optimizer implementing Equation 1 from the paper.

This module implements the performance modeling equation:
duration(op) = max(n_op / Theta_hw * s_comp, v_off / B_hw * r_bytes)

Where:
- n_op: number of elements processed by the operator
- Theta_hw: assigned hardware unit's compute throughput
- s_comp: compute speedup factor from optimizations
- v_off: communication volume
- B_hw: effective memory bandwidth
- r_bytes: memory traffic reduction factor from optimizations
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math

@dataclass
class OperatorMetrics:
    """Metrics extracted from Mapped IR (Table 2 in paper)."""
    n_op: int  # Number of elements processed
    v_off: int  # Communication volume in bytes
    theta_hw: float  # Hardware compute throughput (ops/cycle)
    b_hw: float  # Hardware memory bandwidth (bytes/cycle)
    
@dataclass
class OptimizationFactors:
    """Optimization factors from library (s_comp and r_bytes)."""
    s_comp: float = 1.0  # Compute speedup factor
    r_bytes: float = 1.0  # Memory traffic reduction factor
    applied_strategies: List[str] = None
    
    def __post_init__(self):
        if self.applied_strategies is None:
            self.applied_strategies = []

class EquationBasedOptimizer:
    """
    Implements Equation 1 from the paper for operator duration calculation.
    This replaces the dummy optimizer with the actual performance model.
    """
    
    def __init__(self, optimization_library):
        self.optimization_library = optimization_library
        
    def calculate_duration(self, metrics: OperatorMetrics, factors: OptimizationFactors) -> int:
        """
        Calculate operator duration using Equation 1 from the paper.
        
        Returns duration in cycles.
        """
        # Compute bound: n_op / Theta_hw * s_comp
        compute_cycles = (metrics.n_op / metrics.theta_hw) * factors.s_comp
        
        # Memory bound: v_off / B_hw * r_bytes  
        memory_cycles = (metrics.v_off / metrics.b_hw) * factors.r_bytes
        
        # Take maximum (roofline model)
        duration = max(compute_cycles, memory_cycles)
        
        return int(math.ceil(duration))
    
    def extract_optimization_factors(self, operator_type: str, 
                                    operator_attrs: Dict[str, Any]) -> OptimizationFactors:
        """
        Extract s_comp and r_bytes based on applicable optimization strategies.
        
        This maps optimization strategies to their impact on compute and memory.
        """
        applicable_strategies = self.optimization_library.get_applicable_strategies(operator_type)
        
        s_comp = 1.0
        r_bytes = 1.0
        applied = []
        
        for strategy in applicable_strategies:
            # Tile culling optimization (GSArch/GSCore)
            if strategy.name == "tile_culling" and "active_ratio" in operator_attrs:
                active_ratio = operator_attrs["active_ratio"]
                s_comp *= active_ratio  # Only process active tiles
                applied.append(f"tile_culling(ratio={active_ratio:.2f})")
                
            # Gradient pruning (GSArch)
            elif strategy.name == "gradient_pruning" and operator_type == "GRADIENTCOMPUTE":
                pruning_ratio = strategy.parameters.get("pruning_ratio", 0.4)
                s_comp *= (1 - pruning_ratio)
                r_bytes *= (1 - pruning_ratio)
                applied.append(f"gradient_pruning(p={pruning_ratio:.2f})")
                
            # Row processing optimization (GBU)
            elif strategy.name == "row_processing" and operator_type == "ROWPROCESSING":
                bundle_efficiency = operator_attrs.get("bundle_efficiency", 0.8)
                s_comp *= bundle_efficiency
                applied.append(f"row_processing(eff={bundle_efficiency:.2f})")
                
            # FRM read coalescing (Instant3D)
            elif strategy.name == "frm_coalescing" and operator_type == "FRM":
                coalesce_factor = operator_attrs.get("coalesce_factor", 4)
                r_bytes *= (1.0 / coalesce_factor)
                applied.append(f"frm_coalescing(factor={coalesce_factor})")
                
            # BUM gradient merging (Instant3D)
            elif strategy.name == "bum_merging" and operator_type == "BUM":
                merge_ratio = operator_attrs.get("merge_ratio", 0.6)
                r_bytes *= merge_ratio
                applied.append(f"bum_merging(ratio={merge_ratio:.2f})")
                
            # Early ray termination
            elif strategy.name == "early_ray_termination":
                early_term_ratio = operator_attrs.get("early_termination_ratio", 0.7)
                s_comp *= early_term_ratio
                applied.append(f"early_termination(ratio={early_term_ratio:.2f})")
                
            # Sparse radiance warping (CICERO)
            elif strategy.name == "sparse_radiance_warping":
                reuse_ratio = operator_attrs.get("frame_reuse_ratio", 0.3)
                s_comp *= (1 - reuse_ratio)
                r_bytes *= (1 - reuse_ratio)
                applied.append(f"radiance_warping(reuse={reuse_ratio:.2f})")
                
            # Low precision optimizations
            elif strategy.opt_type.value == "low_bit":
                bit_reduction = operator_attrs.get("bit_reduction_factor", 0.5)
                r_bytes *= bit_reduction
                applied.append(f"low_precision(factor={bit_reduction:.2f})")
        
        return OptimizationFactors(s_comp, r_bytes, applied)
    
    def optimize(self, operator_type: str, operator_attrs: Dict[str, Any],
                 hw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main optimization function that applies Equation 1.
        
        Args:
            operator_type: Type of operator
            operator_attrs: Operator attributes from Mapped IR
            hw_metrics: Hardware metrics (throughput, bandwidth)
            
        Returns:
            Dictionary with duration and optimization metadata
        """
        # Extract metrics from inputs
        metrics = OperatorMetrics(
            n_op=operator_attrs.get("num_elements", 1000),
            v_off=operator_attrs.get("memory_bytes", 4096),
            theta_hw=hw_metrics.get("throughput_ops_per_cycle", 10.0),
            b_hw=hw_metrics.get("bandwidth_bytes_per_cycle", 64.0)
        )
        
        # Get optimization factors
        factors = self.extract_optimization_factors(operator_type, operator_attrs)
        
        # Calculate duration using Equation 1
        duration = self.calculate_duration(metrics, factors)
        
        # Determine if compute or memory bound
        compute_cycles = (metrics.n_op / metrics.theta_hw) * factors.s_comp
        memory_cycles = (metrics.v_off / metrics.b_hw) * factors.r_bytes
        is_compute_bound = compute_cycles >= memory_cycles
        
        return {
            "duration": duration,
            "applied_optimizations": factors.applied_strategies,
            "s_comp": factors.s_comp,
            "r_bytes": factors.r_bytes,
            "is_compute_bound": is_compute_bound,
            "compute_cycles": int(compute_cycles),
            "memory_cycles": int(memory_cycles),
            "optimization_metadata": {
                "n_op": metrics.n_op,
                "v_off": metrics.v_off,
                "theta_hw": metrics.theta_hw,
                "b_hw": metrics.b_hw
            }
        }

def apply_optimizations(mapped_ir):
    """
    Apply operator-level optimizations to produce Op-Scheduled IR.
    
    This is the main entry point that transforms Mapped IR to Op-Scheduled IR
    by applying Equation 1 to each operator.
    """
    from .optimization_library import OptimizationLibrary
    
    # Initialize optimization library and optimizer
    opt_lib = OptimizationLibrary()
    optimizer = EquationBasedOptimizer(opt_lib)
    
    # Create Op-Scheduled IR from Mapped IR
    op_scheduled_ir = mapped_ir.copy()
    
    for node_id, node in op_scheduled_ir.nodes.items():
        # Extract hardware metrics from mapped hardware unit
        hw_metrics = {}
        if hasattr(node, 'hw_units') and node.hw_units:
            hw_unit = node.hw_units[0]
            hw_metrics = {
                "throughput_ops_per_cycle": getattr(hw_unit, 'throughput', 10.0),
                "bandwidth_bytes_per_cycle": getattr(hw_unit, 'bandwidth', 64.0)
            }
        
        # Get operator attributes
        operator_attrs = {}
        if hasattr(node, 'attributes'):
            operator_attrs = node.attributes
        
        # Apply optimization
        result = optimizer.optimize(node.op_type, operator_attrs, hw_metrics)
        
        # Update node with scheduling information
        node.start_cycle = 0  # Will be set by system scheduler
        node.duration = result["duration"]
        node.optimization_info = result
    
    return op_scheduled_ir
