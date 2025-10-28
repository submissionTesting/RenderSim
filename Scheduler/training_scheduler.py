#!/usr/bin/env python3
"""Integrated training-aware scheduler for GSArch, GBU, and Instant3D pipelines."""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "Operators"))
sys.path.insert(0, str(Path(__file__).parent))

from mapping import MappingEngine
from IR import OperatorGraph, OperatorNode, MappedIR, OperatorScheduledIR
from op_sched.training_optimization import TrainingOptimizationLibrary
from op_sched.performance_model import TrainingPerformanceModel, PerformanceMetrics


@dataclass
class TrainingScheduleResult:
    """Result of training-aware scheduling."""
    pipeline_name: str
    total_cycles: int
    forward_cycles: int
    backward_cycles: int
    hw_utilization: Dict[str, float]
    optimizations_applied: List[str]
    performance_metrics: PerformanceMetrics
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "total_cycles": self.total_cycles,
            "forward_cycles": self.forward_cycles,
            "backward_cycles": self.backward_cycles,
            "hw_utilization": self.hw_utilization,
            "optimizations_applied": self.optimizations_applied,
            "performance_metrics": {
                "latency_cycles": self.performance_metrics.latency_cycles,
                "throughput_ops_per_cycle": self.performance_metrics.throughput_ops_per_cycle,
                "memory_bandwidth_gb_s": self.performance_metrics.memory_bandwidth_gb_s,
                "power_watts": self.performance_metrics.power_watts,
                "energy_per_op_joules": self.performance_metrics.energy_per_op_joules,
                "fps": self.performance_metrics.fps,
                "efficiency_gflops_per_watt": self.performance_metrics.efficiency
            }
        }


class TrainingAwareScheduler:
    """Main scheduler for training pipelines with optimization support."""
    
    def __init__(self, hw_config_path: str):
        """
        Initialize the training-aware scheduler.
        
        Args:
            hw_config_path: Path to hardware configuration JSON
        """
        self.mapper = MappingEngine.from_json(hw_config_path)
        self.opt_library = TrainingOptimizationLibrary()
        self.perf_model = TrainingPerformanceModel()
        
    def schedule_pipeline(self, pipeline_name: str, graph: OperatorGraph, 
                         dim: Tuple[int, int]) -> TrainingScheduleResult:
        """
        Schedule a training pipeline with optimizations.
        
        Args:
            pipeline_name: Name of the pipeline (GSArch, GBU, or Instant3D)
            graph: Operator graph from the pipeline
            dim: (batch_size, num_elements) dimensions
            
        Returns:
            TrainingScheduleResult with scheduling details
        """
        # Step 1: Map operators to hardware
        mapped_ir = self.mapper.run(graph)
        
        # Step 2: Apply optimizations and operator-level scheduling
        scheduled_ir, optimizations = self._apply_optimizations(mapped_ir, pipeline_name)
        
        # Step 3: System-level scheduling with training awareness
        schedule = self._system_schedule(scheduled_ir, pipeline_name)
        
        # Step 4: Performance modeling
        perf_metrics = self.perf_model.model_pipeline(pipeline_name, dim)
        
        # Step 5: Compute utilization
        hw_utilization = self._compute_hw_utilization(schedule, scheduled_ir)
        
        return TrainingScheduleResult(
            pipeline_name=pipeline_name,
            total_cycles=schedule["total_cycles"],
            forward_cycles=schedule["forward_cycles"],
            backward_cycles=schedule["backward_cycles"],
            hw_utilization=hw_utilization,
            optimizations_applied=optimizations,
            performance_metrics=perf_metrics
        )
    
    def _apply_optimizations(self, mapped_ir: MappedIR, 
                           pipeline_name: str) -> Tuple[OperatorScheduledIR, List[str]]:
        """Apply training-specific optimizations to mapped operators."""
        scheduled_ir = OperatorScheduledIR()
        scheduled_ir.edges = mapped_ir.edges
        optimizations_applied = []
        
        for node_id, mapped_node in mapped_ir.nodes.items():
            op_type = mapped_node.op_node.op_type
            
            # Apply optimization
            opt_result = self.opt_library.apply_optimization(op_type)
            
            if opt_result.applied:
                optimizations_applied.append(f"{op_type}: {opt_result.optimization_type}")
                
                # Compute optimized duration
                base_duration = self._compute_base_duration(mapped_node)
                optimized_duration = int(base_duration / opt_result.compute_speedup)
            else:
                optimized_duration = self._compute_base_duration(mapped_node)
            
            # Create scheduled node
            from IR import OperatorScheduledIRNode
            scheduled_node = OperatorScheduledIRNode(
                mapped_node=mapped_node,
                start_cycle=0,  # Will be set by system scheduler
                duration=optimized_duration,
                resources={"memory_reduction": opt_result.memory_reduction if opt_result.applied else 1.0}
            )
            
            scheduled_ir.nodes[node_id] = scheduled_node
        
        return scheduled_ir, optimizations_applied
    
    def _compute_base_duration(self, mapped_node) -> int:
        """Compute base duration for an operator without optimizations."""
        # Simple model: duration based on output size
        if mapped_node.op_node.outputs:
            output_size = 1
            for shape in mapped_node.op_node.outputs:
                if hasattr(shape, 'shape'):
                    for dim in shape.shape:
                        output_size *= dim
                elif isinstance(shape, (list, tuple)):
                    for dim in shape:
                        output_size *= dim
            
            # Base cycles = output_size / throughput
            # Assume throughput of 256 ops/cycle for compute units
            return max(1, output_size // 256)
        
        return 100  # Default duration
    
    def _system_schedule(self, scheduled_ir: OperatorScheduledIR, 
                        pipeline_name: str) -> dict:
        """Perform system-level scheduling with training awareness."""
        schedule = {
            "total_cycles": 0,
            "forward_cycles": 0,
            "backward_cycles": 0,
            "node_schedules": {}
        }
        
        # Separate forward and backward operators
        forward_ops = []
        backward_ops = []
        
        for node_id, node in scheduled_ir.nodes.items():
            if "(B)" in node.mapped_node.op_node.op_type:
                backward_ops.append(node_id)
            else:
                forward_ops.append(node_id)
        
        # Schedule forward pass
        current_cycle = 0
        hw_available_at = {}
        
        for op_id in forward_ops:
            node = scheduled_ir.nodes[op_id]
            hw_unit = node.mapped_node.hw_unit
            
            # Find earliest available time
            start_cycle = current_cycle
            if hw_unit in hw_available_at:
                start_cycle = max(start_cycle, hw_available_at[hw_unit])
            
            # Schedule the operation
            node.start_cycle = start_cycle
            end_cycle = start_cycle + node.duration
            
            schedule["node_schedules"][op_id] = {
                "start": start_cycle,
                "end": end_cycle,
                "hw_unit": hw_unit
            }
            
            hw_available_at[hw_unit] = end_cycle
            current_cycle = max(current_cycle, end_cycle)
        
        schedule["forward_cycles"] = current_cycle
        
        # Schedule backward pass
        backward_start = current_cycle
        
        # Apply pipeline-specific scheduling strategies
        if pipeline_name == "GSArch":
            # GSArch: prioritize gradient operations
            backward_ops.sort(key=lambda x: "GRADIENT" in scheduled_ir.nodes[x].mapped_node.op_node.op_type, 
                            reverse=True)
        elif pipeline_name == "GBU":
            # GBU: maintain row-major order
            backward_ops.sort(key=lambda x: "ROW" in scheduled_ir.nodes[x].mapped_node.op_node.op_type, 
                            reverse=True)
        elif pipeline_name == "Instant3D":
            # Instant3D: BUM operations should be serialized
            backward_ops.sort(key=lambda x: "BUM" in scheduled_ir.nodes[x].mapped_node.op_node.op_type, 
                            reverse=True)
        
        for op_id in backward_ops:
            node = scheduled_ir.nodes[op_id]
            hw_unit = node.mapped_node.hw_unit
            
            # Find earliest available time
            start_cycle = current_cycle
            if hw_unit in hw_available_at:
                start_cycle = max(start_cycle, hw_available_at[hw_unit])
            
            # Schedule the operation
            node.start_cycle = start_cycle
            end_cycle = start_cycle + node.duration
            
            schedule["node_schedules"][op_id] = {
                "start": start_cycle,
                "end": end_cycle,
                "hw_unit": hw_unit
            }
            
            hw_available_at[hw_unit] = end_cycle
            current_cycle = max(current_cycle, end_cycle)
        
        schedule["backward_cycles"] = current_cycle - backward_start
        schedule["total_cycles"] = current_cycle
        
        return schedule
    
    def _compute_hw_utilization(self, schedule: dict, scheduled_ir: OperatorScheduledIR) -> Dict[str, float]:
        """Compute hardware utilization for each unit."""
        hw_active_cycles = {}
        hw_total_cycles = {}
        
        for op_id, node_sched in schedule["node_schedules"].items():
            hw_unit = node_sched["hw_unit"]
            duration = node_sched["end"] - node_sched["start"]
            
            if hw_unit not in hw_active_cycles:
                hw_active_cycles[hw_unit] = 0
                hw_total_cycles[hw_unit] = 0
            
            hw_active_cycles[hw_unit] += duration
            hw_total_cycles[hw_unit] = max(hw_total_cycles[hw_unit], node_sched["end"])
        
        utilization = {}
        for hw_unit in hw_active_cycles:
            if hw_total_cycles[hw_unit] > 0:
                utilization[hw_unit] = hw_active_cycles[hw_unit] / hw_total_cycles[hw_unit]
            else:
                utilization[hw_unit] = 0.0
        
        return utilization


def schedule_training_pipeline(pipeline_name: str, dim: Tuple[int, int] = (4, 64)) -> TrainingScheduleResult:
    """
    Schedule a specific training pipeline.
    
    Args:
        pipeline_name: Name of the pipeline (GSArch, GBU, or Instant3D)
        dim: (batch_size, num_elements) dimensions
        
    Returns:
        TrainingScheduleResult with scheduling details
    """
    # Import pipeline builders
    if pipeline_name == "GSArch":
        from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
        graph = build_gsarch_training_pipeline(dim)
        hw_config = "Hardware/examples/hardware_configs/gsarch_config.json"
    elif pipeline_name == "GBU":
        from pipelines.gbu_pipeline import build_gbu_pipeline
        graph = build_gbu_pipeline(dim)
        hw_config = "Hardware/examples/hardware_configs/gbu_config.json"
    elif pipeline_name == "Instant3D":
        from pipelines.instant3d_pipeline import build_instant3d_training_pipeline
        graph = build_instant3d_training_pipeline(dim)
        hw_config = "Hardware/examples/hardware_configs/instant3d_config.json"
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")
    
    # Get absolute path
    hw_config_path = Path(__file__).parent.parent / hw_config
    
    # Create scheduler and schedule
    scheduler = TrainingAwareScheduler(str(hw_config_path))
    result = scheduler.schedule_pipeline(pipeline_name, graph, dim)
    
    return result


def main():
    """Main function to demonstrate training-aware scheduling."""
    print("=" * 60)
    print("Training-Aware Scheduling with Optimizations")
    print("=" * 60)
    
    dim = (4, 1024)  # Batch size 4, 1024 elements
    
    for pipeline in ["GSArch", "GBU", "Instant3D"]:
        print(f"\n--- {pipeline} Pipeline ---")
        
        try:
            result = schedule_training_pipeline(pipeline, dim)
            
            print(f"Total Cycles: {result.total_cycles:,}")
            print(f"Forward Cycles: {result.forward_cycles:,}")
            print(f"Backward Cycles: {result.backward_cycles:,}")
            print(f"Forward/Backward Ratio: {result.backward_cycles/max(result.forward_cycles, 1):.2f}")
            
            print("\nHardware Utilization:")
            for hw_unit, util in result.hw_utilization.items():
                print(f"  {hw_unit}: {util:.1%}")
            
            print("\nOptimizations Applied:")
            for opt in result.optimizations_applied[:5]:  # Show first 5
                print(f"  - {opt}")
            if len(result.optimizations_applied) > 5:
                print(f"  ... and {len(result.optimizations_applied) - 5} more")
            
            print("\nPerformance Metrics:")
            print(f"  FPS: {result.performance_metrics.fps:.2f}")
            print(f"  Throughput: {result.performance_metrics.throughput_ops_per_cycle:.2f} ops/cycle")
            print(f"  Memory BW: {result.performance_metrics.memory_bandwidth_gb_s:.2f} GB/s")
            print(f"  Power: {result.performance_metrics.power_watts:.2f} W")
            print(f"  Efficiency: {result.performance_metrics.efficiency:.2f} GFLOPS/W")
            
            # Save results to file
            output_file = f"schedule_{pipeline.lower()}_result.json"
            with open(output_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to {output_file}")
            
        except Exception as e:
            print(f"Error scheduling {pipeline}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Scheduling Complete!")


if __name__ == "__main__":
    main()
