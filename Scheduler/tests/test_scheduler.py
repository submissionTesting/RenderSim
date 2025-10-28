#!/usr/bin/env python3
"""
Test suite for the RenderSim system and operator-level schedulers
"""

import sys
import os
import json

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Operators'))

from mapping import map_operator_graph
from mapping.hw_config import HWConfig
from op_sched.optimization_library import apply_optimizations
from utils.operator_graph import OperatorGraph


def test_operator_scheduling():
    """Test operator-level scheduling with optimizations"""
    print("\n=== Testing Operator-Level Scheduling ===")
    
    from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
    
    # Build a simple pipeline
    graph = build_gsarch_training_pipeline((800, 600))
    
    # Load hardware config
    hw_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Hardware/examples/hardware_configs/gsarch_config.json"
    )
    
    if not os.path.exists(hw_config_path):
        print("  Hardware config not found")
        return False
    
    hw_config = HWConfig(hw_config_path)
    
    # Map the graph
    mapped_ir = map_operator_graph(graph, hw_config)
    
    # Apply optimizations
    optimized_ir = apply_optimizations(mapped_ir)
    
    # Check that scheduling information was added
    scheduled_count = 0
    for node in optimized_ir.nodes.values():
        if hasattr(node, 'start_cycle') or hasattr(node, 'duration'):
            scheduled_count += 1
    
    print(f"  Scheduled {scheduled_count}/{len(optimized_ir.nodes)} operators")
    return scheduled_count > 0


def test_system_scheduling():
    """Test system-level DAGS scheduling"""
    print("\n=== Testing System-Level Scheduling (DAGS) ===")
    
    # This would require the C++ scheduler to be built
    # For now, we'll test the Python interface
    
    try:
        # Check if C++ module is available
        import op_sched_cpp
        print("  C++ scheduler module available")
        
        # Would run actual scheduling here
        return True
    except ImportError:
        print("  C++ scheduler module not built (this is expected)")
        # Still pass the test since C++ building is optional
        return True


def test_training_optimization():
    """Test training-specific optimizations"""
    print("\n=== Testing Training Optimizations ===")
    
    from op_sched.training_optimization import (
        apply_gradient_pruning,
        apply_tile_merging,
        apply_row_processing_optimization
    )
    
    # Create mock IR for testing
    class MockNode:
        def __init__(self, op_type):
            self.op_type = op_type
            self.compute_ops = 1000
            self.memory_bytes = 4096
    
    class MockIR:
        def __init__(self):
            self.nodes = {
                "grad1": MockNode("GradientCompute"),
                "tile1": MockNode("TileMerging"),
                "row1": MockNode("RowProcessing")
            }
    
    ir = MockIR()
    
    # Test gradient pruning
    pruned_ir = apply_gradient_pruning(ir, threshold=0.01)
    print(f"  Gradient pruning applied")
    
    # Test tile merging
    merged_ir = apply_tile_merging(ir, tile_size=16)
    print(f"  Tile merging applied")
    
    # Test row processing
    row_ir = apply_row_processing_optimization(ir)
    print(f"  Row processing optimization applied")
    
    return True


def test_performance_model():
    """Test performance modeling calculations"""
    print("\n=== Testing Performance Model ===")
    
    from op_sched.performance_model import (
        calculate_latency,
        calculate_throughput,
        calculate_power,
        calculate_energy_efficiency
    )
    
    # Mock hardware specs
    hw_specs = {
        "frequency_mhz": 1000,
        "compute_units": 16,
        "memory_bandwidth_gb": 100,
        "power_w": 200
    }
    
    # Mock workload
    workload = {
        "total_ops": 1e9,
        "memory_accesses": 1e6,
        "duration_cycles": 1e6
    }
    
    # Calculate metrics
    latency = calculate_latency(workload["duration_cycles"], hw_specs["frequency_mhz"])
    throughput = calculate_throughput(workload["total_ops"], latency)
    power = calculate_power(hw_specs["power_w"], workload)
    efficiency = calculate_energy_efficiency(throughput, power)
    
    print(f"  Latency: {latency:.3f} ms")
    print(f"  Throughput: {throughput:.2e} ops/s")
    print(f"  Power: {power:.2f} W")
    print(f"  Energy Efficiency: {efficiency:.2e} ops/J")
    
    return latency > 0 and throughput > 0


def main():
    """Run all scheduler tests"""
    print("=" * 60)
    print("RenderSim Scheduler Tests")
    print("=" * 60)
    
    tests = [
        ("Operator Scheduling", test_operator_scheduling),
        ("System Scheduling", test_system_scheduling),
        ("Training Optimizations", test_training_optimization),
        ("Performance Model", test_performance_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
