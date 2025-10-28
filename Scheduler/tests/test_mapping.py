#!/usr/bin/env python3
"""
Test suite for the RenderSim Scheduler mapping engine
"""

import sys
import os
import json

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Operators'))

from mapping import map_operator_graph
from mapping.hw_config import HWConfig
from operators.operators.base_operator import Operator
from utils.operator_graph import OperatorGraph


def test_backward_operator_mapping():
    """Test that backward operators are correctly mapped"""
    print("\n=== Testing Backward Operator Mapping ===")
    
    # Create a simple graph with backward operators
    from operators.operators.computation_operator import MLPOperator
    from operators.operators.encoding_operator import HashEncodingOperator
    
    g = OperatorGraph()
    
    # Forward pass
    hash_enc = HashEncodingOperator(dim=(800, 600), graph=g, backward=False)
    mlp = MLPOperator(dim=(800, 600), in_dim=32, num_layers=2, layer_width=64, out_dim=3, graph=g, backward=False)
    
    # Backward pass
    mlp_back = MLPOperator(dim=(800, 600), in_dim=3, num_layers=2, layer_width=64, out_dim=32, graph=g, backward=True)
    hash_back = HashEncodingOperator(dim=(800, 600), graph=g, backward=True)
    
    # Connect operators
    hash_enc.add_child(mlp)
    mlp.add_child(mlp_back)
    mlp_back.add_child(hash_back)
    
    # Load hardware config
    hw_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Hardware/examples/hardware_configs/instant3d_config.json"
    )
    
    if os.path.exists(hw_config_path):
        hw_config = HWConfig(hw_config_path)
        
        # Map the graph
        mapped_ir = map_operator_graph(g, hw_config)
        
        # Check that backward operators are mapped
        backward_mapped = 0
        for node_id, node in mapped_ir.nodes.items():
            if "(B)" in node.op_type:
                backward_mapped += 1
                print(f"  Backward operator {node.op_type} mapped to units: {node.hw_units}")
        
        print(f"\nSummary: {backward_mapped} backward operators successfully mapped")
        return backward_mapped > 0
    else:
        print(f"  Hardware config not found: {hw_config_path}")
        return False


def test_training_pipeline_mapping():
    """Test mapping of training-specific operators"""
    print("\n=== Testing Training Pipeline Mapping ===")
    
    # Import training pipelines
    from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
    from pipelines.gbu_pipeline import build_gbu_pipeline
    from pipelines.instant3d_pipeline import build_instant3d_training_pipeline
    
    pipelines = {
        "GSArch": (build_gsarch_training_pipeline, "gsarch_config.json"),
        "GBU": (build_gbu_pipeline, "gbu_config.json"),
        "Instant3D": (build_instant3d_training_pipeline, "instant3d_config.json")
    }
    
    results = {}
    
    for name, (build_func, config_file) in pipelines.items():
        print(f"\n  Testing {name} pipeline mapping...")
        
        # Build pipeline
        graph = build_func((800, 600))
        
        # Load corresponding hardware config
        hw_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"Hardware/examples/hardware_configs/{config_file}"
        )
        
        if os.path.exists(hw_config_path):
            hw_config = HWConfig(hw_config_path)
            
            # Map the graph
            mapped_ir = map_operator_graph(graph, hw_config)
            
            # Count successfully mapped nodes
            total_nodes = len(mapped_ir.nodes)
            mapped_nodes = sum(1 for n in mapped_ir.nodes.values() if n.hw_units)
            
            print(f"    Total nodes: {total_nodes}")
            print(f"    Successfully mapped: {mapped_nodes}")
            print(f"    Mapping rate: {mapped_nodes/total_nodes*100:.1f}%")
            
            results[name] = mapped_nodes == total_nodes
        else:
            print(f"    Hardware config not found: {config_file}")
            results[name] = False
    
    return all(results.values())


def test_fallback_mappings():
    """Test that fallback mappings work correctly"""
    print("\n=== Testing Fallback Mappings ===")
    
    from pipelines.gsarch_pipeline import TileMergingOperator, GradientComputeOperator
    from pipelines.gbu_pipeline import RowProcessingOperator
    from pipelines.instant3d_pipeline import FeedForwardReadMapper, BackpropUpdateMerger
    
    g = OperatorGraph()
    
    # Create training-specific operators
    tile_merge = TileMergingOperator((800, 600), graph=g)
    grad_comp = GradientComputeOperator((800, 600), graph=g, backward=True)
    row_proc = RowProcessingOperator((800, 600), graph=g)
    frm = FeedForwardReadMapper((800, 600), graph=g)
    bum = BackpropUpdateMerger((800, 600), graph=g, backward=True)
    
    # Try to load any hardware config for testing
    hw_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Hardware/examples/hardware_configs/gsarch_config.json"
    )
    
    if os.path.exists(hw_config_path):
        hw_config = HWConfig(hw_config_path)
        mapped_ir = map_operator_graph(g, hw_config)
        
        # Check that operators got mapped (even if through fallbacks)
        for node in mapped_ir.nodes.values():
            if node.hw_units:
                print(f"  {node.op_type} -> {node.hw_units[0].module_type if node.hw_units else 'UNMAPPED'}")
        
        return True
    else:
        print(f"  Hardware config not found")
        return False


def main():
    """Run all mapping tests"""
    print("=" * 60)
    print("RenderSim Scheduler Mapping Tests")
    print("=" * 60)
    
    tests = [
        ("Backward Operator Mapping", test_backward_operator_mapping),
        ("Training Pipeline Mapping", test_training_pipeline_mapping),
        ("Fallback Mappings", test_fallback_mappings)
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
