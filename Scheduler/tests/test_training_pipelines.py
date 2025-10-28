#!/usr/bin/env python3
"""Test script to verify Scheduler integration with GSArch, GBU, and Instant3D training pipelines."""

import sys
import os
from pathlib import Path

# Add paths for imports (adjusted for tests subdirectory)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Operators"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gsarch_pipeline():
    """Test GSArch pipeline with Scheduler."""
    print("\n=== Testing GSArch Pipeline ===")
    
    try:
        # Import pipeline builder
        from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
        from utils.operator_graph import OperatorGraph
        
        # Build pipeline
        graph = build_gsarch_training_pipeline((4, 64))
        print(f"Built GSArch pipeline with {len(list(graph.nodes))} nodes")
        
        # Check for backward nodes
        backward_nodes = [n for n in graph.nodes if hasattr(n, 'is_backward') and n.is_backward]
        print(f"Found {len(backward_nodes)} backward nodes")
        
        # Import Scheduler components
        from mapping import MappingEngine
        from IR import OperatorGraph as SchedulerGraph, OperatorNode
        
        # Convert to Scheduler IR format
        scheduler_graph = SchedulerGraph()
        for node in graph.nodes:
            op_type = node.get_label() if hasattr(node, 'get_label') else node.get_op_type()
            scheduler_node = OperatorNode(
                id=str(id(node)),
                op_type=op_type,
                inputs=[],
                outputs=[]
            )
            scheduler_graph.nodes[scheduler_node.id] = scheduler_node
        
        # Test mapping with GSArch config
        config_path = Path(__file__).parent.parent.parent / "Hardware/examples/hardware_configs/gsarch_config.json"
        if config_path.exists():
            print(f"Loading hardware config from {config_path}")
            mapper = MappingEngine.from_json(str(config_path))
            mapped_ir = mapper.run(scheduler_graph)
            print(f"Mapped {len(mapped_ir.nodes)} operators to hardware")
            
            # Check specific GSArch operators
            gsarch_ops = ["TILEMERGING", "GRADIENTCOMPUTE", "GRADIENTPRUNING", "REARRANGEMENT"]
            for op in gsarch_ops:
                found = any(op in node.op_node.op_type.upper() for node in mapped_ir.nodes.values())
                print(f"  {op}: {'Found' if found else 'Not found'}")
        else:
            print(f"Config file not found: {config_path}")
        
        print("GSArch test: PASSED")
        return True
        
    except Exception as e:
        print(f"GSArch test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gbu_pipeline():
    """Test GBU pipeline with Scheduler."""
    print("\n=== Testing GBU Pipeline ===")
    
    try:
        # Import pipeline builder
        from pipelines.gbu_pipeline import build_gbu_pipeline
        
        # Build pipeline
        graph = build_gbu_pipeline((4, 64))
        print(f"Built GBU pipeline with {len(list(graph.nodes))} nodes")
        
        # Check for backward nodes
        backward_nodes = [n for n in graph.nodes if hasattr(n, 'is_backward') and n.is_backward]
        print(f"Found {len(backward_nodes)} backward nodes")
        
        # Import Scheduler components
        from mapping import MappingEngine
        from IR import OperatorGraph as SchedulerGraph, OperatorNode
        
        # Convert to Scheduler IR format
        scheduler_graph = SchedulerGraph()
        for node in graph.nodes:
            op_type = node.get_label() if hasattr(node, 'get_label') else node.get_op_type()
            scheduler_node = OperatorNode(
                id=str(id(node)),
                op_type=op_type,
                inputs=[],
                outputs=[]
            )
            scheduler_graph.nodes[scheduler_node.id] = scheduler_node
        
        # Test mapping with GBU config
        config_path = Path(__file__).parent.parent.parent / "Hardware/examples/hardware_configs/gbu_config.json"
        if config_path.exists():
            print(f"Loading hardware config from {config_path}")
            mapper = MappingEngine.from_json(str(config_path))
            mapped_ir = mapper.run(scheduler_graph)
            print(f"Mapped {len(mapped_ir.nodes)} operators to hardware")
            
            # Check specific GBU operators
            gbu_ops = ["ROWPROCESSING", "ROWGENERATION", "DECOMPBINNING"]
            for op in gbu_ops:
                found = any(op in node.op_node.op_type.upper() for node in mapped_ir.nodes.values())
                print(f"  {op}: {'Found' if found else 'Not found'}")
        else:
            print(f"Config file not found: {config_path}")
        
        print("GBU test: PASSED")
        return True
        
    except Exception as e:
        print(f"GBU test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instant3d_pipeline():
    """Test Instant3D pipeline with Scheduler."""
    print("\n=== Testing Instant3D Pipeline ===")
    
    try:
        # Import pipeline builder
        from pipelines.instant3d_pipeline import build_instant3d_training_pipeline
        
        # Build pipeline
        graph = build_instant3d_training_pipeline((4, 64))
        print(f"Built Instant3D pipeline with {len(list(graph.nodes))} nodes")
        
        # Check for backward nodes
        backward_nodes = [n for n in graph.nodes if hasattr(n, 'is_backward') and n.is_backward]
        print(f"Found {len(backward_nodes)} backward nodes")
        
        # Import Scheduler components
        from mapping import MappingEngine
        from IR import OperatorGraph as SchedulerGraph, OperatorNode
        
        # Convert to Scheduler IR format
        scheduler_graph = SchedulerGraph()
        for node in graph.nodes:
            op_type = node.get_label() if hasattr(node, 'get_label') else node.get_op_type()
            scheduler_node = OperatorNode(
                id=str(id(node)),
                op_type=op_type,
                inputs=[],
                outputs=[]
            )
            scheduler_graph.nodes[scheduler_node.id] = scheduler_node
        
        # Test mapping with Instant3D config
        config_path = Path(__file__).parent.parent.parent / "Hardware/examples/hardware_configs/instant3d_config.json"
        if config_path.exists():
            print(f"Loading hardware config from {config_path}")
            mapper = MappingEngine.from_json(str(config_path))
            mapped_ir = mapper.run(scheduler_graph)
            print(f"Mapped {len(mapped_ir.nodes)} operators to hardware")
            
            # Check specific Instant3D operators
            instant3d_ops = ["FRM", "BUM", "HASHENCODING"]
            for op in instant3d_ops:
                found = any(op in node.op_node.op_type.upper() for node in mapped_ir.nodes.values())
                print(f"  {op}: {'Found' if found else 'Not found'}")
        else:
            print(f"Config file not found: {config_path}")
        
        print("Instant3D test: PASSED")
        return True
        
    except Exception as e:
        print(f"Instant3D test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_operator_mapping():
    """Test that backward operators are properly mapped."""
    print("\n=== Testing Backward Operator Mapping ===")
    
    try:
        from mapping import MappingEngine
        from mapping.hw_config import HWConfig, HWUnit
        from IR import OperatorGraph, OperatorNode
        
        # Create a simple test graph with backward operators
        graph = OperatorGraph()
        
        # Add forward and backward operators
        operators = [
            ("mlp_forward", "MLP"),
            ("mlp_backward", "MLP (B)"),
            ("hash_forward", "HashEncoding"),
            ("hash_backward", "HashEncoding (B)"),
            ("blend_forward", "GaussianAlphaBlend"),
            ("blend_backward", "GaussianAlphaBlend (B)"),
        ]
        
        for op_id, op_type in operators:
            node = OperatorNode(id=op_id, op_type=op_type, inputs=[], outputs=[])
            graph.nodes[op_id] = node
        
        # Create a simple hardware config
        hw_config = HWConfig(units=[
            HWUnit(id="mlp_unit", type="MLP"),
            HWUnit(id="hash_unit", type="HASH_ENCODE"),
            HWUnit(id="blend_unit", type="BLENDING"),
            HWUnit(id="compute_unit", type="FIELD_COMPUTATION"),
        ])
        
        # Test mapping
        mapper = MappingEngine(hw_config=hw_config)
        mapped_ir = mapper.run(graph)
        
        print(f"Mapped {len(mapped_ir.nodes)} operators")
        for node_id, mapped_node in mapped_ir.nodes.items():
            print(f"  {mapped_node.op_node.op_type} -> {mapped_node.hw_unit}")
        
        # Verify backward operators are mapped
        backward_mapped = sum(1 for n in mapped_ir.nodes.values() if "(B)" in n.op_node.op_type)
        print(f"Successfully mapped {backward_mapped} backward operators")
        
        print("Backward operator mapping test: PASSED")
        return True
        
    except Exception as e:
        print(f"Backward operator mapping test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Scheduler Integration with Training Pipelines")
    print("=" * 60)
    
    results = []
    
    # Test each pipeline
    results.append(("GSArch", test_gsarch_pipeline()))
    results.append(("GBU", test_gbu_pipeline()))
    results.append(("Instant3D", test_instant3d_pipeline()))
    results.append(("Backward Mapping", test_backward_operator_mapping()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
