#!/usr/bin/env python3
"""Focused test suite for training-enabled pipelines (GSArch, GBU, Instant3D).

This script provides detailed verification of backward pass implementation,
operator connectivity, and training-specific features.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, List, Optional
from utils.operator_graph import OperatorGraph


def verify_backward_chain(graph: OperatorGraph, pipeline_name: str) -> bool:
    """Verify that the backward chain is properly connected after forward blending/rendering."""
    nodes = list(graph.nodes)
    
    # Find forward blending/rendering nodes
    blend_nodes = [n for n in nodes 
                   if not getattr(n, 'is_backward', False) 
                   and ('Blend' in n.get_op_type() or 'Render' in n.get_op_type())]
    
    if not blend_nodes:
        print(f"  FAILED: {pipeline_name}: No forward blending/rendering node found")
        return False
    
    # Check if any blending node has backward children
    for blend in blend_nodes:
        children = getattr(blend, 'children', [])
        backward_children = [c for c in children if getattr(c, 'is_backward', False)]
        if backward_children:
            print(f"  SUCCESS: {pipeline_name}: Backward chain connected after {blend.get_op_type()}")
            return True
    
    print(f"  FAILED: {pipeline_name}: No backward chain after blending/rendering")
    return False


def trace_backward_path(node, visited=None) -> List[str]:
    """Trace the backward path from a given node."""
    if visited is None:
        visited = set()
    
    if id(node) in visited:
        return []
    
    visited.add(id(node))
    path = [node.get_label() if hasattr(node, 'get_label') else node.get_op_type()]
    
    for child in getattr(node, 'children', []):
        if getattr(child, 'is_backward', False):
            child_path = trace_backward_path(child, visited)
            if child_path:
                path.extend(child_path)
    
    return path


def test_gsarch_pipeline(dim: Tuple[int, int]) -> bool:
    """Test GSArch training pipeline with gradient computation and pruning."""
    print("\n" + "="*50)
    print("GSArch Training Pipeline")
    print("="*50)
    
    try:
        from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
        graph = build_gsarch_training_pipeline(dim)
        
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        
        print(f"Structure:")
        print(f"  - Total nodes: {len(nodes)}")
        print(f"  - Forward nodes: {len(forward)}")
        print(f"  - Backward nodes: {len(backward)}")
        
        # Check for GSArch-specific operators
        op_types = set(n.get_op_type() for n in nodes)
        required_ops = {
            'FrustumCullingProjection': 'Primitive sampling',
            'Sorting': 'Depth ordering',
            'FeatureCompute': 'Feature computation',
            'TileMerging': 'Tile-based merging',
            'GaussianAlphaBlend': 'Alpha blending',
            'GradientCompute': 'Gradient computation (backward)',
            'GradientPruning': 'Gradient pruning (backward)',
            'Rearrangement': 'Memory rearrangement (backward)'
        }
        
        print("\nKey operators:")
        all_present = True
        for op, desc in required_ops.items():
            if op in op_types:
                print(f"  [OK] {op}: {desc}")
            else:
                print(f"  [MISSING] {op}: {desc}")
                all_present = False
        
        # Verify backward chain
        chain_ok = verify_backward_chain(graph, "GSArch")
        
        # Trace backward path
        blend_node = next((n for n in forward if 'Blend' in n.get_op_type()), None)
        if blend_node:
            backward_path = trace_backward_path(blend_node)
            if len(backward_path) > 1:
                print(f"\nBackward path from blending:")
                for i, op in enumerate(backward_path[:6], 1):  # Show first 6 ops
                    print(f"  {i}. {op}")
                if len(backward_path) > 6:
                    print(f"  ... ({len(backward_path)-6} more operations)")
        
        return all_present and chain_ok and len(backward) > 0
        
    except Exception as e:
        print(f"ERROR: Failed to build GSArch pipeline: {e}")
        return False


def test_gbu_pipeline(dim: Tuple[int, int]) -> bool:
    """Test GBU pipeline with row-based processing."""
    print("\n" + "="*50)
    print("GBU (Gaussian Blending Unit) Pipeline")
    print("="*50)
    
    try:
        from pipelines.gbu_pipeline import build_gbu_pipeline
        graph = build_gbu_pipeline(dim)
        
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        
        print(f"Structure:")
        print(f"  - Total nodes: {len(nodes)}")
        print(f"  - Forward nodes: {len(forward)}")
        print(f"  - Backward nodes: {len(backward)}")
        
        # Check for GBU-specific operators
        op_types = set(n.get_op_type() for n in nodes)
        required_ops = {
            'FrustumCullingProjection': 'Primitive sampling',
            'RowProcessing': 'Row-based processing',
            'RowGeneration': 'Row generation engine',
            'DecompBinning': 'Decomposition and binning',
            'Sorting': 'Depth sorting',
            'GaussianAlphaBlend': 'Alpha blending'
        }
        
        print("\nKey operators:")
        all_present = True
        for op, desc in required_ops.items():
            if op in op_types:
                print(f"  [OK] {op}: {desc}")
            else:
                print(f"  [MISSING] {op}: {desc}")
                all_present = False
        
        # Verify backward chain
        chain_ok = verify_backward_chain(graph, "GBU")
        
        # Check row-based processing flow
        row_ops = ['RowProcessing', 'RowGeneration', 'DecompBinning']
        row_flow_ok = all(op in op_types for op in row_ops)
        print(f"\nRow-based processing flow: {'Complete' if row_flow_ok else 'Incomplete'}")
        
        return all_present and chain_ok and len(backward) > 0
        
    except Exception as e:
        print(f"ERROR: Failed to build GBU pipeline: {e}")
        return False


def test_instant3d_pipeline(dim: Tuple[int, int]) -> bool:
    """Test Instant3D pipeline with asymmetric FRM/BUM architecture."""
    print("\n" + "="*50)
    print("Instant3D Training Pipeline")
    print("="*50)
    
    try:
        from pipelines.instant3d_pipeline import build_instant3d_training_pipeline
        graph = build_instant3d_training_pipeline(dim)
        
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        
        print(f"Structure:")
        print(f"  - Total nodes: {len(nodes)}")
        print(f"  - Forward nodes: {len(forward)}")
        print(f"  - Backward nodes: {len(backward)}")
        
        # Check for Instant3D-specific operators
        op_types = set(n.get_op_type() for n in nodes)
        required_ops = {
            'UniformSampler': 'Ray sampling',
            'FRM': 'Feed-Forward Read Mapper',
            'HashEncoding': 'Hash grid encoding',
            'MLP': 'Neural field computation',
            'RGBRenderer': 'Volume rendering',
            'BUM': 'Back-Propagation Update Merger'
        }
        
        print("\nKey operators:")
        all_present = True
        for op, desc in required_ops.items():
            if op in op_types:
                print(f"  [OK] {op}: {desc}")
            else:
                print(f"  [MISSING] {op}: {desc}")
                all_present = False
        
        # Verify backward chain
        chain_ok = verify_backward_chain(graph, "Instant3D")
        
        # Check asymmetric architecture
        has_frm = 'FRM' in op_types
        has_bum = 'BUM' in op_types
        asymmetric_ok = has_frm and has_bum
        
        print(f"\nAsymmetric architecture:")
        print(f"  {'[OK]' if has_frm else '[MISSING]'} FRM in forward pass")
        print(f"  {'[OK]' if has_bum else '[MISSING]'} BUM in backward pass")
        print(f"  Overall: {'Asymmetric design implemented' if asymmetric_ok else 'Incomplete'}")
        
        return all_present and chain_ok and asymmetric_ok and len(backward) > 0
        
    except Exception as e:
        print(f"ERROR: Failed to build Instant3D pipeline: {e}")
        return False


def test_backward_operator_support():
    """Test that base operator classes support backward parameter."""
    print("\n" + "="*50)
    print("Backward Operator Support")
    print("="*50)
    
    from operators.base_operator import Operator
    from operators.computation_operator import MLPOperator
    from operators.blending_operator import GaussianAlphaBlendOperator
    from operators.encoding_operator import HashEncodingOperator
    
    test_cases = [
        ("Base Operator", Operator),
        ("MLP Operator", MLPOperator),
        ("Gaussian Alpha Blend", GaussianAlphaBlendOperator),
        ("Hash Encoding", HashEncodingOperator)
    ]
    
    all_ok = True
    for name, OpClass in test_cases:
        try:
            if name == "MLP Operator":
                op = OpClass((4, 64), in_dim=32, num_layers=2, layer_width=64, out_dim=4, backward=True)
            elif name == "Hash Encoding":
                op = OpClass((4, 64), num_levels=16, backward=True)
            else:
                op = OpClass((4, 64), backward=True)
            
            has_backward = hasattr(op, 'is_backward') and op.is_backward
            has_label = hasattr(op, 'get_label')
            
            if has_backward and has_label:
                label = op.get_label()
                expected_suffix = " (B)" in label
                if expected_suffix:
                    print(f"  [OK] {name}: backward={has_backward}, label='{label}'")
                else:
                    print(f"  [WARNING] {name}: backward={has_backward}, but label missing (B) suffix")
            else:
                print(f"  [FAILED] {name}: backward support incomplete")
                all_ok = False
                
        except Exception as e:
            print(f"  [ERROR] {name}: Failed to instantiate with backward=True: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Main test runner for training pipelines."""
    print("TRAINING PIPELINE TEST SUITE")
    print("="*60)
    print("Testing forward/backward pass implementation for:")
    print("- GSArch: 3DGS with gradient pruning")
    print("- GBU: Row-based Gaussian processing")
    print("- Instant3D: Asymmetric FRM/BUM architecture")
    
    dim = (4, 64)
    
    # Test backward support in base classes
    backward_ok = test_backward_operator_support()
    
    # Test each training pipeline
    gsarch_ok = test_gsarch_pipeline(dim)
    gbu_ok = test_gbu_pipeline(dim)
    instant3d_ok = test_instant3d_pipeline(dim)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = {
        "Backward operator support": backward_ok,
        "GSArch pipeline": gsarch_ok,
        "GBU pipeline": gbu_ok,
        "Instant3D pipeline": instant3d_ok
    }
    
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
