#!/usr/bin/env python3
"""Comprehensive test suite for all neural rendering pipelines in the Operators framework.

This script tests both existing pipelines (ICARUS, NeuRex, CICERO, GSCore, etc.) 
and new training-enabled pipelines (GSArch, GBU, Instant3D).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, Dict, Any
from utils.operator_graph import OperatorGraph


def test_existing_pipelines(dim: Tuple[int, int]) -> Dict[str, Any]:
    """Test existing inference-only pipelines."""
    results = {}
    
    # Test ICARUS
    try:
        from pipelines.icarus_pipeline import build_icarus_pipeline
        graph = build_icarus_pipeline(dim)
        nodes = list(graph.nodes)
        results['ICARUS'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'has_backward': any(getattr(n, 'is_backward', False) for n in nodes),
            'key_ops': ['UniformSampler', 'RFFEncoding', 'MLP', 'RGBRenderer', 'DensityRenderer']
        }
    except Exception as e:
        results['ICARUS'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test NeuRex
    try:
        from pipelines.neurex_pipeline import build_neurex_pipeline
        graph = build_neurex_pipeline(dim)
        nodes = list(graph.nodes)
        results['NeuRex'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'has_backward': any(getattr(n, 'is_backward', False) for n in nodes),
            'key_ops': ['UniformSampler', 'HashEncoding', 'MLP', 'RGBRenderer', 'DensityRenderer']
        }
    except Exception as e:
        results['NeuRex'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test CICERO
    try:
        from pipelines.cicero_pipeline import build_cicero_pipeline
        graph = build_cicero_pipeline(dim)
        nodes = list(graph.nodes)
        results['CICERO'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'has_backward': any(getattr(n, 'is_backward', False) for n in nodes),
            'key_ops': ['UniformSampler', 'HashEncoding', 'MLP', 'RGBRenderer']
        }
    except Exception as e:
        results['CICERO'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test GSCore
    try:
        from pipelines.gscore_pipeline import build_gscore_pipeline
        graph = build_gscore_pipeline(dim)
        nodes = list(graph.nodes)
        results['GSCore'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'has_backward': any(getattr(n, 'is_backward', False) for n in nodes),
            'key_ops': ['FrustumCullingProjection', 'Sorting', 'GaussianAlphaBlend']
        }
    except Exception as e:
        results['GSCore'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test SRender
    try:
        from pipelines.srender_pipeline import build_srender_pipeline
        graph = build_srender_pipeline(dim)
        nodes = list(graph.nodes)
        results['SRender'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'has_backward': any(getattr(n, 'is_backward', False) for n in nodes),
            'key_ops': ['UniformSampler', 'HashEncoding', 'MLP', 'RGBRenderer']
        }
    except Exception as e:
        results['SRender'] = {'status': 'FAILED', 'error': str(e)}
    
    return results


def test_training_pipelines(dim: Tuple[int, int]) -> Dict[str, Any]:
    """Test new training-enabled pipelines with forward and backward passes."""
    results = {}
    
    # Test GSArch
    try:
        from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
        graph = build_gsarch_training_pipeline(dim)
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        op_types = set(n.get_op_type() for n in nodes)
        
        results['GSArch'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'forward_nodes': len(forward),
            'backward_nodes': len(backward),
            'has_backward': len(backward) > 0,
            'key_ops': ['TileMerging', 'FeatureCompute', 'GradientCompute', 'GradientPruning', 'Rearrangement'],
            'key_ops_present': [op for op in ['TileMerging', 'FeatureCompute', 'GradientCompute', 'GradientPruning', 'Rearrangement'] if op in op_types]
        }
    except Exception as e:
        results['GSArch'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test GBU
    try:
        from pipelines.gbu_pipeline import build_gbu_pipeline
        graph = build_gbu_pipeline(dim)
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        op_types = set(n.get_op_type() for n in nodes)
        
        results['GBU'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'forward_nodes': len(forward),
            'backward_nodes': len(backward),
            'has_backward': len(backward) > 0,
            'key_ops': ['RowProcessing', 'RowGeneration', 'DecompBinning', 'Sorting', 'GaussianAlphaBlend'],
            'key_ops_present': [op for op in ['RowProcessing', 'RowGeneration', 'DecompBinning', 'Sorting', 'GaussianAlphaBlend'] if op in op_types]
        }
    except Exception as e:
        results['GBU'] = {'status': 'FAILED', 'error': str(e)}
    
    # Test Instant3D
    try:
        from pipelines.instant3d_pipeline import build_instant3d_training_pipeline
        graph = build_instant3d_training_pipeline(dim)
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        op_types = set(n.get_op_type() for n in nodes)
        
        results['Instant3D'] = {
            'status': 'SUCCESS',
            'total_nodes': len(nodes),
            'forward_nodes': len(forward),
            'backward_nodes': len(backward),
            'has_backward': len(backward) > 0,
            'key_ops': ['FRM', 'BUM', 'HashEncoding', 'MLP', 'RGBRenderer'],
            'key_ops_present': [op for op in ['FRM', 'BUM', 'HashEncoding', 'MLP', 'RGBRenderer'] if op in op_types]
        }
    except Exception as e:
        results['Instant3D'] = {'status': 'FAILED', 'error': str(e)}
    
    return results


def print_results(results: Dict[str, Any], category: str):
    """Pretty print test results."""
    print(f"\n{'='*60}")
    print(f"{category} PIPELINES")
    print(f"{'='*60}")
    
    for name, info in results.items():
        print(f"\n{name}:")
        if info['status'] == 'SUCCESS':
            print(f"  Status: {info['status']}")
            print(f"  - Total nodes: {info['total_nodes']}")
            if 'forward_nodes' in info:
                print(f"  - Forward nodes: {info['forward_nodes']}")
                print(f"  - Backward nodes: {info['backward_nodes']}")
            print(f"  - Has backward support: {info['has_backward']}")
            if 'key_ops_present' in info:
                print(f"  - Key operators present: {', '.join(info['key_ops_present'])}")
        else:
            print(f"  Status: {info['status']} (FAILED)")
            print(f"  - Error: {info['error']}")


def main():
    """Main test runner."""
    print("COMPREHENSIVE PIPELINE TEST SUITE")
    print("="*60)
    
    dim = (4, 64)  # Standard test dimensions
    
    # Test existing pipelines
    existing_results = test_existing_pipelines(dim)
    print_results(existing_results, "EXISTING (INFERENCE)")
    
    # Test training pipelines
    training_results = test_training_pipelines(dim)
    print_results(training_results, "NEW (TRAINING)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_existing = len(existing_results)
    successful_existing = sum(1 for r in existing_results.values() if r['status'] == 'SUCCESS')
    
    total_training = len(training_results)
    successful_training = sum(1 for r in training_results.values() if r['status'] == 'SUCCESS')
    
    print(f"\nExisting pipelines: {successful_existing}/{total_existing} successful")
    print(f"Training pipelines: {successful_training}/{total_training} successful")
    
    if successful_existing == total_existing and successful_training == total_training:
        print("\nSUCCESS: ALL TESTS PASSED!")
        return 0
    else:
        print("\nFAILURE: SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
