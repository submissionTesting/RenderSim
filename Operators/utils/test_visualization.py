#!/usr/bin/env python3
"""Test pipeline visualization capabilities.

This script generates visual representations of pipeline operator graphs
for debugging and documentation purposes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, Optional
import tempfile


def visualize_pipeline(pipeline_name: str, build_func, dim: Tuple[int, int], 
                       output_dir: Optional[str] = None) -> bool:
    """Generate visualization for a single pipeline."""
    try:
        print(f"\nVisualizing {pipeline_name}...")
        graph = build_func(dim)
        
        # Count nodes
        nodes = list(graph.nodes)
        forward = [n for n in nodes if not getattr(n, 'is_backward', False)]
        backward = [n for n in nodes if getattr(n, 'is_backward', False)]
        
        print(f"  - Total nodes: {len(nodes)}")
        print(f"  - Forward: {len(forward)}, Backward: {len(backward)}")
        
        # Try to visualize
        if output_dir:
            output_path = os.path.join(output_dir, f"{pipeline_name.lower()}_pipeline.png")
        else:
            output_path = f"{pipeline_name.lower()}_pipeline.png"
        
        try:
            graph.visualize(output_path)
            print(f"  SUCCESS: Visualization saved to: {output_path}")
            return True
        except Exception as e:
            print(f"  WARNING: Visualization failed (graphviz may not be installed): {e}")
            return False
            
    except Exception as e:
        print(f"  FAILED: Could not build {pipeline_name}: {e}")
        return False


def main():
    """Main visualization test."""
    print("PIPELINE VISUALIZATION TEST")
    print("="*60)
    
    dim = (4, 64)
    
    # Create output directory
    output_dir = "pipeline_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Test existing pipelines
    print("\n--- Existing Pipelines ---")
    existing_pipelines = [
        ("ICARUS", "pipelines.icarus_pipeline", "build_icarus_pipeline"),
        ("NeuRex", "pipelines.neurex_pipeline", "build_neurex_pipeline"),
        ("CICERO", "pipelines.cicero_pipeline", "build_cicero_pipeline"),
        ("GSCore", "pipelines.gscore_pipeline", "build_gscore_pipeline"),
        ("SRender", "pipelines.srender_pipeline", "build_srender_pipeline"),
    ]
    
    existing_success = 0
    for name, module_path, func_name in existing_pipelines:
        try:
            module = __import__(module_path, fromlist=[func_name])
            build_func = getattr(module, func_name)
            if visualize_pipeline(name, build_func, dim, output_dir):
                existing_success += 1
        except ImportError as e:
            print(f"\nERROR: Could not import {name}: {e}")
    
    # Test training pipelines
    print("\n--- Training Pipelines ---")
    training_pipelines = [
        ("GSArch", "pipelines.gsarch_pipeline", "build_gsarch_training_pipeline"),
        ("GBU", "pipelines.gbu_pipeline", "build_gbu_pipeline"),
        ("Instant3D", "pipelines.instant3d_pipeline", "build_instant3d_training_pipeline"),
    ]
    
    training_success = 0
    for name, module_path, func_name in training_pipelines:
        try:
            module = __import__(module_path, fromlist=[func_name])
            build_func = getattr(module, func_name)
            if visualize_pipeline(name, build_func, dim, output_dir):
                training_success += 1
        except ImportError as e:
            print(f"\nERROR: Could not import {name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Existing pipelines visualized: {existing_success}/{len(existing_pipelines)}")
    print(f"Training pipelines visualized: {training_success}/{len(training_pipelines)}")
    
    if existing_success + training_success > 0:
        print(f"\nSUCCESS: Visualizations saved to: {output_dir}/")
        print("\nNote: If visualization failed, install graphviz:")
        print("  Ubuntu/Debian: sudo apt-get install graphviz")
        print("  MacOS: brew install graphviz")
        print("  Python package: pip install graphviz")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
