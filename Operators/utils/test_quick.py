#!/usr/bin/env python3
"""Quick smoke test for all pipelines.

This is a minimal test to verify that all pipelines can be imported and built
without errors. Use this for rapid validation during development.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def quick_test():
    """Run quick smoke tests on all pipelines."""
    dim = (4, 64)
    results = []
    
    # Test configurations: (name, module, function)
    pipelines = [
        # Existing pipelines
        ("ICARUS", "pipelines.icarus_pipeline", "build_icarus_pipeline"),
        ("NeuRex", "pipelines.neurex_pipeline", "build_neurex_pipeline"),
        ("CICERO", "pipelines.cicero_pipeline", "build_cicero_pipeline"),
        ("GSCore", "pipelines.gscore_pipeline", "build_gscore_pipeline"),
        ("SRender", "pipelines.srender_pipeline", "build_srender_pipeline"),
        # Training pipelines
        ("GSArch", "pipelines.gsarch_pipeline", "build_gsarch_training_pipeline"),
        ("GBU", "pipelines.gbu_pipeline", "build_gbu_pipeline"),
        ("Instant3D", "pipelines.instant3d_pipeline", "build_instant3d_training_pipeline"),
    ]
    
    print("QUICK PIPELINE TEST")
    print("=" * 40)
    
    for name, module_path, func_name in pipelines:
        try:
            module = __import__(module_path, fromlist=[func_name])
            build_func = getattr(module, func_name)
            graph = build_func(dim)
            nodes = list(graph.nodes)
            backward_count = sum(1 for n in nodes if getattr(n, 'is_backward', False))
            
            status = "PASS"
            results.append(True)
            print(f"{status} {name:10} - {len(nodes):3} nodes ({backward_count} backward)")
            
        except Exception as e:
            status = "FAIL"
            results.append(False)
            print(f"{status} {name:10} - FAILED: {str(e)[:50]}")
    
    print("=" * 40)
    success = sum(results)
    total = len(results)
    
    if success == total:
        print(f"SUCCESS: All {total} pipelines passed!")
        return 0
    else:
        print(f"FAILURE: {total - success}/{total} pipelines failed")
        return 1


if __name__ == "__main__":
    sys.exit(quick_test())
