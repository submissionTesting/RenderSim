#!/usr/bin/env python3
"""Test suite specifically for the three new training pipelines.

This is the original test for GSArch, GBU, and Instant3D pipelines,
kept for backwards compatibility and focused testing.
"""

import sys
import os
from typing import Tuple

# Ensure local Operators dir is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_all(dim: Tuple[int, int]):
    """Build all three new training pipelines."""
    from pipelines.gbu_pipeline import build_gbu_pipeline
    from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
    from pipelines.instant3d_pipeline import build_instant3d_training_pipeline

    return {
        "GBU": build_gbu_pipeline(dim),
        "GSArch": build_gsarch_training_pipeline(dim),
        "Instant3D": build_instant3d_training_pipeline(dim),
    }


def summarize(graph, name: str):
    """Summarize and verify a pipeline graph."""
    nodes = list(graph.nodes)
    total = len(nodes)
    bwd = sum(1 for n in nodes if getattr(n, "is_backward", False))
    
    # Count operator types
    kinds = {}
    for n in nodes:
        kinds[n.get_op_type()] = kinds.get(n.get_op_type(), 0) + 1
    
    # Check for backward chain after blending
    fwd_blend = [n for n in nodes 
                 if n.get_op_type() in ("GaussianAlphaBlend", "RGBRenderer") 
                 and not getattr(n, "is_backward", False)]
    bwd_after_blend = any(
        any(getattr(c, "is_backward", False) for c in getattr(n, "children", []))
        for n in fwd_blend
    )
    
    print(f"[{name}] nodes={total} backward={bwd} has_bwd_after_blend={bwd_after_blend}")
    
    # Assertions
    assert bwd > 0, f"{name}: expected at least one backward node"
    assert bwd_after_blend, f"{name}: expected backward chain after blending"
    
    return True


def main():
    """Main test runner."""
    print("Testing new training pipelines...")
    print("-" * 40)
    
    graphs = build_all((4, 64))
    all_passed = True
    
    for name, graph in graphs.items():
        try:
            summarize(graph, name)
        except AssertionError as e:
            print(f"FAILED: {name} - {e}")
            all_passed = False
        except Exception as e:
            print(f"ERROR: {name} - {e}")
            all_passed = False
    
    if all_passed:
        print("-" * 40)
        print("SUCCESS: All new training pipelines OK")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())


