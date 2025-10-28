#!/usr/bin/env python3
"""Minimal test that writes results to file to confirm execution"""

import sys
import os
import traceback
from datetime import datetime

# Output file for results
OUTPUT_FILE = "test_results.txt"

def write_result(msg):
    """Write result to both stdout and file"""
    print(msg)
    with open(OUTPUT_FILE, 'a') as f:
        f.write(msg + '\n')

def main():
    write_result("=" * 60)
    write_result(f"Test execution started at {datetime.now()}")
    write_result("=" * 60)
    
    # Test 1: Basic imports
    try:
        import sys
        import os
        write_result("TEST 1 PASSED: Basic imports work")
    except Exception as e:
        write_result(f"TEST 1 FAILED: {e}")
        traceback.print_exc(file=open(OUTPUT_FILE, 'a'))
    
    # Test 2: Add paths
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Operators'))
        write_result("TEST 2 PASSED: Paths added successfully")
    except Exception as e:
        write_result(f"TEST 2 FAILED: {e}")
    
    # Test 3: Import mapping
    try:
        from mapping import fallback_mappings
        write_result(f"TEST 3 PASSED: Mapping imported, found {len(fallback_mappings)} fallback mappings")
        
        # Check for training-specific mappings
        training_mappings = ["TILEMERGING", "ROWPROCESSING", "FRM", "BUM"]
        found = [m for m in training_mappings if m in fallback_mappings]
        write_result(f"  Training mappings found: {found}")
    except Exception as e:
        write_result(f"TEST 3 FAILED: {e}")
        traceback.print_exc(file=open(OUTPUT_FILE, 'a'))
    
    # Test 4: Import optimization library
    try:
        from op_sched.optimization_library import OptimizationLibrary
        lib = OptimizationLibrary()
        strategies = lib.get_all_strategies()
        write_result(f"TEST 4 PASSED: OptimizationLibrary loaded with {len(strategies)} strategies")
        
        # Check training strategies
        training_strats = ["gradient_pruning", "tile_merging", "row_processing", "frm_coalescing", "bum_merging"]
        found_strats = [s for s in training_strats if any(st.name == s for st in strategies)]
        write_result(f"  Training strategies found: {found_strats}")
    except Exception as e:
        write_result(f"TEST 4 FAILED: {e}")
        traceback.print_exc(file=open(OUTPUT_FILE, 'a'))
    
    # Test 5: Import equation optimizer
    try:
        from op_sched.equation_based_optimizer import EquationBasedOptimizer, OperatorMetrics, OptimizationFactors
        
        # Test Equation 1 calculation
        metrics = OperatorMetrics(n_op=1000, v_off=4096, theta_hw=10.0, b_hw=64.0)
        factors = OptimizationFactors(s_comp=0.8, r_bytes=0.5)
        
        optimizer = EquationBasedOptimizer(None)
        duration = optimizer.calculate_duration(metrics, factors)
        
        write_result(f"TEST 5 PASSED: Equation optimizer works, calculated duration = {duration} cycles")
        write_result(f"  Compute cycles: {(metrics.n_op / metrics.theta_hw) * factors.s_comp}")
        write_result(f"  Memory cycles: {(metrics.v_off / metrics.b_hw) * factors.r_bytes}")
    except Exception as e:
        write_result(f"TEST 5 FAILED: {e}")
        traceback.print_exc(file=open(OUTPUT_FILE, 'a'))
    
    # Test 6: Check hardware configs exist
    try:
        hw_dir = "../../../Hardware/examples/hardware_configs"
        configs = ["gsarch_config.json", "gbu_config.json", "instant3d_config.json"]
        existing = []
        for c in configs:
            path = os.path.join(hw_dir, c)
            if os.path.exists(path):
                existing.append(c)
        
        if len(existing) == len(configs):
            write_result(f"TEST 6 PASSED: All hardware configs found: {existing}")
        else:
            write_result(f"TEST 6 PARTIAL: Found {len(existing)}/{len(configs)} configs: {existing}")
    except Exception as e:
        write_result(f"TEST 6 FAILED: {e}")
    
    # Test 7: Try to import Operators module
    try:
        from utils.operator_graph import OperatorGraph
        from operators.operators.base_operator import Operator
        
        # Create a simple graph
        g = OperatorGraph()
        write_result(f"TEST 7 PASSED: Operators module imported, created graph")
    except Exception as e:
        write_result(f"TEST 7 FAILED: Could not import Operators - {e}")
    
    # Test 8: Try to import training pipelines
    try:
        from pipelines.gsarch_pipeline import build_gsarch_training_pipeline
        from pipelines.gbu_pipeline import build_gbu_pipeline
        from pipelines.instant3d_pipeline import build_instant3d_training_pipeline
        
        write_result("TEST 8 PASSED: All training pipelines can be imported")
    except Exception as e:
        write_result(f"TEST 8 FAILED: Could not import training pipelines - {e}")
    
    write_result("\n" + "=" * 60)
    write_result("Test execution completed")
    write_result("=" * 60)

if __name__ == "__main__":
    # Clear previous results
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Starting minimal test execution...\n")
    
    try:
        main()
    except Exception as e:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"\nFATAL ERROR: {e}\n")
            traceback.print_exc(file=f)
