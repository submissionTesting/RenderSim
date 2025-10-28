#!/usr/bin/env python3
"""Direct validation of test infrastructure"""

print("Starting validation...")

# Test 1: Check basic imports
try:
    import sys
    import os
    print("1. Basic imports: OK")
except Exception as e:
    print(f"1. Basic imports: FAILED - {e}")

# Test 2: Check Scheduler modules
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mapping import fallback_mappings
    print("2. Scheduler mapping: OK")
    print(f"   Found {len(fallback_mappings)} fallback mappings")
except Exception as e:
    print(f"2. Scheduler mapping: FAILED - {e}")

# Test 3: Check optimization library
try:
    from op_sched.optimization_library import OptimizationLibrary
    lib = OptimizationLibrary()
    strategies = lib.get_all_strategies()
    print(f"3. Optimization library: OK")
    print(f"   Loaded {len(strategies)} strategies")
    
    # Check for training strategies
    training = ["gradient_pruning", "tile_merging", "row_processing", "frm_coalescing", "bum_merging"]
    found = [s for s in training if any(st.name == s for st in strategies)]
    print(f"   Training strategies found: {found}")
except Exception as e:
    print(f"3. Optimization library: FAILED - {e}")

# Test 4: Check equation-based optimizer
try:
    from op_sched.equation_based_optimizer import EquationBasedOptimizer, OperatorMetrics
    metrics = OperatorMetrics(n_op=1000, v_off=4096, theta_hw=10.0, b_hw=64.0)
    print("4. Equation-based optimizer: OK")
    print(f"   Test metrics created: n_op={metrics.n_op}, theta_hw={metrics.theta_hw}")
except Exception as e:
    print(f"4. Equation-based optimizer: FAILED - {e}")

# Test 5: Check if Operators module is accessible
try:
    operators_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Operators")
    if os.path.exists(operators_path):
        sys.path.insert(0, operators_path)
        from utils.operator_graph import OperatorGraph
        g = OperatorGraph()
        print("5. Operators module: OK")
        print(f"   Created empty graph")
    else:
        print(f"5. Operators module: Path not found - {operators_path}")
except Exception as e:
    print(f"5. Operators module: FAILED - {e}")

# Test 6: Check hardware configs
try:
    hw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "Hardware/examples/hardware_configs")
    configs = ["gsarch_config.json", "gbu_config.json", "instant3d_config.json"]
    found = []
    for c in configs:
        if os.path.exists(os.path.join(hw_path, c)):
            found.append(c)
    print("6. Hardware configs: OK")
    print(f"   Found configs: {found}")
except Exception as e:
    print(f"6. Hardware configs: FAILED - {e}")

print("\nValidation complete!")
print("\nTo run actual tests:")
print("  python3 test_mapping.py")
print("  python3 test_scheduler.py")
print("  python3 test_training_pipelines.py")
