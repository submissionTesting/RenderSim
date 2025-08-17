"""Test the C++ operator scheduler and optimization library via pybind11 bindings."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_cpp_bindings_import():
    """Test that C++ bindings can be imported."""
    try:
        # Add the build directory to path for the compiled module
        import sys
        build_path = os.path.join(os.path.dirname(__file__), "..", "build", "Scheduler", "cpp")
        sys.path.insert(0, build_path)
        
        import rendersim_cpp
        print("✓ C++ bindings imported successfully")
        print(f"  Available classes: {[name for name in dir(rendersim_cpp) if not name.startswith('_')]}")
        return True
    except ImportError as e:
        print(f"✗ C++ bindings import failed: {e}")
        return False

def test_optimization_library():
    """Test the C++ optimization library."""
    try:
        import sys
        build_path = os.path.join(os.path.dirname(__file__), "..", "build", "Scheduler", "cpp")
        sys.path.insert(0, build_path)
        
        import rendersim_cpp
        
        # Create optimization library
        lib = rendersim_cpp.OptimizationLibrary()
        
        # Check that built-in strategies were loaded
        strategy_count = lib.get_strategy_count()
        assert strategy_count > 0, f"Expected strategies to be loaded, got {strategy_count}"
        
        print(f"✓ OptimizationLibrary created with {strategy_count} strategies")
        
        # Create optimizer
        optimizer = rendersim_cpp.DummyOperatorOptimizer(lib)
        
        # Test optimization
        attrs = {"encoding_type": "hash", "feature_dim": "256"}
        result = optimizer.optimize("HASH_ENCODE", attrs)
        
        assert result.duration > 0, f"Expected positive duration, got {result.duration}"
        assert result.base_duration > 0, f"Expected positive base duration, got {result.base_duration}"
        
        print(f"✓ Optimization completed: duration={result.duration}, speedup={result.speedup_factor:.2f}")
        print(f"  Applied optimizations: {result.applied_optimizations}")
        
        return True
    except Exception as e:
        print(f"✗ Optimization library test failed: {e}")
        return False

def test_operator_scheduler():
    """Test the C++ operator scheduler."""
    try:
        import sys
        build_path = os.path.join(os.path.dirname(__file__), "..", "build", "Scheduler", "cpp")
        sys.path.insert(0, build_path)
        
        import rendersim_cpp
        
        # Create components
        lib = rendersim_cpp.OptimizationLibrary()
        optimizer = rendersim_cpp.DummyOperatorOptimizer(lib)
        scheduler = rendersim_cpp.OperatorLevelScheduler(optimizer)
        
        # Create simple test data
        mapped_ir = rendersim_cpp.MappedIR()
        
        # Create test nodes
        for i, op_type in enumerate(["HASH_ENCODE", "FIELD_COMPUTATION", "VOLUME_RENDERING"]):
            # Create operator node
            inputs = [rendersim_cpp.TensorDesc()]
            inputs[0].shape = [1, 256]
            inputs[0].dtype = "float32"
            
            outputs = [rendersim_cpp.TensorDesc()]
            outputs[0].shape = [1, 512]
            outputs[0].dtype = "float32"
            
            op_node = rendersim_cpp.OperatorNode()
            op_node.id = f"n{i}"
            op_node.op_type = op_type
            op_node.inputs = inputs
            op_node.outputs = outputs
            op_node.call_count = 1
            
            # Create mapped IR node
            mapped_node = rendersim_cpp.MappedIRNode()
            mapped_node.op_node = op_node
            mapped_node.hw_unit = f"hw_unit_{i % 2}"  # Distribute across 2 hardware units
            mapped_node.attrs = {"test": "value"}
            
            mapped_ir.nodes[f"n{i}"] = mapped_node
        
        # Add edges for dependencies
        mapped_ir.edges = [("n0", "n1"), ("n1", "n2")]
        
        # Schedule the operators
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        assert len(scheduled_ir.nodes) == 3, f"Expected 3 nodes, got {len(scheduled_ir.nodes)}"
        
        # Check scheduling results
        for node_id, scheduled_node in scheduled_ir.nodes.items():
            assert scheduled_node.duration > 0, f"Node {node_id} has invalid duration"
            assert scheduled_node.start_cycle >= 0, f"Node {node_id} has invalid start cycle"
            print(f"  Node {node_id}: start={scheduled_node.start_cycle}, duration={scheduled_node.duration}")
        
        # Get and check statistics
        stats = scheduler.get_last_scheduling_stats()
        assert stats.total_operators == 3, f"Expected 3 operators in stats, got {stats.total_operators}"
        
        print(f"✓ Operator scheduling completed successfully")
        print(f"  Total operators: {stats.total_operators}")
        print(f"  Optimized operators: {stats.optimized_operators}")
        print(f"  Total speedup: {stats.total_speedup:.2f}")
        print(f"  Hardware unit usage: {dict(stats.hw_unit_usage)}")
        
        return True
    except Exception as e:
        print(f"✗ Operator scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing C++ operator scheduler components...")
    
    success = True
    success &= test_cpp_bindings_import()
    success &= test_optimization_library()
    success &= test_operator_scheduler()
    
    if success:
        print("\n🎉 All C++ scheduler tests passed!")
    else:
        print("\n❌ Some C++ scheduler tests failed!")
        sys.exit(1) 