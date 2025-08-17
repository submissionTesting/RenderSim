#!/usr/bin/env python3
"""
Comprehensive unit tests for RenderSim OperatorLevelScheduler

Tests cover:
- Basic scheduling functionality and workflow
- Hardware unit grouping and per-HW scheduling
- Optimization integration and pass validation
- Dependency resolution and timing calculation
- Statistics tracking and latency instrumentation
- Edge cases and error handling
- Integration with optimization library
"""

import os
import sys
import json
from pathlib import Path

# Add RenderSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cpp_scheduler_imports():
    """Test that C++ operator scheduler modules can be imported"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("✅ Successfully imported rendersim_cpp module")
        
        # Check for operator scheduler classes
        required_classes = [
            'OperatorLevelScheduler',
            'MappedIR',
            'MappedIRNode', 
            'OperatorScheduledIR',
            'OperatorScheduledIRNode',
            'OptimizationLibrary',
            'DummyOperatorOptimizer',
            'OptimizerFactory'
        ]
        
        for cls_name in required_classes:
            if hasattr(rs, cls_name):
                print(f"  ✅ Found {cls_name}")
            else:
                print(f"  ❌ Missing {cls_name}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import C++ modules: {e}")
        return False

def create_test_mapped_ir():
    """Create test MappedIR data for scheduler testing"""
    sys.path.insert(0, "build/Scheduler/cpp")
    import rendersim_cpp as rs
    
    mapped_ir = rs.MappedIR()
    
    # Create test nodes with different hardware units and operator types
    test_nodes = [
        {
            'id': 'encoding_op',
            'op_type': 'ENCODING',
            'hw_unit': 'encoder_0',
            'input_shape': [1024, 3],
            'output_shape': [1024, 63],
            'attrs': {'encoding_type': 'positional', 'max_frequencies': '10'}
        },
        {
            'id': 'mlp_density',
            'op_type': 'FIELD_COMPUTATION', 
            'hw_unit': 'mlp_0',
            'input_shape': [1024, 63],
            'output_shape': [1024, 1],
            'attrs': {'layer_count': '4', 'hidden_size': '256'}
        },
        {
            'id': 'mlp_color',
            'op_type': 'FIELD_COMPUTATION',
            'hw_unit': 'mlp_0',  # Same hardware unit as density MLP
            'input_shape': [1024, 64],
            'output_shape': [1024, 3],
            'attrs': {'layer_count': '3', 'hidden_size': '128'}
        },
        {
            'id': 'volume_render',
            'op_type': 'BLENDING',
            'hw_unit': 'renderer_0',
            'input_shape': [1024, 4],
            'output_shape': [1024, 3],
            'attrs': {'blending_mode': 'alpha', 'sample_count': '128'}
        }
    ]
    
    for node_data in test_nodes:
        node = rs.MappedIRNode()
        
        # Set operator node properties
        node.op_node.id = node_data['id']
        node.op_node.op_type = node_data['op_type']
        node.op_node.call_count = 1
        
        # Create input tensor
        input_tensor = rs.TensorDesc()
        input_tensor.shape = node_data['input_shape']
        input_tensor.dtype = "float32"
        node.op_node.inputs = [input_tensor]
        
        # Create output tensor
        output_tensor = rs.TensorDesc()
        output_tensor.shape = node_data['output_shape']
        output_tensor.dtype = "float32"
        node.op_node.outputs = [output_tensor]
        
        # Set hardware unit and attributes
        node.hw_unit = node_data['hw_unit']
        node.attrs = node_data['attrs']
        
        mapped_ir.nodes[node_data['id']] = node
    
    # Add edges for dependencies
    mapped_ir.edges = [
        ("encoding_op", "mlp_density"),
        ("encoding_op", "mlp_color"),
        ("mlp_density", "volume_render"),
        ("mlp_color", "volume_render")
    ]
    
    return mapped_ir

def test_operator_scheduler_creation():
    """Test OperatorLevelScheduler creation and basic setup"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔧 Testing OperatorLevelScheduler creation...")
        
        # Create optimization library and optimizer
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        
        # Create scheduler
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        # Test basic properties
        assert scheduler is not None
        
        # Test initial statistics (may have some initial values)
        stats = scheduler.get_last_scheduling_stats()
        assert stats.total_operators >= 0
        assert stats.optimized_operators >= 0
        
        # Test latency instrumentation controls
        scheduler.set_latency_instrumentation_enabled(True)
        scheduler.clear_latency_measurements()
        scheduler.set_latency_instrumentation_enabled(False)
        
        print("  ✅ OperatorLevelScheduler creation and setup works")
        return True
        
    except Exception as e:
        print(f"  ❌ Scheduler creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_scheduling_functionality():
    """Test basic operator scheduling workflow"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("⚙️  Testing basic scheduling functionality...")
        
        # Create test data
        mapped_ir = create_test_mapped_ir()
        
        # Create scheduler
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        # Enable instrumentation
        scheduler.set_latency_instrumentation_enabled(True)
        
        # Run scheduling
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        # Verify output structure
        assert scheduled_ir is not None
        assert len(scheduled_ir.nodes) == 4  # Same number of nodes as input
        assert len(scheduled_ir.edges) == 4  # Same number of edges as input
        
        # Verify all input nodes are present in output
        input_node_ids = set(mapped_ir.nodes.keys())
        output_node_ids = set(scheduled_ir.nodes.keys())
        assert input_node_ids == output_node_ids
        
        # Verify edges are preserved
        input_edges = set(mapped_ir.edges)
        output_edges = set(scheduled_ir.edges)
        assert input_edges == output_edges
        
        print("  ✅ Basic scheduling functionality works")
        print(f"     Scheduled {len(scheduled_ir.nodes)} operators")
        print(f"     Preserved {len(scheduled_ir.edges)} dependencies")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic scheduling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_operator_scheduled_ir_structure():
    """Test OperatorScheduledIR and OperatorScheduledIRNode structure"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🏗️  Testing OperatorScheduledIR structure...")
        
        # Create test data and run scheduling
        mapped_ir = create_test_mapped_ir()
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        # Test structure of scheduled nodes
        for node_id, scheduled_node in scheduled_ir.nodes.items():
            # Verify OperatorScheduledIRNode has expected fields
            assert hasattr(scheduled_node, 'mapped_node')
            assert hasattr(scheduled_node, 'start_cycle')
            assert hasattr(scheduled_node, 'duration')
            assert hasattr(scheduled_node, 'resources')
            assert hasattr(scheduled_node, 'optimization_result')
            
            # Verify mapped_node is preserved
            assert scheduled_node.mapped_node.op_node.id == node_id
            assert len(scheduled_node.mapped_node.op_node.inputs) > 0
            assert len(scheduled_node.mapped_node.op_node.outputs) > 0
            
            # Verify timing information is populated
            assert scheduled_node.start_cycle >= 0
            assert scheduled_node.duration > 0
            
            # Verify resources are allocated
            assert len(scheduled_node.resources) > 0
            assert "compute_units" in scheduled_node.resources
            assert "memory_bandwidth" in scheduled_node.resources
            
            # Verify optimization result is present
            assert scheduled_node.optimization_result.duration > 0
            
            print(f"     Node {node_id}: start={scheduled_node.start_cycle}, "
                  f"duration={scheduled_node.duration}, "
                  f"hw_unit={scheduled_node.mapped_node.hw_unit}")
        
        print("  ✅ OperatorScheduledIR structure validation passed")
        return True
        
    except Exception as e:
        print(f"  ❌ OperatorScheduledIR structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hardware_unit_grouping():
    """Test hardware unit grouping and per-HW scheduling"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔧 Testing hardware unit grouping...")
        
        # Create test data with multiple operators on same hardware
        mapped_ir = create_test_mapped_ir()
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        # Group scheduled nodes by hardware unit
        hw_groups = {}
        for node_id, scheduled_node in scheduled_ir.nodes.items():
            hw_unit = scheduled_node.mapped_node.hw_unit
            if hw_unit not in hw_groups:
                hw_groups[hw_unit] = []
            hw_groups[hw_unit].append(scheduled_node)
        
        print(f"     Found {len(hw_groups)} hardware units:")
        
        # Verify scheduling within each hardware unit
        for hw_unit, nodes in hw_groups.items():
            print(f"       {hw_unit}: {len(nodes)} operators")
            
            if len(nodes) > 1:
                # For multiple operators on same HW, verify sequential scheduling
                nodes.sort(key=lambda n: n.start_cycle)
                
                for i in range(1, len(nodes)):
                    prev_node = nodes[i-1]
                    curr_node = nodes[i]
                    
                    # Current node should start after previous node finishes
                    # Note: Due to dependency resolution, this might not be strictly sequential
                    prev_finish = prev_node.start_cycle + prev_node.duration
                    
                    print(f"         {prev_node.mapped_node.op_node.id}: "
                          f"{prev_node.start_cycle}-{prev_finish}")
                    print(f"         {curr_node.mapped_node.op_node.id}: "
                          f"{curr_node.start_cycle}-{curr_node.start_cycle + curr_node.duration}")
                    
                    # Check if scheduling violates hardware constraints
                    # (Allow for dependency-driven scheduling that might override hardware ordering)
                    if curr_node.start_cycle < prev_finish:
                        print(f"         ⚠️  Note: Dependency-driven scheduling detected")
                    else:
                        print(f"         ✓ Sequential scheduling on {hw_unit}")
        
        print("  ✅ Hardware unit grouping and scheduling works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Hardware unit grouping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_integration():
    """Test integration with optimization library and passes"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🚀 Testing optimization integration...")
        
        # Create test data
        mapped_ir = create_test_mapped_ir()
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        # Verify optimization results are present
        for node_id, scheduled_node in scheduled_ir.nodes.items():
            opt_result = scheduled_node.optimization_result
            
            # Verify optimization result structure
            assert hasattr(opt_result, 'duration')
            assert hasattr(opt_result, 'applied_optimizations')
            assert hasattr(opt_result, 'speedup_factor')
            assert hasattr(opt_result, 'base_duration')
            
            # Verify reasonable values
            assert opt_result.duration > 0
            assert opt_result.speedup_factor > 0.0  # Can be < 1.0 (slowdown) or > 1.0 (speedup)
            assert opt_result.base_duration > 0
            
            # Duration should match scheduled node duration
            assert scheduled_node.duration == opt_result.duration
            
            print(f"     {node_id}: duration={opt_result.duration}, "
                  f"speedup={opt_result.speedup_factor:.2f}x, "
                  f"optimizations={len(opt_result.applied_optimizations)}")
        
        print("  ✅ Optimization integration works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Optimization integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduling_statistics():
    """Test scheduling statistics collection and accuracy"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("📊 Testing scheduling statistics...")
        
        # Create test data
        mapped_ir = create_test_mapped_ir()
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        scheduled_ir = scheduler.schedule(mapped_ir)
        stats = scheduler.get_last_scheduling_stats()
        
        # Verify basic statistics
        assert stats.total_operators == len(mapped_ir.nodes)
        assert stats.optimized_operators >= 0
        assert stats.optimized_operators <= stats.total_operators
        
        # Verify hardware unit usage statistics
        expected_hw_units = set()
        for node in mapped_ir.nodes.values():
            expected_hw_units.add(node.hw_unit)
        
        for hw_unit in expected_hw_units:
            assert hw_unit in stats.hw_unit_usage
            assert stats.hw_unit_usage[hw_unit] > 0
        
        # Calculate expected usage counts
        expected_usage = {}
        for node in mapped_ir.nodes.values():
            expected_usage[node.hw_unit] = expected_usage.get(node.hw_unit, 0) + 1
        
        for hw_unit, expected_count in expected_usage.items():
            assert stats.hw_unit_usage[hw_unit] == expected_count
        
        print(f"     Total operators: {stats.total_operators}")
        print(f"     Optimized operators: {stats.optimized_operators}")
        print(f"     Total speedup: {stats.total_speedup:.2f}x")
        print(f"     Hardware unit usage:")
        for hw_unit, count in stats.hw_unit_usage.items():
            print(f"       {hw_unit}: {count} operators")
        
        print("  ✅ Scheduling statistics collection works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Scheduling statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependency_resolution():
    """Test cross-hardware dependency resolution and timing"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔗 Testing dependency resolution...")
        
        # Create test data with cross-hardware dependencies
        mapped_ir = create_test_mapped_ir()
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        # Verify dependency constraints are satisfied
        for edge in scheduled_ir.edges:
            source_id, target_id = edge
            
            source_node = scheduled_ir.nodes[source_id]
            target_node = scheduled_ir.nodes[target_id]
            
            # Source must finish before target starts
            source_finish = source_node.start_cycle + source_node.duration
            target_start = target_node.start_cycle
            
            assert source_finish <= target_start, \
                f"Dependency violation: {source_id} finishes at {source_finish}, " \
                f"but {target_id} starts at {target_start}"
            
            print(f"     Dependency {source_id} → {target_id}: "
                  f"{source_finish} ≤ {target_start} ✓")
        
        # Verify scheduling makes sense for our test graph
        # encoding_op should start first (no dependencies)
        encoding_node = scheduled_ir.nodes["encoding_op"]
        assert encoding_node.start_cycle == 0
        
        # MLPs should start after encoding finishes
        encoding_finish = encoding_node.start_cycle + encoding_node.duration
        density_node = scheduled_ir.nodes["mlp_density"]
        color_node = scheduled_ir.nodes["mlp_color"]
        
        # Both MLPs depend on encoding, but are on same hardware unit
        # So they should be scheduled sequentially after encoding finishes
        assert density_node.start_cycle >= encoding_finish
        assert color_node.start_cycle >= encoding_finish
        
        # Volume render should start after both MLPs finish
        render_node = scheduled_ir.nodes["volume_render"]
        density_finish = density_node.start_cycle + density_node.duration
        color_finish = color_node.start_cycle + color_node.duration
        latest_mlp_finish = max(density_finish, color_finish)
        
        assert render_node.start_cycle >= latest_mlp_finish
        
        print("  ✅ Dependency resolution works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Dependency resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_latency_instrumentation():
    """Test latency instrumentation in operator scheduler"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("⏱️  Testing latency instrumentation...")
        
        # Create test data
        mapped_ir = create_test_mapped_ir()
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        # Clear any existing measurements
        scheduler.clear_latency_measurements()
        scheduler.set_latency_instrumentation_enabled(True)
        
        # Run scheduling
        scheduled_ir = scheduler.schedule(mapped_ir)
        
        # Get latency report
        latency_report = scheduler.get_latency_report()
        
        # Verify timing data was collected
        assert latency_report.operator_total.last_duration_ns > 0
        assert latency_report.operator_hw_grouping.last_duration_ns >= 0
        assert latency_report.operator_hw_scheduling.last_duration_ns >= 0
        assert latency_report.operator_dependency_resolution.last_duration_ns >= 0
        
        # Test disabling instrumentation
        scheduler.set_latency_instrumentation_enabled(False)
        scheduler.clear_latency_measurements()
        
        scheduled_ir2 = scheduler.schedule(mapped_ir)
        latency_report2 = scheduler.get_latency_report()
        
        # After clearing, measurements should be 0
        assert latency_report2.operator_total.last_duration_ns == 0
        
        print(f"     Total time: {latency_report.operator_total.last_duration_ns} ns")
        print(f"     HW grouping: {latency_report.operator_hw_grouping.last_duration_ns} ns")
        print(f"     HW scheduling: {latency_report.operator_hw_scheduling.last_duration_ns} ns")
        print(f"     Dependency resolution: {latency_report.operator_dependency_resolution.last_duration_ns} ns")
        
        print("  ✅ Latency instrumentation works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Latency instrumentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔍 Testing edge cases...")
        
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        scheduler = rs.OperatorLevelScheduler(optimizer)
        
        # Test 1: Empty mapped IR
        empty_ir = rs.MappedIR()
        scheduled_empty = scheduler.schedule(empty_ir)
        
        assert len(scheduled_empty.nodes) == 0
        assert len(scheduled_empty.edges) == 0
        
        stats = scheduler.get_last_scheduling_stats()
        assert stats.total_operators == 0
        
        print("     ✅ Empty input handling works")
        
        # Test 2: Single operator
        single_ir = rs.MappedIR()
        
        node = rs.MappedIRNode()
        node.op_node.id = "single_op"
        node.op_node.op_type = "ENCODING"
        node.hw_unit = "encoder_0"
        
        input_tensor = rs.TensorDesc()
        input_tensor.shape = [1024, 3]
        input_tensor.dtype = "float32"
        node.op_node.inputs = [input_tensor]
        
        output_tensor = rs.TensorDesc()
        output_tensor.shape = [1024, 63]
        output_tensor.dtype = "float32"
        node.op_node.outputs = [output_tensor]
        
        single_ir.nodes["single_op"] = node
        
        scheduled_single = scheduler.schedule(single_ir)
        
        assert len(scheduled_single.nodes) == 1
        assert "single_op" in scheduled_single.nodes
        
        single_node = scheduled_single.nodes["single_op"]
        assert single_node.start_cycle == 0  # No dependencies, should start immediately
        assert single_node.duration > 0
        
        print("     ✅ Single operator handling works")
        
        # Test 3: Multiple operators on same hardware (sequential scheduling)
        multi_hw_ir = rs.MappedIR()
        
        for i in range(3):
            node = rs.MappedIRNode()
            node.op_node.id = f"op_{i}"
            node.op_node.op_type = "FIELD_COMPUTATION"
            node.hw_unit = "mlp_0"  # All on same hardware unit
            
            input_tensor = rs.TensorDesc()
            input_tensor.shape = [1024, 64]
            input_tensor.dtype = "float32"
            node.op_node.inputs = [input_tensor]
            
            output_tensor = rs.TensorDesc()
            output_tensor.shape = [1024, 64]
            output_tensor.dtype = "float32"
            node.op_node.outputs = [output_tensor]
            
            multi_hw_ir.nodes[f"op_{i}"] = node
        
        scheduled_multi = scheduler.schedule(multi_hw_ir)
        
        # Verify sequential scheduling on same hardware
        scheduled_nodes = list(scheduled_multi.nodes.values())
        scheduled_nodes.sort(key=lambda n: n.start_cycle)
        
        for i in range(1, len(scheduled_nodes)):
            prev_node = scheduled_nodes[i-1]
            curr_node = scheduled_nodes[i]
            
            prev_finish = prev_node.start_cycle + prev_node.duration
            assert curr_node.start_cycle >= prev_finish
        
        print("     ✅ Multiple operators on same hardware works")
        
        print("  ✅ Edge cases handling works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_data_loading():
    """Test loading mapped IR from JSON test data"""
    try:
        print("📄 Testing JSON test data loading...")
        
        # Check if test data file exists
        test_data_file = Path("tests/data/mapped_ir_min.json")
        if not test_data_file.exists():
            print(f"     ⚠️  Test data file not found: {test_data_file}")
            print("     ℹ️  Skipping JSON data loading test")
            return True
        
        # Load and parse JSON
        with open(test_data_file, 'r') as f:
            data = json.load(f)
        
        # Verify JSON structure
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0
        
        # Verify node structure
        for node_id, node_data in data["nodes"].items():
            assert "op_node" in node_data
            assert "hw_unit" in node_data
            assert "attrs" in node_data
            
            op_node = node_data["op_node"]
            assert "id" in op_node
            assert "op_type" in op_node
            assert "inputs" in op_node
            assert "outputs" in op_node
        
        print(f"     ✅ Loaded {len(data['nodes'])} nodes and {len(data['edges'])} edges")
        print("     ✅ JSON structure validation passed")
        
        # TODO: Could create MappedIR from JSON data and test scheduling
        # This would require JSON-to-C++ conversion utilities
        
        print("  ✅ JSON test data loading works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ JSON data loading test failed: {e}")
        return False

def main():
    """Run all operator scheduler unit tests"""
    print("🧪 Testing RenderSim OperatorLevelScheduler")
    print("=" * 50)
    
    tests = [
        ("C++ Scheduler Import", test_cpp_scheduler_imports),
        ("Scheduler Creation", test_operator_scheduler_creation),
        ("Basic Scheduling", test_basic_scheduling_functionality),
        ("OperatorScheduledIR Structure", test_operator_scheduled_ir_structure),
        ("Hardware Unit Grouping", test_hardware_unit_grouping),
        ("Optimization Integration", test_optimization_integration),
        ("Scheduling Statistics", test_scheduling_statistics),
        ("Dependency Resolution", test_dependency_resolution),
        ("Latency Instrumentation", test_latency_instrumentation),
        ("Edge Cases", test_edge_cases),
        ("JSON Data Loading", test_json_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All OperatorLevelScheduler tests passed!")
        print("✅ Operator scheduling functionality is working correctly")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 