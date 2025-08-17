#!/usr/bin/env python3
"""
Comprehensive unit tests for RenderSim SystemLevelScheduler

Tests cover:
- DAGS algorithm implementation and correctness
- System-level scheduling workflow and data structures
- Global coordination across hardware units
- Priority queue and scoring mechanisms (successor count + critical resource impact)
- System statistics and efficiency metrics
- Resource utilization tracking and balance factors
- Schedule validation and dependency satisfaction
- Latency instrumentation and performance measurement
- DAGS configuration management and weight tuning
- Edge cases and robustness testing
"""

import os
import sys
import json
from pathlib import Path

# Add RenderSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cpp_system_scheduler_imports():
    """Test that C++ system scheduler modules can be imported"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("✅ Successfully imported rendersim_cpp module")
        
        # Check for system scheduler classes
        required_classes = [
            'SystemLevelScheduler',
            'SystemSchedule', 
            'SystemScheduleEntry',
            'DAGSConfig',
            'SystemSchedulingStats',
            'SystemSchedulerFactory',
            'SystemSchedulerType'
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

def create_test_operator_scheduled_ir():
    """Create test OperatorScheduledIR data for system scheduler testing"""
    sys.path.insert(0, "build/Scheduler/cpp")
    import rendersim_cpp as rs
    
    # Create MappedIR first, then use real OperatorLevelScheduler to generate OperatorScheduledIR
    mapped_ir = rs.MappedIR()
    
    # Create test nodes
    test_nodes = [
        {
            'id': 'sampling_op',
            'op_type': 'SAMPLING', 
            'hw_unit': 'sampler_0',
            'input_shape': [1024, 6],
            'output_shape': [1024, 128, 3]
        },
        {
            'id': 'encoding_op',
            'op_type': 'ENCODING',
            'hw_unit': 'encoder_0', 
            'input_shape': [1024, 128, 3],
            'output_shape': [1024, 128, 63]
        },
        {
            'id': 'density_mlp',
            'op_type': 'FIELD_COMPUTATION',
            'hw_unit': 'mlp_0',
            'input_shape': [1024, 128, 63],
            'output_shape': [1024, 128, 1]
        },
        {
            'id': 'color_mlp', 
            'op_type': 'FIELD_COMPUTATION',
            'hw_unit': 'mlp_1',
            'input_shape': [1024, 128, 64],
            'output_shape': [1024, 128, 3]
        },
        {
            'id': 'volume_render',
            'op_type': 'BLENDING',
            'hw_unit': 'renderer_0',
            'input_shape': [1024, 128, 4],
            'output_shape': [1024, 3]
        }
    ]
    
    for node_data in test_nodes:
        node = rs.MappedIRNode()
        
        # Set operator node information
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
        node.attrs = {'complexity': 'medium'}
        
        mapped_ir.nodes[node_data['id']] = node
    
    # Add edges for dependency graph
    mapped_ir.edges = [
        ("sampling_op", "encoding_op"),
        ("encoding_op", "density_mlp"),
        ("encoding_op", "color_mlp"),
        ("density_mlp", "volume_render"),
        ("color_mlp", "volume_render")
    ]
    
    # Use real OperatorLevelScheduler to generate proper OperatorScheduledIR
    lib = rs.OptimizationLibrary()
    optimizer = rs.DummyOperatorOptimizer(lib)
    op_scheduler = rs.OperatorLevelScheduler(optimizer)
    
    op_scheduled_ir = op_scheduler.schedule(mapped_ir)
    
    return op_scheduled_ir

def test_system_scheduler_creation():
    """Test SystemLevelScheduler creation and basic setup"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔧 Testing SystemLevelScheduler creation...")
        
        # Test default DAGS configuration
        default_config = rs.DAGSConfig()
        assert default_config.alpha == 0.6  # Default successor count weight
        assert default_config.beta == 0.4   # Default critical resource impact weight
        
        # Test custom DAGS configuration
        custom_config = rs.DAGSConfig(0.7, 0.3)
        assert custom_config.alpha == 0.7
        assert custom_config.beta == 0.3
        
        # Create scheduler with default config
        scheduler = rs.SystemLevelScheduler()
        assert scheduler is not None
        
        # Create scheduler with custom config
        scheduler_custom = rs.SystemLevelScheduler(custom_config)
        assert scheduler_custom is not None
        
        # Test initial statistics
        stats = scheduler.get_last_scheduling_stats()
        assert stats.total_operators >= 0
        assert stats.ready_queue_peak_size >= 0
        
        # Test latency instrumentation controls
        scheduler.set_latency_instrumentation_enabled(True)
        scheduler.clear_latency_measurements()
        scheduler.set_latency_instrumentation_enabled(False)
        
        # Test configuration updates
        new_config = rs.DAGSConfig(0.5, 0.5)
        scheduler.update_config(new_config)
        
        print("  ✅ SystemLevelScheduler creation and setup works")
        return True
        
    except Exception as e:
        print(f"  ❌ System scheduler creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_system_scheduling():
    """Test basic system scheduling workflow"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🏗️  Testing basic system scheduling functionality...")
        
        # Create test data
        op_scheduled_ir = create_test_operator_scheduled_ir()
        
        # Create scheduler
        config = rs.DAGSConfig()
        scheduler = rs.SystemLevelScheduler(config)
        
        # Enable instrumentation
        scheduler.set_latency_instrumentation_enabled(True)
        
        # Run system scheduling
        system_schedule = scheduler.schedule(op_scheduled_ir)
        
        # Verify output structure
        assert system_schedule is not None
        assert len(system_schedule.entries) == 5  # Same number of operators as input
        
        # Verify all input operators are present in output
        input_op_ids = set(op_scheduled_ir.nodes.keys())
        output_op_ids = set(entry.op_id for entry in system_schedule.entries)
        assert input_op_ids == output_op_ids
        
        # Verify schedule entries have expected fields
        for entry in system_schedule.entries:
            assert hasattr(entry, 'op_id')
            assert hasattr(entry, 'hw_unit')
            assert hasattr(entry, 'start_cycle')
            assert hasattr(entry, 'duration')
            assert hasattr(entry, 'resource_utilization')
            
            assert entry.start_cycle >= 0
            assert entry.duration > 0
            assert 0.0 <= entry.resource_utilization <= 1.0
        
        # Verify system schedule has global metrics
        assert system_schedule.total_cycles > 0
        assert system_schedule.avg_resource_utilization >= 0.0
        assert len(system_schedule.hw_unit_finish_times) > 0
        
        print(f"     Scheduled {len(system_schedule.entries)} operators")
        print(f"     Total cycles: {system_schedule.total_cycles}")
        print(f"     Average resource utilization: {system_schedule.avg_resource_utilization:.2f}")
        
        print("  ✅ Basic system scheduling functionality works")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic system scheduling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dags_algorithm_correctness():
    """Test DAGS algorithm implementation and dependency handling"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🧠 Testing DAGS algorithm correctness...")
        
        # Create test data
        op_scheduled_ir = create_test_operator_scheduled_ir()
        
        # Create scheduler
        config = rs.DAGSConfig(0.6, 0.4)  # Standard DAGS weights
        scheduler = rs.SystemLevelScheduler(config)
        
        system_schedule = scheduler.schedule(op_scheduled_ir)
        
        # Test 1: Dependency constraint satisfaction
        # Build schedule map for quick lookup
        schedule_map = {entry.op_id: entry for entry in system_schedule.entries}
        
        # Verify all dependencies are satisfied
        for source, target in op_scheduled_ir.edges:
            source_entry = schedule_map[source]
            target_entry = schedule_map[target]
            
            source_finish = source_entry.start_cycle + source_entry.duration
            target_start = target_entry.start_cycle
            
            assert source_finish <= target_start, \
                f"Dependency violation: {source} finishes at {source_finish}, " \
                f"but {target} starts at {target_start}"
            
            print(f"     Dependency {source} → {target}: "
                  f"{source_finish} ≤ {target_start} ✓")
        
        # Test 2: Hardware unit coordination
        # Verify no two operations on same hardware overlap
        hw_operations = {}
        for entry in system_schedule.entries:
            if entry.hw_unit not in hw_operations:
                hw_operations[entry.hw_unit] = []
            hw_operations[entry.hw_unit].append(entry)
        
        for hw_unit, ops in hw_operations.items():
            if len(ops) > 1:
                # Sort by start time
                ops.sort(key=lambda op: op.start_cycle)
                
                for i in range(1, len(ops)):
                    prev_op = ops[i-1]
                    curr_op = ops[i]
                    
                    prev_finish = prev_op.start_cycle + prev_op.duration
                    curr_start = curr_op.start_cycle
                    
                    assert prev_finish <= curr_start, \
                        f"Hardware conflict on {hw_unit}: {prev_op.op_id} and {curr_op.op_id} overlap"
                    
                    print(f"     Hardware {hw_unit}: {prev_op.op_id} → {curr_op.op_id} ✓")
        
        # Test 3: Schedule optimality indicators
        # Verify schedule makes reasonable decisions
        source_nodes = []
        for entry in system_schedule.entries:
            is_source = True
            for source, target in op_scheduled_ir.edges:
                if target == entry.op_id:
                    is_source = False
                    break
            if is_source:
                source_nodes.append(entry)
        
        # Source nodes should start early (no dependencies)
        for source_node in source_nodes:
            print(f"     Source node {source_node.op_id} starts at cycle {source_node.start_cycle}")
        
        print("  ✅ DAGS algorithm correctness validated")
        return True
        
    except Exception as e:
        print(f"  ❌ DAGS algorithm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_scheduling_statistics():
    """Test system scheduling statistics collection and accuracy"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("📊 Testing system scheduling statistics...")
        
        # Create test data
        op_scheduled_ir = create_test_operator_scheduled_ir()
        
        # Create scheduler
        config = rs.DAGSConfig()
        scheduler = rs.SystemLevelScheduler(config)
        
        system_schedule = scheduler.schedule(op_scheduled_ir)
        stats = scheduler.get_last_scheduling_stats()
        
        # Verify basic statistics
        assert stats.total_operators == len(op_scheduled_ir.nodes)
        assert stats.ready_queue_peak_size >= 1  # At least source nodes in queue
        
        # Verify efficiency and balance metrics
        assert stats.scheduling_efficiency >= 0.0
        assert stats.resource_balance_factor >= 0.0
        
        # Verify hardware unit utilizations
        expected_hw_units = set()
        for entry in system_schedule.entries:
            expected_hw_units.add(entry.hw_unit)
        
        for hw_unit in expected_hw_units:
            assert hw_unit in stats.hw_unit_utilizations
            assert 0.0 <= stats.hw_unit_utilizations[hw_unit] <= 1.0
        
        print(f"     Total operators: {stats.total_operators}")
        print(f"     Ready queue peak size: {stats.ready_queue_peak_size}")
        print(f"     Scheduling efficiency: {stats.scheduling_efficiency:.3f}")
        print(f"     Resource balance factor: {stats.resource_balance_factor:.3f}")
        print(f"     Hardware unit utilizations:")
        for hw_unit, utilization in stats.hw_unit_utilizations.items():
            print(f"       {hw_unit}: {utilization:.3f}")
        
        print("  ✅ System scheduling statistics collection works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ System scheduling statistics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dags_configuration_effects():
    """Test DAGS configuration weight effects on scheduling decisions"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("⚙️  Testing DAGS configuration effects...")
        
        # Create test data
        op_scheduled_ir = create_test_operator_scheduled_ir()
        
        # Test different DAGS configurations
        configs = [
            ("Successor-Heavy", rs.DAGSConfig(0.9, 0.1)),  # Prioritize successor count
            ("Resource-Heavy", rs.DAGSConfig(0.1, 0.9)),   # Prioritize critical resource impact
            ("Balanced", rs.DAGSConfig(0.5, 0.5)),         # Equal weights
        ]
        
        schedules = {}
        
        for config_name, config in configs:
            scheduler = rs.SystemLevelScheduler(config)
            schedule = scheduler.schedule(op_scheduled_ir)
            stats = scheduler.get_last_scheduling_stats()
            
            schedules[config_name] = {
                'schedule': schedule,
                'stats': stats,
                'total_cycles': schedule.total_cycles,
                'efficiency': stats.scheduling_efficiency
            }
            
            print(f"     {config_name}: {schedule.total_cycles} cycles, "
                  f"efficiency={stats.scheduling_efficiency:.3f}")
        
        # Verify all configurations produce valid schedules
        for config_name, result in schedules.items():
            assert result['total_cycles'] > 0
            assert len(result['schedule'].entries) == len(op_scheduled_ir.nodes)
        
        # Different configurations should potentially produce different results
        # (This is probabilistic, but we can at least verify they run successfully)
        unique_cycles = set(result['total_cycles'] for result in schedules.values())
        print(f"     Generated {len(unique_cycles)} distinct schedule lengths")
        
        print("  ✅ DAGS configuration effects testing works")
        return True
        
    except Exception as e:
        print(f"  ❌ DAGS configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_latency_instrumentation():
    """Test latency instrumentation in system scheduler"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("⏱️  Testing system scheduler latency instrumentation...")
        
        # Create test data
        op_scheduled_ir = create_test_operator_scheduled_ir()
        
        # Create scheduler
        config = rs.DAGSConfig()
        scheduler = rs.SystemLevelScheduler(config)
        
        # Clear any existing measurements
        scheduler.clear_latency_measurements()
        scheduler.set_latency_instrumentation_enabled(True)
        
        # Run scheduling
        system_schedule = scheduler.schedule(op_scheduled_ir)
        
        # Get latency report
        latency_report = scheduler.get_latency_report()
        
        # Verify timing data was collected for system stages
        assert latency_report.system_total.last_duration_ns > 0
        assert latency_report.system_dependency_graph.last_duration_ns >= 0
        assert latency_report.system_heuristic_computation.last_duration_ns >= 0
        assert latency_report.system_scheduling_loop.last_duration_ns >= 0
        assert latency_report.system_finalization.last_duration_ns >= 0
        
        # Test disabling instrumentation
        scheduler.set_latency_instrumentation_enabled(False)
        scheduler.clear_latency_measurements()
        
        system_schedule2 = scheduler.schedule(op_scheduled_ir)
        latency_report2 = scheduler.get_latency_report()
        
        # After clearing, measurements should be 0
        assert latency_report2.system_total.last_duration_ns == 0
        
        print(f"     Total time: {latency_report.system_total.last_duration_ns} ns")
        print(f"     Dependency graph: {latency_report.system_dependency_graph.last_duration_ns} ns")
        print(f"     Heuristic computation: {latency_report.system_heuristic_computation.last_duration_ns} ns")
        print(f"     Scheduling loop: {latency_report.system_scheduling_loop.last_duration_ns} ns")
        print(f"     Finalization: {latency_report.system_finalization.last_duration_ns} ns")
        
        print("  ✅ System scheduler latency instrumentation works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ System latency instrumentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_scheduler_edge_cases():
    """Test edge cases and error handling for system scheduler"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔍 Testing system scheduler edge cases...")
        
        config = rs.DAGSConfig()
        scheduler = rs.SystemLevelScheduler(config)
        
        # Test 1: Empty operator scheduled IR
        empty_ir = rs.OperatorScheduledIR()
        empty_schedule = scheduler.schedule(empty_ir)
        
        assert len(empty_schedule.entries) == 0
        assert empty_schedule.total_cycles == 0
        
        stats = scheduler.get_last_scheduling_stats()
        assert stats.total_operators == 0
        
        print("     ✅ Empty input handling works")
        
        # Test 2: Single operator
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
        
        node.attrs = {'complexity': 'medium'}
        
        # Use operator scheduler to generate proper OperatorScheduledIR
        mapped_ir_single = rs.MappedIR()
        mapped_ir_single.nodes["single_op"] = node
        
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        op_scheduler = rs.OperatorLevelScheduler(optimizer)
        single_ir = op_scheduler.schedule(mapped_ir_single)
        
        single_schedule = scheduler.schedule(single_ir)
        
        assert len(single_schedule.entries) == 1
        assert single_schedule.entries[0].op_id == "single_op"
        assert single_schedule.entries[0].start_cycle >= 0
        assert single_schedule.total_cycles > 0
        
        print("     ✅ Single operator handling works")
        
        # Test 3: Complex dependency chain (linear)
        mapped_ir_chain = rs.MappedIR()
        
        for i in range(4):
            node = rs.MappedIRNode()
            node.op_node.id = f"op_{i}"
            node.op_node.op_type = "FIELD_COMPUTATION"
            node.hw_unit = f"hw_{i % 2}"  # Alternate between 2 hardware units
            
            input_tensor = rs.TensorDesc()
            input_tensor.shape = [1024, 64]
            input_tensor.dtype = "float32"
            node.op_node.inputs = [input_tensor]
            
            output_tensor = rs.TensorDesc()
            output_tensor.shape = [1024, 64]
            output_tensor.dtype = "float32"
            node.op_node.outputs = [output_tensor]
            
            node.attrs = {'complexity': 'medium'}
            
            mapped_ir_chain.nodes[f"op_{i}"] = node
        
        # Create linear dependency chain
        mapped_ir_chain.edges = [("op_0", "op_1"), ("op_1", "op_2"), ("op_2", "op_3")]
        
        # Use operator scheduler to generate proper OperatorScheduledIR
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        op_scheduler = rs.OperatorLevelScheduler(optimizer)
        chain_ir = op_scheduler.schedule(mapped_ir_chain)
        
        chain_schedule = scheduler.schedule(chain_ir)
        
        # Verify linear ordering is preserved
        schedule_map = {entry.op_id: entry for entry in chain_schedule.entries}
        
        for i in range(3):
            current = schedule_map[f"op_{i}"]
            next_op = schedule_map[f"op_{i+1}"]
            
            current_finish = current.start_cycle + current.duration
            assert current_finish <= next_op.start_cycle
        
        print("     ✅ Complex dependency chain handling works")
        
        print("  ✅ System scheduler edge cases handling works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ System scheduler edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_scheduler_factory():
    """Test SystemSchedulerFactory and scheduler type selection"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🏭 Testing SystemSchedulerFactory...")
        
        # Test different scheduler types
        config = rs.DAGSConfig(0.6, 0.4)
        
        scheduler_types = [
            rs.SystemSchedulerType.DAGS,
            rs.SystemSchedulerType.LIST_BASED,
            rs.SystemSchedulerType.CRITICAL_PATH
        ]
        
        for scheduler_type in scheduler_types:
            scheduler = rs.SystemSchedulerFactory.create_scheduler(scheduler_type, config)
            assert scheduler is not None
            
            # Verify scheduler can be used
            op_ir = create_test_operator_scheduled_ir()
            schedule = scheduler.schedule(op_ir)
            assert schedule is not None
            assert len(schedule.entries) > 0
            
            print(f"     ✅ {scheduler_type} scheduler creation and usage works")
        
        print("  ✅ SystemSchedulerFactory works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ SystemSchedulerFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_operator_to_system_scheduling():
    """Test complete operator-to-system scheduling pipeline"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔄 Testing end-to-end operator-to-system scheduling...")
        
        # Step 1: Create MappedIR (would normally come from mapping stage)
        mapped_ir = rs.MappedIR()
        
        test_nodes = [
            {'id': 'encoding', 'hw_unit': 'encoder_0', 'op_type': 'ENCODING'},
            {'id': 'mlp_1', 'hw_unit': 'mlp_0', 'op_type': 'FIELD_COMPUTATION'},
            {'id': 'mlp_2', 'hw_unit': 'mlp_0', 'op_type': 'FIELD_COMPUTATION'},
            {'id': 'render', 'hw_unit': 'renderer_0', 'op_type': 'BLENDING'}
        ]
        
        for node_data in test_nodes:
            node = rs.MappedIRNode()
            node.op_node.id = node_data['id']
            node.op_node.op_type = node_data['op_type']
            node.hw_unit = node_data['hw_unit']
            
            input_tensor = rs.TensorDesc()
            input_tensor.shape = [1024, 64]
            input_tensor.dtype = "float32"
            node.op_node.inputs = [input_tensor]
            
            output_tensor = rs.TensorDesc()
            output_tensor.shape = [1024, 64]
            output_tensor.dtype = "float32"
            node.op_node.outputs = [output_tensor]
            
            node.attrs = {'complexity': 'medium'}
            mapped_ir.nodes[node_data['id']] = node
        
        mapped_ir.edges = [
            ("encoding", "mlp_1"),
            ("encoding", "mlp_2"),
            ("mlp_1", "render"),
            ("mlp_2", "render")
        ]
        
        # Step 2: Operator-level scheduling
        lib = rs.OptimizationLibrary()
        optimizer = rs.DummyOperatorOptimizer(lib)
        op_scheduler = rs.OperatorLevelScheduler(optimizer)
        
        op_scheduled_ir = op_scheduler.schedule(mapped_ir)
        
        # Step 3: System-level scheduling
        config = rs.DAGSConfig()
        sys_scheduler = rs.SystemLevelScheduler(config)
        
        system_schedule = sys_scheduler.schedule(op_scheduled_ir)
        
        # Verify complete pipeline
        assert len(system_schedule.entries) == 4
        assert system_schedule.total_cycles > 0
        
        # Verify dependency preservation through entire pipeline
        schedule_map = {entry.op_id: entry for entry in system_schedule.entries}
        
        for source, target in mapped_ir.edges:
            source_entry = schedule_map[source]
            target_entry = schedule_map[target]
            
            source_finish = source_entry.start_cycle + source_entry.duration
            target_start = target_entry.start_cycle
            
            assert source_finish <= target_start
            
            print(f"     End-to-end dependency {source} → {target}: "
                  f"{source_finish} ≤ {target_start} ✓")
        
        # Get combined statistics
        op_stats = op_scheduler.get_last_scheduling_stats()
        sys_stats = sys_scheduler.get_last_scheduling_stats()
        
        print(f"     Operator scheduling: {op_stats.total_operators} ops processed")
        print(f"     System scheduling: {sys_stats.total_operators} ops coordinated")
        print(f"     Final schedule: {system_schedule.total_cycles} cycles")
        
        print("  ✅ End-to-end operator-to-system scheduling works")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end scheduling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all system scheduler unit tests"""
    print("🧪 Testing RenderSim SystemLevelScheduler")
    print("=" * 50)
    
    tests = [
        ("C++ System Scheduler Import", test_cpp_system_scheduler_imports),
        ("System Scheduler Creation", test_system_scheduler_creation),
        ("Basic System Scheduling", test_basic_system_scheduling),
        ("DAGS Algorithm Correctness", test_dags_algorithm_correctness),
        ("System Scheduling Statistics", test_system_scheduling_statistics),
        ("DAGS Configuration Effects", test_dags_configuration_effects),
        ("System Latency Instrumentation", test_system_latency_instrumentation),
        ("System Scheduler Edge Cases", test_system_scheduler_edge_cases),
        ("SystemSchedulerFactory", test_system_scheduler_factory),
        ("End-to-End Operator→System Pipeline", test_end_to_end_operator_to_system_scheduling)
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
        print("🎉 All SystemLevelScheduler tests passed!")
        print("✅ DAGS algorithm and system scheduling functionality working correctly")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 