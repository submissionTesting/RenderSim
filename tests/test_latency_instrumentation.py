#!/usr/bin/env python3
"""
Comprehensive tests for RenderSim C++ scheduler latency instrumentation

Tests cover:
- Performance timer functionality
- Operator-level scheduler latency tracking
- System-level scheduler latency tracking
- Latency report generation and formatting
- Integration with Python bindings
"""

import os
import sys
import time
from pathlib import Path

# Add RenderSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cpp_imports():
    """Test that C++ latency instrumentation modules can be imported"""
    try:
        # Try importing the C++ module
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("✅ Successfully imported rendersim_cpp module")
        
        # Check for latency-related classes
        required_classes = [
            'LatencyStats',
            'SchedulingLatencyReport',
            'OperatorLevelScheduler',
            'SystemLevelScheduler'
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

def test_latency_stats_creation():
    """Test LatencyStats object creation and manipulation"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🕐 Testing LatencyStats creation...")
        
        # Create empty LatencyStats
        stats = rs.LatencyStats()
        assert stats.total_duration_ns == 0
        assert stats.average_duration_ns == 0.0
        assert stats.last_duration_ns == 0
        assert stats.measurement_count == 0
        
        # Test with values
        stats.total_duration_ns = 1000000  # 1ms
        stats.average_duration_ns = 500000.0  # 0.5ms
        stats.last_duration_ns = 750000  # 0.75ms
        stats.measurement_count = 2
        
        assert stats.total_duration_ns == 1000000
        assert stats.average_duration_ns == 500000.0
        assert stats.last_duration_ns == 750000
        assert stats.measurement_count == 2
        
        print("  ✅ LatencyStats creation and manipulation works")
        return True
        
    except Exception as e:
        print(f"  ❌ LatencyStats test failed: {e}")
        return False

def test_scheduling_latency_report():
    """Test SchedulingLatencyReport creation and formatting"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("📊 Testing SchedulingLatencyReport...")
        
        # Create report
        report = rs.SchedulingLatencyReport()
        
        # Set some operator-level stats
        report.operator_total.last_duration_ns = 2500000  # 2.5ms
        report.operator_hw_grouping.last_duration_ns = 500000  # 0.5ms
        report.operator_hw_scheduling.last_duration_ns = 1500000  # 1.5ms
        report.operator_dependency_resolution.last_duration_ns = 500000  # 0.5ms
        
        # Set some system-level stats
        report.system_total.last_duration_ns = 5000000  # 5ms
        report.system_dependency_graph.last_duration_ns = 800000  # 0.8ms
        report.system_heuristic_computation.last_duration_ns = 1200000  # 1.2ms
        report.system_scheduling_loop.last_duration_ns = 2500000  # 2.5ms
        report.system_finalization.last_duration_ns = 500000  # 0.5ms
        
        # Test report generation
        report_text = report.generate_report()
        assert "RenderSim Scheduling Latency Report" in report_text
        assert "Operator-Level Scheduler:" in report_text
        assert "System-Level Scheduler:" in report_text
        
        # Test duration formatting
        assert rs.SchedulingLatencyReport.format_duration(500) == "500 ns"
        assert "μs" in rs.SchedulingLatencyReport.format_duration(5000)
        assert "ms" in rs.SchedulingLatencyReport.format_duration(5000000)
        
        print("  ✅ SchedulingLatencyReport creation and formatting works")
        return True
        
    except Exception as e:
        print(f"  ❌ SchedulingLatencyReport test failed: {e}")
        return False

def create_test_ir_data():
    """Create test data for scheduler testing"""
    sys.path.insert(0, "build/Scheduler/cpp")
    import rendersim_cpp as rs
    
    # Create optimization library and optimizer
    lib = rs.OptimizationLibrary()
    optimizer = rs.DummyOperatorOptimizer(lib)
    
    # Create test mapped IR
    mapped_ir = rs.MappedIR()
    
    # Create test nodes
    for i in range(3):
        node = rs.MappedIRNode()
        node.op_node.id = f"op_{i}"
        node.op_node.op_type = "FIELD_COMPUTATION" if i % 2 == 0 else "ENCODING"
        node.hw_unit = f"hw_unit_{i % 2}"
        
        # Add input/output tensors
        input_tensor = rs.TensorDesc()
        input_tensor.shape = [1024, 64]
        input_tensor.dtype = "float32"
        node.op_node.inputs = [input_tensor]
        
        output_tensor = rs.TensorDesc()
        output_tensor.shape = [1024, 32]
        output_tensor.dtype = "float32"
        node.op_node.outputs = [output_tensor]
        
        mapped_ir.nodes[f"op_{i}"] = node
    
    # Add edges
    mapped_ir.edges = [("op_0", "op_1"), ("op_1", "op_2")]
    
    return mapped_ir, optimizer

def test_operator_scheduler_latency():
    """Test latency instrumentation in OperatorLevelScheduler"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("⚙️  Testing OperatorLevelScheduler latency instrumentation...")
        
        mapped_ir, optimizer = create_test_ir_data()
        
        # Create operator scheduler
        op_scheduler = rs.OperatorLevelScheduler(optimizer)
        
        # Clear any existing measurements
        op_scheduler.clear_latency_measurements()
        
        # Enable instrumentation (should be enabled by default)
        op_scheduler.set_latency_instrumentation_enabled(True)
        
        # Run scheduling
        scheduled_ir = op_scheduler.schedule(mapped_ir)
        
        # Get latency report
        latency_report = op_scheduler.get_latency_report()
        
        # Verify that timing data was collected
        assert latency_report.operator_total.last_duration_ns > 0
        assert latency_report.operator_hw_grouping.last_duration_ns >= 0
        assert latency_report.operator_hw_scheduling.last_duration_ns >= 0
        assert latency_report.operator_dependency_resolution.last_duration_ns >= 0
        
        # Test disabling instrumentation
        op_scheduler.set_latency_instrumentation_enabled(False)
        op_scheduler.clear_latency_measurements()
        
        scheduled_ir2 = op_scheduler.schedule(mapped_ir)
        latency_report2 = op_scheduler.get_latency_report()
        
        # After clearing, all measurements should be 0
        assert latency_report2.operator_total.last_duration_ns == 0
        
        print(f"  ✅ Operator scheduler latency instrumentation works")
        print(f"     Total time: {latency_report.operator_total.last_duration_ns} ns")
        print(f"     HW grouping: {latency_report.operator_hw_grouping.last_duration_ns} ns")
        print(f"     HW scheduling: {latency_report.operator_hw_scheduling.last_duration_ns} ns")
        print(f"     Dependency resolution: {latency_report.operator_dependency_resolution.last_duration_ns} ns")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Operator scheduler latency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_scheduler_latency():
    """Test latency instrumentation in SystemLevelScheduler"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🏗️  Testing SystemLevelScheduler latency instrumentation...")
        
        # Create test data
        mapped_ir, optimizer = create_test_ir_data()
        
        # Run operator scheduling first
        op_scheduler = rs.OperatorLevelScheduler(optimizer)
        op_scheduled_ir = op_scheduler.schedule(mapped_ir)
        
        # Create system scheduler
        config = rs.DAGSConfig()
        sys_scheduler = rs.SystemLevelScheduler(config)
        
        # Clear any existing measurements
        sys_scheduler.clear_latency_measurements()
        
        # Enable instrumentation
        sys_scheduler.set_latency_instrumentation_enabled(True)
        
        # Run system scheduling
        system_schedule = sys_scheduler.schedule(op_scheduled_ir)
        
        # Get latency report
        latency_report = sys_scheduler.get_latency_report()
        
        # Verify that timing data was collected
        assert latency_report.system_total.last_duration_ns > 0
        assert latency_report.system_dependency_graph.last_duration_ns >= 0
        assert latency_report.system_heuristic_computation.last_duration_ns >= 0
        assert latency_report.system_scheduling_loop.last_duration_ns >= 0
        assert latency_report.system_finalization.last_duration_ns >= 0
        
        print(f"  ✅ System scheduler latency instrumentation works")
        print(f"     Total time: {latency_report.system_total.last_duration_ns} ns")
        print(f"     Dependency graph: {latency_report.system_dependency_graph.last_duration_ns} ns")
        print(f"     Heuristic computation: {latency_report.system_heuristic_computation.last_duration_ns} ns")
        print(f"     Scheduling loop: {latency_report.system_scheduling_loop.last_duration_ns} ns")
        print(f"     Finalization: {latency_report.system_finalization.last_duration_ns} ns")
        
        return True
        
    except Exception as e:
        print(f"  ❌ System scheduler latency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_latency_tracking():
    """Test complete end-to-end latency tracking through the scheduling pipeline"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("🔄 Testing end-to-end latency tracking...")
        
        # Create test data
        mapped_ir, optimizer = create_test_ir_data()
        
        # Step 1: Operator-level scheduling
        op_scheduler = rs.OperatorLevelScheduler(optimizer)
        op_scheduler.set_latency_instrumentation_enabled(True)
        op_scheduled_ir = op_scheduler.schedule(mapped_ir)
        op_latency = op_scheduler.get_latency_report()
        
        # Step 2: System-level scheduling
        config = rs.DAGSConfig()
        sys_scheduler = rs.SystemLevelScheduler(config)
        sys_scheduler.set_latency_instrumentation_enabled(True)
        system_schedule = sys_scheduler.schedule(op_scheduled_ir)
        sys_latency = sys_scheduler.get_latency_report()
        
        # Combine latency reports
        combined_report = rs.SchedulingLatencyReport()
        
        # Copy operator-level timings
        combined_report.operator_hw_grouping = op_latency.operator_hw_grouping
        combined_report.operator_hw_scheduling = op_latency.operator_hw_scheduling
        combined_report.operator_dependency_resolution = op_latency.operator_dependency_resolution
        combined_report.operator_total = op_latency.operator_total
        
        # Copy system-level timings
        combined_report.system_dependency_graph = sys_latency.system_dependency_graph
        combined_report.system_heuristic_computation = sys_latency.system_heuristic_computation
        combined_report.system_scheduling_loop = sys_latency.system_scheduling_loop
        combined_report.system_finalization = sys_latency.system_finalization
        combined_report.system_total = sys_latency.system_total
        
        # Calculate pipeline total
        combined_report.pipeline_total.last_duration_ns = (
            op_latency.operator_total.last_duration_ns + 
            sys_latency.system_total.last_duration_ns
        )
        
        # Generate comprehensive report
        full_report = combined_report.generate_report()
        print("\n" + "="*60)
        print(full_report)
        print("="*60)
        
        # Verify all stages have timing data
        assert combined_report.operator_total.last_duration_ns > 0
        assert combined_report.system_total.last_duration_ns > 0
        assert combined_report.pipeline_total.last_duration_ns > 0
        
        print(f"  ✅ End-to-end latency tracking works")
        print(f"     Pipeline total: {combined_report.pipeline_total.last_duration_ns} ns")
        
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end latency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_latency_report_formatting():
    """Test latency report formatting and duration conversion"""
    try:
        sys.path.insert(0, "build/Scheduler/cpp")
        import rendersim_cpp as rs
        
        print("📋 Testing latency report formatting...")
        
        # Test duration formatting at different scales
        test_cases = [
            (500, "ns"),  # nanoseconds
            (5000, "μs"),  # microseconds
            (5000000, "ms"),  # milliseconds
            (5000000000, "s")  # seconds
        ]
        
        for duration_ns, expected_unit in test_cases:
            formatted = rs.SchedulingLatencyReport.format_duration(duration_ns)
            assert expected_unit in formatted
            print(f"     {duration_ns} ns → {formatted}")
        
        # Test report generation with various timing values
        report = rs.SchedulingLatencyReport()
        
        # Set various realistic timing values
        report.operator_hw_grouping.last_duration_ns = 150000  # 150 μs
        report.operator_hw_scheduling.last_duration_ns = 2500000  # 2.5 ms
        report.operator_dependency_resolution.last_duration_ns = 800000  # 800 μs
        report.operator_total.last_duration_ns = 3450000  # 3.45 ms
        
        report.system_dependency_graph.last_duration_ns = 400000  # 400 μs
        report.system_heuristic_computation.last_duration_ns = 1200000  # 1.2 ms
        report.system_scheduling_loop.last_duration_ns = 8500000  # 8.5 ms
        report.system_finalization.last_duration_ns = 300000  # 300 μs
        report.system_total.last_duration_ns = 10400000  # 10.4 ms
        
        report.pipeline_total.last_duration_ns = 13850000  # 13.85 ms
        
        report_text = report.generate_report()
        
        # Verify report contains expected elements
        assert "RenderSim Scheduling Latency Report" in report_text
        assert "Operator-Level Scheduler:" in report_text
        assert "System-Level Scheduler:" in report_text
        assert "Pipeline Total:" in report_text
        assert "μs" in report_text
        assert "ms" in report_text
        
        print("  ✅ Latency report formatting works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Latency report formatting test failed: {e}")
        return False

def main():
    """Run all latency instrumentation tests"""
    print("🧪 Testing RenderSim C++ Scheduler Latency Instrumentation")
    print("=" * 65)
    
    tests = [
        ("C++ Module Import", test_cpp_imports),
        ("LatencyStats Creation", test_latency_stats_creation),
        ("SchedulingLatencyReport", test_scheduling_latency_report),
        ("Operator Scheduler Latency", test_operator_scheduler_latency),
        ("System Scheduler Latency", test_system_scheduler_latency),
        ("End-to-End Latency Tracking", test_end_to_end_latency_tracking),
        ("Latency Report Formatting", test_latency_report_formatting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("=" * 65)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All latency instrumentation tests passed!")
        print("✅ C++ scheduler latency tracking is working correctly")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 