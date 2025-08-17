#!/usr/bin/env python3
"""
Integration test for MappingEngine with complete RenderSim pipeline

This test validates that the mapping engine integrates correctly with:
- CLI interface
- Scheduling pipeline  
- Example operator graphs
- Hardware configurations
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path

# Add RenderSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cli_mapping_integration():
    """Test mapping engine integration with CLI interface"""
    try:
        print("🔗 Testing CLI mapping integration...")
        
        # Use the sample DAG created earlier
        sample_dag = "examples/sample_dag.pkl"
        if not Path(sample_dag).exists():
            print(f"   ❌ Sample DAG not found: {sample_dag}")
            return False
        
        # Test with each hardware configuration
        hw_configs = [
            "Hardware/examples/hardware_configs/icarus_config.json",
            "Hardware/examples/hardware_configs/neurex_config.json"
        ]
        
        success_count = 0
        
        for hw_config in hw_configs:
            if not Path(hw_config).exists():
                print(f"   ⚠️  Hardware config not found: {hw_config}")
                continue
            
            accelerator = Path(hw_config).stem.replace("_config", "")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            try:
                # Test CLI mapping command
                cmd = [
                    "python", "CLI/main.py", "map",
                    sample_dag, hw_config,
                    "-o", output_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Verify output file was created and has content
                    if Path(output_file).exists() and Path(output_file).stat().st_size > 0:
                        with open(output_file, 'r') as f:
                            try:
                                mapped_data = json.load(f)
                                print(f"     ✅ {accelerator.upper()} CLI mapping successful")
                                success_count += 1
                            except json.JSONDecodeError:
                                print(f"     ❌ {accelerator.upper()} invalid JSON output")
                    else:
                        print(f"     ❌ {accelerator.upper()} no output file generated")
                else:
                    print(f"     ❌ {accelerator.upper()} CLI mapping failed: {result.stderr}")
                    
            finally:
                if Path(output_file).exists():
                    os.unlink(output_file)
        
        if success_count > 0:
            print(f"   ✅ CLI integration successful ({success_count} accelerators)")
            return True
        else:
            print(f"   ❌ CLI integration failed")
            return False
            
    except Exception as e:
        print(f"   ❌ CLI integration test failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline including mapping"""
    try:
        print("🔄 Testing end-to-end pipeline...")
        
        sample_dag = "examples/sample_dag.pkl"
        hw_config = "Hardware/examples/hardware_configs/icarus_config.json"
        
        if not all(Path(f).exists() for f in [sample_dag, hw_config]):
            print("   ❌ Required files not found")
            return False
        
        # Create temporary output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            mapped_file = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            scheduled_file = f.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            report_file = f.name
        
        try:
            # Step 1: Mapping
            map_cmd = ["python", "CLI/main.py", "map", sample_dag, hw_config, "-o", mapped_file]
            result = subprocess.run(map_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"     ❌ Mapping step failed: {result.stderr}")
                return False
            
            print("     ✅ Mapping step completed")
            
            # Step 2: Scheduling
            schedule_cmd = ["python", "CLI/main.py", "schedule", mapped_file, "-o", scheduled_file]
            result = subprocess.run(schedule_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"     ❌ Scheduling step failed: {result.stderr}")
                return False
            
            print("     ✅ Scheduling step completed")
            
            # Step 3: Report generation
            report_cmd = ["python", "CLI/main.py", "report", scheduled_file, "-o", report_file]
            result = subprocess.run(report_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"     ❌ Report step failed: {result.stderr}")
                return False
            
            print("     ✅ Report generation completed")
            
            # Verify all output files exist and have content
            if all(Path(f).exists() and Path(f).stat().st_size > 0 
                   for f in [mapped_file, scheduled_file, report_file]):
                print("   ✅ End-to-end pipeline successful")
                return True
            else:
                print("   ❌ Some output files missing or empty")
                return False
            
        finally:
            # Cleanup
            for f in [mapped_file, scheduled_file, report_file]:
                if Path(f).exists():
                    os.unlink(f)
                    
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        return False

def test_mapping_with_networkx_dag():
    """Test mapping engine with NetworkX DAGs from actual trace data"""
    try:
        print("🕸️  Testing mapping with NetworkX DAGs...")
        
        # Create a NetworkX DAG similar to what nerfstudio would generate
        import pickle
        import networkx as nx
        
        # Create a more complex DAG
        G = nx.DiGraph()
        
        # Add nodes with neural rendering operators
        nodes_data = [
            ('ray_generation', {
                'op_type': 'SAMPLING',
                'input_shape': [1024, 3],
                'output_shape': [1024, 256, 3],
                'flops': 1024 * 256 * 10,
                'memory_bytes': 1024 * 256 * 3 * 4
            }),
            ('hash_encoding', {
                'op_type': 'ENCODING',
                'input_shape': [1024, 256, 3], 
                'output_shape': [1024, 256, 32],
                'flops': 1024 * 256 * 32 * 20,
                'memory_bytes': 1024 * 256 * 32 * 4
            }),
            ('density_mlp', {
                'op_type': 'FIELD_COMPUTATION',
                'input_shape': [1024, 256, 32],
                'output_shape': [1024, 256, 1],
                'flops': 1024 * 256 * 64 * 4,
                'memory_bytes': 1024 * 256 * 64 * 4
            }),
            ('color_mlp', {
                'op_type': 'FIELD_COMPUTATION',
                'input_shape': [1024, 256, 32],
                'output_shape': [1024, 256, 3],
                'flops': 1024 * 256 * 64 * 4,
                'memory_bytes': 1024 * 256 * 64 * 4
            }),
            ('volume_rendering', {
                'op_type': 'BLENDING',
                'input_shape': [1024, 256, 4],
                'output_shape': [1024, 3],
                'flops': 1024 * 256 * 8,
                'memory_bytes': 1024 * 256 * 4 * 4
            })
        ]
        
        for node_id, attrs in nodes_data:
            G.add_node(node_id, **attrs)
        
        # Add edges
        edges = [
            ('ray_generation', 'hash_encoding'),
            ('hash_encoding', 'density_mlp'),
            ('hash_encoding', 'color_mlp'),
            ('density_mlp', 'volume_rendering'),
            ('color_mlp', 'volume_rendering')
        ]
        
        G.add_edges_from(edges)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(G, f)
            temp_dag = f.name
        
        try:
            # Test with CLI
            hw_config = "Hardware/examples/hardware_configs/neurex_config.json"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            try:
                cmd = [
                    "python", "CLI/main.py", "map",
                    temp_dag, hw_config,
                    "-o", output_file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and Path(output_file).exists():
                    print("     ✅ NetworkX DAG mapping successful")
                    return True
                else:
                    print(f"     ❌ NetworkX DAG mapping failed: {result.stderr}")
                    return False
                    
            finally:
                if Path(output_file).exists():
                    os.unlink(output_file)
                    
        finally:
            if Path(temp_dag).exists():
                os.unlink(temp_dag)
            
    except Exception as e:
        print(f"   ❌ NetworkX DAG test failed: {e}")
        return False

def test_mapping_performance():
    """Test mapping engine performance with different graph sizes"""
    try:
        print("⚡ Testing mapping performance...")
        
        from Scheduler.mapping import MappingEngine
        from Scheduler.mapping.hw_config import HWUnit, HWConfig
        from Scheduler.IR import OperatorNode, OperatorGraph, TensorDesc
        import time
        
        # Create hardware config
        units = [
            HWUnit(id=f"unit_{i}", type="GENERIC", throughput=100e6) 
            for i in range(10)
        ]
        hw_config = HWConfig(units=units)
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Test different graph sizes
        graph_sizes = [5, 20, 50, 100]
        
        for size in graph_sizes:
            # Create operator graph of specified size
            nodes = {
                f"op_{i}": OperatorNode(
                    id=f"op_{i}",
                    op_type="GENERIC",
                    inputs=[TensorDesc([1024, 64])],
                    outputs=[TensorDesc([1024, 64])]
                )
                for i in range(size)
            }
            
            op_graph = OperatorGraph(nodes=nodes)
            
            # Measure mapping time
            start_time = time.time()
            mapped_ir = mapping_engine.run(op_graph)
            mapping_time = time.time() - start_time
            
            assert len(mapped_ir.nodes) == size
            
            print(f"     ✅ {size} operators mapped in {mapping_time:.3f}s")
        
        print("   ✅ Performance tests passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def main():
    """Run all mapping integration tests"""
    print("🧪 MappingEngine Integration Tests")
    print("=" * 50)
    
    tests = [
        ("CLI Integration", test_cli_mapping_integration),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
        ("NetworkX DAG Support", test_mapping_with_networkx_dag),
        ("Performance", test_mapping_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print("=" * 50)
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mapping integration tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 