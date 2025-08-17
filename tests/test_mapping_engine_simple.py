#!/usr/bin/env python3
"""
Simplified unit tests for RenderSim MappingEngine that work with current imports

This tests the core mapping functionality without complex pytest setup.
"""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add RenderSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that mapping engine can be imported"""
    try:
        from Scheduler.mapping import MappingEngine
        from Scheduler.mapping.hw_config import HWUnit, HWConfig, load_hw_config
        from Scheduler.IR import OperatorNode, OperatorGraph, TensorDesc, MappedIR
        print("✅ MappingEngine imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_hw_config_creation():
    """Test hardware configuration creation"""
    try:
        from Scheduler.mapping.hw_config import HWUnit, HWConfig
        
        units = [
            HWUnit(id="mlp0", type="FIELD_COMPUTATION", throughput=128e6),
            HWUnit(id="encoder0", type="ENCODING", throughput=64e6)
        ]
        hw_config = HWConfig(units=units)
        
        assert len(hw_config.units) == 2
        assert hw_config.units[0].id == "mlp0"
        
        units_by_type = hw_config.units_by_type()
        assert "FIELD_COMPUTATION" in units_by_type
        assert "ENCODING" in units_by_type
        
        print("✅ Hardware configuration creation works")
        return True
    except Exception as e:
        print(f"❌ Hardware config test failed: {e}")
        return False

def test_hw_config_loading():
    """Test loading hardware config from JSON"""
    try:
        from Scheduler.mapping.hw_config import load_hw_config
        
        config_data = {
            "hw_units": [
                {"id": "test_unit", "type": "ENCODING", "throughput": 64e6, "memory_kb": 128}
            ]
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            hw_config = load_hw_config(temp_path)
            assert len(hw_config.units) == 1
            assert hw_config.units[0].id == "test_unit"
            print("✅ Hardware configuration loading works")
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Hardware config loading test failed: {e}")
        return False

def test_operator_graph_creation():
    """Test operator graph creation"""
    try:
        from Scheduler.IR import OperatorNode, OperatorGraph, TensorDesc
        
        node = OperatorNode(
            id="test_op",
            op_type="ENCODING",
            inputs=[TensorDesc([1024, 64])],
            outputs=[TensorDesc([1024, 32])]
        )
        
        graph = OperatorGraph(nodes={"test_op": node})
        assert len(graph.nodes) == 1
        assert "test_op" in graph.nodes
        
        print("✅ Operator graph creation works")
        return True
    except Exception as e:
        print(f"❌ Operator graph test failed: {e}")
        return False

def test_basic_mapping():
    """Test basic mapping functionality"""
    try:
        from Scheduler.mapping import MappingEngine
        from Scheduler.mapping.hw_config import HWUnit, HWConfig
        from Scheduler.IR import OperatorNode, OperatorGraph, TensorDesc
        
        # Create hardware config
        units = [
            HWUnit(id="encoder0", type="ENCODING", throughput=80e6),
            HWUnit(id="mlp0", type="FIELD_COMPUTATION", throughput=128e6),
            HWUnit(id="renderer0", type="BLENDING", throughput=64e6)
        ]
        hw_config = HWConfig(units=units)
        
        # Create operator graph
        nodes = {
            "encoding": OperatorNode(
                id="encoding",
                op_type="ENCODING", 
                inputs=[TensorDesc([1024, 3])],
                outputs=[TensorDesc([1024, 63])]
            ),
            "mlp": OperatorNode(
                id="mlp",
                op_type="FIELD_COMPUTATION",
                inputs=[TensorDesc([1024, 63])], 
                outputs=[TensorDesc([1024, 4])]
            ),
            "rendering": OperatorNode(
                id="rendering",
                op_type="BLENDING",
                inputs=[TensorDesc([1024, 4])],
                outputs=[TensorDesc([1024, 3])]
            )
        }
        
        edges = [("encoding", "mlp"), ("mlp", "rendering")]
        op_graph = OperatorGraph(nodes=nodes, edges=edges)
        
        # Run mapping
        mapping_engine = MappingEngine(hw_config=hw_config)
        mapped_ir = mapping_engine.run(op_graph)
        
        # Verify mapping
        assert len(mapped_ir.nodes) == 3
        assert mapped_ir.nodes["encoding"].hw_unit == "encoder0"
        assert mapped_ir.nodes["mlp"].hw_unit == "mlp0"
        assert mapped_ir.nodes["rendering"].hw_unit == "renderer0"
        assert mapped_ir.edges == edges
        
        print("✅ Basic mapping functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Basic mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mapping_error_handling():
    """Test mapping error handling for unsupported operators"""
    try:
        from Scheduler.mapping import MappingEngine
        from Scheduler.mapping.hw_config import HWUnit, HWConfig
        from Scheduler.IR import OperatorNode, OperatorGraph, TensorDesc
        
        # Create limited hardware config (no GENERIC fallback)
        units = [HWUnit(id="encoder0", type="ENCODING", throughput=80e6)]
        hw_config = HWConfig(units=units)
        
        # Create operator with unsupported type
        nodes = {
            "unknown": OperatorNode(
                id="unknown",
                op_type="UNSUPPORTED_TYPE",
                inputs=[TensorDesc([1024, 64])],
                outputs=[TensorDesc([1024, 32])]
            )
        }
        
        op_graph = OperatorGraph(nodes=nodes)
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Should raise RuntimeError
        try:
            mapping_engine.run(op_graph)
            print("❌ Expected RuntimeError but mapping succeeded")
            return False
        except RuntimeError as e:
            if "No HW unit found" in str(e):
                print("✅ Error handling works correctly")
                return True
            else:
                print(f"❌ Wrong error message: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_factory_method():
    """Test MappingEngine.from_json factory method"""
    try:
        from Scheduler.mapping import MappingEngine
        
        config_data = {
            "hw_units": [
                {"id": "test_unit", "type": "ENCODING", "throughput": 64e6, "memory_kb": 128}
            ]
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            mapping_engine = MappingEngine.from_json(temp_path)
            assert len(mapping_engine.hw_config.units) == 1
            assert mapping_engine.hw_config.units[0].id == "test_unit"
            
            print("✅ Factory method works")
            return True
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ Factory method test failed: {e}")
        return False

def main():
    """Run all mapping engine tests"""
    print("🧪 Running MappingEngine Unit Tests (Simplified)")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_hw_config_creation,
        test_hw_config_loading,
        test_operator_graph_creation,
        test_basic_mapping,
        test_mapping_error_handling,
        test_factory_method
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All MappingEngine tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 