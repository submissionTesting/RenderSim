"""Basic import and instantiation tests for RenderSim components."""

import sys
import os
# Add the parent directory of RenderSim to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_imports():
    """Test that core components can be imported."""
    try:
        from RenderSim.Scheduler.mapping import MappingEngine
        from RenderSim.Scheduler.IR import OperatorGraph, OperatorNode, TensorDesc
        import RenderSim.Scheduler.parse_dag as parse_dag
        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_instantiation():
    """Test that core classes can be instantiated."""
    try:
        from RenderSim.Scheduler.IR import OperatorGraph, OperatorNode, TensorDesc
        
        # Test TensorDesc
        tensor = TensorDesc(shape=[1, 256, 256, 3], dtype="float32")
        assert tensor.shape == [1, 256, 256, 3]
        assert tensor.dtype == "float32"
        
        # Test OperatorNode
        input_tensor = TensorDesc(shape=[1, 3], dtype="float32")
        output_tensor = TensorDesc(shape=[1, 256], dtype="float32")
        node = OperatorNode(id="n0", op_type="HASH_ENCODE", inputs=[input_tensor], outputs=[output_tensor])
        assert node.id == "n0"
        assert node.op_type == "HASH_ENCODE"
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        
        # Test OperatorGraph
        graph = OperatorGraph()
        assert hasattr(graph, 'nodes')
        assert hasattr(graph, 'edges')
        
        print("✓ Basic instantiation successful")
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic RenderSim component tests...")
    
    success = True
    success &= test_imports()
    success &= test_basic_instantiation()
    
    if success:
        print("\n🎉 All basic tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 