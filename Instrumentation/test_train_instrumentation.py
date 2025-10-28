#!/usr/bin/env python
"""Test script to verify training instrumentation is working correctly."""

import sys
from pathlib import Path

def test_imports():
    """Test that all instrumentation modules can be imported."""
    print("Testing imports...")
    
    try:
        # Add nerfstudio to path if needed
        nerfstudio_path = Path(__file__).parent.parent / "nerfstudio"
        if nerfstudio_path.exists():
            sys.path.insert(0, str(nerfstudio_path))
        
        # Test importing the instrumented train script
        from nerfstudio.scripts import train_instrumented
        print("[OK] train_instrumented imported successfully")
        
        # Test importing train utilities
        from nerfstudio.utils import train_utils
        print("[OK] train_utils imported successfully")
        
        # Test importing tracing module
        from nerfstudio.instrumentation import tracing
        print("[OK] tracing module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[FAILED] Import failed: {e}")
        return False


def test_tracer():
    """Test the TrainingTracer class."""
    print("\nTesting TrainingTracer...")
    
    try:
        from nerfstudio.utils.train_utils import TrainingTracer
        
        # Create a tracer
        tracer = TrainingTracer(output_dir=Path("/tmp/test_traces"))
        print("[OK] TrainingTracer created successfully")
        
        # Test starting an iteration
        tracer.start_iteration(100)
        print("[OK] Started tracing iteration 100")
        
        # Test ending an iteration
        tracer.end_iteration()
        print("[OK] Ended tracing iteration")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] TrainingTracer test failed: {e}")
        return False


def test_operator_extraction():
    """Test operator extraction from DAG."""
    print("\nTesting operator extraction...")
    
    try:
        import networkx as nx
        from nerfstudio.utils.train_utils import extract_training_operators
        
        # Create a sample DAG
        dag = nx.DiGraph()
        dag.add_node("mlp_forward", func_name="mlp_forward")
        dag.add_node("loss_compute", func_name="loss_compute")
        dag.add_node("backward_pass", func_name="backward_pass")
        dag.add_node("adam_step", func_name="adam_optimizer_step")
        
        # Extract operators
        result = extract_training_operators(dag)
        
        assert result['statistics']['total_nodes'] == 4
        assert result['statistics']['forward_ops'] >= 1
        assert result['statistics']['backward_ops'] >= 1
        assert result['statistics']['optimizer_ops'] >= 1
        assert result['statistics']['loss_ops'] >= 1
        
        print("[OK] Operator extraction working correctly")
        print(f"  - Forward ops: {result['statistics']['forward_ops']}")
        print(f"  - Backward ops: {result['statistics']['backward_ops']}")
        print(f"  - Optimizer ops: {result['statistics']['optimizer_ops']}")
        print(f"  - Loss ops: {result['statistics']['loss_ops']}")
        
        return True
        
    except Exception as e:
        print(f"[FAILED] Operator extraction failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Training Instrumentation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("TrainingTracer Test", test_tracer),
        ("Operator Extraction Test", test_operator_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "[OK]" if success else "[FAILED]"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[ERROR] Some tests failed. Please check the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
