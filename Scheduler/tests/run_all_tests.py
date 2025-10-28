#!/usr/bin/env python3
"""
Run all Scheduler tests
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Operators"))

def run_test_file(test_file, test_name):
    """Run a single test file and return results."""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print('='*60)
    
    try:
        # Import and run the test
        module_name = test_file.stem
        exec(f"from {module_name} import main")
        result = locals()['main']()
        return result == 0
    except Exception as e:
        print(f"Error running {test_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test files in the tests directory."""
    print("="*60)
    print("RenderSim Scheduler - Complete Test Suite")
    print("="*60)
    
    test_dir = Path(__file__).parent
    test_files = [
        (test_dir / "test_training_pipelines.py", "Training Pipeline Tests"),
        (test_dir / "test_mapping.py", "Mapping Engine Tests"),
        (test_dir / "test_scheduler.py", "Scheduler Component Tests"),
    ]
    
    results = []
    
    for test_file, test_name in test_files:
        if test_file.exists():
            passed = run_test_file(test_file, test_name)
            results.append((test_name, passed))
        else:
            print(f"Warning: Test file not found: {test_file}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("OVERALL TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:30} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print("SOME TESTS FAILED!")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
