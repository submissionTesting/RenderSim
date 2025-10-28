# RenderSim Scheduler Test Suite

This directory contains comprehensive tests for the RenderSim Scheduler components, including support for training pipelines (GSArch, GBU, Instant3D) and backward pass operations.

## Test Structure

### Core Test Files

#### `test_training_pipelines.py`
Tests the integration of training-specific pipelines with the Scheduler:
- **GSArch Pipeline**: Validates gradient-aware 3DGS training scheduling
- **GBU Pipeline**: Tests row-based Gaussian bundle processing
- **Instant3D Pipeline**: Verifies asymmetric FRM/BUM architecture scheduling
- **Backward Operator Mapping**: Ensures correct mapping of backward pass operators

#### `test_mapping.py`
Tests the mapping engine functionality:
- **Backward Operator Mapping**: Validates "(B)" suffix detection and mapping
- **Training Pipeline Mapping**: Tests GSArch, GBU, and Instant3D specific operators
- **Fallback Mappings**: Verifies fallback strategies for unmapped operators

#### `test_scheduler.py`
Tests scheduling algorithms and optimizations:
- **Operator-Level Scheduling**: Tests local optimization within hardware modules
- **System-Level Scheduling**: Validates DAGS (Dependency-Aware Greedy Scheduler)
- **Training Optimizations**: Tests gradient pruning, tile merging, row processing
- **Performance Model**: Validates latency, throughput, power calculations

### Validation Test Files

#### `validate_simple.py`
Simple validation script to verify basic functionality:
- Tests core imports work correctly
- Validates optimization library loads
- Checks hardware configurations exist
- Verifies Equation 1 implementation

#### `minimal_test.py`
Minimal test that writes results to file:
- Tests basic Python environment
- Validates all key imports
- Writes results to `test_results.txt` for debugging

## Running Tests

### Run All Tests
```bash
cd Scheduler/tests
python -m pytest  # If pytest is installed
# OR
python test_training_pipelines.py
python test_mapping.py
python test_scheduler.py
```

### Run Individual Test Suites
```bash
# Test training pipeline integration
python test_training_pipelines.py

# Test mapping engine
python test_mapping.py

# Test scheduler components
python test_scheduler.py
```

### Run Specific Tests
```python
# Within Python
from test_training_pipelines import test_gsarch_pipeline
test_gsarch_pipeline()
```

## Test Coverage

### Training Pipeline Support
- GSArch: TileMerging, GradientCompute, GradientPruning, Rearrangement
- GBU: RowProcessing, RowGeneration, DecompBinning
- Instant3D: FeedForwardReadMapper (FRM), BackpropUpdateMerger (BUM)

### Backward Pass Support
- Operator suffix detection: "(B)" marking
- Backward mapping strategies
- Gradient flow validation
- Training-specific optimizations

### Hardware Mapping
- Direct operator-to-hardware mapping
- Fallback mapping strategies
- Training-specific hardware modules
- Resource allocation and constraints

### Scheduling Algorithms
- Operator-level scheduling with optimizations
- System-level DAGS scheduling
- Dependency resolution
- Critical path analysis

## Expected Output

### Successful Test Run
```
============================================================
Testing Scheduler Integration with Training Pipelines
============================================================

=== Testing GSArch Pipeline ===
Built GSArch pipeline with 14 nodes
Found 7 backward nodes
Mapped 14 operators to hardware
  TILEMERGING: Found
  GRADIENTCOMPUTE: Found
  GRADIENTPRUNING: Found
  REARRANGEMENT: Found
GSArch test: PASSED

=== Testing GBU Pipeline ===
Built GBU pipeline with 12 nodes
Found 5 backward nodes
Mapped 12 operators to hardware
  ROWPROCESSING: Found
  ROWGENERATION: Found
  DECOMPBINNING: Found
GBU test: PASSED

=== Testing Instant3D Pipeline ===
Built Instant3D pipeline with 10 nodes
Found 4 backward nodes
Mapped 10 operators to hardware
  FRM: Found
  BUM: Found
  HASHENCODING: Found
Instant3D test: PASSED

============================================================
TEST SUMMARY
============================================================
GSArch               PASSED
GBU                  PASSED
Instant3D            PASSED
Backward Mapping     PASSED

All tests PASSED!
```

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
1. You're running from the correct directory
2. The parent directories are in the Python path
3. The Operators module is accessible

### Hardware Config Not Found
Ensure hardware configuration files exist in:
```
Hardware/examples/hardware_configs/
  - gsarch_config.json
  - gbu_config.json
  - instant3d_config.json
```

### C++ Module Not Available
The C++ scheduler module is optional. Tests will still pass if it's not built:
```
C++ scheduler module not built (this is expected)
```

To build the C++ module:
```bash
cd Scheduler/cpp
mkdir build && cd build
cmake ..
make
```

## Adding New Tests

To add new tests, follow this template:

```python
def test_new_feature():
    """Test description."""
    print("\n=== Testing New Feature ===")
    
    try:
        # Test implementation
        result = perform_test()
        
        # Validation
        assert result == expected_value
        
        print("New feature test: PASSED")
        return True
        
    except Exception as e:
        print(f"New feature test: FAILED - {e}")
        return False
```

Then add the test to the main() function in the appropriate test file.

## Dependencies

- Python 3.6+
- RenderSim Operators module
- RenderSim Hardware configurations
- Optional: pytest for advanced testing features
- Optional: C++ scheduler module for full functionality

## Notes

- Tests are designed to be independent and can run in any order
- Each test validates a specific aspect of the Scheduler
- Tests use realistic pipeline configurations from research papers
- Performance tests use representative workloads
