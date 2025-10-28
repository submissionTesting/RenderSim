# Operators Utils Directory

This directory contains utility modules and test suites for the RenderSim Operators framework.

## Core Utilities

### operator_graph.py
**Purpose**: Core graph data structure for representing neural rendering pipelines.
- Manages operator nodes and their dependencies
- Provides graph traversal and analysis capabilities
- Supports visualization via Graphviz
- Key class: `OperatorGraph`

### unit.py
**Purpose**: Unit conversion and standardization utilities.
- Handles different unit representations (e.g., memory, compute)
- Provides consistent formatting for metrics

### system.py
**Purpose**: System-level utilities for pipeline execution.
- System configuration management
- Resource allocation helpers

### analysis_model.py
**Purpose**: Performance analysis and modeling utilities.
- Roofline model analysis
- Performance prediction helpers

### roofline_plot.py
**Purpose**: Visualization for roofline performance models.
- Generates roofline plots for operators
- Helps identify compute vs. memory bottlenecks

### export_pipeline.py
**Purpose**: Pipeline export utilities.
- Exports pipeline configurations to various formats
- Serialization helpers for pipeline definitions

### export_graph.py
**Purpose**: Graph export and serialization.
- Exports operator graphs to different formats (JSON, DOT, etc.)
- Supports graph interchange between tools

## Test Suites

### test_quick.py
**Purpose**: Quick smoke test for all pipelines.
- **Use case**: Rapid validation during development
- **What it tests**: Basic import and build functionality for all pipelines
- **Runtime**: < 1 second
- **Command**: `python utils/test_quick.py`

### test_new_pipelines.py
**Purpose**: Focused test for the three new training pipelines (GSArch, GBU, Instant3D).
- **Use case**: Verify training pipeline implementation
- **What it tests**: 
  - Backward node presence
  - Backward chain connectivity after blending
- **Command**: `python utils/test_new_pipelines.py`

### test_all_pipelines.py
**Purpose**: Comprehensive test for all pipelines (existing + new).
- **Use case**: Full system validation
- **What it tests**:
  - All existing inference pipelines (ICARUS, NeuRex, CICERO, GSCore, SRender)
  - All new training pipelines (GSArch, GBU, Instant3D)
  - Node counts and structure
  - Key operator presence
- **Command**: `python utils/test_all_pipelines.py`

### test_training_pipelines.py
**Purpose**: Detailed verification of training-specific features.
- **Use case**: Deep validation of backward pass implementation
- **What it tests**:
  - Backward operator support in base classes
  - Pipeline-specific operators (e.g., GradientCompute, FRM, BUM)
  - Backward chain connectivity and path tracing
  - Asymmetric architectures (Instant3D)
  - Row-based processing (GBU)
- **Command**: `python utils/test_training_pipelines.py`

### test_visualization.py
**Purpose**: Generate visual representations of pipeline graphs.
- **Use case**: Debugging and documentation
- **What it does**:
  - Creates PNG visualizations of all pipelines
  - Saves to `pipeline_visualizations/` directory
  - Requires Graphviz installation
- **Command**: `python utils/test_visualization.py`

### test_verify_training.py
**Purpose**: Comprehensive verification of training support implementation.
- **Use case**: Verify complete training infrastructure
- **What it tests**:
  - Base Operator class backward parameter support
  - All operator subclasses backward compatibility
  - GSArch, GBU, Instant3D pipeline implementations
  - Custom operator presence and configuration
  - Backward chain connectivity
- **Command**: `python utils/test_verify_training.py`

## Running Tests

### Quick Validation
```bash
# From Operators directory
python utils/test_quick.py
```

### Full Test Suite
```bash
# Test all pipelines
python utils/test_all_pipelines.py

# Test only training pipelines in detail
python utils/test_training_pipelines.py

# Test only the three new pipelines
python utils/test_new_pipelines.py
```

### Generate Visualizations
```bash
# Requires graphviz: sudo apt-get install graphviz
python utils/test_visualization.py
```

## Test Coverage

| Pipeline | Inference | Training | Backward Chain | Visualization |
|----------|-----------|----------|----------------|---------------|
| ICARUS   | Yes       | -        | -              | Yes           |
| NeuRex   | Yes       | -        | -              | Yes           |
| CICERO   | Yes       | -        | -              | Yes           |
| GSCore   | Yes       | -        | -              | Yes           |
| SRender  | Yes       | -        | -              | Yes           |
| GSArch   | Yes       | Yes      | Yes            | Yes           |
| GBU      | Yes       | Yes      | Yes            | Yes           |
| Instant3D| Yes       | Yes      | Yes            | Yes           |

## Key Features Tested

### Training Pipeline Features
- **GSArch**: GradientCompute, GradientPruning, Rearrangement, TileMerging
- **GBU**: RowProcessing, RowGeneration, DecompBinning
- **Instant3D**: FRM (Feed-Forward Read Mapper), BUM (Back-Propagation Update Merger)

### Backward Pass Verification
- All operators support `backward=True` parameter
- Backward nodes are labeled with "(B)" suffix
- Backward chains properly connect after forward blending/rendering
- Operator-specific backward methods (get_backward_num_ops, etc.)

## Dependencies

- Python 3.6+
- Graphviz (optional, for visualization)
  - Ubuntu/Debian: `sudo apt-get install graphviz`
  - MacOS: `brew install graphviz`
  - Python package: `pip install graphviz`

## Notes

- All test scripts are self-contained and handle import paths automatically
- Tests can be run from any directory within the project
- Visualization tests will gracefully handle missing Graphviz installation
- Test results include both success indicators (Yes/No) and detailed error messages
