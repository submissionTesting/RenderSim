# Instrumentation

# Operator

- Modified `Operators/operators` base classes
    - Added backward pass support via `backward` parameter
    - Implemented `get_backward_*` methods for training
- Added training pipelines under `Operators/pipelines`
    - GSArch: TileMerging, GradientCompute, GradientPruning operators
    - GBU: RowProcessing, RowGeneration, DecompBinning operators
    - Instant3D: FeedForwardReadMapper (FRM), BackpropUpdateMerger (BUM) operators
- Reorganized tests to `Operators/utils` with comprehensive validation
- Updated visualization tools (plot_operator.py, plot_roofline.py) for training pipelines 

# Hardware

- Added `Hardware/specification.md` for details
- Added GSArch hardware modules
    - Blending: Sorting, Tile Merging, Feature Computing, Gradient Computing, Gradient Pruning, Rearrangement
- Added GBU hardware modules
    - Blending: Row Processing Element, Row Generation Engine, Decomposition and Binning Engine
- Added Instant3d hardware modules
    - FieldCompute: MLP
    - Encode: RFM, BUM

# Instrumentation

- Added training instrumentation support (`Instrumentation/nerfstudio_vendor/scripts/train.py`)
    - Collect execution traces during training iterations
    - Configurable iteration-specific tracing
    - Training utilities for operator extraction and analysis
- Enhanced documentation with training trace collection examples

# Scheduler

- Enhanced mapping engine (`Scheduler/mapping`)
    - Added backward operator detection via "(B)" suffix
    - Implemented fallback mappings for training operators
- Added optimization library (`Scheduler/op_sched`)
    - Implemented Equation 1 from paper for duration calculation
    - Added training-specific optimizations (gradient pruning, tile merging, etc.)
    - Created 3D optimization framework (Type, Scope, Criteria)
- Updated C++ implementation (`Scheduler/cpp`)
    - Extended DAGS algorithm for training workloads
    - Added training-aware system scheduling
- Created comprehensive test suite in `Scheduler/tests`
- Integrated GSArch, GBU, and Instant3D pipeline support