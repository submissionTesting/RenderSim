# RenderSim Modular Hardware Library Specification

This document specifies the modular hardware libraries under `Hardware/A1_cmod`, the supported modules, their interface contracts (I/O datatypes), configurable architectural parameters, and legal (portable) HLS optimizations. All modules share consistent HLS coding rules and type packs so they can be composed across accelerators.

## Module summary

| Category | Modules |
| --- | --- |
| [Sampling](#sampling) | [CullingConversionUnit](#cullingconversionunit), [Skipping Controller](#skippingcontroller), [Sampling Unit](#samplingunit), [HAMAT](#hamat), [RFM](#rfm) |
| [Encoding](#encoding-instant3d) | [FRM](#frm), [BUM](#bum), [AG](#ag), [IGU](#igu), [ICU](#icu), [DCU](#dcu), [CU](#cu), [SPE](#spe), [PEU](#peu), [Reducer](#reducer) |
| [Field Compute](#field-compute) | [NPU variants](#npu), [Adder Tree](#adder-tree), [MLP_vanilla](#mlp-vanilla), [MLP_ssa](#mlp-ssa), [MLP_monb](#mlp-monb), [MLP_sonb](#mlp-sonb), [MLP_block](#mlp-block), [PE](#pe), [PE_array](#pe-array), [fixedpoint_mul](#fixedpoint-mul) |
| [Blending](#blending--gsarchgbu) | [Sorting](#sorting--bitonic), [BSU](#bsu), [QSU](#qsu), [TileMerging](#tilemerging), [FeatureCompute](#featurecompute-vru-stage), [VRU variant 1](#vru_v1), [VRU variant 2](#vru_v2), [VRU variant 3](#vru_v3), [VRU variant 4](#vru_v4), [GradientCompute](#gradientcompute), [GradientPruning](#gradientpruning), [Rearrangement](#rearrangement-ru), [RowProcessing](#rowprocessing-gbu-row-pe), [RowGeneration](#rowgeneration), [DecompBinning](#decompbinning), [PE Arrays](#pe-arrays), [IE/UNIIE/PRU](#ie-uniie-pru), [IE](#ie), [UNIIE](#uniie) |

| Utilities/Test | [SRAMTest](#sramtest) |


## Module catalog by taxonomy

Below, each module lists I/O, core behavior, key parameters, and legal optimizations.

### Sampling

<a id="sampling"></a>

<a id="cullingconversionunit"></a>
1) CullingConversionUnit (`A1_cmod/Sample/CullingConversionUnit`) · [Back to summary](#module-summary-click-to-jump)
- <a id="hamat"></a>
1.3) HAMAT (`A1_cmod/Sample/HAMAT`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `BSU_IN_OUT_TYPE`; Out `BSU_IN_OUT_TYPE`.
- Function: Bitonic sort over an n‑element vector in IRIS pack; educational/sample module.
- Params: `SORT_NUM`, datatype per `IRISPackDef.h`.
- Optimizations: Pipeline at inner compare‑swap loop; II=1 per stage.

- <a id="rfm"></a>
1.4) RFM (`A1_cmod/Sample/RFM`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `RFM_IN_OUT_TYPE` X/Y plus mode; Out `RFM_IN_OUT_TYPE`.
- Function: Reference functional module (placeholder arithmetic kernel) used in legacy IRIS samples.
- Params: Mode enum; numeric precision from IRIS pack.
- Optimizations: Simple streaming; II=1.
- I/O: In `CameraParams`, `Stage1a_In`; Out `Stage1f_Out`.
- Function: View transform → near/far cull → project to uv → screen cull; emits per‑pixel meta (downstream VRU composes color).
- Params: Numeric aliases (FP16 default, fixed‑point available). Camera near/far/intrinsics supplied at runtime.
- Optimizations: II tuning; reciprocal via PWL or polynomial; optional LUTs; channel depths.

- <a id="skippingcontroller"></a>
1.1) Skipping Controller (optional) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `Stage1a_In` plus optional scene/ROI metadata; Out gated `Stage1a_In` (or tags) for downstream.
- Function: Early discard of samples based on ROI/frustum/heuristics to reduce load before projection.
- Params: Visibility/ROI thresholds; enable flags; precision aliases as in Sample pack.
- Optimizations: Pure control‑plane gating; II=1; channel buffering for burstiness.

- <a id="samplingunit"></a>
1.2) Sampling Unit (optional) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `Stage1a_In` (or post‑CCU meta); Out scheduled/gathered samples for encode.
- Function: Prioritizes/strides samples (e.g., strided rows or importance sampling) while preserving ordering constraints.
- Params: Stride, priority weights, maximum outstanding samples.
- Optimizations: Small FIFOs for reordering; II=1; optional priority heap (resource‑shared comparator).

### Encoding (Instant3D)

<a id="encoding-instant3d"></a>

<a id="frm"></a>
2) FRM – Feed‑Forward Read Mapper (`A1_cmod/Encode/FRM`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `FRM_IN_TYPE { f:FP16, addr:u16 }`; Out `FRM_OUT_TYPE { g:FP16, addr:u16 }`.
- Function: Maps requests to banks by `addr & (NUM_BANKS-1)`, accepts one per bank per window, then commits a dense sequence (banks 0..N-1).
- Params: `NUM_BANKS` (default 8); commit window size/hash policy (compile‑time knobs).
- Optimizations: Window size/hash variants; II tuning; resource sharing for detectors; one commit per channel per cycle.

<a id="bum"></a>
3) BUM – Back‑Propagation Update Merger (`A1_cmod/Encode/BUM`) · [Back to summary](#module-summary-click-to-jump)
<a id="ag"></a>
4) AG – Address/VID Generator (`A1_cmod/Encode/AG`) · [Back to summary](#module-summary-click-to-jump)
- I/O (CICERO pack: `Encode/include_v1/CICEROPack_Def.h`): In `AG_VIDsWs` (vector of IDs and weights); Out `AG_VID`, `AG_W`.
- Function: For each input request, emits a stream of 8 VID/weight pairs (one per cycle) used to address downstream tables.
- Params: Vector length (8 by default), VID/weight widths.
- Optimizations: II=1; unroll factor for inner emit loop; FIFO depth between mem interface and emit.

<a id="igu"></a>
5) IGU – Index Generation Unit (`A1_cmod/Encode/IGU`) · [Back to summary](#module-summary-click-to-jump)
- I/O (NEUREX pack: `Encode/include_v3/NEUREXPackDef.h`): In `IGU_Grid_Res` (resolution), `IGU_In_Type` (position); Out `IGU_Grid_Res` (grid_id), `Hashed_addr` (8 hashed neighbors), `IGU_Weight` (8 interpolation weights).
- Function: Scales position by grid resolution; computes lower/frac parts; enumerates 8 cell corners; hashes to table indices; computes trilinear weights.
- Params: Hash constants (P1, P2), table size mask, dimensionality (fixed 3D), numeric precision.
- Optimizations: Loop unrolling across 8 corners; resource sharing in hash and weight MACs; II tuning.

<a id="icu"></a>
6) ICU – Index Computation Unit (`A1_cmod/Encode/ICU`) · [Back to summary](#module-summary-click-to-jump)
- I/O (NEUREX/SRENDER packs depending on version): typical inputs are cell coordinates/attrs; outputs per‑point indices for table access.
- Function: Deterministic mapping from local coordinates to linearized table addresses; often shared with IGU.
- Params: Strides per dimension; table extents; precision.
- Optimizations: Strength reduction; II=1; optional unroll over small dims.

<a id="dcu"></a>
7) DCU – Distance Compute Unit (`A1_cmod/Encode/DCU`) · [Back to summary](#module-summary-click-to-jump)
- I/O (SRENDER pack: `Encode/include_v4/SRENDERPackDef.h`): In `DCU_IN_TYPE { informative_pixel:RGB, candidate_pixel:RGB }`; Out `DCU_OUT_TYPE { distance }`.
- Function: Per‑channel diff → abs (sign detect + mux) → sum to L1 distance.
- Params: RGB precision; thresholding left to caller.
- Optimizations: Channel‑parallel unroll (3‑way); II=1.

<a id="cu"></a>
8) CU – Comparison Unit (`A1_cmod/Encode/CU`) · [Back to summary](#module-summary-click-to-jump)
- I/O (SRENDER pack): In `{ input_value, threshold, id }`; Out `{ result, id }`.
- Function: Single compare (≥ or >) with ID pass‑through; used for sensitivity predicates.
- Params: Predicate selection (≥/>) compile‑time; precision.
- Optimizations: Fully combinational; II=1.

<a id="spe"></a>
9) SPE – Sensitivity Prediction Engine (`A1_cmod/Encode/SPE`) · [Back to summary](#module-summary-click-to-jump)
- I/O (SRENDER pack): In `SPE_IN_TYPE` (rendered pixels, thresholds, counts); Out `SPE_OUT_TYPE` (sensitive rays and points with boundaries).
- Function: Color‑based ray prediction via DCU+CU over 3×3 neighborhoods; weight‑factor‑based point prediction via CU. Instantiates arrays of DCU/CU sub‑modules for throughput.
- Params: Parallelism degrees (e.g., 16 DCU, 16 CU for rays, 48 CU for points); thresholds T/D; maximum rays/points.
- Optimizations: Module replication counts; channel depths; II=1 across orchestration thread.

<a id="peu"></a>
10) PEU – Position Encoding Unit (`A1_cmod/Encode/PEU`) · [Back to summary](#module-summary-click-to-jump)
- I/O (ICARUS pack: `Encode/include_v2/ICARUSPackDef.h`): In `PEU_In_Type (x,y,z)`; Out `PEU_Out_Type` (2×N features: cos/sin of projected inputs).
- Function: MatVec (frequency projection) → sin/cos (CORDIC) to form 2N‑dim encoding (e.g., 256 dims from N=128).
- Params: Input dim (3), output dim (2N), projection kernel (fixed vs programmable), CORDIC iterations.
- Optimizations: Unroll across N; share CORDICs; II tuning; stream decomposition.

<a id="reducer"></a>
11) Reducer / Tree Reducer (`A1_cmod/Encode/reducer`) · [Back to summary](#module-summary-click-to-jump)
- I/O: Tree‑style reduction inputs/outputs defined in local pack.
- Function: Parallel reduction (sum/min/max) for partial features or losses.
- Params: Fan‑in, tree depth, op (sum/min/max), precision.
- Optimizations: Balanced tree vs pipelined array; resource sharing of adders; II=1.
- I/O: In `BUM_IN_TYPE { addr:u16, grad:FP16, last }`; Out `BUM_OUT_TYPE { addr:u16, upd:FP16 }`.
- Function: One‑to‑all match in 16‑entry table; merge `grad` on match; allocate on miss; on `last`, commit entries one per cycle (apply learning‑rate scale) and clear.
- Params: `NUM_ENTRIES` (default 16); learning rate constant/port.
- Optimizations: CAM vs serialized match; commit stride (respect channel contract); II tuning.

### Field Compute

<a id="field-compute"></a>

<a id="npu"></a>
4) NPU (MLP engines) variants (`A1_cmod/FieldComp/NPU_*`) · [Back to summary](#module-summary-click-to-jump)
- <a id="mlp-vanilla"></a>
4.2) MLP_vanilla (`A1_cmod/FieldComp/MLP_vanilla`) · [Back to summary](#module-summary-click-to-jump)
- I/O: `MLP_In_Type` → `MLP_Out_Type` (per ICARUS pack).
- Function: Baseline dense MLP pipeline; software‑like ordering with streaming stages.
- Params: Layer dims; activation; precision per pack.
- Optimizations: Stage pipelining; resource sharing of MACs; II tuning.

- <a id="mlp-ssa"></a>
4.3) MLP_ssa (`A1_cmod/FieldComp/MLP_ssa`) · [Back to summary](#module-summary-click-to-jump)
- Function: Single‑shape array MLP; demonstrates systolic accumulation flow.
- Params/Optimizations: As above.

- <a id="mlp-monb"></a>
4.4) MLP_monb (`A1_cmod/FieldComp/MLP_monb`) · [Back to summary](#module-summary-click-to-jump)
- Function: Multi‑Output Network Block producer side; writes intermediate activations.
- Params/Optimizations: Block size, sample pipelining.

- <a id="mlp-sonb"></a>
4.5) MLP_sonb (`A1_cmod/FieldComp/MLP_sonb`) · [Back to summary](#module-summary-click-to-jump)
- Function: Single‑Output Network Block consumer side; reads intermediate activations to produce final outputs.
- Params/Optimizations: Matches MONB settings; op‑balanced loops.

- <a id="mlp-block"></a>
4.6) MLP_block (`A1_cmod/FieldComp/MLP_block`) · [Back to summary](#module-summary-click-to-jump)
- I/O: `MLP_In_Type` → `MLP_Out_Type` with internal `MemReq` for weight init.
- Function: Two‑layer block (256→256, 256→4) demo; includes init path to RAMs.
- Params: Block size, dims.
- Optimizations: Tiled matmul; on‑chip buffering of activations.

- <a id="pe"></a>
4.7) PE (`A1_cmod/FieldComp/PE`) · [Back to summary](#module-summary-click-to-jump)
- I/O: Pixel, Gaussian features, enables → partial output.
- Function: Per‑pixel Gaussian blend micro‑kernel with three staged threads.
- Params: Covariance precision; color width; control signals.
- Optimizations: Stage FIFOs; cordic/PWL for exp; II=1.

- <a id="pe-array"></a>
4.8) PE_array (`A1_cmod/FieldComp/PE_array`) · [Back to summary](#module-summary-click-to-jump)
- I/O: Array input/output wrapping N `PE` instances.
- Function: Broadcast Gaussian and control; gather PE results.
- Params: `NUM_PEs`.
- Optimizations: Broadcast/gather pipelines; II=1 per PE.

- <a id="fixedpoint-mul"></a>
4.9) fixedpoint_mul (`A1_cmod/FieldComp/fixedpoint_mul`) · [Back to summary](#module-summary-click-to-jump)
- I/O: Fixed‑point operands; Out product.
- Function: Standalone test of fixed‑point multiply and type interactions.
- Params: Word length, frac bits.
- Optimizations: None needed; II=1.
- I/O: As defined in the respective include packs.
- Function: On‑chip MLP compute (SSA/monolithic/systolic). Activation via PWL or LUT.
- Params: Array sizes (PE count, layer dims), activation method, precision.
- Optimizations: Unroll per layer; systolic tiling; PWL vs LUT; II tuning.

- <a id="adder-tree"></a>
4.1) Adder Tree / Reduction Core (generic) · [Back to summary](#module-summary-click-to-jump)
- I/O: In vector stream (e.g., partial sums/activations); Out reduced value (sum/min/max) or reduced vector.
- Function: Balanced or pipelined tree that reduces fan‑in (e.g., 8/16/32) to scalar/vector outputs; used behind MLPs and encoders.
- Params: Fan‑in (power of two preferred), op (sum/min/max), internal precision.
- Optimizations: Tree unroll factor; pipeline register insertion per level; resource sharing for ops; II=1.

### Blending – GSArch/GBU

<a id="blending--gsarchgbu"></a>

<a id="sorting--bitonic"></a>
5) Sorting – Bitonic (`A1_cmod/Blending/Sorting`) · [Back to summary](#module-summary-click-to-jump)
- <a id="bsu"></a>
5.0) BSU – Bitonic Sort Unit (`A1_cmod/Blending/BSU`) · [Back to summary](#module-summary-click-to-jump)
- I/O: `BSU_IN_OUT_TYPE` (per older packs).
- Function: Classic bitonic sort sample used in IRIS/GAURAST demos.
- Params: `SORT_NUM`.
- Optimizations: Stage pipelining; II=1.

<a id="qsu"></a>
5.1) QSU – Quick Sort Unit (`A1_cmod/Blending/QSU`) · [Back to summary](#module-summary-click-to-jump)
- I/O: Same contract as Sorting unless configured for streaming mode (then key/payload streams). Default uses `SortingVec` with parallel key/id arrays.
- Function: Quicksort‑based total order or partition (top‑k) depending on configuration. Provides lower area for moderate `SORT_NUM` when full bitonic is unnecessary.
- Params: `SORT_NUM`, top‑k enable, pivot policy (middle/median‑of‑three), precision.
- Optimizations: Tail recursion removal (iterative stack), loop unroll on partition compare/swap, II=1.

- I/O: `SortingVec { key[SORT_NUM], id[SORT_NUM] }`.
- Function: Bitonic network on `key`; payload alignment by parallel arrays.
- Params: `SORT_NUM`, `SORT_ASCENDING`, key/id widths.
- Optimizations: Stage unroll/resource sharing; precision.

<a id="tilemerging"></a>
6) TileMerging (`A1_cmod/Blending/TileMerging`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `TM_IndexVec` ×2; Out `TM_HCOut { hot_count, hot_id[], hot_addr[], cold_count, cold[] }`.
- Function: Frequency merge of per‑tile IDs; hot/cold classify; hot address allocation; optional SRAM model via `TM_SRAM`.
- Params: `TM_NUM_ENTRIES`, `TM_NUM_BANKS`, `TM_HOT_THRESH`, `TM_LIST_LEN`.
- Optimizations: FSM staging; banking/hash; early exit if hot_count=0; II tuning.

<a id="featurecompute-vru-stage"></a>
7) FeatureCompute (VRU stage) (`A1_cmod/Blending/FeatureCompute`) · [Back to summary](#module-summary-click-to-jump)
<a id="vru_v1"></a>
7.0) VRU variant 1 (`A1_cmod/Blending/VRU_v1`) · [Back to summary](#module-summary-click-to-jump)
- Function: Multi‑stage VRU with explicit inter‑stage channels; FP32 in GAURAST pack example.
- Notes: Educational variant demonstrating stage factoring and exp PWL usage.

<a id="vru_v2"></a>
7.1) VRU variant 2 (`A1_cmod/Blending/VRU_v2`) · [Back to summary](#module-summary-click-to-jump)
- I/O: `VRU_IN_TYPE` → `VRU_OUT_TYPE` from the GSArch pack.
- Function: Same math as FeatureCompute with a two/three‑stage pipeline split (alpha/T/render) and explicit Connections channels between stages.
- Params: Stage partition, exp implementation (PWL/poly), thresholds for early exit; precision.
- Optimizations: Inter‑stage FIFO depth; unroll in color accum; II=1 per stage.

<a id="vru_v3"></a>
7.2) VRU_v3 (`A1_cmod/Blending/VRU_v3`) · [Back to summary](#module-summary-click-to-jump)
- Function: Variant focusing on tighter coupling between alpha and T update, reducing register pressure; same I/O and op counts.
- Params/Optimizations: As in VRU_v2; can share common sub‑expressions across stages.

<a id="vru_v4"></a>
7.3) VRU_v4 (`A1_cmod/Blending/VRU_v4`) · [Back to summary](#module-summary-click-to-jump)
- Function: Vectorized/micro‑fused VRU variant for higher throughput at modest area increase; identical interface.
- Params: Vector factor, pipeline depth; numeric precision.
- Optimizations: Unroll factor matches vector lanes; II per lane kept at 1.

- I/O: In `VRU_IN_TYPE`; Out `VRU_OUT_TYPE`.
- Ops: Quadratic form (6 mul, 3 add + 2 sub), exp approx (2 mul + 2 add), alpha (1 mul); transmittance (1 sub + 1 mul); color acc (3 mul + 3 add).
- Params: Polynomial vs PWL; thresholds for pruning; precision aliases.
- Optimizations: Pipeline across sub‑stages; II=1 default.

<a id="gradientcompute"></a>
8) GradientCompute (`A1_cmod/Blending/GradientCompute`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `VRU_IN_TYPE`; Out `VRU_IN_TYPE` (grads packed in fields).
- Ops: Reuses alpha/T path; simple grads (color 1 mul; opacity 1 mul + 2 add; mu 2 mul; cov 3 mul).
- Optimizations: Same as FeatureCompute.

<a id="gradientpruning"></a>
9) GradientPruning (`A1_cmod/Blending/GradientPruning`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `VRU_IN_TYPE` with scalar `grad`; Out pruned top‑K `VRU_IN_TYPE`.
- Function: In‑place quick‑select FSM to find K‑th; then prune and stream; final `last` on final kept.
- Params: `GP_MAX`, `GP_TOPK`.
- Optimizations: Selection network for fixed K; streaming min‑replacement for small K; II tuning.

<a id="rearrangement-ru"></a>
10) Rearrangement (RU) (`A1_cmod/Blending/Rearrangement`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `RU_Req { addr, grad, last }`; Out `RU_Update { bank, addr, grad, last }`.
- Function: Mask controller selects up to `RU_NUM_BANKS` non‑conflicting banks per cycle, merges duplicates within selection, emits one update per cycle; marks `last` at row end.
- Params: `RU_NUM_BANKS`; pending pool depth.
- Optimizations: Selection policy; merge fan‑in; queue depth; II tuning.

<a id="rowprocessing-gbu-row-pe"></a>
11) RowProcessing (GBU Row PE) (`A1_cmod/Blending/RowProcessing`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `ROW_IN_TYPE { dxpp,xpp,yn2,thr,opacity,color,pix_idx,last }`; Out `ROW_OUT_TYPE { accum:RGB16, T, pix_idx, last }`.
- Ops: Threshold: 2 mul (dx''·x'', y''²) + 1 add + 1 cmp. Color: 4 mul (T·α + 3×Ta·rgb) + 1 sub (1−α) + 3 add (accum). T update across row.
- Params: Optional internal pixel buffer (register array, not SRAM); can be elided if external buffer is used.
- Optimizations: Stream increment mode; II tuning; buffer size.

<a id="rowgeneration"></a>
12) RowGeneration (`A1_cmod/Blending/RowGeneration`) · [Back to summary](#module-summary-click-to-jump)
- I/O: In `BIN_DESC_TYPE { bin_idx, count }`; Out `ROW_IN_TYPE` sequence; LUT‑driven dx''/x'' base/y²/threshold/opacity/color.
- Params: `GBU_LUT_SIZE` (default 16); config struct `RG_LUT_CFG` for optional programming.
- Optimizations: Config port addition; II tuning; LUT resource sharing.

<a id="decompbinning"></a>
13) DecompBinning (`A1_cmod/Blending/DecompBinning`) · [Back to summary](#module-summary-click-to-jump)
<a id="pe-arrays"></a>
14) PE Arrays (Tile/Row PE replication) · [Back to summary](#module-summary-click-to-jump)
- I/O: Arrayed versions of RowProcessing or per‑tile PEs connected via a crossbar/demux.
- Function: Spatially parallel PEs for rows/tiles; each PE follows RowProcessing contract; aggregator performs address‑aware merge.
- Params: Number of PEs, demux policy (round‑robin or address‑based), bank interface width.
- Optimizations: Balance selection depth vs PE count; pipeline selectors; II=1 per PE.

<a id="ie-uniie-pru"></a>
15) IE / UNIIE / PRU · [Back to summary](#module-summary-click-to-jump)
- <a id="ie"></a>
15.1) IE – Interpolation Engine (`A1_cmod/Blending/IE`) · [Back to summary](#module-summary-click-to-jump)
- I/O: `VRU_IN_TYPE`, four scalar alpha inputs; Out `VRU_OUT_INT_TYPE`.
- Function: Accumulates color with external alphas; demonstrates stream fusion and arithmetic in integer formats.
- Params: Integer widths; optional bitmap.
- Optimizations: II=1; resource sharing across channel reads.

- <a id="uniie"></a>
15.2) UNIIE – Unified Nonlinear Interpolation Engine (`A1_cmod/Blending/UNIIE`) · [Back to summary](#module-summary-click-to-jump)
- I/O: `alpha_TYPE` → `alpha_type`.
- Function: Sum and shift‑average a 4‑way alpha; PWL shift for efficiency.
- Params: Alpha widths and lanes.
- Optimizations: II=1; avoid dead code; PWL for shifts.
- IE (Interpolation Engine): produces interpolated values (e.g., features or weights) for downstream blending; contracts match local include packs; FP16 default.
- UNIIE (Unified Nonlinear Interpolation Engine): IE extension with nonlinear activation/PWL; parameterized table sizes and piecewise segments.
- PRU (Pixel/Point Reduction Unit): tree/fan‑in reduction (sum/min/max) for intermediate products; compatible with `Reducer` in Encode; selectable op and arity.
- Optimizations: LUT vs PWL, tree depth, unroll, precision.
- I/O: In `ROW_IN_TYPE`; Out `BIN_DESC_TYPE` runs; exactly one push per cycle (staged).
- Function: `bin_idx = pix_idx / TILE_WIDTH`; coalesce runs; flush on row `last`.
- Params: `TILE_WIDTH`, `NUM_TILES_X` → `NUM_BINS`.
- Optimizations: Alternative mapping; II tuning.

## Architectural configuration surface

- Precision: FP16 default; fixed‑point alternatives provided behind typedefs in the include packs. All connected modules must agree on packs to be bit‑compatible.
- Structural: Sorting width; SRAM banks/entries; K/top‑K; RU banks; LUT size; tile width/cols. Compile‑time macros control these.
- Interfaces: Channels can be deepened via `Fifo<>`; commit bandwidth limited to one push per channel per cycle unless explicitly widened with parallel channels.

## Legal HLS optimizations

- Pipelining (initiation interval), loop unrolling (compute or traversal loops), resource sharing (comparators/MACs), and dataflow partitioning with Connections FIFOs.
- Math substitutions: PWL vs polynomial vs LUT within the module, provided the functional contract is preserved (e.g., exp/reciprocal approximations).
- Banking/hash policy changes as long as one‑per‑bank constraints and channel handshake rules remain valid.

## Integration & system‑level notes

- `S0_scripts/linker.json` enumerates accelerators and per‑stage modules. The scripts in `S0_scripts/` drive HLS/FC/PWR flows and summarize PPA.
- Two‑level pipeline (GBU): chunked depth‑ordered binning (DecompBinning) enables Tile PEs to start early; overlap with GPU is realized by the system controller (outside module library). Modules expose clean streaming interfaces to enable this scheduling.


### Utilities / Test Modules

<a id="sramtest"></a>
SRAMTest (`A1_cmod/SRAMTest`) · [Back to summary](#module-summary-click-to-jump)
- Testing SRAM