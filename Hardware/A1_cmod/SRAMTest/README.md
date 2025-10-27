# SRAMTest

## Overview
- Demonstrates on‑chip SRAM access through request/response Connections channels.
- The module `SRAMTest` follows the GBR (Gradient Boosting Regressor) model scaffold and exercises two SRAMs via `mem_array_sep` wrapped by `GBR_SRAM` in `HybridDef.h`.
- The testbench drives a small configuration stream and runs a toy inference over sample JSON data bundled locally.

## Build & run

```bash
cd Hardware/A1_cmod/SRAMTest
cmake .
make -j
./sim_SRAMTest
```
