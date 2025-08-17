#!/usr/bin/env bash

modules=(
  # Encode modules
  "Encode/AG"
  "Encode/reducer"
  "Encode/DCU"
  "Encode/CU"
  "Encode/SPE"
  "Encode/IGU"
  "Encode/ICU"
  "Encode/PEU"
  
  # FieldComp modules
  "FieldComp/NPU_v1"
  "FieldComp/NPU_v2"
  "FieldComp/NPU_PE_v1"
  "FieldComp/NPU_PE_v2"
  "FieldComp/fixedpoint_mul"
  "FieldComp/PE_array"
  "FieldComp/PE"
  "FieldComp/MLP_shared"
  "FieldComp/MLP_sonb"
  "FieldComp/MLP_ssa"
  "FieldComp/MLP_vanilla"
  "FieldComp/MLP_monb"
  "FieldComp/MLP_block"
  
  # Blending modules
  "Blending/BSU"
  "Blending/VRU_v1"
  "Blending/VRU_v2"
  "Blending/VRU_v3"
  "Blending/VRU_v4"
  "Blending/PRU"
  "Blending/UNIIE"
  "Blending/IE"
  "Blending/NPU_PE"
  "Blending/NPU"
  "Blending/QSU"
  
  # Sample modules
  "Sample/HAMAT"
  "Sample/RFM"
)

for module in "${modules[@]}"; do
  echo -e "\n==================== $module ===================="

  # Run build flow (HLS, Fusion, Power) for the current module.
  make hls PROJ_PATH="$module" HLS_BUILD_NAME=build_hls FC_BUILD_NAME=build_fc CLK_PERIOD=1.0 TECH_NODE=tn28rvt9t
  make fc  PROJ_PATH="$module" HLS_BUILD_NAME=build_hls FC_BUILD_NAME=build_fc CLK_PERIOD=1.0 TECH_NODE=tn28rvt9t
  make pwr PROJ_PATH="$module" HLS_BUILD_NAME=build_hls FC_BUILD_NAME=build_fc CLK_PERIOD=1.0 TECH_NODE=tn28rvt9t

  # Summarise the generated reports.
  ./report_module.sh "$module"
done