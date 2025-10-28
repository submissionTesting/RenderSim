#!/usr/bin/env python3
"""
Operator Mapping System for RenderSim Neural Rendering Instrumentation

This module provides mapping between traced nerfstudio function names and 
standardized RenderSim operator types defined in the /Operators taxonomy.

Categories:
- SAMPLING: Ray sampling, point sampling, frustum operations
- ENCODING: Hash encoding, positional encoding, feature encoding  
- COMPUTATION: MLP, field computation, spherical harmonics
- BLENDING: Volume rendering, RGB/density rendering, alpha blending
"""

import re
from typing import Dict, Optional

def sanitize_function_name(raw: str) -> str:
    """Strip bracketed shape signatures and trailing instance/hash suffixes from traced names,
    and collapse module prefixes to Class.method so direct mappings can match.

    Examples:
      'nerfstudio.model_components.renderers.RGBRenderer.forward[rgb:(4096,64,3)->(4096,3)]#1' -> 'RGBRenderer.forward'
      'nerfstudio.model_components.ray_samplers.UniformSampler.generate_ray_samples' -> 'UniformSampler.generate_ray_samples'
    """
    s = str(raw)
    # Remove bracketed signature: [....]
    s = re.sub(r"\[.*?\]", "", s)
    # Remove hash and field suffixes like #1, #2.self, #1.rgb, etc.
    s = re.sub(r"#\d+.*$", "", s)
    s = s.strip()
    # Collapse module prefixes to last two segments (Class.method) when possible
    parts = s.split('.')
    if len(parts) >= 2:
        s = '.'.join(parts[-2:])
    return s

# Neural Rendering Function Name to Operator Type Mapping
FUNCTION_TO_OPERATOR_MAP = {
    # ===== SAMPLING OPERATIONS =====
    # Only count explicit sampler generation calls as SAMPLING
    "UniformSampler.generate_ray_samples": "UNIFORM_SAMPLING",
    "PDFSampler.generate_ray_samples": "PDF_SAMPLING",
    "SpacedSampler.generate_ray_samples": "HELPER",
    # Treat other sampling-related helpers as non-operators
    "Sampler.forward": "RAY_SAMPLING",
    "UniformSampler.forward": "HELPER",
    "PDFSampler.forward": "HELPER",
    "RayBundle.get_ray_samples": "HELPER",
    "Frustums.get_positions": "HELPER",
    "RaySamples.get_weights": "HELPER",
    
    # ===== ENCODING OPERATIONS =====
    "NeRFEncoding.forward": "POSITIONAL_ENCODING",
    "HashEncoding.forward": "HASH_ENCODING", 
    "RFFEncoding.forward": "RFF_ENCODING",
    "FFEncoding.forward": "FOURIER_ENCODING",
    "MLPWithHashEncoding.forward": "HASH_MLP_ENCODING",
    
    # ===== COMPUTATION OPERATIONS =====
    "MLP.forward": "MLP_COMPUTATION",
    "DensityFieldHead.forward": "DENSITY_FIELD_COMPUTATION", 
    "RGBFieldHead.forward": "RGB_FIELD_COMPUTATION",
    # Treat model-level get_outputs as wrappers, not schedulable ops
    "NeRFModel.get_outputs": "MODEL_WRAPPER",
    "NGPModel.get_outputs": "MODEL_WRAPPER",
    "SplatFactoModel.get_outputs": "MODEL_WRAPPER",
    
    # ===== BLENDING OPERATIONS =====
    "RGBRenderer.forward": "RGB_VOLUME_RENDERING",
    "DepthRenderer.forward": "DEPTH_RENDERING", 
    "AccumulationRenderer.forward": "ALPHA_BLENDING",
    "UncertaintyRenderer.forward": "UNCERTAINTY_RENDERING",
    "SemanticRenderer.forward": "SEMANTIC_RENDERING",
    "NormalsRenderer.forward": "NORMALS_RENDERING",
}

# Fallback patterns for function name matching
FUNCTION_PATTERN_MAP = {
    # Sampling patterns (restrict broad matches to HELPER to avoid over-counting)
    r".*[Ss]ampl.*": "HELPER",
    r".*[Rr]ay.*": "HELPER", 
    r".*[Ff]rustum.*": "HELPER",
    r".*[Pp]osition.*": "HELPER",
    
    # Encoding patterns
    r".*[Ee]ncod.*": "POSITIONAL_ENCODING",
    r".*[Hh]ash.*": "HASH_ENCODING",
    r".*[Pp]ositional.*": "POSITIONAL_ENCODING",
    
    # Computation patterns  
    r".*MLP.*": "MLP_COMPUTATION",
    r".*[Ff]ield.*": "FIELD_COMPUTATION",
    r".*[Mm]odel.*get_outputs.*": "MODEL_WRAPPER",
    r".*[Mm]odel.*": "MODEL_COMPUTATION",
    
    # Rendering patterns
    r".*[Rr]ender.*": "VOLUME_RENDERING",
    r".*[Bb]lend.*": "ALPHA_BLENDING",
    r".*RGB.*": "RGB_RENDERING",
    r".*[Dd]ensity.*": "DENSITY_RENDERING",
}

# 4-Stage taxonomy to hardware mapping (unified neural rendering taxonomy)
OPERATOR_TO_HARDWARE_MAP = {
    # Field Sampler (SAMPLING) -> Volume Rendering Unit (VRU)
    "SAMPLING": "VOLUME_RENDERING",
    "UNIFORM_SAMPLING": "VOLUME_RENDERING",
    "PDF_SAMPLING": "VOLUME_RENDERING", 
    "RAY_SAMPLING": "VOLUME_RENDERING",
    "FRUSTUM_SAMPLING": "VOLUME_RENDERING",
    "WEIGHT_SAMPLING": "VOLUME_RENDERING",
    "POINT_SAMPLING": "VOLUME_RENDERING",
    
    # Encoding (ENCODING) -> Positional Encoding Unit (PEU)
    "ENCODING": "POSITIONAL_ENCODE",            # hardware mapping
    "POSITIONAL_ENCODING": "POSITIONAL_ENCODE",  # hardware mapping
    "HASH_ENCODING": "HASH_ENCODE",             # hardware mapping
    "RFF_ENCODING": "POSITIONAL_ENCODE",        # hardware mapping
    "FOURIER_ENCODING": "POSITIONAL_ENCODE",    # hardware mapping  
    "HASH_MLP_ENCODING": "HASH_ENCODE",         # hardware mapping
    
    # Field Computation (FIELD_COMPUTATION) -> MLP Engine
    "FIELD_COMPUTATION": "FIELD_COMPUTATION",   # hardware mapping
    "MLP_COMPUTATION": "MLP",                   # hardware mapping
    "DENSITY_FIELD_COMPUTATION": "FIELD_COMPUTATION",
    "RGB_FIELD_COMPUTATION": "FIELD_COMPUTATION",
    "NERF_MODEL_COMPUTATION": "FIELD_COMPUTATION", 
    "INSTANT_NGP_COMPUTATION": "FIELD_COMPUTATION",
    "GAUSSIAN_SPLATTING_COMPUTATION": "FIELD_COMPUTATION",
    "MODEL_COMPUTATION": "FIELD_COMPUTATION",
    
    # Wrapper (ignore for scheduling)
    "MODEL_WRAPPER": "IGNORE",
    "HELPER": "IGNORE",
    
    # Blending (BLENDING) -> Volume Rendering Unit (VRU)
    "BLENDING": "VOLUME_RENDERING",             # hardware mapping
    "RGB_VOLUME_RENDERING": "VOLUME_RENDERING",
    "DEPTH_RENDERING": "VOLUME_RENDERING", 
    "UNCERTAINTY_RENDERING": "VOLUME_RENDERING",
    "SEMANTIC_RENDERING": "VOLUME_RENDERING",
    "NORMALS_RENDERING": "VOLUME_RENDERING",
    "VOLUME_RENDERING": "VOLUME_RENDERING",
    "ALPHA_BLENDING": "BLENDING",              # hardware mapping
    "RGB_RENDERING": "VOLUME_RENDERING",
    "DENSITY_RENDERING": "VOLUME_RENDERING",
}

def map_function_to_operator_type(function_name: str) -> str:
    """
    Map a traced function name to standardized operator type.
    
    Args:
        function_name: Function name from instrumentation (e.g., "NeRFEncoding.forward")
        
    Returns:
        Standardized operator type (e.g., "POSITIONAL_ENCODING")
    """
    # Inspect raw name for chunk/micro tags before sanitizing
    raw_name = str(function_name)
    if "|CHUNK:PDF" in raw_name:
        return "PDF_SAMPLING"
    if "|CHUNK:UNIFORM" in raw_name:
        return "UNIFORM_SAMPLING"
    if "|MICRO:" in raw_name:
        return "HELPER"
    # Sanitize traced names to canonical form first
    function_name = sanitize_function_name(function_name)
    # Direct mapping first (exact match on Class.method)
    if function_name in FUNCTION_TO_OPERATOR_MAP:
        return FUNCTION_TO_OPERATOR_MAP[function_name]
    
    # Treat low-level tcnn primitives as helpers to avoid duplication inflation
    if function_name.startswith("tcnn."):
        return "HELPER"
    
    # Pattern-based fallback
    for pattern, op_type in FUNCTION_PATTERN_MAP.items():
        if re.match(pattern, function_name, re.IGNORECASE):
            return op_type
    
    # Default fallback
    return "FIELD_COMPUTATION"

def map_operator_to_hardware_type(operator_type: str) -> str:
    """
    Map an operator type to hardware module type.
    
    Args:
        operator_type: Standardized operator type
        
    Returns:
        Hardware module type for mapping engine
    """
    return OPERATOR_TO_HARDWARE_MAP.get(operator_type, "FIELD_COMPUTATION")

def enhance_dag_with_operator_types(dag_data: dict) -> dict:
    """
    Enhance a loaded DAG with proper operator type classifications.
    
    Args:
        dag_data: Raw DAG data from instrumentation (can be NetworkX or dict format)
        
    Returns:
        Enhanced DAG with operator types mapped to RenderSim taxonomy
    """
    enhanced_dag = dag_data.copy()
    
    # Process nodes and update operator types
    nodes_updated = 0
    nodes_dict = enhanced_dag.get('nodes', {})
    
    for node_id, node_info in nodes_dict.items():
        # For NetworkX format, the function name might be in the node_id itself
        function_name = sanitize_function_name(node_info.get('function_name', str(node_id)))
        
        # Extract the actual function name (ensure Class.method form)
        if '.' in function_name and not function_name.endswith('.self'):
            operator_type = map_function_to_operator_type(function_name)
            hardware_type = map_operator_to_hardware_type(operator_type)
            
            node_info['function_name'] = function_name
            node_info['op_type'] = operator_type
            node_info['hardware_type'] = hardware_type
            nodes_updated += 1
    
    print(f"[OK] Enhanced DAG: Updated {nodes_updated} nodes with operator types")
    return enhanced_dag

def get_operator_statistics(dag_data: dict) -> Dict[str, int]:
    """
    Generate operator type statistics from enhanced DAG.
    
    Args:
        dag_data: Enhanced DAG data
        
    Returns:
        Dictionary with operator type counts
    """
    stats = {}
    for node_info in dag_data.get('nodes', {}).values():
        op_type = node_info.get('op_type', 'unknown')
        stats[op_type] = stats.get(op_type, 0) + 1
    
    return stats

if __name__ == "__main__":
    # Test the mapping system
    test_functions = [
        "nerfstudio.field_components.encodings.NeRFEncoding.forward",
        "nerfstudio.model_components.ray_samplers.UniformSampler.generate_ray_samples", 
        "nerfstudio.field_components.mlp.MLP.forward",
        "nerfstudio.model_components.renderers.RGBRenderer.forward",
        "nerfstudio.model_components.renderers.AccumulationRenderer.forward",
        "unknown_function"
    ]
    
    print("🧪 Testing Operator Mapping System")
    print("=" * 50)
    for func in test_functions:
        op_type = map_function_to_operator_type(func)
        hw_type = map_operator_to_hardware_type(op_type)
        print(f"{func:35} -> {op_type:25} -> {hw_type}") 