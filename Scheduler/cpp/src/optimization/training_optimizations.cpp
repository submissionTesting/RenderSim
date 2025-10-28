#include "RenderSim/optimization_library.hpp"
#include <cmath>
#include <algorithm>

namespace rendersim {

// GSArch Optimizations
class GSArchOptimizations {
public:
    // Tile-based merging optimization for GSArch
    static OptimizationResult tileMergingOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "tile_merging";
        
        // Based on GSArch paper: hierarchical tile merging reduces memory access
        // by processing tiles in spatial locality groups
        int tile_size = 16;  // 16x16 tiles as per paper
        int num_tiles = node.op_node.outputs[0].shape[0] / (tile_size * tile_size);
        
        // Compute speedup from tile merging (reduces redundant memory accesses)
        double merge_efficiency = 0.85;  // 85% efficiency from paper
        result.compute_speedup = 1.0 / (1.0 - (1.0 - merge_efficiency) * 0.3);
        
        // Memory traffic reduction from tile coalescing
        result.memory_reduction = 0.65;  // 35% reduction in memory traffic
        
        // Optimization scope: region-level (tile-based)
        result.scope = "region";
        result.applied = true;
        
        return result;
    }
    
    // Gradient pruning optimization for GSArch
    static OptimizationResult gradientPruningOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "gradient_pruning";
        
        // Based on GSArch: informativeness-based pruning
        // Prunes gradients below threshold to reduce computation
        double pruning_threshold = 0.01;  // From paper
        double expected_pruning_ratio = 0.4;  // 40% gradients pruned on average
        
        // Compute speedup from pruning
        result.compute_speedup = 1.0 / (1.0 - expected_pruning_ratio);
        
        // Memory reduction from not storing pruned gradients
        result.memory_reduction = 1.0 - expected_pruning_ratio;
        
        // Decision criteria: threshold-based
        result.scope = "element";
        result.applied = true;
        
        return result;
    }
    
    // Rearrangement optimization for memory coalescing
    static OptimizationResult rearrangementOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "request_rearrangement";
        
        // Rearranges gradient accumulation requests for better memory access
        // Based on bucket sorting from paper
        int bucket_size = 256;
        int num_elements = node.op_node.outputs[0].shape[0];
        int num_buckets = (num_elements + bucket_size - 1) / bucket_size;
        
        // Logarithmic overhead for bucket sorting
        double sort_overhead = std::log2(num_buckets) / num_elements;
        result.compute_speedup = 1.0 - sort_overhead;
        
        // Improved memory access pattern
        result.memory_reduction = 0.8;  // 20% reduction from coalescing
        
        result.scope = "region";
        result.applied = true;
        
        return result;
    }
};

// GBU Optimizations
class GBUOptimizations {
public:
    // Row-based processing optimization
    static OptimizationResult rowProcessingOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "row_processing";
        
        // GBU processes Gaussians in row-major order for better cache locality
        int row_width = 256;  // From paper
        int bundle_size = 32;  // Gaussian bundle size
        
        // Row-wise processing improves cache hit rate
        double cache_hit_improvement = 1.75;  // 75% improvement from paper
        result.compute_speedup = cache_hit_improvement;
        
        // Memory bandwidth utilization improvement
        result.memory_reduction = 0.6;  // 40% reduction in DRAM accesses
        
        result.scope = "region";  // Row-level
        result.applied = true;
        
        return result;
    }
    
    // Decomposition and binning optimization
    static OptimizationResult decompBinningOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "decomp_binning";
        
        // Hierarchical decomposition into bins for parallel processing
        int decomp_levels = 4;  // From paper
        int bin_size = 64;
        
        // Parallel processing speedup from binning
        double parallel_efficiency = 0.9;  // 90% parallel efficiency
        int num_bins = node.op_node.outputs[0].shape[0] / bin_size;
        result.compute_speedup = std::min(4.0, parallel_efficiency * std::sqrt(num_bins));
        
        // Memory access pattern improvement from binning
        result.memory_reduction = 0.5;  // 50% reduction
        
        result.scope = "region";
        result.applied = true;
        
        return result;
    }
    
    // Bundle generation optimization
    static OptimizationResult rowGenerationOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "row_generation";
        
        // Dynamic row generation based on Gaussian density
        double generation_efficiency = 0.8;  // From paper
        
        // Compute savings from dynamic generation
        result.compute_speedup = 1.0 / (1.0 - (1.0 - generation_efficiency) * 0.5);
        
        // Memory savings from on-demand generation
        result.memory_reduction = 0.7;  // 30% reduction
        
        result.scope = "region";
        result.applied = true;
        
        return result;
    }
};

// Instant3D Optimizations
class Instant3DOptimizations {
public:
    // Feed-forward read mapper optimization
    static OptimizationResult frmOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "frm";
        
        // FRM consolidates hash table reads in forward pass
        // Asymmetric design: optimized for forward reads
        int hash_levels = 16;  // Multi-resolution hash encoding
        int features_per_level = 2;
        
        // Read coalescing efficiency
        double coalescing_ratio = 0.7;  // 70% of reads can be coalesced
        result.compute_speedup = 1.0 + coalescing_ratio * 0.5;
        
        // Memory bandwidth savings from consolidated reads
        result.memory_reduction = coalescing_ratio * 0.6;
        
        result.scope = "region";  // Hash block level
        result.applied = true;
        
        return result;
    }
    
    // Backprop update merger optimization
    static OptimizationResult bumOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "bum";
        
        // BUM merges gradient updates hierarchically in backward pass
        // Asymmetric design: optimized for backward updates
        int merge_tree_depth = 4;  // Hierarchical merging levels
        
        // Update merging reduces write conflicts
        double conflict_reduction = 0.8;  // 80% conflict reduction
        result.compute_speedup = 1.0 / (1.0 - conflict_reduction * 0.4);
        
        // Memory traffic reduction from merged updates
        result.memory_reduction = 0.45;  // 55% traffic remains
        
        result.scope = "region";
        result.applied = true;
        
        return result;
    }
    
    // Color-density decomposition optimization
    static OptimizationResult colorDensityDecompOptimization(
        const MappedIRNode& node,
        const OptimizationLibrary& lib) {
        
        OptimizationResult result;
        result.optimization_type = "color_density_decomp";
        
        // Asymmetric processing of color (3 channels) vs density (1 channel)
        // Allows different precision and processing paths
        double decomp_efficiency = 0.85;
        
        // Compute savings from specialized paths
        result.compute_speedup = 1.0 + decomp_efficiency * 0.3;
        
        // Memory savings from reduced precision for density
        result.memory_reduction = 0.75;  // 25% reduction
        
        result.scope = "element";
        result.applied = true;
        
        return result;
    }
};

// Extension to OptimizationLibrary for training optimizations
OptimizationResult applyTrainingOptimization(
    const MappedIRNode& node,
    const OptimizationLibrary& lib) {
    
    std::string op_type = node.op_node.op_type;
    
    // Convert to uppercase for comparison
    std::transform(op_type.begin(), op_type.end(), op_type.begin(), ::toupper);
    
    // GSArch optimizations
    if (op_type == "TILEMERGING" || op_type == "TILEMERGING (B)") {
        return GSArchOptimizations::tileMergingOptimization(node, lib);
    }
    else if (op_type == "GRADIENTCOMPUTE" || op_type == "GRADIENTPRUNING") {
        return GSArchOptimizations::gradientPruningOptimization(node, lib);
    }
    else if (op_type == "REARRANGEMENT") {
        return GSArchOptimizations::rearrangementOptimization(node, lib);
    }
    // GBU optimizations
    else if (op_type == "ROWPROCESSING") {
        return GBUOptimizations::rowProcessingOptimization(node, lib);
    }
    else if (op_type == "DECOMPBINNING") {
        return GBUOptimizations::decompBinningOptimization(node, lib);
    }
    else if (op_type == "ROWGENERATION") {
        return GBUOptimizations::rowGenerationOptimization(node, lib);
    }
    // Instant3D optimizations
    else if (op_type == "FRM") {
        return Instant3DOptimizations::frmOptimization(node, lib);
    }
    else if (op_type == "BUM") {
        return Instant3DOptimizations::bumOptimization(node, lib);
    }
    else if (op_type.find("HASH") != std::string::npos && 
             op_type.find("(B)") != std::string::npos) {
        // Backward hash encoding uses BUM optimization
        return Instant3DOptimizations::bumOptimization(node, lib);
    }
    
    // Default: no optimization
    OptimizationResult result;
    result.applied = false;
    result.compute_speedup = 1.0;
    result.memory_reduction = 1.0;
    return result;
}

} // namespace rendersim
