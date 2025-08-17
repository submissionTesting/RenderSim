#include "RenderSim/optimization_library.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace rendersim {

// OptimizationStrategy implementation
OptimizationStrategy::OptimizationStrategy(
    const std::string& name,
    OptimizationType opt_type,
    OptimizationScope scope,
    DecisionCriteria criteria,
    const std::string& description,
    const std::vector<std::string>& applicable_operators,
    const OptimizationParameters& parameters
) : name_(name), opt_type_(opt_type), scope_(scope), criteria_(criteria),
    description_(description), applicable_operators_(applicable_operators), 
    parameters_(parameters) {
    
    if (name_.empty()) {
        throw std::invalid_argument("Optimization strategy must have a name");
    }
    if (applicable_operators_.empty()) {
        throw std::invalid_argument("Optimization strategy must specify applicable operators");
    }
}

bool OptimizationStrategy::isApplicableTo(const std::string& operator_type) const {
    return std::find(applicable_operators_.begin(), applicable_operators_.end(), operator_type) != applicable_operators_.end() ||
           std::find(applicable_operators_.begin(), applicable_operators_.end(), "*") != applicable_operators_.end();
}

// OptimizationLibrary implementation
OptimizationLibrary::OptimizationLibrary() {
    registerBuiltinStrategies();
}

void OptimizationLibrary::registerStrategy(std::unique_ptr<OptimizationStrategy> strategy) {
    const std::string& name = strategy->getName();
    strategies_[name] = std::move(strategy);
}

std::vector<const OptimizationStrategy*> OptimizationLibrary::getApplicableStrategies(const std::string& operator_type) const {
    std::vector<const OptimizationStrategy*> result;
    for (const auto& pair : strategies_) {
        if (pair.second->isApplicableTo(operator_type)) {
            result.push_back(pair.second.get());
        }
    }
    return result;
}

std::vector<const OptimizationStrategy*> OptimizationLibrary::getStrategiesByType(OptimizationType opt_type) const {
    std::vector<const OptimizationStrategy*> result;
    for (const auto& pair : strategies_) {
        if (pair.second->getOptType() == opt_type) {
            result.push_back(pair.second.get());
        }
    }
    return result;
}

std::vector<const OptimizationStrategy*> OptimizationLibrary::getStrategiesByScope(OptimizationScope scope) const {
    std::vector<const OptimizationStrategy*> result;
    for (const auto& pair : strategies_) {
        if (pair.second->getScope() == scope) {
            result.push_back(pair.second.get());
        }
    }
    return result;
}

const OptimizationStrategy* OptimizationLibrary::getStrategy(const std::string& name) const {
    auto it = strategies_.find(name);
    return it != strategies_.end() ? it->second.get() : nullptr;
}

void OptimizationLibrary::registerBuiltinStrategies() {
    // Reuse optimizations
    {
        OptimizationParameters params;
        params.double_params["reuse_threshold"] = 0.95;
        params.int_params["max_reuse_distance"] = 4;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "exponential_value_reuse",
            OptimizationType::REUSE,
            OptimizationScope::ELEMENT_LEVEL,
            DecisionCriteria::THRESHOLD_BASED,
            "Share exponential computations across multiple Gaussians in hybrid arrays",
            std::vector<std::string>{"GAUSSIAN_SPLATTING", "FIELD_COMPUTATION"},
            params
        ));
    }
    
    {
        OptimizationParameters params;
        params.vec_params["subgrid_size"] = {16, 16, 16};
        params.int_params["hash_table_size"] = 262144;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "restricted_hashing",
            OptimizationType::REUSE,
            OptimizationScope::REGION_LEVEL,
            DecisionCriteria::BOUNDARY_BASED,
            "Process rays within spatial subgrids for hash table locality",
            std::vector<std::string>{"HASH_ENCODE"},
            params
        ));
    }
    
    {
        OptimizationParameters params;
        params.double_params["angular_threshold"] = 0.1;
        params.int_params["temporal_window"] = 3;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "sparse_radiance_warping",
            OptimizationType::REUSE,
            OptimizationScope::FRAME_LEVEL,
            DecisionCriteria::THRESHOLD_BASED,
            "Reuse pixels with similar ray directions across frames",
            std::vector<std::string>{"VOLUME_RENDERING", "*"},
            params
        ));
    }
    
    // Skip optimizations
    {
        OptimizationParameters params;
        params.double_params["alpha_threshold"] = 0.005;
        params.double_params["distance_threshold"] = 100.0;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "gaussian_skipping",
            OptimizationType::SKIP,
            OptimizationScope::ELEMENT_LEVEL,
            DecisionCriteria::THRESHOLD_BASED,
            "Skip rendering individual Gaussians based on contribution scores",
            std::vector<std::string>{"GAUSSIAN_SPLATTING"},
            params
        ));
    }
    
    {
        OptimizationParameters params;
        params.double_params["opacity_threshold"] = 0.99;
        params.int_params["min_samples"] = 8;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "early_ray_termination",
            OptimizationType::SKIP,
            OptimizationScope::ELEMENT_LEVEL,
            DecisionCriteria::THRESHOLD_BASED,
            "Terminate rays early based on accumulated opacity",
            std::vector<std::string>{"VOLUME_RENDERING"},
            params
        ));
    }
    
    {
        OptimizationParameters params;
        params.vec_params["tile_size"] = {16, 16};
        params.int_params["culling_margin"] = 2;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "tile_culling",
            OptimizationType::SKIP,
            OptimizationScope::REGION_LEVEL,
            DecisionCriteria::BOUNDARY_BASED,
            "Skip entire tiles based on bounding box tests",
            std::vector<std::string>{"GAUSSIAN_SPLATTING", "RASTERIZATION"},
            params
        ));
    }
    
    // Low bit optimizations
    {
        OptimizationParameters params;
        params.int_params["precision_bits"] = 8;
        params.double_params["sensitivity_threshold"] = 0.01;
        
        registerStrategy(std::make_unique<OptimizationStrategy>(
            "low_precision_sampling",
            OptimizationType::LOW_BIT,
            OptimizationScope::ELEMENT_LEVEL,
            DecisionCriteria::THRESHOLD_BASED,
            "Use reduced precision for importance sampling computations",
            std::vector<std::string>{"SAMPLING"},
            params
        ));
    }
}

// OperatorOptimizer implementation
OperatorOptimizer::OperatorOptimizer(std::shared_ptr<OptimizationLibrary> library) 
    : optimization_library_(library) {}

// DummyOperatorOptimizer implementation
DummyOperatorOptimizer::DummyOperatorOptimizer(std::shared_ptr<OptimizationLibrary> library)
    : OperatorOptimizer(library) {
    initializeDefaultDurations();
}

void DummyOperatorOptimizer::initializeDefaultDurations() {
    default_durations_["HASH_ENCODE"] = 10;
    default_durations_["FIELD_COMPUTATION"] = 50;
    default_durations_["SAMPLING"] = 20;
    default_durations_["VOLUME_RENDERING"] = 100;
    default_durations_["GAUSSIAN_SPLATTING"] = 80;
    default_durations_["RASTERIZATION"] = 30;
    default_durations_["BLENDING"] = 15;
}

OptimizationResult DummyOperatorOptimizer::optimize(
    const std::string& operator_type, 
    const std::unordered_map<std::string, std::string>& operator_attrs
) const {
    auto applicable_strategies = optimization_library_->getApplicableStrategies(operator_type);
    
    // Remove placeholder base durations: rely solely on workload and memory terms
    int32_t base_duration = 0;
    
    // Rough throughput model in elements/cycle per operator type
    double elems_per_cycle = 1.0;
    if (operator_type == "FIELD_COMPUTATION") elems_per_cycle = 256.0;      // 32x32 array approx
    else if (operator_type == "HASH_ENCODE" || operator_type == "ENCODING") elems_per_cycle = 64.0;
    else if (operator_type == "SAMPLING") elems_per_cycle = 64.0;
    else if (operator_type == "VOLUME_RENDERING" || operator_type == "BLENDING") elems_per_cycle = 64.0;
    
    // Workload: number of input elements if provided
    auto w_it = operator_attrs.find("work_elems");
    double work = 0.0;
    if (w_it != operator_attrs.end()) {
        try { work = std::stod(w_it->second); } catch (...) { work = 0.0; }
    }
    // If FLOPs provided, bias workload upward
    auto f_it = operator_attrs.find("flop_count");
    if (f_it != operator_attrs.end()) {
        try {
            double flops = std::stod(f_it->second);
            // simple adjust: equivalent elements += flops/32
            work = std::max(work, flops / 32.0);
        } catch (...) {}
    }
    
    int32_t workload_cycles = 0;
    if (work > 0.0 && elems_per_cycle > 0.0) {
        workload_cycles = static_cast<int32_t>(std::ceil(work / elems_per_cycle));
    }
    
    // Apply simple speedup factors for demonstration
    double speedup_factor = 1.0;
    std::vector<std::string> applied_optimizations;
    
    for (const auto* strategy : applicable_strategies) {
        if (strategy->getOptType() == OptimizationType::SKIP || 
            strategy->getOptType() == OptimizationType::REUSE) {
            speedup_factor *= 0.8;  // 20% speedup
            applied_optimizations.push_back(strategy->getName());
        }
    }
    
    // Combine base + workload, then apply speedup
    int32_t combined = base_duration + workload_cycles;
    int32_t compute_cycles = static_cast<int32_t>(std::max(1.0, combined * speedup_factor));

    // Memory-bound time: bytes / (bytes_per_cycle)
    double bytes_per_cycle = 16.0; // default fallback
    if (operator_type == "FIELD_COMPUTATION") bytes_per_cycle = 64.0;
    else if (operator_type == "HASH_ENCODE" || operator_type == "ENCODING") bytes_per_cycle = 32.0;
    else if (operator_type == "SAMPLING") bytes_per_cycle = 32.0;
    else if (operator_type == "VOLUME_RENDERING" || operator_type == "BLENDING") bytes_per_cycle = 32.0;

    double bytes = 0.0;
    auto b_it = operator_attrs.find("bytes");
    if (b_it != operator_attrs.end()) {
        try { bytes = std::stod(b_it->second); } catch (...) { bytes = 0.0; }
    }
    int32_t mem_cycles = 0;
    if (bytes > 0.0 && bytes_per_cycle > 0.0) {
        mem_cycles = static_cast<int32_t>(std::ceil(bytes / bytes_per_cycle));
    }

    int32_t final_duration = std::max(compute_cycles, mem_cycles);

    return OptimizationResult{
        final_duration,
        applied_optimizations,
        speedup_factor,
        base_duration,
        applicable_strategies.size()
    };
}

// ------------------ AnalyticalOperatorOptimizer ------------------

AnalyticalOperatorOptimizer::AnalyticalOperatorOptimizer(std::shared_ptr<OptimizationLibrary> library)
    : OperatorOptimizer(library) {
    initBaseCosts();
}

void AnalyticalOperatorOptimizer::initBaseCosts() {
    base_cost_["HASH_ENCODE"] = 400;
    base_cost_["FIELD_COMPUTATION"] = 1200;
    base_cost_["SAMPLING"] = 600;
    base_cost_["VOLUME_RENDERING"] = 900;
    base_cost_["GAUSSIAN_SPLATTING"] = 1500;
    base_cost_["RASTERIZATION"] = 700;
    base_cost_["BLENDING"] = 500;
}

OptimizationResult AnalyticalOperatorOptimizer::optimize(
    const std::string& operator_type,
    const std::unordered_map<std::string, std::string>& operator_attrs) const {
    // base cost lookup
    auto it = base_cost_.find(operator_type);
    int32_t base_cycles = (it != base_cost_.end()) ? it->second : 800;

    // adjust by FLOPs if provided (attrs["flop_count"])
    auto fit = operator_attrs.find("flop_count");
    if (fit != operator_attrs.end()) {
        try {
            double flops = std::stod(fit->second);
            // simple model: cycles = base + log10(FLOPs)
            base_cycles += static_cast<int32_t>(std::log10(std::max(1.0, flops)) * 100);
        } catch (...) {}
    }

    // consider applicable optimisation strategies to get speedup factor
    auto strategies = optimization_library_->getApplicableStrategies(operator_type);
    double speedup = 1.0;
    std::vector<std::string> applied;
    for (const auto* s : strategies) {
        if (s->getOptType() == OptimizationType::REUSE) speedup *= 0.9;
        if (s->getOptType() == OptimizationType::SKIP)  speedup *= 0.85;
        if (s->getOptType() == OptimizationType::LOW_BIT) speedup *= 0.95;
        applied.push_back(s->getName());
    }

    // ------------------------------------------------------------------
    // Hint/criteria-gated adjustments from mapped IR attributes
    // ------------------------------------------------------------------
    auto get_double = [&](const char* key, double def_v) -> double {
        auto it2 = operator_attrs.find(key);
        if (it2 == operator_attrs.end()) return def_v;
        try { return std::stod(it2->second); } catch (...) { return def_v; }
    };
    auto get_int = [&](const char* key, int def_v) -> int {
        auto it2 = operator_attrs.find(key);
        if (it2 == operator_attrs.end()) return def_v;
        try { return std::stoi(it2->second); } catch (...) { return def_v; }
    };
    auto get_bool = [&](const char* key, bool def_v) -> bool {
        auto it2 = operator_attrs.find(key);
        if (it2 == operator_attrs.end()) return def_v;
        std::string v = it2->second;
        for (auto& c : v) c = static_cast<char>(std::tolower(c));
        if (v == "1" || v == "true" || v == "yes" || v == "y") return true;
        if (v == "0" || v == "false" || v == "no" || v == "n") return false;
        return def_v;
    };

    double gating = 1.0;

    if (operator_type == "VOLUME_RENDERING" || operator_type == "SAMPLING") {
        // Early ray termination and sampling activity
        double opacity_threshold = get_double("opacity_threshold", 0.99);
        double avg_opacity = get_double("avg_opacity", -1.0);
        double active_ratio = get_double("active_samples_ratio", -1.0);
        if (avg_opacity >= 0.0 && avg_opacity >= opacity_threshold) {
            gating *= 0.85; // terminate earlier when opacity high
            applied.push_back("hint_early_ray_termination");
        }
        if (active_ratio >= 0.0) {
            // Map [0,1] to [0.55,1.0] speed multiplier
            double mult = std::max(0.55, std::min(1.0, 0.55 + 0.45 * active_ratio));
            gating *= mult;
            applied.push_back("hint_sampling_activity");
        }
    }

    if (operator_type == "HASH_ENCODE" || operator_type == "ENCODING") {
        // Restricted hashing / locality
        bool hash_active = get_bool("hash_index_activity", false);
        double locality = get_double("locality_score", -1.0);
        if (hash_active) {
            gating *= 0.9;
            applied.push_back("hint_hash_activity");
        }
        if (locality >= 0.0 && locality >= 0.7) {
            gating *= 0.9;
            applied.push_back("hint_locality");
        }
    }

    if (operator_type == "FIELD_COMPUTATION") {
        // Low precision compute
        bool low_bit = get_bool("low_bit_observed", false);
        int bits = get_int("precision_bits", 16);
        if (low_bit || bits <= 8) {
            gating *= 0.9;
            applied.push_back("hint_low_bit");
        }
        if (bits <= 4) {
            gating *= 0.85;
        }
    }

    speedup *= gating;

    int32_t final_cycles = static_cast<int32_t>(base_cycles * speedup);

    return OptimizationResult{
        final_cycles,
        applied,
        speedup,
        base_cycles,
        strategies.size()
    };
}

} // namespace rendersim 