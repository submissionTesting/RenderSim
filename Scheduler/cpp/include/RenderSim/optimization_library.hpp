#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace rendersim {

/** Optimization framework for neural rendering accelerators. */

enum class OptimizationType {
    REUSE,    // Share computation results when multiple operations require same values
    SKIP,     // Avoid unnecessary computation when results are negligible  
    LOW_BIT   // Use low-precision arithmetic for reduced bandwidth/energy
};

enum class OptimizationScope {
    ELEMENT_LEVEL,  // Individual rays, points, or primitives
    REGION_LEVEL,   // Spatial groups (tiles, subgrids, blocks)
    FRAME_LEVEL     // Temporal boundaries between consecutive frames
};

enum class DecisionCriteria {
    BOUNDARY_BASED,  // Geometric boundaries and spatial partitions
    THRESHOLD_BASED  // Computed metrics against predefined values
};

struct OptimizationParameters {
    std::unordered_map<std::string, double> double_params;
    std::unordered_map<std::string, int32_t> int_params;
    std::unordered_map<std::string, std::vector<int32_t>> vec_params;
};

class OptimizationStrategy {
public:
    OptimizationStrategy(
        const std::string& name,
        OptimizationType opt_type,
        OptimizationScope scope,
        DecisionCriteria criteria,
        const std::string& description,
        const std::vector<std::string>& applicable_operators,
        const OptimizationParameters& parameters
    );

    const std::string& getName() const { return name_; }
    OptimizationType getOptType() const { return opt_type_; }
    OptimizationScope getScope() const { return scope_; }
    DecisionCriteria getCriteria() const { return criteria_; }
    const std::string& getDescription() const { return description_; }
    const std::vector<std::string>& getApplicableOperators() const { return applicable_operators_; }
    const OptimizationParameters& getParameters() const { return parameters_; }

    bool isApplicableTo(const std::string& operator_type) const;

private:
    std::string name_;
    OptimizationType opt_type_;
    OptimizationScope scope_;
    DecisionCriteria criteria_;
    std::string description_;
    std::vector<std::string> applicable_operators_;
    OptimizationParameters parameters_;
};

class OptimizationLibrary {
public:
    OptimizationLibrary();
    
    void registerStrategy(std::unique_ptr<OptimizationStrategy> strategy);
    std::vector<const OptimizationStrategy*> getApplicableStrategies(const std::string& operator_type) const;
    std::vector<const OptimizationStrategy*> getStrategiesByType(OptimizationType opt_type) const;
    std::vector<const OptimizationStrategy*> getStrategiesByScope(OptimizationScope scope) const;
    
    size_t getStrategyCount() const { return strategies_.size(); }
    const OptimizationStrategy* getStrategy(const std::string& name) const;

private:
    std::unordered_map<std::string, std::unique_ptr<OptimizationStrategy>> strategies_;
    
    void registerBuiltinStrategies();
};

struct OptimizationResult {
    int32_t duration;
    std::vector<std::string> applied_optimizations;
    double speedup_factor;
    int32_t base_duration;
    size_t total_strategies_considered;
};

class OperatorOptimizer {
public:
    explicit OperatorOptimizer(std::shared_ptr<OptimizationLibrary> library);
    virtual ~OperatorOptimizer() = default;
    
    virtual OptimizationResult optimize(
        const std::string& operator_type, 
        const std::unordered_map<std::string, std::string>& operator_attrs
    ) const = 0;

protected:
    std::shared_ptr<OptimizationLibrary> optimization_library_;
};

class DummyOperatorOptimizer : public OperatorOptimizer {
public:
    explicit DummyOperatorOptimizer(std::shared_ptr<OptimizationLibrary> library);
    
    OptimizationResult optimize(
        const std::string& operator_type, 
        const std::unordered_map<std::string, std::string>& operator_attrs
    ) const override;

private:
    std::unordered_map<std::string, int32_t> default_durations_;
    void initializeDefaultDurations();
};

/**
 * AnalyticalOperatorOptimizer – estimates duration based on simple analytical
 * models (FLOP count heuristics, memory footprint, etc.).  First version uses a
 * per-operator-type base cost plus logarithmic scaling with an optional
 * “flop_count” attribute if present.
 */
class AnalyticalOperatorOptimizer : public OperatorOptimizer {
public:
    explicit AnalyticalOperatorOptimizer(std::shared_ptr<OptimizationLibrary> library);

    OptimizationResult optimize(
        const std::string& operator_type,
        const std::unordered_map<std::string, std::string>& operator_attrs
    ) const override;

private:
    std::unordered_map<std::string, int32_t> base_cost_;
    void initBaseCosts();
};

} // namespace rendersim 