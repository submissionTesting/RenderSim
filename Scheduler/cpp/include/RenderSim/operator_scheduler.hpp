#pragma once

#include "ir.hpp"
#include "optimization_library.hpp"
#include "performance_timer.hpp"
#include <memory>
#include <unordered_map>

namespace rendersim {

/**
 * Extended IR structures for operator-level scheduling
 */

struct MappedIRNode {
    OperatorNode op_node;
    std::string hw_unit;
    std::unordered_map<std::string, std::string> attrs;
};

struct MappedIR {
    std::unordered_map<std::string, MappedIRNode> nodes;
    std::vector<std::pair<std::string, std::string>> edges;
};

struct OperatorScheduledIRNode {
    MappedIRNode mapped_node;
    int32_t start_cycle;
    int32_t duration;
    std::unordered_map<std::string, std::string> resources;
    OptimizationResult optimization_result;
};

struct OperatorScheduledIR {
    std::unordered_map<std::string, OperatorScheduledIRNode> nodes;
    std::vector<std::pair<std::string, std::string>> edges;
};

/** Operator-level scheduler producing timed/resource-annotated IR. */
class OperatorLevelScheduler {
public:
    explicit OperatorLevelScheduler(std::shared_ptr<OperatorOptimizer> optimizer);
    
    /**
     * Schedule operators within each hardware module
     * 
     * @param mapped_ir Input mapped IR from mapping stage
     * @return Operator-scheduled IR with timing and optimization information
     */
    OperatorScheduledIR schedule(const MappedIR& mapped_ir);
    
    /**
     * Get statistics about the last scheduling run
     */
    struct SchedulingStats {
        size_t total_operators;
        size_t optimized_operators;
        double total_speedup;
        std::unordered_map<std::string, size_t> hw_unit_usage;
    };
    
    SchedulingStats getLastSchedulingStats() const { return last_stats_; }
    
    /**
     * Get latency statistics for the last scheduling run
     */
    SchedulingLatencyReport getLatencyReport() const;
    
    /**
     * Enable/disable latency instrumentation (enabled by default)
     */
    void setLatencyInstrumentationEnabled(bool enabled) { 
        latency_instrumentation_enabled_ = enabled; 
    }
    
    /**
     * Clear all latency measurements
     */
    void clearLatencyMeasurements() {
        if (timer_) timer_->clear();
    }

private:
    std::shared_ptr<OperatorOptimizer> optimizer_;
    SchedulingStats last_stats_;
    
    // Latency instrumentation
    std::shared_ptr<PerformanceTimer> timer_;
    bool latency_instrumentation_enabled_;
    
    /**
     * Schedule operators for a specific hardware unit
     */
    std::vector<OperatorScheduledIRNode> scheduleHardwareUnit(
        const std::vector<MappedIRNode>& nodes_for_hw_unit,
        const std::string& hw_unit
    );
    
    /**
     * Calculate start times based on dependencies and hardware constraints
     */
    void calculateStartTimes(
        std::vector<OperatorScheduledIRNode*>& scheduled_nodes,
        const std::vector<std::pair<std::string, std::string>>& edges
    );
    
    /**
     * Update scheduling statistics
     */
    void updateStats(const OperatorScheduledIR& result);
};

/**
 * Factory for creating different types of operator optimizers
 */
class OptimizerFactory {
public:
    enum class OptimizerType {
        DUMMY,          // Simple constant-duration optimizer
        ANALYTICAL,     // Analytical performance models
        ML_BASED        // Machine learning-based optimizer (future)
    };
    
    static std::unique_ptr<OperatorOptimizer> createOptimizer(
        OptimizerType type, 
        std::shared_ptr<OptimizationLibrary> library
    );
};

} // namespace rendersim 