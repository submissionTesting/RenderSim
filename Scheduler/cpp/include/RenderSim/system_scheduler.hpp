#pragma once

#include "operator_scheduler.hpp"
#include "performance_timer.hpp"
#include <queue>
#include <functional>
#include <unordered_set>

namespace rendersim {

/**
 * System Schedule Entry - final scheduling result for an operator
 */
struct SystemScheduleEntry {
    std::string op_id;
    std::string hw_unit;
    int64_t start_cycle;
    int64_t duration;
    double resource_utilization;
};

/**
 * System Schedule - complete scheduling result for entire pipeline
 */
struct SystemSchedule {
    std::vector<SystemScheduleEntry> entries;
    int64_t total_cycles;
    double avg_resource_utilization;
    std::unordered_map<std::string, int64_t> hw_unit_finish_times;
    
    SystemSchedule() : total_cycles(0), avg_resource_utilization(0.0) {}
};

/**
 * DAGS Algorithm Configuration
 */
struct DAGSConfig {
    double alpha;  // Weight for successor count heuristic
    double beta;   // Weight for critical resource impact heuristic
    
    // Default weights from research validation
    DAGSConfig() : alpha(0.6), beta(0.4) {}
    DAGSConfig(double a, double b) : alpha(a), beta(b) {}
};

/** System-level scheduler implementing a DAGS heuristic. */
class SystemLevelScheduler {
public:
    explicit SystemLevelScheduler(const DAGSConfig& config = DAGSConfig());
    
    /**
     * Schedule entire system using DAGS algorithm
     * 
     * @param op_scheduled_ir Input from operator-level scheduling
     * @return Complete system schedule with global coordination
     */
    SystemSchedule schedule(const OperatorScheduledIR& op_scheduled_ir);
    
    /**
     * Get detailed statistics about the last scheduling run
     */
    struct SystemSchedulingStats {
        size_t total_operators;
        size_t ready_queue_peak_size;
        double scheduling_efficiency;     // actual_cycles / critical_path_cycles
        double resource_balance_factor;  // std_dev of hw unit utilizations
        std::unordered_map<std::string, double> hw_unit_utilizations;
    };
    
    SystemSchedulingStats getLastSchedulingStats() const { return last_stats_; }
    
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
    
    /**
     * Update DAGS configuration weights
     */
    void updateConfig(const DAGSConfig& config) { config_ = config; }

private:
    DAGSConfig config_;
    SystemSchedulingStats last_stats_;
    
    // Latency instrumentation
    std::shared_ptr<PerformanceTimer> timer_;
    bool latency_instrumentation_enabled_;
    
    /**
     * Build dependency graph from edges
     */
    std::unordered_map<std::string, std::vector<std::string>> buildDependencyGraph(
        const std::vector<std::pair<std::string, std::string>>& edges
    );
    
    /**
     * Count successors for each node (heuristic 1)
     */
    std::unordered_map<std::string, int32_t> computeSuccessorCounts(
        const std::unordered_map<std::string, std::vector<std::string>>& dependencies,
        const std::unordered_map<std::string, OperatorScheduledIRNode>& nodes
    );
    
    /**
     * Compute critical resource impact (heuristic 2)
     * Combines latency, bandwidth utilization, and power consumption
     */
    std::unordered_map<std::string, double> computeCriticalResourceImpact(
        const std::unordered_map<std::string, OperatorScheduledIRNode>& nodes
    );
    
    /**
     * Calculate DAGS scoring function: α*successor_count + β*critical_impact
     */
    double calculateDAGSScore(
        const std::string& op_id,
        const std::unordered_map<std::string, int32_t>& successor_counts,
        const std::unordered_map<std::string, double>& critical_impacts
    );
    
    /**
     * Find earliest available slot for an operator considering:
     * - Hardware unit availability
     * - Data dependencies  
     * - Resource constraints
     */
    int64_t findEarliestSlot(
        const OperatorScheduledIRNode& node,
        const std::unordered_map<std::string, int64_t>& hw_finish_times,
        const std::unordered_map<std::string, int64_t>& op_finish_times
    );
    
    /**
     * Update ready queue with newly available operators
     */
    void updateReadyQueue(
        std::priority_queue<std::pair<double, std::string>>& ready_queue,
        const std::unordered_map<std::string, std::vector<std::string>>& dependencies,
        const std::unordered_set<std::string>& scheduled_ops,
        const std::unordered_map<std::string, OperatorScheduledIRNode>& all_nodes,
        const std::unordered_map<std::string, int32_t>& successor_counts,
        const std::unordered_map<std::string, double>& critical_impacts
    );
    
    /**
     * Calculate final statistics
     */
    void updateSystemStats(
        const SystemSchedule& schedule,
        const std::unordered_map<std::string, OperatorScheduledIRNode>& nodes
    );
    
    /**
     * Validate scheduling correctness (dependencies, resource constraints)
     */
    bool validateSchedule(
        const SystemSchedule& schedule,
        const std::vector<std::pair<std::string, std::string>>& edges
    );
};

/**
 * System Scheduler Factory for different algorithms
 */
class SystemSchedulerFactory {
public:
    enum class SchedulerType {
        DAGS,          // Dependency-Aware Greedy Scheduler (default)
        LIST_BASED,    // Classical list scheduling
        CRITICAL_PATH  // Critical path scheduling
    };
    
    static std::unique_ptr<SystemLevelScheduler> createScheduler(
        SchedulerType type,
        const DAGSConfig& config = DAGSConfig()
    );
};

} // namespace rendersim 