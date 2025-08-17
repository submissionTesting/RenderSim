#include "RenderSim/operator_scheduler.hpp"
#include <unordered_map>
#include <algorithm>
#include <numeric>

namespace rendersim {

// OperatorLevelScheduler implementation
OperatorLevelScheduler::OperatorLevelScheduler(std::shared_ptr<OperatorOptimizer> optimizer)
    : optimizer_(optimizer), timer_(std::make_shared<PerformanceTimer>()), 
      latency_instrumentation_enabled_(true) {}

OperatorScheduledIR OperatorLevelScheduler::schedule(const MappedIR& mapped_ir) {
    // Start overall timing
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("operator_total");
    }
    
    OperatorScheduledIR result;
    result.edges = mapped_ir.edges;
    
    // Reset statistics
    last_stats_ = SchedulingStats{};
    last_stats_.total_operators = mapped_ir.nodes.size();
    
    // Stage 1: Group nodes by hardware unit
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("operator_hw_grouping");
    }
    
    std::unordered_map<std::string, std::vector<MappedIRNode>> hw_unit_groups;
    for (const auto& node_pair : mapped_ir.nodes) {
        const auto& node = node_pair.second;
        hw_unit_groups[node.hw_unit].push_back(node);
        last_stats_.hw_unit_usage[node.hw_unit]++;
    }
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("operator_hw_grouping");
    }
    
    // Stage 2: Schedule each hardware unit independently
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("operator_hw_scheduling");
    }
    
    for (const auto& hw_group : hw_unit_groups) {
        const std::string& hw_unit = hw_group.first;
        const auto& nodes_for_hw = hw_group.second;
        
        auto scheduled_nodes = scheduleHardwareUnit(nodes_for_hw, hw_unit);
        
        // Add scheduled nodes to result
        for (const auto& scheduled_node : scheduled_nodes) {
            result.nodes[scheduled_node.mapped_node.op_node.id] = scheduled_node;
        }
    }
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("operator_hw_scheduling");
    }
    
    // Stage 3: Calculate start times considering cross-hardware dependencies
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("operator_dependency_resolution");
    }
    
    std::vector<OperatorScheduledIRNode*> all_nodes;
    for (auto& node_pair : result.nodes) {
        all_nodes.push_back(&node_pair.second);
    }
    
    calculateStartTimes(all_nodes, mapped_ir.edges);
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("operator_dependency_resolution");
    }
    
    // Update statistics
    updateStats(result);
    
    // End overall timing
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("operator_total");
    }
    
    return result;
}

std::vector<OperatorScheduledIRNode> OperatorLevelScheduler::scheduleHardwareUnit(
    const std::vector<MappedIRNode>& nodes_for_hw_unit,
    const std::string& hw_unit
) {
    std::vector<OperatorScheduledIRNode> scheduled_nodes;
    scheduled_nodes.reserve(nodes_for_hw_unit.size());
    
    int32_t current_cycle = 0;
    
    for (const auto& mapped_node : nodes_for_hw_unit) {
        // Apply optimizations to get duration and optimization metadata
        OptimizationResult opt_result = optimizer_->optimize(
            mapped_node.op_node.op_type, 
            mapped_node.attrs
        );
        
        // Create scheduled node
        OperatorScheduledIRNode scheduled_node;
        scheduled_node.mapped_node = mapped_node;
        scheduled_node.start_cycle = current_cycle;
        scheduled_node.duration = opt_result.duration;
        scheduled_node.optimization_result = opt_result;
        
        // Simple resource allocation (placeholder)
        scheduled_node.resources["compute_units"] = "1";
        scheduled_node.resources["memory_bandwidth"] = "high";
        
        scheduled_nodes.push_back(scheduled_node);
        
        // Update current cycle for next operator on this hardware unit
        current_cycle += opt_result.duration;
        
        // Update optimization statistics
        if (!opt_result.applied_optimizations.empty()) {
            last_stats_.optimized_operators++;
        }
    }
    
    return scheduled_nodes;
}

void OperatorLevelScheduler::calculateStartTimes(
    std::vector<OperatorScheduledIRNode*>& scheduled_nodes,
    const std::vector<std::pair<std::string, std::string>>& edges
) {
    // Build dependency graph
    std::unordered_map<std::string, std::vector<std::string>> dependencies;
    std::unordered_map<std::string, OperatorScheduledIRNode*> node_map;
    
    for (auto* node : scheduled_nodes) {
        node_map[node->mapped_node.op_node.id] = node;
    }
    
    for (const auto& edge : edges) {
        dependencies[edge.second].push_back(edge.first);
    }
    
    // Simple topological scheduling considering hardware-local start times
    std::unordered_map<std::string, int32_t> earliest_start_times;
    
    for (auto* node : scheduled_nodes) {
        int32_t earliest_start = 0;
        
        // Check dependencies
        const auto& deps = dependencies[node->mapped_node.op_node.id];
        for (const std::string& dep_id : deps) {
            auto dep_it = node_map.find(dep_id);
            if (dep_it != node_map.end()) {
                int32_t dep_finish_time = dep_it->second->start_cycle + dep_it->second->duration;
                earliest_start = std::max(earliest_start, dep_finish_time);
            }
        }
        
        // Ensure we don't conflict with hardware-local scheduling
        int32_t hw_local_start = node->start_cycle;
        node->start_cycle = std::max(earliest_start, hw_local_start);
        
        earliest_start_times[node->mapped_node.op_node.id] = node->start_cycle;
    }
}

void OperatorLevelScheduler::updateStats(const OperatorScheduledIR& result) {
    // Calculate total speedup
    double total_base_duration = 0.0;
    double total_optimized_duration = 0.0;
    
    for (const auto& node_pair : result.nodes) {
        const auto& node = node_pair.second;
        total_base_duration += node.optimization_result.base_duration;
        total_optimized_duration += node.optimization_result.duration;
    }
    
    last_stats_.total_speedup = (total_optimized_duration > 0) ? 
        total_base_duration / total_optimized_duration : 1.0;
}

// OptimizerFactory implementation
std::unique_ptr<OperatorOptimizer> OptimizerFactory::createOptimizer(
    OptimizerType type, 
    std::shared_ptr<OptimizationLibrary> library
) {
    switch (type) {
        case OptimizerType::DUMMY:
            return std::make_unique<DummyOperatorOptimizer>(library);
        case OptimizerType::ANALYTICAL:
            // TODO: Implement analytical optimizer
            return std::make_unique<DummyOperatorOptimizer>(library);
        case OptimizerType::ML_BASED:
            // TODO: Implement ML-based optimizer
            return std::make_unique<DummyOperatorOptimizer>(library);
        default:
            return std::make_unique<DummyOperatorOptimizer>(library);
    }
}

SchedulingLatencyReport OperatorLevelScheduler::getLatencyReport() const {
    SchedulingLatencyReport report;
    
    if (timer_) {
        // Operator-level scheduler stages
        report.operator_hw_grouping = LatencyStats(
            timer_->getTotalDuration("operator_hw_grouping"),
            timer_->getAverageDuration("operator_hw_grouping"),
            timer_->getLastDuration("operator_hw_grouping"),
            timer_->getMeasurementCount("operator_hw_grouping")
        );
        
        report.operator_hw_scheduling = LatencyStats(
            timer_->getTotalDuration("operator_hw_scheduling"),
            timer_->getAverageDuration("operator_hw_scheduling"),
            timer_->getLastDuration("operator_hw_scheduling"),
            timer_->getMeasurementCount("operator_hw_scheduling")
        );
        
        report.operator_dependency_resolution = LatencyStats(
            timer_->getTotalDuration("operator_dependency_resolution"),
            timer_->getAverageDuration("operator_dependency_resolution"),
            timer_->getLastDuration("operator_dependency_resolution"),
            timer_->getMeasurementCount("operator_dependency_resolution")
        );
        
        report.operator_total = LatencyStats(
            timer_->getTotalDuration("operator_total"),
            timer_->getAverageDuration("operator_total"),
            timer_->getLastDuration("operator_total"),
            timer_->getMeasurementCount("operator_total")
        );
    }
    
    return report;
}

} // namespace rendersim 