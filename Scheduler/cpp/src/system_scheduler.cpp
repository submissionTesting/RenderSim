#include "RenderSim/system_scheduler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace rendersim {

// SystemLevelScheduler implementation
SystemLevelScheduler::SystemLevelScheduler(const DAGSConfig& config) 
    : config_(config), timer_(std::make_shared<PerformanceTimer>()),
      latency_instrumentation_enabled_(true) {}

SystemSchedule SystemLevelScheduler::schedule(const OperatorScheduledIR& op_scheduled_ir) {
    // Start overall timing
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("system_total");
    }
    
    SystemSchedule result;
    
    // Reset statistics
    last_stats_ = SystemSchedulingStats{};
    last_stats_.total_operators = op_scheduled_ir.nodes.size();
    
    if (op_scheduled_ir.nodes.empty()) {
        if (latency_instrumentation_enabled_ && timer_) {
            timer_->end("system_total");
        }
        return result;
    }
    
    // Stage 1: Build dependency graph
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("system_dependency_graph");
    }
    // Filter edges to only those whose endpoints exist in the node set
    std::unordered_map<std::string, std::vector<std::string>> dependencies;
    {
        std::unordered_set<std::string> node_ids;
        node_ids.reserve(op_scheduled_ir.nodes.size());
        for (const auto& p : op_scheduled_ir.nodes) node_ids.insert(p.first);
        for (const auto& e : op_scheduled_ir.edges) {
            if (node_ids.count(e.first) && node_ids.count(e.second)) {
                dependencies[e.second].push_back(e.first);
            }
        }
    }
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("system_dependency_graph");
    }
    
    // Stage 2: Compute DAGS heuristics
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("system_heuristic_computation");
    }
    
    auto successor_counts = computeSuccessorCounts(dependencies, op_scheduled_ir.nodes);
    auto critical_impacts = computeCriticalResourceImpact(op_scheduled_ir.nodes);
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("system_heuristic_computation");
    }
    
    // Build successors map and remaining predecessor counts for O(V+E) ready-queue maintenance
    std::unordered_map<std::string, std::vector<std::string>> successors;
    std::unordered_map<std::string, int32_t> remaining_pred_count;
    // Initialize all nodes with zero
    for (const auto& p : op_scheduled_ir.nodes) {
        remaining_pred_count[p.first] = 0;
    }
    // Fill from dependencies: deps[target] = [sources...]
    for (const auto& dep_pair : dependencies) {
        const std::string& target = dep_pair.first;
        const auto& sources = dep_pair.second;
        remaining_pred_count[target] = static_cast<int32_t>(sources.size());
        for (const std::string& source : sources) {
            successors[source].push_back(target);
        }
    }
    
    // Initialize tracking structures
    std::unordered_map<std::string, int64_t> hw_finish_times;
    std::unordered_map<std::string, int64_t> op_finish_times;
    std::unordered_set<std::string> scheduled_ops;
    std::unordered_set<std::string> enqueued_ops; // prevent duplicate PQ entries
    
    // Priority queue for ready operations (max-heap by DAGS score)
    std::priority_queue<std::pair<double, std::string>> ready_queue;
    
    // Initialize ready queue with source nodes (no remaining dependencies)
    for (const auto& node_pair : op_scheduled_ir.nodes) {
        const std::string& op_id = node_pair.first;
        auto it = remaining_pred_count.find(op_id);
        int32_t deg = (it != remaining_pred_count.end()) ? it->second : 0;
        if (deg == 0) {
            double score = calculateDAGSScore(op_id, successor_counts, critical_impacts);
            ready_queue.push({score, op_id});
            enqueued_ops.insert(op_id);
        }
    }
    
    // Stage 3: DAGS main scheduling loop
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("system_scheduling_loop");
    }
    
    size_t peak_queue_size = ready_queue.size();
    while (!ready_queue.empty()) {
        // Select highest scoring operation
        auto top_entry = ready_queue.top();
        double score = top_entry.first;
        std::string selected_op = top_entry.second;
        ready_queue.pop();
        
        // Skip if already scheduled (can happen due to queue updates)
        if (scheduled_ops.count(selected_op)) {
            continue;
        }
        
        const auto& node = op_scheduled_ir.nodes.at(selected_op);
        
        // Find earliest available slot considering dependencies
        int64_t hw_available = 0;
        auto hw_it = hw_finish_times.find(node.mapped_node.hw_unit);
        if (hw_it != hw_finish_times.end()) {
            hw_available = hw_it->second;
        }
        
        // Check dependency availability
        int64_t dep_available = 0;
        auto deps_it = dependencies.find(selected_op);
        if (deps_it != dependencies.end()) {
            for (const std::string& dep_id : deps_it->second) {
                auto dep_finish_it = op_finish_times.find(dep_id);
                if (dep_finish_it != op_finish_times.end()) {
                    dep_available = std::max(dep_available, dep_finish_it->second);
                }
            }
        }
        
        int64_t start_time = std::max(hw_available, dep_available);
        int64_t finish_time = start_time + static_cast<int64_t>(node.duration);
        
        // Create system schedule entry
        SystemScheduleEntry entry;
        entry.op_id = selected_op;
        entry.hw_unit = node.mapped_node.hw_unit;
        entry.start_cycle = start_time;
        entry.duration = node.duration;
        entry.resource_utilization = 1.0; // Simplified for now
        
        result.entries.push_back(entry);
        
        // Update tracking structures
        scheduled_ops.insert(selected_op);
        hw_finish_times[node.mapped_node.hw_unit] = finish_time;
        op_finish_times[selected_op] = finish_time;
        
        // Decrement remaining predecessor counts of successors and enqueue newly ready ops once
        auto succ_it = successors.find(selected_op);
        if (succ_it != successors.end()) {
            for (const std::string& succ_id : succ_it->second) {
                auto rem_it = remaining_pred_count.find(succ_id);
                if (rem_it != remaining_pred_count.end() && rem_it->second > 0) {
                    rem_it->second -= 1;
                }
                int32_t rem = (rem_it != remaining_pred_count.end()) ? rem_it->second : 0;
                if (rem == 0 && !scheduled_ops.count(succ_id) && !enqueued_ops.count(succ_id)) {
                    double s = calculateDAGSScore(succ_id, successor_counts, critical_impacts);
                    ready_queue.push({s, succ_id});
                    enqueued_ops.insert(succ_id);
                }
            }
        }
        
        peak_queue_size = std::max(peak_queue_size, ready_queue.size());
    }
    
    // Fallback: schedule any remaining unscheduled ops greedily by HW availability
    if (scheduled_ops.size() < op_scheduled_ir.nodes.size()) {
        for (const auto& node_pair : op_scheduled_ir.nodes) {
            const std::string& op_id = node_pair.first;
            if (scheduled_ops.count(op_id)) continue;
            const auto& node = node_pair.second;
            int64_t hw_available = 0;
            auto hw_it = hw_finish_times.find(node.mapped_node.hw_unit);
            if (hw_it != hw_finish_times.end()) hw_available = hw_it->second;
            int64_t start_time = hw_available;
            int64_t finish_time = start_time + static_cast<int64_t>(node.duration);
            SystemScheduleEntry entry;
            entry.op_id = op_id;
            entry.hw_unit = node.mapped_node.hw_unit;
            entry.start_cycle = start_time;
            entry.duration = node.duration;
            entry.resource_utilization = 1.0;
            result.entries.push_back(entry);
            scheduled_ops.insert(op_id);
            hw_finish_times[node.mapped_node.hw_unit] = finish_time;
            op_finish_times[op_id] = finish_time;
        }
    }
    
    // Calculate final results
    result.total_cycles = 0;
    // Primary: max finish time across all scheduled entries
    for (const auto& entry : result.entries) {
        int64_t finish = entry.start_cycle + entry.duration;
        if (finish > result.total_cycles) result.total_cycles = finish;
        // Track per-HW finish times as well
        auto it = result.hw_unit_finish_times.find(entry.hw_unit);
        if (it == result.hw_unit_finish_times.end()) result.hw_unit_finish_times[entry.hw_unit] = finish;
        else it->second = std::max(it->second, finish);
    }
    // Fallback: if entries were empty (shouldn't happen), use hw_finish_times
    if (result.total_cycles == 0) {
        for (const auto& hw_pair : hw_finish_times) {
            result.total_cycles = std::max(result.total_cycles, hw_pair.second);
            result.hw_unit_finish_times[hw_pair.first] = hw_pair.second;
        }
    }
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("system_scheduling_loop");
    }
    
    // Stage 4: Finalization and statistics
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->start("system_finalization");
    }
    
    // Update statistics
    last_stats_.ready_queue_peak_size = peak_queue_size;
    updateSystemStats(result, op_scheduled_ir.nodes);
    
    // Validate result
    if (!validateSchedule(result, op_scheduled_ir.edges)) {
        // Log warning but continue
    }
    
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("system_finalization");
    }
    
    // End overall timing
    if (latency_instrumentation_enabled_ && timer_) {
        timer_->end("system_total");
    }
    
    return result;
}

std::unordered_map<std::string, std::vector<std::string>> 
SystemLevelScheduler::buildDependencyGraph(
    const std::vector<std::pair<std::string, std::string>>& edges
) {
    std::unordered_map<std::string, std::vector<std::string>> dependencies;
    
    for (const auto& edge : edges) {
        dependencies[edge.second].push_back(edge.first);
    }
    
    return dependencies;
}

std::unordered_map<std::string, int32_t> 
SystemLevelScheduler::computeSuccessorCounts(
    const std::unordered_map<std::string, std::vector<std::string>>& dependencies,
    const std::unordered_map<std::string, OperatorScheduledIRNode>& nodes
) {
    std::unordered_map<std::string, int32_t> successor_counts;
    
    // Build reverse dependency graph (successors)
    std::unordered_map<std::string, std::vector<std::string>> successors;
    for (const auto& dep_pair : dependencies) {
        const std::string& target = dep_pair.first;
        for (const std::string& source : dep_pair.second) {
            successors[source].push_back(target);
        }
    }
    
    // Count successors recursively (DFS)
    std::unordered_map<std::string, bool> visited;
    std::function<int32_t(const std::string&)> countSuccessors = 
        [&](const std::string& op) -> int32_t {
        if (visited[op]) return successor_counts[op];
        
        visited[op] = true;
        int32_t count = 0;
        
        for (const std::string& successor : successors[op]) {
            count += 1 + countSuccessors(successor);
        }
        
        successor_counts[op] = count;
        return count;
    };
    
    // Compute for all nodes
    for (const auto& node_pair : nodes) {
        if (!visited[node_pair.first]) {
            countSuccessors(node_pair.first);
        }
    }
    
    return successor_counts;
}

std::unordered_map<std::string, double> 
SystemLevelScheduler::computeCriticalResourceImpact(
    const std::unordered_map<std::string, OperatorScheduledIRNode>& nodes
) {
    std::unordered_map<std::string, double> critical_impacts;
    
    // Normalize factors: duration, memory usage, compute intensity
    double max_duration = 1.0;
    double max_memory = 1.0;
    double max_compute = 1.0;
    
    // Find maximums for normalization
    for (const auto& node_pair : nodes) {
        const auto& node = node_pair.second;
        max_duration = std::max(max_duration, static_cast<double>(node.duration));
        
        // Estimate memory usage from input/output tensors
        double memory_usage = 0.0;
        for (const auto& input : node.mapped_node.op_node.inputs) {
            memory_usage += 1.0; // Simplified tensor size estimation
        }
        max_memory = std::max(max_memory, memory_usage);
        
        // Estimate compute intensity from optimization speedup
        double compute_intensity = node.optimization_result.base_duration / 
                                 std::max(1.0, static_cast<double>(node.duration));
        max_compute = std::max(max_compute, compute_intensity);
    }
    
    // Compute normalized critical impact
    for (const auto& node_pair : nodes) {
        const std::string& op_id = node_pair.first;
        const auto& node = node_pair.second;
        
        // Normalized components
        double duration_factor = node.duration / max_duration;
        
        double memory_factor = 0.5; // Simplified
        for (const auto& input : node.mapped_node.op_node.inputs) {
            memory_factor += 0.1; // Simplified
        }
        memory_factor = std::min(memory_factor / max_memory, 1.0);
        
        double compute_factor = (node.optimization_result.base_duration / 
                               std::max(1.0, static_cast<double>(node.duration))) / max_compute;
        
        // Combined impact (weighted sum)
        critical_impacts[op_id] = 0.5 * duration_factor + 
                                 0.3 * memory_factor + 
                                 0.2 * compute_factor;
    }
    
    return critical_impacts;
}

double SystemLevelScheduler::calculateDAGSScore(
    const std::string& op_id,
    const std::unordered_map<std::string, int32_t>& successor_counts,
    const std::unordered_map<std::string, double>& critical_impacts
) {
    double successor_score = successor_counts.count(op_id) ? successor_counts.at(op_id) : 0.0;
    double impact_score = critical_impacts.count(op_id) ? critical_impacts.at(op_id) : 0.0;
    
    return config_.alpha * successor_score + config_.beta * impact_score;
}

int64_t SystemLevelScheduler::findEarliestSlot(
    const OperatorScheduledIRNode& node,
    const std::unordered_map<std::string, int64_t>& hw_finish_times,
    const std::unordered_map<std::string, int64_t>& op_finish_times
) {
    // Hardware unit availability
    int64_t hw_available = 0;
    auto hw_it = hw_finish_times.find(node.mapped_node.hw_unit);
    if (hw_it != hw_finish_times.end()) {
        hw_available = hw_it->second;
    }
    
    // Dependency availability - find the latest finish time of all dependencies
    int64_t dep_available = 0;
    // Note: We need to check dependencies explicitly here since ready queue 
    // only ensures they're completed, not when they finish
    
    return std::max(hw_available, dep_available);
}

void SystemLevelScheduler::updateReadyQueue(
    std::priority_queue<std::pair<double, std::string>>& ready_queue,
    const std::unordered_map<std::string, std::vector<std::string>>& dependencies,
    const std::unordered_set<std::string>& scheduled_ops,
    const std::unordered_map<std::string, OperatorScheduledIRNode>& all_nodes,
    const std::unordered_map<std::string, int32_t>& successor_counts,
    const std::unordered_map<std::string, double>& critical_impacts
) {
    // Find newly ready operations
    for (const auto& node_pair : all_nodes) {
        const std::string& op_id = node_pair.first;
        
        // Skip if already scheduled
        if (scheduled_ops.count(op_id)) {
            continue;
        }
        
        // Check if all dependencies are satisfied
        bool all_deps_satisfied = true;
        auto deps_it = dependencies.find(op_id);
        if (deps_it != dependencies.end()) {
            for (const std::string& dep : deps_it->second) {
                if (!scheduled_ops.count(dep)) {
                    all_deps_satisfied = false;
                    break;
                }
            }
        }
        
        if (all_deps_satisfied) {
            double score = calculateDAGSScore(op_id, successor_counts, critical_impacts);
            ready_queue.push({score, op_id});
        }
    }
}

void SystemLevelScheduler::updateSystemStats(
    const SystemSchedule& schedule,
    const std::unordered_map<std::string, OperatorScheduledIRNode>& nodes
) {
    // Calculate hardware unit utilizations
    std::unordered_map<std::string, double> hw_work_time;
    double total_work_time = 0.0;
    
    for (const auto& entry : schedule.entries) {
        hw_work_time[entry.hw_unit] += entry.duration;
        total_work_time += entry.duration;
    }
    
    for (const auto& work_pair : hw_work_time) {
        const std::string& hw_unit = work_pair.first;
        double utilization = work_pair.second / schedule.total_cycles;
        last_stats_.hw_unit_utilizations[hw_unit] = utilization;
    }
    
    // Calculate scheduling efficiency (vs critical path)
    double critical_path_estimate = 0.0;
    for (const auto& node_pair : nodes) {
        critical_path_estimate += node_pair.second.duration;
    }
    critical_path_estimate /= last_stats_.hw_unit_utilizations.size(); // Rough estimate
    
    last_stats_.scheduling_efficiency = critical_path_estimate / schedule.total_cycles;
    
    // Calculate resource balance factor (lower is better)
    if (!last_stats_.hw_unit_utilizations.empty()) {
        double mean_util = total_work_time / (schedule.total_cycles * last_stats_.hw_unit_utilizations.size());
        double variance = 0.0;
        for (const auto& util_pair : last_stats_.hw_unit_utilizations) {
            double diff = util_pair.second - mean_util;
            variance += diff * diff;
        }
        variance /= last_stats_.hw_unit_utilizations.size();
        last_stats_.resource_balance_factor = std::sqrt(variance);
    }
}

bool SystemLevelScheduler::validateSchedule(
    const SystemSchedule& schedule,
    const std::vector<std::pair<std::string, std::string>>& edges
) {
    // Build schedule map for quick lookup
    std::unordered_map<std::string, const SystemScheduleEntry*> schedule_map;
    for (const auto& entry : schedule.entries) {
        schedule_map[entry.op_id] = &entry;
    }
    
    // Check dependency ordering
    for (const auto& edge : edges) {
        const std::string& source = edge.first;
        const std::string& target = edge.second;
        
        auto source_it = schedule_map.find(source);
        auto target_it = schedule_map.find(target);
        
        if (source_it != schedule_map.end() && target_it != schedule_map.end()) {
            int64_t source_finish = source_it->second->start_cycle + source_it->second->duration;
            int64_t target_start = target_it->second->start_cycle;
            
            if (source_finish > target_start) {
                return false; // Dependency violation
            }
        }
    }
    
    return true;
}

// SystemSchedulerFactory implementation
std::unique_ptr<SystemLevelScheduler> SystemSchedulerFactory::createScheduler(
    SchedulerType type,
    const DAGSConfig& config
) {
    switch (type) {
        case SchedulerType::DAGS:
            return std::make_unique<SystemLevelScheduler>(config);
        case SchedulerType::LIST_BASED:
            // TODO: Implement list-based scheduler
            return std::make_unique<SystemLevelScheduler>(config);
        case SchedulerType::CRITICAL_PATH:
            // TODO: Implement critical path scheduler  
            return std::make_unique<SystemLevelScheduler>(config);
        default:
            return std::make_unique<SystemLevelScheduler>(config);
    }
}

SchedulingLatencyReport SystemLevelScheduler::getLatencyReport() const {
    SchedulingLatencyReport report;
    
    if (timer_) {
        // System-level scheduler stages
        report.system_dependency_graph = LatencyStats(
            timer_->getTotalDuration("system_dependency_graph"),
            timer_->getAverageDuration("system_dependency_graph"),
            timer_->getLastDuration("system_dependency_graph"),
            timer_->getMeasurementCount("system_dependency_graph")
        );
        
        report.system_heuristic_computation = LatencyStats(
            timer_->getTotalDuration("system_heuristic_computation"),
            timer_->getAverageDuration("system_heuristic_computation"),
            timer_->getLastDuration("system_heuristic_computation"),
            timer_->getMeasurementCount("system_heuristic_computation")
        );
        
        report.system_scheduling_loop = LatencyStats(
            timer_->getTotalDuration("system_scheduling_loop"),
            timer_->getAverageDuration("system_scheduling_loop"),
            timer_->getLastDuration("system_scheduling_loop"),
            timer_->getMeasurementCount("system_scheduling_loop")
        );
        
        report.system_finalization = LatencyStats(
            timer_->getTotalDuration("system_finalization"),
            timer_->getAverageDuration("system_finalization"),
            timer_->getLastDuration("system_finalization"),
            timer_->getMeasurementCount("system_finalization")
        );
        
        report.system_total = LatencyStats(
            timer_->getTotalDuration("system_total"),
            timer_->getAverageDuration("system_total"),
            timer_->getLastDuration("system_total"),
            timer_->getMeasurementCount("system_total")
        );
    }
    
    return report;
}

} // namespace rendersim 