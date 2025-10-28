#include "RenderSim/system_scheduler.hpp"
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace rendersim {

// Training-aware system scheduler extensions
class TrainingAwareSystemScheduler : public SystemLevelScheduler {
public:
    SystemSchedule scheduleTraining(const OperatorScheduledIR& op_scheduled_ir) {
        SystemSchedule schedule;
        
        // Separate forward and backward operators
        std::vector<std::string> forward_ops, backward_ops;
        for (const auto& [id, node] : op_scheduled_ir.nodes) {
            if (node.mapped_node.op_node.op_type.find("(B)") != std::string::npos) {
                backward_ops.push_back(id);
            } else {
                forward_ops.push_back(id);
            }
        }
        
        // Schedule forward pass first
        auto forward_schedule = schedulePhase(op_scheduled_ir, forward_ops, 0);
        
        // Schedule backward pass with dependency on forward completion
        int backward_start = forward_schedule.total_cycles;
        auto backward_schedule = schedulePhase(op_scheduled_ir, backward_ops, backward_start);
        
        // Merge schedules
        schedule.total_cycles = backward_schedule.total_cycles;
        schedule.node_schedules = forward_schedule.node_schedules;
        for (const auto& [id, node_sched] : backward_schedule.node_schedules) {
            schedule.node_schedules[id] = node_sched;
        }
        
        // Add training-specific metrics
        computeTrainingMetrics(schedule, op_scheduled_ir);
        
        return schedule;
    }
    
private:
    SystemSchedule schedulePhase(
        const OperatorScheduledIR& op_scheduled_ir,
        const std::vector<std::string>& phase_ops,
        int start_cycle) {
        
        SystemSchedule schedule;
        
        // Build dependency graph for this phase
        std::unordered_map<std::string, std::vector<std::string>> dependencies;
        std::unordered_map<std::string, int> in_degree;
        
        for (const auto& edge : op_scheduled_ir.edges) {
            if (std::find(phase_ops.begin(), phase_ops.end(), edge.first) != phase_ops.end() &&
                std::find(phase_ops.begin(), phase_ops.end(), edge.second) != phase_ops.end()) {
                dependencies[edge.first].push_back(edge.second);
                in_degree[edge.second]++;
            }
        }
        
        // Initialize in_degree for all nodes
        for (const auto& op : phase_ops) {
            if (in_degree.find(op) == in_degree.end()) {
                in_degree[op] = 0;
            }
        }
        
        // Priority queue for ready operations (prioritize by critical path)
        auto priority_comp = [&](const std::string& a, const std::string& b) {
            return computeCriticalPath(a, op_scheduled_ir) < 
                   computeCriticalPath(b, op_scheduled_ir);
        };
        std::priority_queue<std::string, std::vector<std::string>, 
                          decltype(priority_comp)> ready_queue(priority_comp);
        
        // Initialize with nodes that have no dependencies
        for (const auto& [op, degree] : in_degree) {
            if (degree == 0) {
                ready_queue.push(op);
            }
        }
        
        // Track resource availability
        std::unordered_map<std::string, int> hw_unit_available_at;
        int current_cycle = start_cycle;
        
        // Schedule operations
        while (!ready_queue.empty()) {
            std::string op = ready_queue.top();
            ready_queue.pop();
            
            const auto& node = op_scheduled_ir.nodes.at(op);
            std::string hw_unit = node.mapped_node.hw_unit;
            
            // Find earliest time this operation can start
            int earliest_start = current_cycle;
            
            // Check hardware availability
            if (hw_unit_available_at.find(hw_unit) != hw_unit_available_at.end()) {
                earliest_start = std::max(earliest_start, hw_unit_available_at[hw_unit]);
            }
            
            // Check data dependencies
            for (const auto& [pred_id, pred_sched] : schedule.node_schedules) {
                bool is_dependency = false;
                for (const auto& edge : op_scheduled_ir.edges) {
                    if (edge.first == pred_id && edge.second == op) {
                        is_dependency = true;
                        break;
                    }
                }
                if (is_dependency) {
                    earliest_start = std::max(earliest_start, pred_sched.end_cycle);
                }
            }
            
            // Schedule the operation
            SystemNodeSchedule node_schedule;
            node_schedule.start_cycle = earliest_start;
            node_schedule.end_cycle = earliest_start + node.duration;
            node_schedule.hw_unit = hw_unit;
            
            schedule.node_schedules[op] = node_schedule;
            
            // Update hardware availability
            hw_unit_available_at[hw_unit] = node_schedule.end_cycle;
            
            // Update ready queue with newly ready operations
            if (dependencies.find(op) != dependencies.end()) {
                for (const auto& successor : dependencies[op]) {
                    in_degree[successor]--;
                    if (in_degree[successor] == 0) {
                        ready_queue.push(successor);
                    }
                }
            }
            
            // Update total cycles
            schedule.total_cycles = std::max(schedule.total_cycles, node_schedule.end_cycle);
        }
        
        return schedule;
    }
    
    int computeCriticalPath(const std::string& op, 
                           const OperatorScheduledIR& op_scheduled_ir) {
        // Compute critical path length from this operation
        // This is a simplified version - full implementation would use dynamic programming
        const auto& node = op_scheduled_ir.nodes.at(op);
        int path_length = node.duration;
        
        // Find longest path through successors
        int max_successor_path = 0;
        for (const auto& edge : op_scheduled_ir.edges) {
            if (edge.first == op) {
                int successor_path = computeCriticalPath(edge.second, op_scheduled_ir);
                max_successor_path = std::max(max_successor_path, successor_path);
            }
        }
        
        return path_length + max_successor_path;
    }
    
    void computeTrainingMetrics(SystemSchedule& schedule, 
                               const OperatorScheduledIR& op_scheduled_ir) {
        // Compute training-specific metrics
        
        // 1. Forward-backward ratio
        int forward_cycles = 0, backward_cycles = 0;
        for (const auto& [id, node_sched] : schedule.node_schedules) {
            int duration = node_sched.end_cycle - node_sched.start_cycle;
            if (op_scheduled_ir.nodes.at(id).mapped_node.op_node.op_type.find("(B)") 
                != std::string::npos) {
                backward_cycles += duration;
            } else {
                forward_cycles += duration;
            }
        }
        
        // 2. Hardware utilization
        std::unordered_map<std::string, int> hw_active_cycles;
        std::unordered_map<std::string, int> hw_total_cycles;
        
        for (const auto& [id, node_sched] : schedule.node_schedules) {
            std::string hw_unit = node_sched.hw_unit;
            int duration = node_sched.end_cycle - node_sched.start_cycle;
            hw_active_cycles[hw_unit] += duration;
            hw_total_cycles[hw_unit] = std::max(hw_total_cycles[hw_unit], 
                                               node_sched.end_cycle);
        }
        
        double total_utilization = 0.0;
        int num_units = 0;
        for (const auto& [hw_unit, active] : hw_active_cycles) {
            if (hw_total_cycles[hw_unit] > 0) {
                double utilization = static_cast<double>(active) / hw_total_cycles[hw_unit];
                total_utilization += utilization;
                num_units++;
            }
        }
        
        // Store metrics in schedule (would need to extend SystemSchedule struct)
        // For now, just compute them
        double avg_utilization = num_units > 0 ? total_utilization / num_units : 0.0;
        double fb_ratio = forward_cycles > 0 ? 
                         static_cast<double>(backward_cycles) / forward_cycles : 0.0;
        
        // Log metrics (in real implementation, store in schedule)
        // schedule.metrics["avg_hw_utilization"] = avg_utilization;
        // schedule.metrics["forward_backward_ratio"] = fb_ratio;
        // schedule.metrics["forward_cycles"] = forward_cycles;
        // schedule.metrics["backward_cycles"] = backward_cycles;
    }
};

// Pipeline-specific scheduling strategies
class GSArchSchedulingStrategy {
public:
    static void applyTileMergingSchedule(SystemSchedule& schedule,
                                         const OperatorScheduledIR& ir) {
        // GSArch-specific: Schedule tile merging operations to maximize locality
        // Group tile operations that access the same memory regions
        
        std::vector<std::string> tile_ops;
        for (const auto& [id, node] : ir.nodes) {
            if (node.mapped_node.op_node.op_type.find("TILE") != std::string::npos) {
                tile_ops.push_back(id);
            }
        }
        
        // Sort by spatial locality (simplified - would use tile coordinates in practice)
        std::sort(tile_ops.begin(), tile_ops.end());
        
        // Adjust schedule to process tiles in locality-aware order
        int prev_end = 0;
        for (const auto& op : tile_ops) {
            if (schedule.node_schedules.find(op) != schedule.node_schedules.end()) {
                auto& node_sched = schedule.node_schedules[op];
                int duration = node_sched.end_cycle - node_sched.start_cycle;
                node_sched.start_cycle = prev_end;
                node_sched.end_cycle = prev_end + duration;
                prev_end = node_sched.end_cycle;
            }
        }
    }
};

class GBUSchedulingStrategy {
public:
    static void applyRowMajorSchedule(SystemSchedule& schedule,
                                      const OperatorScheduledIR& ir) {
        // GBU-specific: Schedule row processing in row-major order
        // This improves cache locality for Gaussian bundle processing
        
        std::vector<std::string> row_ops;
        for (const auto& [id, node] : ir.nodes) {
            if (node.mapped_node.op_node.op_type.find("ROW") != std::string::npos) {
                row_ops.push_back(id);
            }
        }
        
        // Process rows sequentially for better cache behavior
        int row_stride = 256;  // From paper
        for (size_t i = 0; i < row_ops.size(); ++i) {
            if (schedule.node_schedules.find(row_ops[i]) != schedule.node_schedules.end()) {
                auto& node_sched = schedule.node_schedules[row_ops[i]];
                // Ensure sequential row processing
                if (i > 0 && schedule.node_schedules.find(row_ops[i-1]) != 
                    schedule.node_schedules.end()) {
                    auto& prev_sched = schedule.node_schedules[row_ops[i-1]];
                    if (node_sched.start_cycle < prev_sched.end_cycle) {
                        int duration = node_sched.end_cycle - node_sched.start_cycle;
                        node_sched.start_cycle = prev_sched.end_cycle;
                        node_sched.end_cycle = node_sched.start_cycle + duration;
                    }
                }
            }
        }
    }
};

class Instant3DSchedulingStrategy {
public:
    static void applyAsymmetricSchedule(SystemSchedule& schedule,
                                        const OperatorScheduledIR& ir) {
        // Instant3D-specific: Different scheduling for FRM (forward) vs BUM (backward)
        
        std::vector<std::string> frm_ops, bum_ops;
        for (const auto& [id, node] : ir.nodes) {
            if (node.mapped_node.op_node.op_type == "FRM") {
                frm_ops.push_back(id);
            } else if (node.mapped_node.op_node.op_type == "BUM") {
                bum_ops.push_back(id);
            }
        }
        
        // FRM operations: maximize parallelism for reads
        // Schedule FRM ops to overlap when possible
        if (frm_ops.size() > 1) {
            int parallel_factor = 2;  // From paper: 2 FRM units
            for (size_t i = 0; i < frm_ops.size(); ++i) {
                if (schedule.node_schedules.find(frm_ops[i]) != schedule.node_schedules.end()) {
                    auto& node_sched = schedule.node_schedules[frm_ops[i]];
                    // Distribute across parallel FRM units
                    int unit_id = i % parallel_factor;
                    node_sched.hw_unit = "FRM_" + std::to_string(unit_id);
                }
            }
        }
        
        // BUM operations: serialize to avoid write conflicts
        // Process BUM operations sequentially for correct gradient accumulation
        for (size_t i = 1; i < bum_ops.size(); ++i) {
            if (schedule.node_schedules.find(bum_ops[i]) != schedule.node_schedules.end() &&
                schedule.node_schedules.find(bum_ops[i-1]) != schedule.node_schedules.end()) {
                auto& curr_sched = schedule.node_schedules[bum_ops[i]];
                auto& prev_sched = schedule.node_schedules[bum_ops[i-1]];
                
                if (curr_sched.start_cycle < prev_sched.end_cycle) {
                    int duration = curr_sched.end_cycle - curr_sched.start_cycle;
                    curr_sched.start_cycle = prev_sched.end_cycle;
                    curr_sched.end_cycle = curr_sched.start_cycle + duration;
                }
            }
        }
    }
};

} // namespace rendersim
