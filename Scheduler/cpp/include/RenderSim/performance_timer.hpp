#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace rendersim {

/**
 * High-resolution performance timer for latency instrumentation
 */
class PerformanceTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::nanoseconds;
    
    PerformanceTimer() = default;
    
    /**
     * Start timing for a named stage
     */
    void start(const std::string& stage_name) {
        start_times_[stage_name] = Clock::now();
    }
    
    /**
     * End timing for a named stage and record the duration
     */
    void end(const std::string& stage_name) {
        auto end_time = Clock::now();
        auto start_it = start_times_.find(stage_name);
        
        if (start_it != start_times_.end()) {
            auto duration = std::chrono::duration_cast<Duration>(end_time - start_it->second);
            durations_[stage_name].push_back(duration);
            start_times_.erase(start_it);
        }
    }
    
    /**
     * Get the last recorded duration for a stage in nanoseconds
     */
    int64_t getLastDuration(const std::string& stage_name) const {
        auto it = durations_.find(stage_name);
        if (it != durations_.end() && !it->second.empty()) {
            return it->second.back().count();
        }
        return 0;
    }
    
    /**
     * Get the average duration for a stage in nanoseconds
     */
    double getAverageDuration(const std::string& stage_name) const {
        auto it = durations_.find(stage_name);
        if (it != durations_.end() && !it->second.empty()) {
            int64_t total = 0;
            for (const auto& duration : it->second) {
                total += duration.count();
            }
            return static_cast<double>(total) / it->second.size();
        }
        return 0.0;
    }
    
    /**
     * Get total duration for a stage across all measurements in nanoseconds
     */
    int64_t getTotalDuration(const std::string& stage_name) const {
        auto it = durations_.find(stage_name);
        if (it != durations_.end()) {
            int64_t total = 0;
            for (const auto& duration : it->second) {
                total += duration.count();
            }
            return total;
        }
        return 0;
    }
    
    /**
     * Get number of measurements for a stage
     */
    size_t getMeasurementCount(const std::string& stage_name) const {
        auto it = durations_.find(stage_name);
        return it != durations_.end() ? it->second.size() : 0;
    }
    
    /**
     * Get all stage names that have been measured
     */
    std::vector<std::string> getStageNames() const {
        std::vector<std::string> names;
        for (const auto& pair : durations_) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    /**
     * Clear all timing data
     */
    void clear() {
        start_times_.clear();
        durations_.clear();
    }
    
    /**
     * Reset timing data for a specific stage
     */
    void reset(const std::string& stage_name) {
        start_times_.erase(stage_name);
        durations_.erase(stage_name);
    }

private:
    std::unordered_map<std::string, TimePoint> start_times_;
    std::unordered_map<std::string, std::vector<Duration>> durations_;
};

/**
 * RAII timer that automatically starts and ends timing for a stage
 */
class ScopedTimer {
public:
    ScopedTimer(std::shared_ptr<PerformanceTimer> timer, const std::string& stage_name)
        : timer_(timer), stage_name_(stage_name) {
        if (timer_) {
            timer_->start(stage_name_);
        }
    }
    
    ~ScopedTimer() {
        if (timer_) {
            timer_->end(stage_name_);
        }
    }
    
    // Non-copyable, movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = default;
    ScopedTimer& operator=(ScopedTimer&&) = default;

private:
    std::shared_ptr<PerformanceTimer> timer_;
    std::string stage_name_;
};

/**
 * Latency statistics for scheduling stages
 */
struct LatencyStats {
    int64_t total_duration_ns;
    double average_duration_ns;
    int64_t last_duration_ns;
    size_t measurement_count;
    
    LatencyStats() 
        : total_duration_ns(0), average_duration_ns(0.0), 
          last_duration_ns(0), measurement_count(0) {}
    
    LatencyStats(int64_t total, double average, int64_t last, size_t count)
        : total_duration_ns(total), average_duration_ns(average),
          last_duration_ns(last), measurement_count(count) {}
};

/**
 * Comprehensive latency report for scheduling pipeline
 */
struct SchedulingLatencyReport {
    // Operator-level scheduler stages
    LatencyStats operator_hw_grouping;
    LatencyStats operator_hw_scheduling;
    LatencyStats operator_dependency_resolution;
    LatencyStats operator_total;
    
    // System-level scheduler stages  
    LatencyStats system_dependency_graph;
    LatencyStats system_heuristic_computation;
    LatencyStats system_scheduling_loop;
    LatencyStats system_finalization;
    LatencyStats system_total;
    
    // Overall pipeline
    LatencyStats pipeline_total;
    
    /**
     * Convert nanoseconds to human-readable string
     */
    static std::string formatDuration(int64_t nanoseconds) {
        if (nanoseconds < 1000) {
            return std::to_string(nanoseconds) + " ns";
        } else if (nanoseconds < 1000000) {
            return std::to_string(nanoseconds / 1000.0) + " μs";
        } else if (nanoseconds < 1000000000) {
            return std::to_string(nanoseconds / 1000000.0) + " ms";
        } else {
            return std::to_string(nanoseconds / 1000000000.0) + " s";
        }
    }
    
    /**
     * Generate human-readable report
     */
    std::string generateReport() const {
        std::string report;
        report += "=== RenderSim Scheduling Latency Report ===\n\n";
        
        report += "Operator-Level Scheduler:\n";
        report += "  Hardware Grouping: " + formatDuration(operator_hw_grouping.last_duration_ns) + "\n";
        report += "  Hardware Scheduling: " + formatDuration(operator_hw_scheduling.last_duration_ns) + "\n"; 
        report += "  Dependency Resolution: " + formatDuration(operator_dependency_resolution.last_duration_ns) + "\n";
        report += "  Total: " + formatDuration(operator_total.last_duration_ns) + "\n\n";
        
        report += "System-Level Scheduler:\n";
        report += "  Dependency Graph: " + formatDuration(system_dependency_graph.last_duration_ns) + "\n";
        report += "  Heuristic Computation: " + formatDuration(system_heuristic_computation.last_duration_ns) + "\n";
        report += "  Scheduling Loop: " + formatDuration(system_scheduling_loop.last_duration_ns) + "\n";
        report += "  Finalization: " + formatDuration(system_finalization.last_duration_ns) + "\n";
        report += "  Total: " + formatDuration(system_total.last_duration_ns) + "\n\n";
        
        report += "Pipeline Total: " + formatDuration(pipeline_total.last_duration_ns) + "\n";
        
        return report;
    }
};

/**
 * Macro for convenient scoped timing
 */
#define RENDERSIM_TIME_SCOPE(timer, stage_name) \
    rendersim::ScopedTimer scoped_timer_##__LINE__(timer, stage_name)

} // namespace rendersim 