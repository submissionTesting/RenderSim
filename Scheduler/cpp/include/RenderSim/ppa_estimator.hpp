#pragma once

#include "system_scheduler.hpp"
#include "ramulator_interface.hpp"
#include <string>
#include <unordered_map>
#include <memory>

namespace rendersim {

/**
 * PPA (Power, Performance, Area) Metrics
 */
struct PPAMetrics {
    // Performance metrics
    int32_t latency_cycles;
    double throughput_ops_per_sec;
    
    // Area metrics (μm²)
    double area_um2;
    double area_mm2() const { return area_um2 / 1e6; }
    
    // Power metrics (μW)
    double static_power_uw;
    double dynamic_power_uw;
    double total_power_uw() const { return static_power_uw + dynamic_power_uw; }
    
    // Memory metrics
    double dram_latency_ns;
    double dram_bandwidth_gb_s;
    
    PPAMetrics() : latency_cycles(0), throughput_ops_per_sec(0.0), area_um2(0.0),
                   static_power_uw(0.0), dynamic_power_uw(0.0), 
                   dram_latency_ns(0.0), dram_bandwidth_gb_s(0.0) {}
};

/**
 * Hardware Module Configuration for PPA Analysis
 */
struct HardwareModuleConfig {
    std::string module_name;
    std::string accelerator_type;  // ICARUS, NeuRex, CICERO, GSCore
    std::string hardware_path;     // Path in Hardware/ folder
    double clock_period_ns;
    std::string technology_node;   // e.g., "tn28rvt9t"
    
    HardwareModuleConfig() : clock_period_ns(1.0), technology_node("tn28rvt9t") {}
    
    HardwareModuleConfig(const std::string& name, const std::string& accel_type, 
                        const std::string& hw_path, double clk_period = 1.0,
                        const std::string& tech = "tn28rvt9t")
        : module_name(name), accelerator_type(accel_type), hardware_path(hw_path),
          clock_period_ns(clk_period), technology_node(tech) {}
};

/**
 * DRAM Configuration for Ramulator Integration
 */
struct DRAMConfig {
    std::string dram_type;         // DDR4, DDR5, HBM2, etc.
    int32_t capacity_gb;
    int32_t channels;
    int32_t ranks_per_channel;
    double frequency_mhz;
    
    DRAMConfig() : dram_type("DDR4"), capacity_gb(8), channels(4), 
                   ranks_per_channel(1), frequency_mhz(3200.0) {}
};

/** PPA estimator with Ramulator 2.0 integration. */

/** Hardware module PPA analyzer. */
class HardwareModuleAnalyzer {
public:
    explicit HardwareModuleAnalyzer(const std::string& hardware_base_path = "Hardware/");
    
    /**
     * Get PPA metrics for a hardware module by running HLS analysis
     */
    PPAMetrics analyzePPA(const HardwareModuleConfig& config);
    
    /**
     * Get cached PPA metrics (for validated accelerator modules)
     */
    PPAMetrics getCachedPPA(const std::string& accelerator_type, const std::string& module_name);
    
    /**
     * Register a new hardware module configuration
     */
    void registerModule(const HardwareModuleConfig& config);

private:
    std::string hardware_base_path_;
    std::unordered_map<std::string, PPAMetrics> ppa_cache_;
    
    void initializeValidatedModules();
    PPAMetrics parseHLSReport(const std::string& report_path);
    std::string generateHLSCommand(const HardwareModuleConfig& config);
};

/** Complete PPA estimator combining hardware and DRAM timing. */
class PPAEstimator {
public:
    explicit PPAEstimator(const Ramulator2Config& dram_config = Ramulator2Config(), 
                         const std::string& hardware_path = "Hardware/");
    
    // Set global clock period (ns) used to convert cycles to time
    void setClockPeriodNs(double clk_ns) { clock_period_ns_ = clk_ns; }
    double getClockPeriodNs() const { return clock_period_ns_; }
    
    /**
     * Estimate complete PPA metrics for a system schedule
     */
    struct SystemPPAMetrics {
        PPAMetrics total_metrics;
        std::unordered_map<std::string, PPAMetrics> per_hw_unit_metrics;
        
        // System-level metrics
        double total_execution_time_ns;
        double total_area_mm2;
        double total_power_mw;
        double average_memory_bandwidth_gb_s;
        
        // Validation metrics (compared to reference implementations)
        double area_error_percentage;
        double power_error_percentage;
        double latency_error_percentage;
        
        SystemPPAMetrics() : total_execution_time_ns(0.0), total_area_mm2(0.0), 
                           total_power_mw(0.0), average_memory_bandwidth_gb_s(0.0),
                           area_error_percentage(0.0), power_error_percentage(0.0), 
                           latency_error_percentage(0.0) {}
    };
    
    SystemPPAMetrics estimateSystemPPA(const SystemSchedule& schedule,
                                      const std::unordered_map<std::string, HardwareModuleConfig>& hw_configs);
    
    /**
     * Validate PPA estimation against reference implementation
     */
    struct ValidationResult {
        double area_error_percent;
        double power_error_percent;
        double latency_error_percent;
        double overall_error_percent;
        bool meets_target_accuracy;  // <10% error target
        
        ValidationResult() : area_error_percent(0.0), power_error_percent(0.0),
                           latency_error_percent(0.0), overall_error_percent(0.0),
                           meets_target_accuracy(false) {}
    };
    
    ValidationResult validateAccuracy(const SystemPPAMetrics& estimated,
                                    const SystemPPAMetrics& reference);
    
    /**
     * Get built-in hardware configurations for validated accelerators
     */
    std::unordered_map<std::string, HardwareModuleConfig> getValidatedConfigs(const std::string& accelerator);

private:
    std::unique_ptr<Ramulator2Interface> ramulator_;
    std::unique_ptr<HardwareModuleAnalyzer> hw_analyzer_;
    double clock_period_ns_ {1.0};
    
    void initializeValidatedAccelerators();
    PPAMetrics estimateMemorySubsystem(const SystemSchedule& schedule);
    double calculateTotalMemoryAccess(const SystemSchedule& schedule);
    MemoryAccessPattern analyzeScheduleMemoryPattern(const SystemSchedule& schedule);
};

/** PPA report generator. */
class PPAReportGenerator {
public:
    struct AcceleratorComparison {
        std::string accelerator_name;
        std::vector<std::string> module_names;
        std::vector<PPAMetrics> rendersim_results;
        std::vector<PPAMetrics> reference_results;
        double average_error_percent;
    };
    
    /** Generate a comparison table for hardware modules. */
    static std::string generateModuleComparisonTable(const std::vector<AcceleratorComparison>& comparisons);
    
    /**
     * Generate end-to-end comparison table
     */
    static std::string generateEndToEndTable(const std::unordered_map<std::string, PPAEstimator::SystemPPAMetrics>& results);
    
    /**
     * Generate detailed PPA report for a single accelerator
     */
    static std::string generateDetailedReport(const std::string& accelerator_name,
                                            const PPAEstimator::SystemPPAMetrics& metrics);
};

} // namespace rendersim 