#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace rendersim {

/** Ramulator 2.0 integration for DRAM timing statistics. */

struct MemoryAccessPattern {
    std::string pattern_type;        // "sequential", "random", "streaming", etc.
    size_t access_size_bytes;        // Size of each access
    std::vector<uint64_t> addresses; // Memory addresses being accessed
    double access_frequency_hz;      // Frequency of accesses
    
    MemoryAccessPattern() : access_size_bytes(64), access_frequency_hz(1e9) {}
};

struct DRAMTimingResult {
    double average_latency_ns;
    double peak_bandwidth_gb_s;
    double effective_bandwidth_gb_s;
    double power_consumption_mw;
    double row_buffer_hit_rate;
    size_t total_accesses;
    size_t row_buffer_hits;
    size_t row_buffer_misses;
    
    DRAMTimingResult() : average_latency_ns(0.0), peak_bandwidth_gb_s(0.0), 
                        effective_bandwidth_gb_s(0.0), power_consumption_mw(0.0),
                        row_buffer_hit_rate(0.0), total_accesses(0), 
                        row_buffer_hits(0), row_buffer_misses(0) {}
};

/** Ramulator 2.0 configuration. */
struct Ramulator2Config {
    // DRAM Configuration
    std::string dram_type;           // "DDR4", "DDR5", "HBM2", "LPDDR4", etc.
    std::string dram_density;        // "8Gb", "16Gb", etc.
    std::string dram_width;          // "x8", "x16", etc.
    int32_t frequency_mhz;           // DRAM frequency
    
    // System Configuration
    int32_t channels;                // Number of channels
    int32_t ranks_per_channel;       // Ranks per channel
    int32_t banks_per_rank;          // Banks per rank
    
    // Memory Controller Configuration
    std::string scheduling_policy;   // "FCFS", "FR_FCFS", "PAR_BS", etc.
    std::string rowpolicy;          // "opened", "closed", "timeout"
    int32_t req_queue_size;         // Request queue size
    
    // Power Model
    bool enable_power_model;
    std::string power_config;
    
    Ramulator2Config() : dram_type("DDR4"), dram_density("8Gb"), dram_width("x8"),
                        frequency_mhz(3200), channels(4), ranks_per_channel(1),
                        banks_per_rank(16), scheduling_policy("FR_FCFS"),
                        rowpolicy("opened"), req_queue_size(128),
                        enable_power_model(true), power_config("default") {}
};

/** C++ interface to Ramulator 2.0. */
class Ramulator2Interface {
public:
    explicit Ramulator2Interface(const Ramulator2Config& config = Ramulator2Config());
    ~Ramulator2Interface();
    
    /**
     * Initialize Ramulator 2.0 with the given configuration
     */
    bool initialize(const std::string& ramulator_path = "Hardware/ramulator2");
    
    /**
     * Run simulation with memory access pattern and return timing statistics
     */
    DRAMTimingResult simulateMemoryAccess(const MemoryAccessPattern& pattern,
                                         double simulation_time_ns = 100000.0);
    
    /** Simulate a neural rendering memory pattern. */
    DRAMTimingResult simulateNeuralRenderingWorkload(
        size_t hash_table_accesses,
        size_t weight_matrix_accesses,
        size_t volume_data_accesses,
        double execution_time_ns);
    
    /**
     * Get DRAM configuration details
     */
    const Ramulator2Config& getConfig() const { return config_; }
    
    /**
     * Update configuration and reinitialize
     */
    bool updateConfig(const Ramulator2Config& new_config);
    
    /**
     * Generate Ramulator 2.0 YAML configuration file
     */
    std::string generateConfigYAML() const;
    
    /**
     * Get supported DRAM types
     */
    static std::vector<std::string> getSupportedDRAMTypes();

private:
    Ramulator2Config config_;
    std::string ramulator_path_;
    std::string config_file_path_;
    std::string trace_file_path_;
    bool initialized_;
    
    // Internal methods
    bool writeConfigFile();
    bool generateTraceFile(const MemoryAccessPattern& pattern);
    DRAMTimingResult parseRamulatorOutput(const std::string& output_file);
    std::string runRamulatorSimulation(const std::string& trace_file);
    
    // Neural rendering specific trace generation
    void generateHashTableTrace(std::vector<uint64_t>& addresses, size_t num_accesses);
    void generateWeightMatrixTrace(std::vector<uint64_t>& addresses, size_t num_accesses);
    void generateVolumeDataTrace(std::vector<uint64_t>& addresses, size_t num_accesses);
};

/** Factory for Ramulator configurations. */
class NeuralRenderingDRAMConfigFactory {
public:
    /**
     * Get optimized DRAM configuration for specific accelerator
     */
    static Ramulator2Config getConfigForAccelerator(const std::string& accelerator_name);
    
    /**
     * Get configuration optimized for high bandwidth (e.g., for volume rendering)
     */
    static Ramulator2Config getHighBandwidthConfig();
    
    /**
     * Get configuration optimized for low latency (e.g., for hash table lookups)
     */
    static Ramulator2Config getLowLatencyConfig();
    
    /**
     * Get configuration optimized for power efficiency
     */
    static Ramulator2Config getPowerEfficientConfig();
};

} // namespace rendersim 