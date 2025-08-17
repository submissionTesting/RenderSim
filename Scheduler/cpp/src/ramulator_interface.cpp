#include "RenderSim/ramulator_interface.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <algorithm>
#include <regex>

namespace rendersim {

// =============================================================================
// Ramulator2Interface Implementation
// =============================================================================

Ramulator2Interface::Ramulator2Interface(const Ramulator2Config& config) 
    : config_(config), initialized_(false) {
    ramulator_path_ = "Hardware/ramulator2";
    config_file_path_ = "/tmp/rendersim_ramulator_config.yaml";
    trace_file_path_ = "/tmp/rendersim_memory_trace.txt";
}

Ramulator2Interface::~Ramulator2Interface() {
    // Clean up temporary files
    std::filesystem::remove(config_file_path_);
    std::filesystem::remove(trace_file_path_);
}

bool Ramulator2Interface::initialize(const std::string& ramulator_path) {
    ramulator_path_ = ramulator_path;
    
    // Check if Ramulator 2.0 exists and is built
    std::string ramulator_binary = ramulator_path_ + "/ramulator2";
    if (!std::filesystem::exists(ramulator_binary)) {
        // Try to build Ramulator 2.0
        std::string build_cmd = "cd " + ramulator_path_ + " && mkdir -p build && cd build && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .. && make -j$(nproc)";
        int result = std::system(build_cmd.c_str());
        if (result != 0) {
            return false;
        }
        ramulator_binary = ramulator_path_ + "/build/ramulator2";
    }
    
    if (!std::filesystem::exists(ramulator_binary)) {
        return false;
    }
    
    // Write initial configuration file
    if (!writeConfigFile()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool Ramulator2Interface::writeConfigFile() {
    std::ofstream config_file(config_file_path_);
    if (!config_file.is_open()) {
        return false;
    }
    
    config_file << generateConfigYAML();
    config_file.close();
    return true;
}

std::string Ramulator2Interface::generateConfigYAML() const {
    std::stringstream yaml;
    
    yaml << "# RenderSim Generated Ramulator 2.0 Configuration\n";
    yaml << "# DRAM timing statistics via Ramulator 2.0\n\n";
    
    // Frontend Configuration
    yaml << "Frontend:\n";
    yaml << "  impl: SimpleO3\n";
    yaml << "  expected_limit_insts: 1000000\n\n";
    
    // MemorySystem Configuration
    yaml << "MemorySystem:\n";
    yaml << "  impl: GenericDRAMSystem\n";
    yaml << "  clock_freq: " << config_.frequency_mhz << "\n";
    yaml << "  DRAM:\n";
    yaml << "    impl: " << config_.dram_type << "\n";
    yaml << "    timing_preset: " << config_.dram_type << "_" << config_.frequency_mhz << "\n";
    yaml << "    org:\n";
    yaml << "      preset: " << config_.dram_type << "_" << config_.dram_density << "_" << config_.dram_width << "\n";
    yaml << "      channel: " << config_.channels << "\n";
    yaml << "      rank: " << config_.ranks_per_channel << "\n";
    
    if (config_.banks_per_rank > 0) {
        yaml << "      bank: " << config_.banks_per_rank << "\n";
    }
    
    // Controller Configuration
    yaml << "  Controller:\n";
    yaml << "    impl: Generic\n";
    yaml << "    Scheduler:\n";
    yaml << "      impl: " << config_.scheduling_policy << "\n";
    yaml << "    RowPolicy:\n";
    yaml << "      impl: " << config_.rowpolicy << "\n";
    yaml << "    Refresh:\n";
    yaml << "      impl: AllBank\n";
    yaml << "    req_queue_size_per_bank: " << config_.req_queue_size << "\n";
    
    // Power Model (if enabled)
    if (config_.enable_power_model) {
        yaml << "    PowerModel:\n";
        yaml << "      impl: DRAMPower\n";
    }
    
    // Statistics and Output
    yaml << "\n# Statistics Configuration\n";
    yaml << "Statistics:\n";
    yaml << "  impl: Default\n";
    yaml << "  print_stats: true\n";
    yaml << "  file_prefix: /tmp/rendersim_ramulator_stats\n\n";
    
    return yaml.str();
}

bool Ramulator2Interface::generateTraceFile(const MemoryAccessPattern& pattern) {
    std::ofstream trace_file(trace_file_path_);
    if (!trace_file.is_open()) {
        return false;
    }
    
    // Generate trace in Ramulator 2.0 format
    // Format: <access_type> <address> <# of bytes>
    // access_type: 0 = read, 1 = write
    
    for (size_t i = 0; i < pattern.addresses.size(); ++i) {
        uint64_t address = pattern.addresses[i];
        
        // For neural rendering, we mostly do reads (weights, hash tables, volumes)
        // with occasional writes (intermediate results)
        int access_type = (i % 10 == 0) ? 1 : 0;  // 10% writes, 90% reads
        
        trace_file << access_type << " " << std::hex << address << " " 
                   << std::dec << pattern.access_size_bytes << "\n";
    }
    
    trace_file.close();
    return true;
}

std::string Ramulator2Interface::runRamulatorSimulation(const std::string& trace_file) {
    if (!initialized_) {
        return "";
    }
    
    std::string ramulator_binary = ramulator_path_ + "/build/ramulator2";
    if (!std::filesystem::exists(ramulator_binary)) {
        ramulator_binary = ramulator_path_ + "/ramulator2";
    }
    
    std::string output_file = "/tmp/rendersim_ramulator_output.txt";
    std::string cmd = ramulator_binary + " -f " + config_file_path_ + 
                     " -t " + trace_file + " > " + output_file + " 2>&1";
    
    int result = std::system(cmd.c_str());
    if (result != 0) {
        return "";
    }
    
    return output_file;
}

DRAMTimingResult Ramulator2Interface::parseRamulatorOutput(const std::string& output_file) {
    DRAMTimingResult result;
    
    std::ifstream file(output_file);
    if (!file.is_open()) {
        return result;
    }
    
    std::string line;
    std::regex latency_regex(R"(Average\s+Memory\s+Access\s+Latency\s*:\s*([0-9.]+))");
    std::regex bandwidth_regex(R"(Memory\s+Bandwidth\s*:\s*([0-9.]+))");
    std::regex hit_rate_regex(R"(Row\s+Buffer\s+Hit\s+Rate\s*:\s*([0-9.]+))");
    std::regex power_regex(R"(Average\s+Power\s*:\s*([0-9.]+))");
    std::regex accesses_regex(R"(Total\s+Memory\s+Accesses\s*:\s*([0-9]+))");
    
    std::smatch match;
    
    while (std::getline(file, line)) {
        if (std::regex_search(line, match, latency_regex)) {
            result.average_latency_ns = std::stod(match[1].str());
        } else if (std::regex_search(line, match, bandwidth_regex)) {
            result.effective_bandwidth_gb_s = std::stod(match[1].str());
        } else if (std::regex_search(line, match, hit_rate_regex)) {
            result.row_buffer_hit_rate = std::stod(match[1].str());
        } else if (std::regex_search(line, match, power_regex)) {
            result.power_consumption_mw = std::stod(match[1].str());
        } else if (std::regex_search(line, match, accesses_regex)) {
            result.total_accesses = std::stoull(match[1].str());
        }
    }
    
    // Calculate derived metrics
    result.row_buffer_hits = static_cast<size_t>(result.total_accesses * result.row_buffer_hit_rate);
    result.row_buffer_misses = result.total_accesses - result.row_buffer_hits;
    
    // Estimate peak bandwidth based on DRAM configuration
    result.peak_bandwidth_gb_s = (config_.frequency_mhz * config_.channels * 8.0) / 1000.0;
    
    file.close();
    return result;
}

DRAMTimingResult Ramulator2Interface::simulateMemoryAccess(const MemoryAccessPattern& pattern,
                                                          double simulation_time_ns) {
    if (!initialized_ || pattern.addresses.empty()) {
        return DRAMTimingResult{};
    }
    
    // Generate trace file
    if (!generateTraceFile(pattern)) {
        return DRAMTimingResult{};
    }
    
    // Run Ramulator simulation
    std::string output_file = runRamulatorSimulation(trace_file_path_);
    if (output_file.empty()) {
        return DRAMTimingResult{};
    }
    
    // Parse results
    return parseRamulatorOutput(output_file);
}

void Ramulator2Interface::generateHashTableTrace(std::vector<uint64_t>& addresses, size_t num_accesses) {
    // Random access pattern for hash table lookups
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0x1000000, 0x10000000);  // 16MB-256MB range
    
    for (size_t i = 0; i < num_accesses; ++i) {
        addresses.push_back(dist(gen));
    }
}

void Ramulator2Interface::generateWeightMatrixTrace(std::vector<uint64_t>& addresses, size_t num_accesses) {
    // Sequential access pattern for weight matrices
    uint64_t base_addr = 0x20000000;  // 512MB base
    uint64_t cache_line_size = 64;
    
    for (size_t i = 0; i < num_accesses; ++i) {
        addresses.push_back(base_addr + i * cache_line_size);
    }
}

void Ramulator2Interface::generateVolumeDataTrace(std::vector<uint64_t>& addresses, size_t num_accesses) {
    // Spatial locality pattern for volume data
    uint64_t base_addr = 0x40000000;  // 1GB base
    uint64_t block_size = 256;  // 256-byte blocks
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> locality_dist(-2, 2);  // ±2 blocks locality
    
    uint64_t current_addr = base_addr;
    for (size_t i = 0; i < num_accesses; ++i) {
        addresses.push_back(current_addr);
        current_addr += block_size * (1 + locality_dist(gen));
    }
}

DRAMTimingResult Ramulator2Interface::simulateNeuralRenderingWorkload(
    size_t hash_table_accesses,
    size_t weight_matrix_accesses,
    size_t volume_data_accesses,
    double execution_time_ns) {
    
    MemoryAccessPattern pattern;
    pattern.pattern_type = "neural_rendering_mixed";
    pattern.access_size_bytes = 64;  // Cache line size
    
    // Generate mixed access pattern
    generateHashTableTrace(pattern.addresses, hash_table_accesses);
    generateWeightMatrixTrace(pattern.addresses, weight_matrix_accesses);
    generateVolumeDataTrace(pattern.addresses, volume_data_accesses);
    
    // Shuffle to simulate interleaved access pattern
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(pattern.addresses.begin(), pattern.addresses.end(), gen);
    
    return simulateMemoryAccess(pattern, execution_time_ns);
}

bool Ramulator2Interface::updateConfig(const Ramulator2Config& new_config) {
    config_ = new_config;
    return writeConfigFile();
}

std::vector<std::string> Ramulator2Interface::getSupportedDRAMTypes() {
    return {"DDR4", "DDR5", "HBM2", "LPDDR4", "LPDDR5", "GDDR6"};
}

// =============================================================================
// NeuralRenderingDRAMConfigFactory Implementation
// =============================================================================

Ramulator2Config NeuralRenderingDRAMConfigFactory::getConfigForAccelerator(const std::string& accelerator_name) {
    Ramulator2Config config;
    
    if (accelerator_name == "ICARUS") {
        // ICARUS optimized for NeRF workloads
        config.dram_type = "DDR4";
        config.frequency_mhz = 3200;
        config.channels = 4;
        config.scheduling_policy = "FR_FCFS";
        config.rowpolicy = "opened";
    } else if (accelerator_name == "NeuRex") {
        // NeuRex optimized for high-bandwidth volume rendering
        config.dram_type = "HBM2";
        config.frequency_mhz = 2000;
        config.channels = 8;
        config.scheduling_policy = "PAR_BS";
        config.rowpolicy = "opened";
    } else if (accelerator_name == "CICERO") {
        // CICERO optimized for compression and sparsity
        config.dram_type = "DDR5";
        config.frequency_mhz = 4800;
        config.channels = 2;
        config.scheduling_policy = "FR_FCFS";
        config.rowpolicy = "closed";
    } else if (accelerator_name == "GSCore") {
        // GSCore optimized for Gaussian Splatting
        config.dram_type = "DDR4";
        config.frequency_mhz = 3200;
        config.channels = 4;
        config.scheduling_policy = "FR_FCFS";
        config.rowpolicy = "opened";
    }
    
    return config;
}

Ramulator2Config NeuralRenderingDRAMConfigFactory::getHighBandwidthConfig() {
    Ramulator2Config config;
    config.dram_type = "HBM2";
    config.frequency_mhz = 2000;
    config.channels = 8;
    config.dram_width = "x1024";  // Wide interface
    config.scheduling_policy = "PAR_BS";
    config.rowpolicy = "opened";
    config.req_queue_size = 256;
    return config;
}

Ramulator2Config NeuralRenderingDRAMConfigFactory::getLowLatencyConfig() {
    Ramulator2Config config;
    config.dram_type = "DDR5";
    config.frequency_mhz = 5600;
    config.channels = 2;
    config.scheduling_policy = "FCFS";
    config.rowpolicy = "closed";  // Minimize row conflicts
    config.req_queue_size = 64;   // Smaller queue for lower latency
    return config;
}

Ramulator2Config NeuralRenderingDRAMConfigFactory::getPowerEfficientConfig() {
    Ramulator2Config config;
    config.dram_type = "LPDDR5";
    config.frequency_mhz = 3200;
    config.channels = 4;
    config.scheduling_policy = "FR_FCFS";
    config.rowpolicy = "timeout";  // Power-aware row management
    config.enable_power_model = true;
    return config;
}

} // namespace rendersim 