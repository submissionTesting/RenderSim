#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "RenderSim/ir.hpp"
#include "RenderSim/optimization_library.hpp"
#include "RenderSim/operator_scheduler.hpp"
#include "RenderSim/system_scheduler.hpp"
#include "RenderSim/ppa_estimator.hpp"
#include "RenderSim/ramulator_interface.hpp"
#include "RenderSim/performance_timer.hpp"
#include "RenderSim/hw_config.hpp"
#include "RenderSim/mapping_engine.hpp"
#include "RenderSim/mapping_loader.hpp"  // NEW: MappedIR JSON loader

namespace py = pybind11;
using namespace rendersim;

// Bind STL containers for proper dict-like access
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, rendersim::MappedIRNode>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, rendersim::OperatorScheduledIRNode>);

PYBIND11_MODULE(rendersim_cpp, m) {
    m.doc() = "RenderSim C++ core bindings (stub)";

    py::class_<TensorDesc>(m, "TensorDesc")
        .def(py::init<>())
        .def_readwrite("shape", &TensorDesc::shape)
        .def_readwrite("dtype", &TensorDesc::dtype);

    py::class_<OperatorNode>(m, "OperatorNode")
        .def(py::init<>())
        .def_readwrite("id", &OperatorNode::id)
        .def_readwrite("op_type", &OperatorNode::op_type)
        .def_readwrite("inputs", &OperatorNode::inputs)
        .def_readwrite("outputs", &OperatorNode::outputs)
        .def_readwrite("call_count", &OperatorNode::call_count);

    py::class_<OperatorGraph>(m, "OperatorGraph")
        .def(py::init<>())
        .def_readwrite("nodes", &OperatorGraph::nodes)
        .def_readwrite("edges", &OperatorGraph::edges);

    // Optimization Library bindings
    py::enum_<OptimizationType>(m, "OptimizationType")
        .value("REUSE", OptimizationType::REUSE)
        .value("SKIP", OptimizationType::SKIP)
        .value("LOW_BIT", OptimizationType::LOW_BIT);

    py::enum_<OptimizationScope>(m, "OptimizationScope")
        .value("ELEMENT_LEVEL", OptimizationScope::ELEMENT_LEVEL)
        .value("REGION_LEVEL", OptimizationScope::REGION_LEVEL)
        .value("FRAME_LEVEL", OptimizationScope::FRAME_LEVEL);

    py::enum_<DecisionCriteria>(m, "DecisionCriteria")
        .value("BOUNDARY_BASED", DecisionCriteria::BOUNDARY_BASED)
        .value("THRESHOLD_BASED", DecisionCriteria::THRESHOLD_BASED);

    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def_readwrite("duration", &OptimizationResult::duration)
        .def_readwrite("applied_optimizations", &OptimizationResult::applied_optimizations)
        .def_readwrite("speedup_factor", &OptimizationResult::speedup_factor)
        .def_readwrite("base_duration", &OptimizationResult::base_duration)
        .def_readwrite("total_strategies_considered", &OptimizationResult::total_strategies_considered);

    py::class_<OptimizationLibrary, std::shared_ptr<OptimizationLibrary>>(m, "OptimizationLibrary")
        .def(py::init<>())
        .def("get_strategy_count", &OptimizationLibrary::getStrategyCount);

    py::class_<OperatorOptimizer, std::shared_ptr<OperatorOptimizer>>(m, "OperatorOptimizer");
        
    py::class_<DummyOperatorOptimizer, OperatorOptimizer, std::shared_ptr<DummyOperatorOptimizer>>(m, "DummyOperatorOptimizer")
        .def(py::init<std::shared_ptr<OptimizationLibrary>>())
        .def("optimize", &DummyOperatorOptimizer::optimize);

    py::class_<AnalyticalOperatorOptimizer, OperatorOptimizer, std::shared_ptr<AnalyticalOperatorOptimizer>>(m, "AnalyticalOperatorOptimizer")
        .def(py::init<std::shared_ptr<OptimizationLibrary>>())
        .def("optimize", &AnalyticalOperatorOptimizer::optimize);

    // Latency instrumentation bindings
    py::class_<LatencyStats>(m, "LatencyStats")
        .def(py::init<>())
        .def_readwrite("total_duration_ns", &LatencyStats::total_duration_ns)
        .def_readwrite("average_duration_ns", &LatencyStats::average_duration_ns)
        .def_readwrite("last_duration_ns", &LatencyStats::last_duration_ns)
        .def_readwrite("measurement_count", &LatencyStats::measurement_count);

    py::class_<SchedulingLatencyReport>(m, "SchedulingLatencyReport")
        .def(py::init<>())
        .def_readwrite("operator_hw_grouping", &SchedulingLatencyReport::operator_hw_grouping)
        .def_readwrite("operator_hw_scheduling", &SchedulingLatencyReport::operator_hw_scheduling)
        .def_readwrite("operator_dependency_resolution", &SchedulingLatencyReport::operator_dependency_resolution)
        .def_readwrite("operator_total", &SchedulingLatencyReport::operator_total)
        .def_readwrite("system_dependency_graph", &SchedulingLatencyReport::system_dependency_graph)
        .def_readwrite("system_heuristic_computation", &SchedulingLatencyReport::system_heuristic_computation)
        .def_readwrite("system_scheduling_loop", &SchedulingLatencyReport::system_scheduling_loop)
        .def_readwrite("system_finalization", &SchedulingLatencyReport::system_finalization)
        .def_readwrite("system_total", &SchedulingLatencyReport::system_total)
        .def_readwrite("pipeline_total", &SchedulingLatencyReport::pipeline_total)
        .def("generate_report", &SchedulingLatencyReport::generateReport)
        .def_static("format_duration", &SchedulingLatencyReport::formatDuration);

    // Bind map types for proper Python dict interface
    // --------------------------------------------------
    // Hardware config bindings
    // --------------------------------------------------
    py::class_<HWUnit>(m, "HWUnit")
        .def_readonly("id", &HWUnit::id)
        .def_readonly("type", &HWUnit::type);

    py::class_<HWConfig>(m, "HWConfig")
        .def_readonly("accelerator_name", &HWConfig::accelerator_name)
        .def_readonly("units", &HWConfig::units);

    m.def("load_hw_config_from_json", &load_hw_config_from_json, py::arg("path"),
          R"pbdoc(Load hardware configuration JSON and return HWConfig.)pbdoc");

    py::bind_map<std::unordered_map<std::string, MappedIRNode>>(m, "MappedIRNodeMap");
    py::bind_map<std::unordered_map<std::string, OperatorScheduledIRNode>>(m, "OperatorScheduledIRNodeMap");

    // Operator Scheduler bindings
    py::class_<MappedIRNode>(m, "MappedIRNode")
        .def(py::init<>())
        .def_readwrite("op_node", &MappedIRNode::op_node)
        .def_readwrite("hw_unit", &MappedIRNode::hw_unit)
        .def_readwrite("attrs", &MappedIRNode::attrs);

    py::class_<MappedIR>(m, "MappedIR")
        .def(py::init<>())
        .def_readwrite("nodes", &MappedIR::nodes)
        .def_readwrite("edges", &MappedIR::edges);

    // ---------------------------------------------------------------------
    // Utility: load MappedIR directly from JSON (native C++ implementation)
    // ---------------------------------------------------------------------
    m.def("load_mapped_ir_from_json", &rendersim::load_mapped_ir_from_json,
          py::arg("path"),
          R"pbdoc(Load a mapped IR from a JSON file on disk and return a MappedIR object.)pbdoc");

    py::class_<OperatorScheduledIRNode>(m, "OperatorScheduledIRNode")
        .def(py::init<>())
        .def_readwrite("mapped_node", &OperatorScheduledIRNode::mapped_node)
        .def_readwrite("start_cycle", &OperatorScheduledIRNode::start_cycle)
        .def_readwrite("duration", &OperatorScheduledIRNode::duration)
        .def_readwrite("resources", &OperatorScheduledIRNode::resources)
        .def_readwrite("optimization_result", &OperatorScheduledIRNode::optimization_result);

    py::class_<OperatorScheduledIR>(m, "OperatorScheduledIR")
        .def(py::init<>())
        .def_readwrite("nodes", &OperatorScheduledIR::nodes)
        .def_readwrite("edges", &OperatorScheduledIR::edges);

    py::class_<OperatorLevelScheduler::SchedulingStats>(m, "SchedulingStats")
        .def_readwrite("total_operators", &OperatorLevelScheduler::SchedulingStats::total_operators)
        .def_readwrite("optimized_operators", &OperatorLevelScheduler::SchedulingStats::optimized_operators)
        .def_readwrite("total_speedup", &OperatorLevelScheduler::SchedulingStats::total_speedup)
        .def_readwrite("hw_unit_usage", &OperatorLevelScheduler::SchedulingStats::hw_unit_usage);

    py::class_<OperatorLevelScheduler>(m, "OperatorLevelScheduler")
        .def(py::init<std::shared_ptr<OperatorOptimizer>>())
        .def("schedule", &OperatorLevelScheduler::schedule)
        .def("get_last_scheduling_stats", &OperatorLevelScheduler::getLastSchedulingStats)
        .def("get_latency_report", &OperatorLevelScheduler::getLatencyReport)
        .def("set_latency_instrumentation_enabled", &OperatorLevelScheduler::setLatencyInstrumentationEnabled)
        .def("clear_latency_measurements", &OperatorLevelScheduler::clearLatencyMeasurements);

    py::enum_<OptimizerFactory::OptimizerType>(m, "OptimizerType")
        .value("DUMMY", OptimizerFactory::OptimizerType::DUMMY)
        .value("ANALYTICAL", OptimizerFactory::OptimizerType::ANALYTICAL)
        .value("ML_BASED", OptimizerFactory::OptimizerType::ML_BASED);

    py::class_<OptimizerFactory>(m, "OptimizerFactory")
        .def_static("create_optimizer", &OptimizerFactory::createOptimizer);

    // System Scheduler bindings
    py::class_<SystemScheduleEntry>(m, "SystemScheduleEntry")
        .def(py::init<>())
        .def_readwrite("op_id", &SystemScheduleEntry::op_id)
        .def_readwrite("hw_unit", &SystemScheduleEntry::hw_unit)
        .def_readwrite("start_cycle", &SystemScheduleEntry::start_cycle)
        .def_readwrite("duration", &SystemScheduleEntry::duration)
        .def_readwrite("resource_utilization", &SystemScheduleEntry::resource_utilization);

    py::class_<SystemSchedule>(m, "SystemSchedule")
        .def(py::init<>())
        .def_readwrite("entries", &SystemSchedule::entries)
        .def_readwrite("total_cycles", &SystemSchedule::total_cycles)
        .def_readwrite("avg_resource_utilization", &SystemSchedule::avg_resource_utilization)
        .def_readwrite("hw_unit_finish_times", &SystemSchedule::hw_unit_finish_times);

    py::class_<DAGSConfig>(m, "DAGSConfig")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("alpha", &DAGSConfig::alpha)
        .def_readwrite("beta", &DAGSConfig::beta);

    py::class_<SystemLevelScheduler::SystemSchedulingStats>(m, "SystemSchedulingStats")
        .def_readwrite("total_operators", &SystemLevelScheduler::SystemSchedulingStats::total_operators)
        .def_readwrite("ready_queue_peak_size", &SystemLevelScheduler::SystemSchedulingStats::ready_queue_peak_size)
        .def_readwrite("scheduling_efficiency", &SystemLevelScheduler::SystemSchedulingStats::scheduling_efficiency)
        .def_readwrite("resource_balance_factor", &SystemLevelScheduler::SystemSchedulingStats::resource_balance_factor)
        .def_readwrite("hw_unit_utilizations", &SystemLevelScheduler::SystemSchedulingStats::hw_unit_utilizations);

    py::class_<SystemLevelScheduler>(m, "SystemLevelScheduler")
        .def(py::init<const DAGSConfig&>(), py::arg("config") = DAGSConfig())
        .def("schedule", &SystemLevelScheduler::schedule)
        .def("get_last_scheduling_stats", &SystemLevelScheduler::getLastSchedulingStats)
        .def("get_latency_report", &SystemLevelScheduler::getLatencyReport)
        .def("set_latency_instrumentation_enabled", &SystemLevelScheduler::setLatencyInstrumentationEnabled)
        .def("clear_latency_measurements", &SystemLevelScheduler::clearLatencyMeasurements)
        .def("update_config", &SystemLevelScheduler::updateConfig);

    py::enum_<SystemSchedulerFactory::SchedulerType>(m, "SystemSchedulerType")
        .value("DAGS", SystemSchedulerFactory::SchedulerType::DAGS)
        .value("LIST_BASED", SystemSchedulerFactory::SchedulerType::LIST_BASED)
        .value("CRITICAL_PATH", SystemSchedulerFactory::SchedulerType::CRITICAL_PATH);

    py::class_<SystemSchedulerFactory>(m, "SystemSchedulerFactory")
        .def_static("create_scheduler", &SystemSchedulerFactory::createScheduler);

    // PPA Estimator bindings
    py::class_<PPAMetrics>(m, "PPAMetrics")
        .def(py::init<>())
        .def_readwrite("latency_cycles", &PPAMetrics::latency_cycles)
        .def_readwrite("throughput_ops_per_sec", &PPAMetrics::throughput_ops_per_sec)
        .def_readwrite("area_um2", &PPAMetrics::area_um2)
        .def_readwrite("static_power_uw", &PPAMetrics::static_power_uw)
        .def_readwrite("dynamic_power_uw", &PPAMetrics::dynamic_power_uw)
        .def_readwrite("dram_latency_ns", &PPAMetrics::dram_latency_ns)
        .def_readwrite("dram_bandwidth_gb_s", &PPAMetrics::dram_bandwidth_gb_s)
        .def("area_mm2", &PPAMetrics::area_mm2)
        .def("total_power_uw", &PPAMetrics::total_power_uw);

    // Real Ramulator 2.0 Integration
    py::class_<Ramulator2Config>(m, "Ramulator2Config")
        .def(py::init<>())
        .def_readwrite("dram_type", &Ramulator2Config::dram_type)
        .def_readwrite("dram_density", &Ramulator2Config::dram_density)
        .def_readwrite("dram_width", &Ramulator2Config::dram_width)
        .def_readwrite("frequency_mhz", &Ramulator2Config::frequency_mhz)
        .def_readwrite("channels", &Ramulator2Config::channels)
        .def_readwrite("ranks_per_channel", &Ramulator2Config::ranks_per_channel)
        .def_readwrite("banks_per_rank", &Ramulator2Config::banks_per_rank)
        .def_readwrite("scheduling_policy", &Ramulator2Config::scheduling_policy)
        .def_readwrite("rowpolicy", &Ramulator2Config::rowpolicy)
        .def_readwrite("req_queue_size", &Ramulator2Config::req_queue_size)
        .def_readwrite("enable_power_model", &Ramulator2Config::enable_power_model)
        .def_readwrite("power_config", &Ramulator2Config::power_config);

    py::class_<MemoryAccessPattern>(m, "MemoryAccessPattern")
        .def(py::init<>())
        .def_readwrite("pattern_type", &MemoryAccessPattern::pattern_type)
        .def_readwrite("access_size_bytes", &MemoryAccessPattern::access_size_bytes)
        .def_readwrite("addresses", &MemoryAccessPattern::addresses)
        .def_readwrite("access_frequency_hz", &MemoryAccessPattern::access_frequency_hz);

    py::class_<DRAMTimingResult>(m, "DRAMTimingResult")
        .def(py::init<>())
        .def_readwrite("average_latency_ns", &DRAMTimingResult::average_latency_ns)
        .def_readwrite("peak_bandwidth_gb_s", &DRAMTimingResult::peak_bandwidth_gb_s)
        .def_readwrite("effective_bandwidth_gb_s", &DRAMTimingResult::effective_bandwidth_gb_s)
        .def_readwrite("power_consumption_mw", &DRAMTimingResult::power_consumption_mw)
        .def_readwrite("row_buffer_hit_rate", &DRAMTimingResult::row_buffer_hit_rate)
        .def_readwrite("total_accesses", &DRAMTimingResult::total_accesses)
        .def_readwrite("row_buffer_hits", &DRAMTimingResult::row_buffer_hits)
        .def_readwrite("row_buffer_misses", &DRAMTimingResult::row_buffer_misses);

    py::class_<HardwareModuleConfig>(m, "HardwareModuleConfig")
        .def(py::init<const std::string&, const std::string&, const std::string&, double, const std::string&>(),
             py::arg("name"), py::arg("accel_type"), py::arg("hw_path"), 
             py::arg("clk_period") = 1.0, py::arg("tech") = "tn28rvt9t")
        .def_readwrite("module_name", &HardwareModuleConfig::module_name)
        .def_readwrite("accelerator_type", &HardwareModuleConfig::accelerator_type)
        .def_readwrite("hardware_path", &HardwareModuleConfig::hardware_path)
        .def_readwrite("clock_period_ns", &HardwareModuleConfig::clock_period_ns)
        .def_readwrite("technology_node", &HardwareModuleConfig::technology_node);

    py::class_<Ramulator2Interface>(m, "Ramulator2Interface")
        .def(py::init<const Ramulator2Config&>())
        .def("initialize", &Ramulator2Interface::initialize, py::arg("ramulator_path") = "Hardware/ramulator2")
        .def("simulate_memory_access", &Ramulator2Interface::simulateMemoryAccess)
        .def("simulate_neural_rendering_workload", &Ramulator2Interface::simulateNeuralRenderingWorkload)
        .def("update_config", &Ramulator2Interface::updateConfig)
        .def("generate_config_yaml", &Ramulator2Interface::generateConfigYAML)
        .def("get_config", &Ramulator2Interface::getConfig, py::return_value_policy::reference)
        .def_static("get_supported_dram_types", &Ramulator2Interface::getSupportedDRAMTypes);

    py::class_<NeuralRenderingDRAMConfigFactory>(m, "NeuralRenderingDRAMConfigFactory")
        .def_static("get_config_for_accelerator", &NeuralRenderingDRAMConfigFactory::getConfigForAccelerator)
        .def_static("get_high_bandwidth_config", &NeuralRenderingDRAMConfigFactory::getHighBandwidthConfig)
        .def_static("get_low_latency_config", &NeuralRenderingDRAMConfigFactory::getLowLatencyConfig)
        .def_static("get_power_efficient_config", &NeuralRenderingDRAMConfigFactory::getPowerEfficientConfig);

    py::class_<PPAEstimator::SystemPPAMetrics>(m, "SystemPPAMetrics")
        .def(py::init<>())
        .def_readwrite("total_metrics", &PPAEstimator::SystemPPAMetrics::total_metrics)
        .def_readwrite("per_hw_unit_metrics", &PPAEstimator::SystemPPAMetrics::per_hw_unit_metrics)
        .def_readwrite("total_execution_time_ns", &PPAEstimator::SystemPPAMetrics::total_execution_time_ns)
        .def_readwrite("total_area_mm2", &PPAEstimator::SystemPPAMetrics::total_area_mm2)
        .def_readwrite("total_power_mw", &PPAEstimator::SystemPPAMetrics::total_power_mw)
        .def_readwrite("average_memory_bandwidth_gb_s", &PPAEstimator::SystemPPAMetrics::average_memory_bandwidth_gb_s)
        .def_readwrite("area_error_percentage", &PPAEstimator::SystemPPAMetrics::area_error_percentage)
        .def_readwrite("power_error_percentage", &PPAEstimator::SystemPPAMetrics::power_error_percentage)
        .def_readwrite("latency_error_percentage", &PPAEstimator::SystemPPAMetrics::latency_error_percentage);

    py::class_<PPAEstimator::ValidationResult>(m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("area_error_percent", &PPAEstimator::ValidationResult::area_error_percent)
        .def_readwrite("power_error_percent", &PPAEstimator::ValidationResult::power_error_percent)
        .def_readwrite("latency_error_percent", &PPAEstimator::ValidationResult::latency_error_percent)
        .def_readwrite("overall_error_percent", &PPAEstimator::ValidationResult::overall_error_percent)
        .def_readwrite("meets_target_accuracy", &PPAEstimator::ValidationResult::meets_target_accuracy);

    py::class_<PPAEstimator>(m, "PPAEstimator")
        .def(py::init<const Ramulator2Config&, const std::string&>(), 
             py::arg("dram_config") = Ramulator2Config(), py::arg("hardware_path") = "Hardware/")
        .def("set_clock_period_ns", &PPAEstimator::setClockPeriodNs)
        .def("get_clock_period_ns", &PPAEstimator::getClockPeriodNs)
        .def("estimate_system_ppa", &PPAEstimator::estimateSystemPPA)
        .def("validate_accuracy", &PPAEstimator::validateAccuracy)
        .def("get_validated_configs", &PPAEstimator::getValidatedConfigs);

    py::class_<PPAReportGenerator::AcceleratorComparison>(m, "AcceleratorComparison")
        .def(py::init<>())
        .def_readwrite("accelerator_name", &PPAReportGenerator::AcceleratorComparison::accelerator_name)
        .def_readwrite("module_names", &PPAReportGenerator::AcceleratorComparison::module_names)
        .def_readwrite("rendersim_results", &PPAReportGenerator::AcceleratorComparison::rendersim_results)
        .def_readwrite("reference_results", &PPAReportGenerator::AcceleratorComparison::reference_results)
        .def_readwrite("average_error_percent", &PPAReportGenerator::AcceleratorComparison::average_error_percent);

    py::class_<PPAReportGenerator>(m, "PPAReportGenerator")
        .def_static("generate_module_comparison_table", &PPAReportGenerator::generateModuleComparisonTable)
        .def_static("generate_detailed_report", &PPAReportGenerator::generateDetailedReport);

    // --------------------------------------------------
    // Mapping engine
    // --------------------------------------------------
    m.def("map_operator_graph", &map_operator_graph,
          py::arg("operator_graph"), py::arg("hw_config"),
          R"pbdoc(Map an OperatorGraph to hardware units using native C++ greedy mapper.)pbdoc");
} 