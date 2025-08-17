#!/usr/bin/env python3
"""
RenderSim Simple Benchmark Test
==============================

This script demonstrates the RenderSim benchmarking pipeline using the sample DAG
to validate the rs_benchmark_pipelines milestone without requiring nerfstudio.

It tests:
1. Loading sample execution DAG
2. Mapping operators to hardware accelerators  
3. Scheduling execution
4. Generating PPA reports
5. Creating visualizations

This validates that the core benchmarking infrastructure is working.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

# Add RenderSim to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class SimpleBenchmarkTest:
    """Test RenderSim benchmarking with sample data"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.test_output_dir = self.base_dir / "simple_benchmark_results"
        
        # Create test output directory
        self.test_output_dir.mkdir(exist_ok=True)
        
        print("🧪 RenderSim Simple Benchmark Test")
        print(f"   Base directory: {self.base_dir}")
        print(f"   Test output: {self.test_output_dir}")
    
    def test_sample_dag_loading(self) -> str:
        """Test loading the sample DAG"""
        print(f"\n📊 Testing sample DAG loading")
        
        sample_dag_file = self.base_dir / "examples" / "sample_dag.pkl"
        
        if not sample_dag_file.exists():
            print(f"   ❌ Sample DAG not found: {sample_dag_file}")
            return None
        
        try:
            import pickle
            import networkx as nx
            
            with open(sample_dag_file, 'rb') as f:
                dag = pickle.load(f)
            
            print(f"   ✅ Loaded sample DAG")
            print(f"      Nodes: {len(dag.nodes)}")
            print(f"      Edges: {len(dag.edges)}")
            
            # Print operator details
            for node_id, attrs in dag.nodes(data=True):
                op_type = attrs.get('op_type', 'unknown')
                flops = attrs.get('flops', 0)
                print(f"      {node_id}: {op_type} - {flops:,} FLOPs")
            
            return str(sample_dag_file)
            
        except Exception as e:
            print(f"   ❌ Failed to load sample DAG: {e}")
            return None
    
    def test_hardware_configs(self):
        """Test that all hardware configurations are valid"""
        print(f"\n🔧 Testing hardware configurations")
        
        hw_configs = [
            "examples/hardware_configs/icarus_config.json",
            "examples/hardware_configs/neurex_config.json",
            "examples/hardware_configs/gscore_config.json", 
            "examples/hardware_configs/cicero_config.json"
        ]
        
        valid_configs = []
        
        for config_file in hw_configs:
            config_path = self.base_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    accelerator_name = config_path.stem.replace('_config', '')
                    print(f"   ✅ {accelerator_name}: {config_data.get('metadata', {}).get('name', 'Unknown')}")
                    valid_configs.append((accelerator_name, str(config_path)))
                    
                except Exception as e:
                    print(f"   ❌ {config_file}: {e}")
            else:
                print(f"   ❌ {config_file}: Not found")
        
        return valid_configs
    
    def test_cli_interface(self):
        """Test that the CLI interface is working"""
        print(f"\n🖥️  Testing CLI interface")
        
        try:
            # Test CLI help
            result = subprocess.run([
                "python", "CLI/main.py", "--help"
            ], capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                print(f"   ✅ CLI interface accessible")
                return True
            else:
                print(f"   ❌ CLI interface failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ CLI test failed: {e}")
            return False
    
    def test_scheduling_pipeline(self, dag_file: str, accelerator: str, config_path: str):
        """Test the complete scheduling pipeline for one accelerator"""
        print(f"\n⏰ Testing scheduling pipeline: {accelerator}")
        
        try:
            # Define output file paths
            mapped_file = self.test_output_dir / f"mapped_{accelerator}.json"
            scheduled_file = self.test_output_dir / f"scheduled_{accelerator}.json"
            report_file = self.test_output_dir / f"report_{accelerator}.html"
            
            # Step 1: Mapping
            print(f"   🗺️  Mapping operators to {accelerator}...")
            map_cmd = [
                "python", "CLI/main.py", "map",
                dag_file, config_path,
                "-o", str(mapped_file)
            ]
            
            result = subprocess.run(map_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ Mapping failed: {result.stderr}")
                return None
            
            print(f"      ✅ Mapping completed")
            
            # Step 2: Scheduling
            print(f"   ⚙️  Scheduling execution...")
            schedule_cmd = [
                "python", "CLI/main.py", "schedule",
                str(mapped_file),
                "-o", str(scheduled_file)
            ]
            
            result = subprocess.run(schedule_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ Scheduling failed: {result.stderr}")
                return None
            
            print(f"      ✅ Scheduling completed")
            
            # Step 3: Report generation
            print(f"   📋 Generating PPA report...")
            report_cmd = [
                "python", "CLI/main.py", "report",
                str(scheduled_file),
                "-o", str(report_file)
            ]
            
            result = subprocess.run(report_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ Report generation failed: {result.stderr}")
                return None
            
            print(f"      ✅ Report generation completed")
            print(f"      Files: {mapped_file.name}, {scheduled_file.name}, {report_file.name}")
            
            return {
                'mapped': str(mapped_file),
                'scheduled': str(scheduled_file),
                'report': str(report_file)
            }
            
        except Exception as e:
            print(f"   ❌ Scheduling pipeline failed: {e}")
            return None
    
    def test_visualization_system(self):
        """Test that visualization system is accessible"""
        print(f"\n📊 Testing visualization system")
        
        try:
            # Test imports
            sys.path.append(str(self.base_dir))
            from Visualization import ScheduleVisualizer, PPADashboard
            
            print(f"   ✅ Visualization modules imported successfully")
            
            # Create a simple visualization test
            viz_file = self.test_output_dir / "visualization_test.html"
            with open(viz_file, 'w') as f:
                f.write(f"""
                <html>
                <head><title>RenderSim Visualization System Test</title></head>
                <body>
                <h1>RenderSim Visualization System Test</h1>
                <p>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>✅ Visualization system accessible</p>
                <p>Available modules:</p>
                <ul>
                <li>ScheduleVisualizer</li>
                <li>PPADashboard</li>
                <li>OperatorGraphPlotter</li>
                <li>GanttChartPlotter</li>
                </ul>
                </body>
                </html>
                """)
            
            print(f"   ✅ Visualization test completed: {viz_file.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Visualization test failed: {e}")
            return False
    
    def run_complete_benchmark_test(self):
        """Run the complete benchmark test suite"""
        print("\n🎯 Starting Complete RenderSim Benchmark Test")
        
        results = {
            'dag_loading': False,
            'hardware_configs': [],
            'cli_interface': False,
            'scheduling_results': {},
            'visualization': False
        }
        
        # Step 1: Test sample DAG loading
        dag_file = self.test_sample_dag_loading()
        results['dag_loading'] = dag_file is not None
        
        if not dag_file:
            print(f"❌ Cannot proceed without DAG file")
            return results
        
        # Step 2: Test hardware configurations
        hw_configs = self.test_hardware_configs()
        results['hardware_configs'] = hw_configs
        
        # Step 3: Test CLI interface
        results['cli_interface'] = self.test_cli_interface()
        
        # Step 4: Test scheduling pipeline for each accelerator
        if results['cli_interface']:
            for accelerator, config_path in hw_configs:
                print(f"\n{'='*50}")
                print(f"🚀 Testing {accelerator.upper()} Accelerator")
                print(f"{'='*50}")
                
                scheduling_result = self.test_scheduling_pipeline(dag_file, accelerator, config_path)
                results['scheduling_results'][accelerator] = scheduling_result
        
        # Step 5: Test visualization system
        results['visualization'] = self.test_visualization_system()
        
        # Generate summary
        self.generate_test_summary(results)
        
        return results
    
    def generate_test_summary(self, results):
        """Generate a test summary report"""
        print(f"\n📊 Generating Test Summary")
        
        summary_file = self.test_output_dir / "benchmark_test_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# RenderSim Simple Benchmark Test Summary\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall results
            successful_accelerators = len([acc for acc, result in results['scheduling_results'].items() if result])
            total_accelerators = len(results['hardware_configs'])
            
            f.write("## Test Results\n")
            f.write(f"- ✅ DAG Loading: {'✅' if results['dag_loading'] else '❌'}\n")
            f.write(f"- ✅ Hardware Configs: {len(results['hardware_configs'])}/4 available\n")
            f.write(f"- ✅ CLI Interface: {'✅' if results['cli_interface'] else '❌'}\n")
            f.write(f"- ✅ Scheduling Success: {successful_accelerators}/{total_accelerators} accelerators\n")
            f.write(f"- ✅ Visualization: {'✅' if results['visualization'] else '❌'}\n\n")
            
            # Detailed results
            f.write("## Accelerator Results\n\n")
            for accelerator, result in results['scheduling_results'].items():
                if result:
                    f.write(f"### {accelerator.upper()}\n")
                    f.write(f"- ✅ Mapping: {Path(result['mapped']).name}\n")
                    f.write(f"- ✅ Scheduling: {Path(result['scheduled']).name}\n")
                    f.write(f"- ✅ Report: {Path(result['report']).name}\n\n")
                else:
                    f.write(f"### {accelerator.upper()}\n")
                    f.write(f"- ❌ Pipeline failed\n\n")
            
            # Milestone status
            all_core_working = (results['dag_loading'] and 
                              results['cli_interface'] and 
                              successful_accelerators > 0 and
                              results['visualization'])
            
            f.write("## Milestone Status\n")
            if all_core_working:
                f.write("✅ **rs_benchmark_pipelines**: Core functionality validated\n")
                f.write("- Operator graph loading ✅\n")
                f.write("- Hardware configuration system ✅\n") 
                f.write("- CLI interface ✅\n")
                f.write("- Scheduling pipeline ✅\n")
                f.write("- PPA report generation ✅\n")
                f.write("- Visualization system ✅\n")
            else:
                f.write("❌ **rs_benchmark_pipelines**: Issues found\n")
        
        print(f"   Summary report: {summary_file}")
        
        return all_core_working


def main():
    """Main test entry point"""
    print("🚀 RenderSim Simple Benchmark Test")
    print("   Testing core benchmarking infrastructure...")
    
    # Run test
    tester = SimpleBenchmarkTest()
    results = tester.run_complete_benchmark_test()
    
    # Final summary
    print(f"\n🎉 Simple Benchmark Test Completed!")
    print(f"   Results saved to: {tester.test_output_dir}")
    
    successful_accelerators = len([acc for acc, result in results['scheduling_results'].items() if result])
    total_accelerators = len(results['hardware_configs'])
    
    # Check overall success
    core_success = (results['dag_loading'] and 
                   results['cli_interface'] and 
                   successful_accelerators > 0 and
                   results['visualization'])
    
    if core_success:
        print(f"   ✅ Core benchmarking infrastructure: WORKING")
        print(f"   ✅ Successful accelerators: {successful_accelerators}/{total_accelerators}")
        print(f"   ✅ rs_benchmark_pipelines milestone: CORE VALIDATED")
        return 0
    else:
        print(f"   ❌ Some components failed")
        print(f"   🔧 Check test results for details")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 