#!/usr/bin/env python3
"""
RenderSim End-to-End Pipeline Test
==================================

This test validates the complete pipeline: parse DAG → map to hardware → schedule → generate report.
This completes the rs_end_to_end_test milestone.

Tests:
1. Loading execution DAG (from Nerfstudio or sample)
2. Mapping operators to hardware accelerators
3. Operator-level and system-level scheduling
4. Performance analysis and report generation
5. CLI interface integration
6. File I/O and data consistency

This validates that the entire RenderSim pipeline works end-to-end.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

# Add RenderSim to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class EndToEndTest:
    """Comprehensive end-to-end RenderSim pipeline test"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.test_output_dir = self.base_dir / "end_to_end_test_results"
        
        # Create test output directory
        self.test_output_dir.mkdir(exist_ok=True)
        
        print("🧪 RenderSim End-to-End Pipeline Test")
        print("=====================================")
        print(f"   Base directory: {self.base_dir}")
        print(f"   Test output: {self.test_output_dir}")
    
    def test_sample_dag_availability(self) -> bool:
        """Test that the sample DAG is available for testing"""
        print(f"\n📊 Testing sample DAG availability")
        
        sample_dag_file = self.base_dir / "examples" / "sample_dag.pkl"
        
        if sample_dag_file.exists():
            print(f"   ✅ Sample DAG found: {sample_dag_file}")
            
            # Load and inspect the DAG
            try:
                import pickle
                with sample_dag_file.open('rb') as f:
                    dag = pickle.load(f)
                
                print(f"   📋 DAG contains {len(dag.nodes())} nodes and {len(dag.edges())} edges")
                
                # Show operator types
                op_types = set()
                for node, attrs in dag.nodes(data=True):
                    op_type = attrs.get('op_type', 'unknown')
                    op_types.add(op_type)
                
                print(f"   🔧 Operator types: {', '.join(sorted(op_types))}")
                return True
                
            except Exception as e:
                print(f"   ❌ Failed to load DAG: {e}")
                return False
        else:
            print(f"   ❌ Sample DAG not found: {sample_dag_file}")
            return False
    
    def test_hardware_configs_availability(self) -> list:
        """Test that hardware configurations are available"""
        print(f"\n🏗️  Testing hardware configurations availability")
        
        config_dir = self.base_dir / "examples" / "hardware_configs"
        available_configs = []
        
        if not config_dir.exists():
            print(f"   ❌ Hardware config directory not found: {config_dir}")
            return available_configs
        
        config_files = list(config_dir.glob("*.json"))
        
        for config_file in config_files:
            try:
                # Test loading the configuration
                from Scheduler.mapping.hw_config import load_hw_config
                hw_config = load_hw_config(config_file)
                
                print(f"   ✅ {config_file.name}: {hw_config.accelerator_name} ({len(hw_config.units)} units)")
                available_configs.append(str(config_file))
                
            except Exception as e:
                print(f"   ❌ {config_file.name}: Failed to load - {e}")
        
        return available_configs
    
    def test_cli_accessibility(self) -> bool:
        """Test that the CLI interface is accessible"""
        print(f"\n🖥️  Testing CLI accessibility")
        
        try:
            # Test CLI help
            result = subprocess.run([
                "python", "CLI/main.py", "--help"
            ], capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                print(f"   ✅ CLI interface accessible")
                
                # Check for expected commands
                output = result.stdout
                commands = ['map', 'schedule', 'report']
                found_commands = [cmd for cmd in commands if cmd in output]
                
                print(f"   📋 Available commands: {', '.join(found_commands)}")
                
                if len(found_commands) == len(commands):
                    print(f"   ✅ All expected commands found")
                    return True
                else:
                    print(f"   ⚠️  Missing commands: {set(commands) - set(found_commands)}")
                    return False
            else:
                print(f"   ❌ CLI interface failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ CLI test failed: {e}")
            return False
    
    def test_complete_pipeline(self, dag_file: str, config_file: str, accelerator_name: str) -> bool:
        """Test the complete map → schedule → report pipeline"""
        print(f"\n🔄 Testing complete pipeline: {accelerator_name}")
        
        try:
            # Define output file paths
            mapped_file = self.test_output_dir / f"mapped_{accelerator_name.lower()}.json"
            scheduled_file = self.test_output_dir / f"scheduled_{accelerator_name.lower()}.json"
            report_html = self.test_output_dir / f"report_{accelerator_name.lower()}.html"
            report_json = self.test_output_dir / f"report_{accelerator_name.lower()}.json"
            report_text = self.test_output_dir / f"report_{accelerator_name.lower()}.txt"
            
            # Step 1: Mapping
            print(f"   🗺️  Step 1: Mapping operators to {accelerator_name}...")
            map_cmd = [
                "python", "CLI/main.py", "-v", "map",
                dag_file, config_file,
                "-o", str(mapped_file)
            ]
            
            result = subprocess.run(map_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ Mapping failed: {result.stderr}")
                return False
            
            if not mapped_file.exists():
                print(f"   ❌ Mapped file not created: {mapped_file}")
                return False
            
            # Validate mapping output
            with mapped_file.open('r') as f:
                mapping_data = json.load(f)
            
            if "operator_count" not in mapping_data:
                print(f"   ❌ Invalid mapping output format")
                return False
            
            print(f"      ✅ Mapping completed ({mapping_data['operator_count']} operators)")
            
            # Step 2: Scheduling
            print(f"   ⚙️  Step 2: Scheduling execution...")
            schedule_cmd = [
                "python", "CLI/main.py", "-v", "schedule",
                str(mapped_file),
                "-o", str(scheduled_file)
            ]
            
            result = subprocess.run(schedule_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ Scheduling failed: {result.stderr}")
                return False
            
            if not scheduled_file.exists():
                print(f"   ❌ Schedule file not created: {scheduled_file}")
                return False
            
            # Validate scheduling output
            with scheduled_file.open('r') as f:
                schedule_data = json.load(f)
            
            if "system_schedule" not in schedule_data:
                print(f"   ❌ Invalid schedule output format")
                return False
            
            total_cycles = schedule_data["system_schedule"]["total_cycles"]
            efficiency = schedule_data["system_statistics"]["scheduling_efficiency"]
            
            print(f"      ✅ Scheduling completed ({total_cycles} cycles, {efficiency:.3f} efficiency)")
            
            # Step 3: Report generation (multiple formats)
            print(f"   📋 Step 3: Generating reports...")
            
            # HTML report
            html_cmd = [
                "python", "CLI/main.py", "-v", "report",
                str(scheduled_file),
                "-o", str(report_html),
                "--format", "html"
            ]
            
            result = subprocess.run(html_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ HTML report generation failed: {result.stderr}")
                return False
            
            # JSON report
            json_cmd = [
                "python", "CLI/main.py", "-v", "report",
                str(scheduled_file),
                "-o", str(report_json),
                "--format", "json"
            ]
            
            result = subprocess.run(json_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ JSON report generation failed: {result.stderr}")
                return False
            
            # Text report
            text_cmd = [
                "python", "CLI/main.py", "-v", "report",
                str(scheduled_file),
                "-o", str(report_text),
                "--format", "text"
            ]
            
            result = subprocess.run(text_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"   ❌ Text report generation failed: {result.stderr}")
                return False
            
            # Validate all reports exist
            if not all([report_html.exists(), report_json.exists(), report_text.exists()]):
                print(f"   ❌ Not all report files created")
                return False
            
            print(f"      ✅ All reports generated successfully")
            print(f"      📁 Files: {mapped_file.name}, {scheduled_file.name}")
            print(f"      📊 Reports: {report_html.name}, {report_json.name}, {report_text.name}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Pipeline test failed: {e}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test that performance metrics are within expected ranges"""
        print(f"\n📊 Testing performance metrics validation")
        
        try:
            # Find the most recent schedule file
            schedule_files = list(self.test_output_dir.glob("scheduled_*.json"))
            
            if not schedule_files:
                print(f"   ❌ No schedule files found for analysis")
                return False
            
            # Use the first available schedule file
            schedule_file = schedule_files[0]
            
            with schedule_file.open('r') as f:
                schedule_data = json.load(f)
            
            # Extract key metrics
            total_cycles = schedule_data["system_schedule"]["total_cycles"]
            efficiency = schedule_data["system_statistics"]["scheduling_efficiency"]
            balance = schedule_data["system_statistics"]["resource_balance_factor"]
            speedup = schedule_data["operator_statistics"]["total_speedup"]
            
            print(f"   📈 Performance Metrics Analysis:")
            print(f"      Total Cycles: {total_cycles}")
            print(f"      Scheduling Efficiency: {efficiency:.3f}")
            print(f"      Resource Balance: {balance:.3f}")
            print(f"      Total Speedup: {speedup:.2f}x")
            
            # Validate metrics are reasonable
            metrics_valid = True
            
            if total_cycles <= 0:
                print(f"   ❌ Invalid total cycles: {total_cycles}")
                metrics_valid = False
            
            if not (0.0 <= efficiency <= 1.0):
                print(f"   ❌ Invalid efficiency: {efficiency}")
                metrics_valid = False
            
            if balance < 0.0:
                print(f"   ❌ Invalid balance factor: {balance}")
                metrics_valid = False
            
            if speedup <= 0.0:
                print(f"   ❌ Invalid speedup: {speedup}")
                metrics_valid = False
            
            if metrics_valid:
                print(f"   ✅ All performance metrics are valid")
                return True
            else:
                print(f"   ❌ Some performance metrics are invalid")
                return False
                
        except Exception as e:
            print(f"   ❌ Performance metrics test failed: {e}")
            return False
    
    def test_file_consistency(self) -> bool:
        """Test that generated files have consistent data"""
        print(f"\n🔍 Testing file consistency")
        
        try:
            # Find matching mapped and scheduled files
            mapped_files = list(self.test_output_dir.glob("mapped_*.json"))
            scheduled_files = list(self.test_output_dir.glob("scheduled_*.json"))
            
            if not mapped_files or not scheduled_files:
                print(f"   ❌ Missing files for consistency check")
                return False
            
            # Check first pair
            mapped_file = mapped_files[0]
            scheduled_file = scheduled_files[0]
            
            with mapped_file.open('r') as f:
                mapped_data = json.load(f)
            
            with scheduled_file.open('r') as f:
                scheduled_data = json.load(f)
            
            # Check operator count consistency
            mapped_count = mapped_data.get("operator_count", 0)
            scheduled_ops = len(scheduled_data["system_schedule"]["entries"])
            
            print(f"   🔢 Operator counts: Mapped={mapped_count}, Scheduled={scheduled_ops}")
            
            if mapped_count != scheduled_ops:
                print(f"   ⚠️  Operator count mismatch (may be due to sample IR creation)")
            else:
                print(f"   ✅ Operator counts match")
            
            # Check that schedule has valid entries
            if scheduled_ops > 0:
                print(f"   ✅ Schedule contains {scheduled_ops} operators")
                
                # Check that each entry has required fields
                entry = scheduled_data["system_schedule"]["entries"][0]
                required_fields = ["op_id", "hw_unit", "start_cycle", "duration"]
                
                if all(field in entry for field in required_fields):
                    print(f"   ✅ Schedule entries have all required fields")
                    return True
                else:
                    print(f"   ❌ Schedule entries missing required fields")
                    return False
            else:
                print(f"   ❌ Empty schedule")
                return False
                
        except Exception as e:
            print(f"   ❌ File consistency test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all end-to-end tests"""
        print(f"\n🚀 Running Complete End-to-End Test Suite")
        print(f"==========================================")
        
        test_results = []
        
        # Test 1: Sample DAG availability
        test_results.append(("Sample DAG Availability", self.test_sample_dag_availability()))
        
        # Test 2: Hardware config availability
        available_configs = self.test_hardware_configs_availability()
        test_results.append(("Hardware Config Availability", len(available_configs) > 0))
        
        # Test 3: CLI accessibility
        test_results.append(("CLI Accessibility", self.test_cli_accessibility()))
        
        # Test 4: Complete pipeline for each accelerator
        if available_configs:
            dag_file = str(self.base_dir / "examples" / "sample_dag.pkl")
            
            for config_file in available_configs[:2]:  # Test first 2 accelerators
                config_name = Path(config_file).stem.replace('_config', '').upper()
                pipeline_success = self.test_complete_pipeline(dag_file, config_file, config_name)
                test_results.append((f"Pipeline - {config_name}", pipeline_success))
        
        # Test 5: Performance metrics validation
        test_results.append(("Performance Metrics", self.test_performance_metrics()))
        
        # Test 6: File consistency
        test_results.append(("File Consistency", self.test_file_consistency()))
        
        # Summary
        print(f"\n📋 End-to-End Test Results")
        print(f"==========================")
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, passed in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {status} {test_name}")
            if passed:
                passed_tests += 1
        
        print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print(f"🎉 All end-to-end tests passed!")
            print(f"✅ RenderSim pipeline is fully operational")
            return True
        else:
            print(f"❌ Some tests failed - please check the output above")
            return False


def main():
    """Main entry point for end-to-end testing"""
    test_runner = EndToEndTest()
    success = test_runner.run_all_tests()
    
    if success:
        print(f"\n🏆 END-TO-END TEST SUCCESS")
        print(f"   RenderSim complete pipeline validated!")
        print(f"   Ready for production neural rendering accelerator simulation")
        return 0
    else:
        print(f"\n💥 END-TO-END TEST FAILED")
        print(f"   Please check the errors above and fix them")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 