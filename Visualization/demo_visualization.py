#!/usr/bin/env python3
"""
RenderSim Visualization Demo

Demonstrates all visualization capabilities including:
- Operator graph plotting
- Gantt chart visualization  
- PPA dashboard creation
- Comprehensive analysis reports
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .graph_plotter import OperatorGraphPlotter
from .gantt_plotter import GanttChartPlotter
from .ppa_dashboard import PPADashboard
from .schedule_visualizer import ScheduleVisualizer


def create_sample_data() -> Dict[str, Any]:
    """Create sample data for demonstration."""
    
    # Sample execution DAG
    execution_dag = {
        'nodes': {
            'op_1': {'op_type': 'HASH_ENCODE', 'hw_unit': 'hash_unit'},
            'op_2': {'op_type': 'FIELD_COMPUTATION', 'hw_unit': 'mlp_engine'},
            'op_3': {'op_type': 'SAMPLING', 'hw_unit': 'memory_controller'},
            'op_4': {'op_type': 'BLENDING', 'hw_unit': 'rendering_unit'},
            'op_5': {'op_type': 'VOLUME_RENDERING', 'hw_unit': 'rendering_unit'},
            'op_6': {'op_type': 'MLP', 'hw_unit': 'mlp_engine'}
        },
        'edges': [
            {'source': 'op_1', 'target': 'op_2'},
            {'source': 'op_2', 'target': 'op_3'},
            {'source': 'op_3', 'target': 'op_4'},
            {'source': 'op_4', 'target': 'op_5'},
            {'source': 'op_2', 'target': 'op_6'},
            {'source': 'op_6', 'target': 'op_5'}
        ]
    }
    
    # Sample schedule data
    schedule_data = {
        'metadata': {
            'operators_count': 6,
            'total_execution_time': 150
        },
        'schedule': {
            'operators': [
                {
                    'op_id': 'op_1',
                    'hw_unit': 'hash_unit',
                    'start_cycle': 0,
                    'duration': 25
                },
                {
                    'op_id': 'op_2',
                    'hw_unit': 'mlp_engine', 
                    'start_cycle': 25,
                    'duration': 40
                },
                {
                    'op_id': 'op_3',
                    'hw_unit': 'memory_controller',
                    'start_cycle': 65,
                    'duration': 15
                },
                {
                    'op_id': 'op_4',
                    'hw_unit': 'rendering_unit',
                    'start_cycle': 80,
                    'duration': 35
                },
                {
                    'op_id': 'op_5',
                    'hw_unit': 'rendering_unit',
                    'start_cycle': 115,
                    'duration': 35,
                    'is_critical_path': True
                },
                {
                    'op_id': 'op_6',
                    'hw_unit': 'mlp_engine',
                    'start_cycle': 65,
                    'duration': 30
                }
            ]
        }
    }
    
    # Sample PPA data
    ppa_data = {
        'system_metrics': {
            'total_power_mw': 62.5,
            'total_area_mm2': 8.4,
            'performance_fps': 213.3,
            'power_efficiency_fps_per_watt': 3413.0,
            'area_efficiency_fps_per_mm2': 25.4
        },
        'operator_metrics': {
            'op_1': {
                'power_uw': 15000,
                'area_um2': 850000,
                'latency_cycles': 25,
                'hw_unit': 'hash_unit',
                'utilization': 0.167
            },
            'op_2': {
                'power_uw': 18000,
                'area_um2': 2100000,
                'latency_cycles': 40,
                'hw_unit': 'mlp_engine',
                'utilization': 0.267
            },
            'op_3': {
                'power_uw': 8000,
                'area_um2': 450000,
                'latency_cycles': 15,
                'hw_unit': 'memory_controller',
                'utilization': 0.1
            },
            'op_4': {
                'power_uw': 12000,
                'area_um2': 1200000,
                'latency_cycles': 35,
                'hw_unit': 'rendering_unit',
                'utilization': 0.233
            },
            'op_5': {
                'power_uw': 15000,
                'area_um2': 1200000,
                'latency_cycles': 35,
                'hw_unit': 'rendering_unit',
                'utilization': 0.233
            },
            'op_6': {
                'power_uw': 10000,
                'area_um2': 2100000,
                'latency_cycles': 30,
                'hw_unit': 'mlp_engine',
                'utilization': 0.2
            }
        }
    }
    
    return {
        'execution_dag': execution_dag,
        'schedule_data': schedule_data,
        'ppa_data': ppa_data
    }


def demo_visualization_suite():
    """Demonstrate the complete visualization suite."""
    
    print("🎨 RenderSim Visualization Suite Demo")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Initialize schedule visualizer
    visualizer = ScheduleVisualizer(output_dir="demo_visualization_output")
    
    # Create complete analysis
    print("\n🎯 Creating Complete Analysis Report...")
    output_files = visualizer.create_complete_analysis(
        schedule_data=sample_data['schedule_data'],
        ppa_data=sample_data['ppa_data'],
        execution_dag=sample_data['execution_dag'],
        report_name="demo_neural_rendering"
    )
    
    print(f"\n📈 Generated {len(output_files)} visualization files")
    print("\n🎉 Demo completed successfully!")
    print("📁 Check demo_visualization_output/ directory for results")
    
    return output_files


if __name__ == "__main__":
    demo_visualization_suite()
