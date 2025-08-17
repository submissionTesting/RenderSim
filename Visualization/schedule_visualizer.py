"""
RenderSim Schedule Visualizer

Unified interface for comprehensive visualization of neural rendering execution schedules.
Combines operator graphs, Gantt charts, and PPA analysis into cohesive reports.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .graph_plotter import OperatorGraphPlotter
from .gantt_plotter import GanttChartPlotter
from .ppa_dashboard import PPADashboard


class ScheduleVisualizer:
    """Unified interface for comprehensive schedule visualization and analysis."""
    
    def __init__(self, output_dir: str = "visualization_output"):
        """
        Initialize the schedule visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize component visualizers
        self.graph_plotter = OperatorGraphPlotter()
        self.gantt_plotter = GanttChartPlotter()
        self.ppa_dashboard = PPADashboard()
    
    def create_complete_analysis(self,
                               schedule_data: Dict[str, Any],
                               ppa_data: Optional[Dict[str, Any]] = None,
                               execution_dag: Optional[Dict[str, Any]] = None,
                               report_name: str = "neural_rendering_analysis") -> Dict[str, str]:
        """
        Create complete visualization analysis with all components.
        
        Args:
            schedule_data: Execution schedule from scheduler
            ppa_data: PPA analysis data from report command
            execution_dag: Original execution DAG for graph visualization
            report_name: Base name for output files
            
        Returns:
            Dictionary mapping visualization type to output file paths
        """
        
        output_files = {}
        
        print("🎨 Creating comprehensive neural rendering analysis...")
        
        # 1. Execution Schedule Gantt Chart
        if schedule_data:
            gantt_path = self.output_dir / f"{report_name}_gantt_chart.png"
            self.gantt_plotter.plot_execution_schedule(
                schedule_data,
                str(gantt_path),
                title=f"Execution Schedule - {report_name.replace('_', ' ').title()}",
                show_critical_path=True,
                show_hardware_utilization=True
            )
            output_files['gantt_chart'] = str(gantt_path)
        
        # 2. Operator Graph Visualization
        if execution_dag:
            graph_path = self.output_dir / f"{report_name}_operator_graph.png"
            self.graph_plotter.plot_operator_graph(
                execution_dag,
                str(graph_path),
                title=f"Operator Graph - {report_name.replace('_', ' ').title()}",
                show_hardware_assignments=True,
                show_data_flow=True,
                layout='hierarchical'
            )
            output_files['operator_graph'] = str(graph_path)
        
        # 3. PPA Comprehensive Dashboard
        if ppa_data:
            ppa_dashboard_path = self.output_dir / f"{report_name}_ppa_dashboard.png"
            self.ppa_dashboard.create_comprehensive_dashboard(
                ppa_data,
                str(ppa_dashboard_path),
                title=f"PPA Analysis Dashboard - {report_name.replace('_', ' ').title()}"
            )
            output_files['ppa_dashboard'] = str(ppa_dashboard_path)
        
        # 4. Create Summary Report
        summary_path = self.output_dir / f"{report_name}_summary.html"
        self._create_summary_html(output_files, summary_path, report_name, 
                                schedule_data, ppa_data)
        output_files['summary_report'] = str(summary_path)
        
        print(f"\n✅ Complete analysis saved to: {self.output_dir}")
        print("📊 Generated visualizations:")
        for viz_type, path in output_files.items():
            print(f"   • {viz_type.replace('_', ' ').title()}: {Path(path).name}")
        
        return output_files
    
    def _create_summary_html(self,
                           output_files: Dict[str, str],
                           summary_path: Path,
                           report_name: str,
                           schedule_data: Optional[Dict[str, Any]],
                           ppa_data: Optional[Dict[str, Any]]) -> None:
        """Create HTML summary report."""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RenderSim Analysis Report - {report_name.replace('_', ' ').title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .viz-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa; }}
        .viz-image {{ width: 100%; height: auto; border-radius: 4px; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .metrics-table th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RenderSim Analysis Report</h1>
            <h2>{report_name.replace('_', ' ').title()}</h2>
            <p>Comprehensive neural rendering accelerator analysis</p>
        </div>
        
        <div class="section">
            <h3>Executive Summary</h3>
            {self._get_executive_summary(schedule_data, ppa_data)}
        </div>
        
        <div class="section">
            <h3>Visualizations</h3>
            <div class="viz-grid">
        """
        
        # Add visualization cards
        viz_titles = {
            'gantt_chart': 'Execution Schedule',
            'operator_graph': 'Operator Graph',
            'ppa_dashboard': 'PPA Dashboard'
        }
        
        for viz_type, path in output_files.items():
            if viz_type in viz_titles and Path(path).suffix == '.png':
                relative_path = Path(path).name  # Just filename for relative paths
                html_content += f"""
                <div class="viz-card">
                    <h4>{viz_titles[viz_type]}</h4>
                    <img src="{relative_path}" alt="{viz_titles[viz_type]}" class="viz-image">
                </div>
                """
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <p><em>Generated by RenderSim Visualization Suite</em></p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(summary_path, 'w') as f:
            f.write(html_content)
    
    def _get_executive_summary(self,
                             schedule_data: Optional[Dict[str, Any]],
                             ppa_data: Optional[Dict[str, Any]]) -> str:
        """Generate executive summary text."""
        
        summary = "<table class='metrics-table'>"
        summary += "<tr><th>Metric</th><th>Value</th></tr>"
        
        if schedule_data and 'metadata' in schedule_data:
            metadata = schedule_data['metadata']
            summary += f"<tr><td>Total Operators</td><td>{metadata.get('operators_count', 'N/A')}</td></tr>"
            summary += f"<tr><td>Execution Time</td><td>{metadata.get('total_execution_time', 'N/A')} cycles</td></tr>"
        
        if ppa_data and 'system_metrics' in ppa_data:
            metrics = ppa_data['system_metrics']
            summary += f"<tr><td>Performance</td><td>{metrics.get('performance_fps', 'N/A'):.1f} FPS</td></tr>"
            summary += f"<tr><td>Total Power</td><td>{metrics.get('total_power_mw', 'N/A'):.1f} mW</td></tr>"
            summary += f"<tr><td>Total Area</td><td>{metrics.get('total_area_mm2', 'N/A'):.1f} mm²</td></tr>"
            summary += f"<tr><td>Power Efficiency</td><td>{metrics.get('power_efficiency_fps_per_watt', 'N/A'):.1f} FPS/W</td></tr>"
        
        summary += "</table>"
        return summary
