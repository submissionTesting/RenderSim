# RenderSim Visualization Suite

Complete visualization package for neural rendering accelerator analysis.

## Components:
- OperatorGraphPlotter: DAG visualization with hardware assignments
- GanttChartPlotter: Execution schedule and critical path analysis  
- PPADashboard: Comprehensive PPA metrics dashboards
- ScheduleVisualizer: Unified interface for complete analysis

## Features:
- Static plots with matplotlib
- Interactive charts with plotly
- HTML report generation
- Comparison analysis
- Professional visualizations

## Usage:
```python
from RenderSim.Visualization import ScheduleVisualizer

visualizer = ScheduleVisualizer()
visualizer.create_complete_analysis(schedule_data, ppa_data, execution_dag)
```
