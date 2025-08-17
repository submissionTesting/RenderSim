"""
RenderSim Gantt Chart Plotter

Utilities for visualizing execution schedules as Gantt charts.
Shows operator execution timelines, hardware unit utilization, and critical path analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


class GanttChartPlotter:
    """Visualizes execution schedules as Gantt charts with hardware utilization."""
    
    # Color scheme for different hardware units
    HW_UNIT_COLORS = {
        'systolic_array': '#FF6B6B',      # Red
        'hash_unit': '#4ECDC4',           # Teal
        'mlp_engine': '#45B7D1',          # Blue
        'rendering_unit': '#96CEB4',      # Green
        'memory_controller': '#FFEAA7',   # Yellow
        'custom_unit': '#DDA0DD',         # Plum
        'default': '#95A5A6'              # Gray
    }
    
    def __init__(self):
        """Initialize the Gantt chart plotter."""
        self.figure_size = (14, 10)
        self.dpi = 300
    
    def plot_execution_schedule(self, 
                              schedule_data: Dict[str, Any],
                              output_path: str,
                              title: str = "Neural Rendering Execution Schedule",
                              show_critical_path: bool = True,
                              show_hardware_utilization: bool = True,
                              time_unit: str = 'cycles') -> None:
        """
        Plot execution schedule as a Gantt chart.
        """
        
        # Extract operators and their scheduling info
        operators = self._extract_operators(schedule_data)
        
        # Group by hardware unit
        hw_groups = self._group_by_hardware(operators)
        
        # Set up the plot
        fig, (ax_main, ax_util) = plt.subplots(2, 1, 
                                             figsize=self.figure_size, 
                                             dpi=self.dpi,
                                             height_ratios=[3, 1] if show_hardware_utilization else [1, 0])
        
        # Plot main Gantt chart
        self._plot_gantt_bars(ax_main, hw_groups, show_critical_path, time_unit)
        
        # Plot hardware utilization
        if show_hardware_utilization:
            self._plot_hardware_utilization(ax_util, hw_groups, time_unit)
        else:
            ax_util.remove()
        
        # Format the plot
        self._format_gantt_plot(ax_main, title, time_unit)
        
        # Add legend
        self._add_gantt_legend(ax_main)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gantt chart saved to: {output_path}")
    
    def _extract_operators(self, schedule_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract operator information from schedule data."""
        if 'schedule' in schedule_data and 'operators' in schedule_data['schedule']:
            return schedule_data['schedule']['operators']
        elif 'operators' in schedule_data:
            return schedule_data['operators']
        else:
            return []
    
    def _group_by_hardware(self, operators: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group operators by hardware unit."""
        hw_groups = {}
        
        for op in operators:
            hw_unit = op.get('hw_unit', 'default')
            if hw_unit not in hw_groups:
                hw_groups[hw_unit] = []
            hw_groups[hw_unit].append(op)
        
        # Sort operators within each group by start time
        for hw_unit in hw_groups:
            hw_groups[hw_unit].sort(key=lambda x: x.get('start_cycle', 0))
        
        return hw_groups
    
    def _plot_gantt_bars(self, ax: plt.Axes, hw_groups: Dict[str, List[Dict[str, Any]]], 
                        show_critical_path: bool, time_unit: str) -> None:
        """Plot the main Gantt chart bars."""
        
        y_positions = {}
        y_pos = 0
        
        for hw_unit, operators in hw_groups.items():
            y_positions[hw_unit] = y_pos
            
            # Get color for this hardware unit
            color = self.HW_UNIT_COLORS.get(hw_unit, self.HW_UNIT_COLORS['default'])
            
            for op in operators:
                start = op.get('start_cycle', 0)
                duration = op.get('duration', 1)
                
                # Determine bar color (critical path vs regular)
                bar_color = color
                if show_critical_path and op.get('is_critical_path', False):
                    bar_color = '#E74C3C'  # Red for critical path
                    edge_color = '#C0392B'
                    line_width = 3
                else:
                    edge_color = 'black'
                    line_width = 1
                
                # Plot bar
                ax.barh(y_pos, duration, left=start, height=0.8,
                       color=bar_color, alpha=0.8, 
                       edgecolor=edge_color, linewidth=line_width)
                
                # Add operator label
                op_id = op.get('op_id', 'Unknown')[:8]  # Truncate long IDs
                ax.text(start + duration/2, y_pos, op_id, 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            y_pos += 1
        
        # Set y-axis labels
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([hw_unit.replace('_', ' ').title() for hw_unit in y_positions.keys()])
    
    def _plot_hardware_utilization(self, ax: plt.Axes, hw_groups: Dict[str, List[Dict[str, Any]]], 
                                  time_unit: str) -> None:
        """Plot hardware utilization over time."""
        
        # Calculate total execution time
        max_time = 0
        for operators in hw_groups.values():
            for op in operators:
                end_time = op.get('start_cycle', 0) + op.get('duration', 1)
                max_time = max(max_time, end_time)
        
        # Create time bins for utilization calculation
        time_bins = np.arange(0, max_time + 1, max(1, max_time // 100))
        utilization = np.zeros(len(time_bins) - 1)
        
        # Calculate utilization for each time bin
        for i in range(len(time_bins) - 1):
            bin_start, bin_end = time_bins[i], time_bins[i + 1]
            active_units = 0
            
            for operators in hw_groups.values():
                unit_active = False
                for op in operators:
                    op_start = op.get('start_cycle', 0)
                    op_end = op_start + op.get('duration', 1)
                    
                    # Check if operator overlaps with this time bin
                    if op_start < bin_end and op_end > bin_start:
                        unit_active = True
                        break
                
                if unit_active:
                    active_units += 1
            
            utilization[i] = active_units / len(hw_groups) * 100  # Percentage
        
        # Plot utilization
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        ax.plot(bin_centers, utilization, linewidth=2, color='#3498DB')
        ax.fill_between(bin_centers, utilization, alpha=0.3, color='#3498DB')
        
        ax.set_ylabel('Utilization (%)', fontweight='bold')
        ax.set_xlabel(f'Time ({time_unit})', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    
    def _format_gantt_plot(self, ax: plt.Axes, title: str, time_unit: str) -> None:
        """Format the Gantt chart plot."""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(f'Time ({time_unit})', fontweight='bold')
        ax.set_ylabel('Hardware Units', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()  # Hardware units from top to bottom
    
    def _add_gantt_legend(self, ax: plt.Axes) -> None:
        """Add legend for Gantt chart."""
        
        # Hardware unit legend
        hw_patches = []
        for hw_unit, color in self.HW_UNIT_COLORS.items():
            if hw_unit != 'default':
                patch = mpatches.Patch(color=color, 
                                     label=hw_unit.replace('_', ' ').title())
                hw_patches.append(patch)
        
        # Add critical path indicator
        critical_patch = mpatches.Patch(color='#E74C3C', label='Critical Path')
        hw_patches.append(critical_patch)
        
        legend = ax.legend(handles=hw_patches, loc='upper right', 
                         title='Hardware Units', fontsize=9)
        legend.get_title().set_fontweight('bold')
