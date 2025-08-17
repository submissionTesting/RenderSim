"""
RenderSim PPA Dashboard

Comprehensive dashboard for visualizing Power, Performance, Area (PPA) metrics.
Includes breakdown charts, efficiency analysis, and comparison visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


class PPADashboard:
    """Creates comprehensive PPA analysis dashboards with multiple metrics."""
    
    # Color schemes for different metric types
    POWER_COLORS = {
        'dynamic': '#E74C3C',      # Red
        'static': '#F39C12',       # Orange
        'leakage': '#F1C40F',      # Yellow
        'total': '#C0392B'         # Dark Red
    }
    
    PERFORMANCE_COLORS = {
        'latency': '#3498DB',      # Blue
        'throughput': '#2ECC71',   # Green
        'fps': '#1ABC9C',          # Turquoise
        'efficiency': '#9B59B6'    # Purple
    }
    
    AREA_COLORS = {
        'logic': '#E67E22',        # Orange
        'memory': '#34495E',       # Dark Blue
        'interconnect': '#95A5A6', # Gray
        'total': '#D35400'         # Dark Orange
    }
    
    def __init__(self):
        """Initialize the PPA dashboard."""
        self.figure_size = (16, 12)
        self.dpi = 300
    
    def create_comprehensive_dashboard(self, 
                                     ppa_data: Dict[str, Any],
                                     output_path: str,
                                     title: str = "Neural Rendering Accelerator PPA Analysis") -> None:
        """
        Create comprehensive PPA dashboard with multiple visualizations.
        """
        
        # Set up the dashboard layout
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)
        
        # Extract metrics
        system_metrics = ppa_data.get('system_metrics', {})
        operator_metrics = ppa_data.get('operator_metrics', {})
        metadata = ppa_data.get('metadata', {})
        
        # Top row: System-level overview
        ax_power_overview = fig.add_subplot(gs[0, 0])
        ax_perf_overview = fig.add_subplot(gs[0, 1])
        ax_area_overview = fig.add_subplot(gs[0, 2])
        ax_efficiency = fig.add_subplot(gs[0, 3])
        
        self._plot_power_overview(ax_power_overview, system_metrics)
        self._plot_performance_overview(ax_perf_overview, system_metrics)
        self._plot_area_overview(ax_area_overview, system_metrics)
        self._plot_efficiency_metrics(ax_efficiency, system_metrics)
        
        # Middle row: Detailed breakdowns
        ax_power_breakdown = fig.add_subplot(gs[1, :2])
        ax_area_breakdown = fig.add_subplot(gs[1, 2:])
        
        self._plot_power_breakdown_by_hw(ax_power_breakdown, operator_metrics)
        self._plot_area_breakdown_by_hw(ax_area_breakdown, operator_metrics)
        
        # Bottom row: Analysis and trends
        ax_utilization = fig.add_subplot(gs[2, :2])
        ax_pareto = fig.add_subplot(gs[2, 2:])
        
        self._plot_hardware_utilization(ax_utilization, operator_metrics)
        self._plot_ppa_pareto_analysis(ax_pareto, operator_metrics)
        
        # Add main title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Add configuration info
        config_text = self._get_config_text(metadata)
        fig.text(0.02, 0.02, config_text, fontsize=9, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PPA dashboard saved to: {output_path}")
    
    def _plot_power_overview(self, ax: plt.Axes, system_metrics: Dict[str, Any]) -> None:
        """Plot system-level power overview."""
        total_power = system_metrics.get('total_power_mw', 0)
        
        # Create gauge-style plot
        ax.pie([total_power, 100 - total_power], 
               colors=['#E74C3C', '#ECF0F1'],
               startangle=90,
               counterclock=False,
               wedgeprops=dict(width=0.3))
        
        ax.text(0, 0, f'{total_power:.1f}\nmW', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title('Total Power', fontweight='bold')
    
    def _plot_performance_overview(self, ax: plt.Axes, system_metrics: Dict[str, Any]) -> None:
        """Plot system-level performance overview."""
        fps = system_metrics.get('performance_fps', 0)
        
        # Create gauge-style plot  
        ax.pie([fps, max(0, 60 - fps)], 
               colors=['#2ECC71', '#ECF0F1'],
               startangle=90,
               counterclock=False,
               wedgeprops=dict(width=0.3))
        
        ax.text(0, 0, f'{fps:.1f}\nFPS', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title('Performance', fontweight='bold')
    
    def _plot_area_overview(self, ax: plt.Axes, system_metrics: Dict[str, Any]) -> None:
        """Plot system-level area overview."""
        total_area = system_metrics.get('total_area_mm2', 0)
        
        # Create gauge-style plot
        ax.pie([total_area, max(0, 50 - total_area)], 
               colors=['#E67E22', '#ECF0F1'],
               startangle=90,
               counterclock=False,
               wedgeprops=dict(width=0.3))
        
        ax.text(0, 0, f'{total_area:.1f}\nmm²', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_title('Total Area', fontweight='bold')
    
    def _plot_efficiency_metrics(self, ax: plt.Axes, system_metrics: Dict[str, Any]) -> None:
        """Plot efficiency metrics."""
        power_eff = system_metrics.get('power_efficiency_fps_per_watt', 0)
        area_eff = system_metrics.get('area_efficiency_fps_per_mm2', 0)
        
        metrics = ['Power\nEfficiency', 'Area\nEfficiency']
        values = [power_eff, area_eff]
        colors = ['#9B59B6', '#1ABC9C']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Efficiency Metrics', fontweight='bold')
        ax.set_ylabel('FPS/W or FPS/mm²', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_power_breakdown_by_hw(self, ax: plt.Axes, operator_metrics: Dict[str, Any]) -> None:
        """Plot power breakdown by hardware unit."""
        hw_power = {}
        
        for op_id, metrics in operator_metrics.items():
            hw_unit = metrics.get('hw_unit', 'unknown')
            power = metrics.get('power_uw', 0) / 1000  # Convert to mW
            
            if hw_unit not in hw_power:
                hw_power[hw_unit] = 0
            hw_power[hw_unit] += power
        
        # Create horizontal bar chart
        hw_units = list(hw_power.keys())
        powers = list(hw_power.values())
        colors = [self.POWER_COLORS.get('dynamic', '#E74C3C') for _ in hw_units]
        
        bars = ax.barh(hw_units, powers, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, power in zip(bars, powers):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                   f'{power:.1f} mW', ha='left', va='center', fontweight='bold')
        
        ax.set_title('Power Breakdown by Hardware Unit', fontweight='bold')
        ax.set_xlabel('Power (mW)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_area_breakdown_by_hw(self, ax: plt.Axes, operator_metrics: Dict[str, Any]) -> None:
        """Plot area breakdown by hardware unit."""
        hw_area = {}
        
        for op_id, metrics in operator_metrics.items():
            hw_unit = metrics.get('hw_unit', 'unknown')
            area = metrics.get('area_um2', 0) / 1e6  # Convert to mm²
            
            if hw_unit not in hw_area:
                hw_area[hw_unit] = 0
            hw_area[hw_unit] += area
        
        # Create pie chart
        labels = list(hw_area.keys())
        sizes = list(hw_area.values())
        colors = [self.AREA_COLORS.get('logic', '#E67E22') for _ in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        # Beautify text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Area Breakdown by Hardware Unit', fontweight='bold')
    
    def _plot_hardware_utilization(self, ax: plt.Axes, operator_metrics: Dict[str, Any]) -> None:
        """Plot hardware utilization analysis."""
        hw_utilization = {}
        
        for op_id, metrics in operator_metrics.items():
            hw_unit = metrics.get('hw_unit', 'unknown')
            utilization = metrics.get('utilization', 0) * 100  # Convert to percentage
            
            if hw_unit not in hw_utilization:
                hw_utilization[hw_unit] = []
            hw_utilization[hw_unit].append(utilization)
        
        # Calculate average utilization per hardware unit
        hw_names = []
        avg_utilizations = []
        
        for hw_unit, utils in hw_utilization.items():
            hw_names.append(hw_unit.replace('_', ' ').title())
            avg_utilizations.append(np.mean(utils))
        
        # Create bar chart with color coding
        bars = ax.bar(range(len(hw_names)), avg_utilizations, 
                     edgecolor='black', alpha=0.8)
        
        # Color code based on utilization level
        for bar, util in zip(bars, avg_utilizations):
            if util < 30:
                bar.set_color('#E74C3C')  # Red for low utilization
            elif util < 70:
                bar.set_color('#F39C12')  # Orange for medium utilization
            else:
                bar.set_color('#2ECC71')  # Green for high utilization
        
        ax.set_xticks(range(len(hw_names)))
        ax.set_xticklabels(hw_names, rotation=45, ha='right')
        ax.set_ylabel('Average Utilization (%)', fontweight='bold')
        ax.set_title('Hardware Unit Utilization', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def _plot_ppa_pareto_analysis(self, ax: plt.Axes, operator_metrics: Dict[str, Any]) -> None:
        """Plot PPA Pareto frontier analysis."""
        power_values = []
        area_values = []
        performance_values = []
        
        for op_id, metrics in operator_metrics.items():
            power = metrics.get('power_uw', 0) / 1000  # mW
            area = metrics.get('area_um2', 0) / 1e6    # mm²
            perf = 1 / max(metrics.get('latency_cycles', 1), 1)  # Inverse latency as performance proxy
            
            power_values.append(power)
            area_values.append(area)
            performance_values.append(perf)
        
        # Create scatter plot (Power vs Area, color by Performance)
        scatter = ax.scatter(power_values, area_values, c=performance_values, 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Power (mW)', fontweight='bold')
        ax.set_ylabel('Area (mm²)', fontweight='bold')
        ax.set_title('PPA Trade-off Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Performance (1/latency)', fontweight='bold')
    
    def _get_config_text(self, metadata: Dict[str, Any]) -> str:
        """Get configuration text for dashboard."""
        accelerator = metadata.get('accelerator_type', 'Unknown')
        dram_config = metadata.get('dram_configuration', {})
        dram_type = dram_config.get('type', 'Unknown')
        dram_freq = dram_config.get('frequency_mhz', 0)
        
        return f"Configuration: {accelerator} | DRAM: {dram_type} @ {dram_freq} MHz"
