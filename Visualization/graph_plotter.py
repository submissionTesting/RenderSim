"""
RenderSim Graph Plotter

Utilities for visualizing operator graphs from neural rendering pipelines.
Supports DAG visualization with operator types, hardware assignments, and data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


class OperatorGraphPlotter:
    """Visualizes operator graphs with hardware assignments and data flow."""
    
    # Color scheme for different operator types
    OPERATOR_COLORS = {
        'HASH_ENCODE': '#FF6B6B',      # Red
        'FIELD_COMPUTATION': '#4ECDC4', # Teal  
        'SAMPLING': '#45B7D1',         # Blue
        'BLENDING': '#96CEB4',         # Green
        'VOLUME_RENDERING': '#FFEAA7', # Yellow
        'MLP': '#DDA0DD',              # Plum
        'ATTENTION': '#F39C12',        # Orange
        'UNKNOWN': '#95A5A6'           # Gray
    }
    
    # Hardware unit shapes
    HW_SHAPES = {
        'systolic_array': 's',      # Square
        'hash_unit': 'o',           # Circle
        'mlp_engine': '^',          # Triangle
        'rendering_unit': 'D',      # Diamond
        'memory_controller': 'h',   # Hexagon
        'default': 'o'              # Circle
    }
    
    def __init__(self):
        """Initialize the graph plotter."""
        self.figure_size = (12, 8)
        self.dpi = 300
        
    def plot_operator_graph(self, 
                          graph_data: Dict[str, Any], 
                          output_path: str,
                          title: str = "Neural Rendering Operator Graph",
                          show_hardware_assignments: bool = True,
                          show_data_flow: bool = True,
                          layout: str = 'hierarchical') -> None:
        """
        Plot operator graph with optional hardware assignments and data flow.
        """
        
        # Create NetworkX graph
        G = self._build_networkx_graph(graph_data)
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Choose layout algorithm
        if layout == 'hierarchical':
            pos = self._hierarchical_layout(G)
        elif layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw edges (data flow)
        if show_data_flow:
            self._draw_edges(G, pos, ax)
        
        # Draw nodes (operators)
        self._draw_nodes(G, pos, ax, show_hardware_assignments)
        
        # Add labels
        self._draw_labels(G, pos, ax)
        
        # Add legend
        self._add_legend(ax, show_hardware_assignments)
        
        # Format plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Operator graph saved to: {output_path}")
    
    def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        """Build NetworkX graph from graph data."""
        G = nx.DiGraph()
        
        # Handle different input formats
        if 'operators' in graph_data:
            # Mapped IR format
            operators = graph_data['operators']
            for op in operators:
                G.add_node(op['op_id'], **op)
                
            # Add edges if dependency information exists
            if 'dependencies' in graph_data:
                for dep in graph_data['dependencies']:
                    G.add_edge(dep['from'], dep['to'])
                    
        elif 'nodes' in graph_data:
            # Direct graph format
            for node_id, node_data in graph_data['nodes'].items():
                G.add_node(node_id, **node_data)
                
            for edge in graph_data.get('edges', []):
                G.add_edge(edge['source'], edge['target'])
        
        return G
    
    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on topological sorting."""
        try:
            # Topological sort to get hierarchy levels
            topo_order = list(nx.topological_sort(G))
            levels = {}
            
            # Assign levels using longest path
            for node in topo_order:
                if not list(G.predecessors(node)):
                    levels[node] = 0
                else:
                    levels[node] = max(levels[pred] for pred in G.predecessors(node)) + 1
            
            # Group nodes by level
            level_groups = {}
            for node, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            # Position nodes
            pos = {}
            for level, nodes in level_groups.items():
                for i, node in enumerate(nodes):
                    x = level * 2
                    y = (i - len(nodes)/2) * 1.5
                    pos[node] = (x, y)
            
            return pos
            
        except nx.NetworkXError:
            # Fallback to spring layout if graph has cycles
            return nx.spring_layout(G, k=2)
    
    def _draw_edges(self, G: nx.DiGraph, pos: Dict, ax: plt.Axes) -> None:
        """Draw edges with arrows indicating data flow."""
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#666666',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=1.5,
            alpha=0.7
        )
    
    def _draw_nodes(self, G: nx.DiGraph, pos: Dict, ax: plt.Axes, 
                   show_hardware_assignments: bool) -> None:
        """Draw nodes with colors based on operator type and shapes for HW units."""
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Get operator type and color
            op_type = node_data.get('op_type', 'UNKNOWN')
            color = self.OPERATOR_COLORS.get(op_type, self.OPERATOR_COLORS['UNKNOWN'])
            
            # Get hardware unit and shape
            hw_unit = node_data.get('hw_unit', 'default')
            shape = self.HW_SHAPES.get(hw_unit, self.HW_SHAPES['default'])
            
            # Draw node
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node], ax=ax,
                node_color=color,
                node_shape=shape,
                node_size=800,
                alpha=0.8,
                edgecolors='black',
                linewidths=2
            )
    
    def _draw_labels(self, G: nx.DiGraph, pos: Dict, ax: plt.Axes) -> None:
        """Draw node labels."""
        labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            # Create compact label
            op_type = node_data.get('op_type', 'UNK')[:4]  # First 4 chars
            labels[node] = f"{node[:6]}\n{op_type}"  # Node ID + op type
        
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=8,
            font_weight='bold'
        )
    
    def _add_legend(self, ax: plt.Axes, show_hardware_assignments: bool) -> None:
        """Add legend for operator types and hardware units."""
        
        # Operator type legend
        op_patches = []
        for op_type, color in self.OPERATOR_COLORS.items():
            if op_type != 'UNKNOWN':  # Skip unknown for cleaner legend
                patch = mpatches.Patch(color=color, label=op_type.replace('_', ' ').title())
                op_patches.append(patch)
        
        legend1 = ax.legend(handles=op_patches, loc='upper left', 
                          title='Operator Types', fontsize=9)
        legend1.get_title().set_fontweight('bold')
