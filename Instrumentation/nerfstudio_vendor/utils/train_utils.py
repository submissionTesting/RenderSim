# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training utilities with instrumentation support."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from nerfstudio.utils.rich_utils import CONSOLE


class TrainingTracer:
    """Manages execution tracing during training iterations."""
    
    def __init__(self, 
                 trace_config_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """Initialize the training tracer.
        
        Args:
            trace_config_path: Path to trace configuration JSON
            output_dir: Directory to save trace files
        """
        self.trace_config_path = trace_config_path
        self.output_dir = output_dir or Path("traces")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_iteration = None
        self.iteration_traces = {}
        self.is_tracing = False
        
        # Try to load tracing module
        try:
            from nerfstudio.instrumentation import tracing
            self.tracing_module = tracing
            self._setup_tracing()
        except ImportError:
            CONSOLE.log("[yellow]Warning: Tracing module not available[/yellow]")
            self.tracing_module = None
    
    def _setup_tracing(self):
        """Setup the tracing configuration."""
        if self.tracing_module and self.trace_config_path:
            if self.trace_config_path.exists():
                self.tracing_module.setup_tracing(str(self.trace_config_path))
            else:
                CONSOLE.log(f"[red]Trace config not found: {self.trace_config_path}[/red]")
    
    def start_iteration(self, iteration: int):
        """Start tracing for a specific iteration.
        
        Args:
            iteration: The training iteration number
        """
        if not self.tracing_module:
            return
        
        self.current_iteration = iteration
        self.is_tracing = True
        
        # Clear previous DAG
        self.tracing_module.execution_dag.clear()
        CONSOLE.log(f"[cyan]Started tracing iteration {iteration}[/cyan]")
    
    def end_iteration(self):
        """End tracing for the current iteration and save the trace."""
        if not self.tracing_module or not self.is_tracing:
            return
        
        iteration = self.current_iteration
        if iteration is None:
            return
        
        # Save the execution DAG for this iteration
        dag = self.tracing_module.execution_dag.copy()
        self.iteration_traces[iteration] = dag
        
        # Save to file
        trace_file = self.output_dir / f"execution_dag_iter_{iteration}.pkl"
        with open(trace_file, 'wb') as f:
            pickle.dump(dag, f)
        
        CONSOLE.log(f"[green]Saved trace for iteration {iteration} to {trace_file}[/green]")
        CONSOLE.log(f"  - Nodes: {dag.number_of_nodes()}, Edges: {dag.number_of_edges()}")
        
        self.is_tracing = False
        self.current_iteration = None
    
    def save_summary(self):
        """Save a summary of all collected traces."""
        if not self.iteration_traces:
            return
        
        summary_file = self.output_dir / "training_trace_summary.pkl"
        summary = {
            'iterations': list(self.iteration_traces.keys()),
            'traces': self.iteration_traces,
            'config_path': str(self.trace_config_path) if self.trace_config_path else None,
        }
        
        with open(summary_file, 'wb') as f:
            pickle.dump(summary, f)
        
        CONSOLE.log(f"[green]Saved training trace summary to {summary_file}[/green]")
        CONSOLE.log(f"  - Total iterations traced: {len(self.iteration_traces)}")


def extract_training_operators(dag: nx.DiGraph) -> Dict[str, Any]:
    """Extract training-specific operators from an execution DAG.
    
    Args:
        dag: The execution DAG
        
    Returns:
        Dictionary containing extracted operators and statistics
    """
    operators = {
        'forward': [],
        'backward': [],
        'optimizer': [],
        'loss': [],
        'other': [],
    }
    
    for node in dag.nodes():
        node_data = dag.nodes[node]
        func_name = node_data.get('func_name', '')
        
        # Categorize based on function name patterns
        if 'backward' in func_name.lower() or 'grad' in func_name.lower():
            operators['backward'].append(node)
        elif 'optimizer' in func_name.lower() or 'adam' in func_name.lower() or 'sgd' in func_name.lower():
            operators['optimizer'].append(node)
        elif 'loss' in func_name.lower() or 'criterion' in func_name.lower():
            operators['loss'].append(node)
        elif 'forward' in func_name.lower() or 'mlp' in func_name.lower() or 'encode' in func_name.lower():
            operators['forward'].append(node)
        else:
            operators['other'].append(node)
    
    # Compute statistics
    stats = {
        'total_nodes': dag.number_of_nodes(),
        'total_edges': dag.number_of_edges(),
        'forward_ops': len(operators['forward']),
        'backward_ops': len(operators['backward']),
        'optimizer_ops': len(operators['optimizer']),
        'loss_ops': len(operators['loss']),
        'other_ops': len(operators['other']),
    }
    
    return {
        'operators': operators,
        'statistics': stats,
        'dag': dag,
    }


def compare_training_iterations(traces: Dict[int, nx.DiGraph]) -> Dict[str, Any]:
    """Compare execution DAGs across training iterations.
    
    Args:
        traces: Dictionary mapping iteration numbers to DAGs
        
    Returns:
        Comparison results and statistics
    """
    if not traces:
        return {}
    
    comparisons = []
    iterations = sorted(traces.keys())
    
    for i, iter_num in enumerate(iterations):
        dag = traces[iter_num]
        ops = extract_training_operators(dag)
        
        comparison = {
            'iteration': iter_num,
            'stats': ops['statistics'],
        }
        
        # Compare with previous iteration if available
        if i > 0:
            prev_iter = iterations[i-1]
            prev_dag = traces[prev_iter]
            
            # Calculate differences
            node_diff = dag.number_of_nodes() - prev_dag.number_of_nodes()
            edge_diff = dag.number_of_edges() - prev_dag.number_of_edges()
            
            comparison['diff_from_prev'] = {
                'node_change': node_diff,
                'edge_change': edge_diff,
                'prev_iteration': prev_iter,
            }
        
        comparisons.append(comparison)
    
    return {
        'iterations': iterations,
        'comparisons': comparisons,
        'total_iterations': len(traces),
    }
