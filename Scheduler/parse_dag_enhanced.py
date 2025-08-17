#!/usr/bin/env python3
"""
Enhanced DAG Parser with Operator Taxonomy Integration

This module extends the basic DAG parsing with proper operator type classification
using the neural rendering operator taxonomy from /Operators.
"""

import pickle
import networkx as nx
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '.')

from Scheduler.IR import OperatorGraph, OperatorNode, TensorDesc
from Instrumentation.operator_mapping import (
    map_function_to_operator_type, 
    map_operator_to_hardware_type,
    enhance_dag_with_operator_types,
    get_operator_statistics
)
from Instrumentation.dag_to_operators_integration import load_and_transform_traced_dag

def load_execution_dag_with_operators(dag_path: str) -> OperatorGraph:
    """
    Load execution DAG with full /Operators transformation for realistic performance modeling.
    
    This provides the most accurate neural rendering analysis by:
    1. Loading traced DAG data
    2. Transforming to actual /Operators instances
    3. Converting to Scheduler.IR format with realistic characteristics
    
    Args:
        dag_path: Path to the execution DAG pickle file
        
    Returns:
        OperatorGraph with realistic operator characteristics from /Operators framework
    """
    print(f"Loading DAG with /Operators transformation: {dag_path}")
    
    try:
        scheduler_graph, impact = load_and_transform_traced_dag(dag_path)
        
        print("Transformation Impact:")
        print(f"   Realistic operators: {impact['transformation_summary']['nodes_processed']}")
        print(f"   Total FLOPs: {impact['transformation_summary']['total_flops']}")
        print(f"   Memory workload: {impact['transformation_summary']['total_memory_mb']}")
        print(f"   Operator types: {list(impact['transformation_summary']['operator_distribution'].keys())}")
        
        return scheduler_graph
        
    except Exception as e:
        print(f"/Operators transformation failed ({e}), falling back to enhanced parsing")
        return load_execution_dag_enhanced(dag_path)

def load_execution_dag_enhanced(dag_path: str) -> OperatorGraph:
    """
    Load execution DAG from pickle file with enhanced operator type classification.
    
    Args:
        dag_path: Path to the execution DAG pickle file
        
    Returns:
        OperatorGraph with properly classified operator types
    """
    print(f"Loading enhanced DAG from: {dag_path}")
    
    # Load raw DAG data
    with open(dag_path, 'rb') as f:
        dag_data = pickle.load(f)
    
    if isinstance(dag_data, nx.DiGraph):
        # Convert NetworkX graph to our format
        raw_dag = {"nodes": {}, "edges": list(dag_data.edges())}
        for node_id, node_data in dag_data.nodes(data=True):
            # For NetworkX format, the node_id often contains the function name
            node_data_copy = node_data.copy()
            node_data_copy['function_name'] = str(node_id)
            raw_dag["nodes"][node_id] = node_data_copy
        dag_data = raw_dag
    
    print(f"Raw DAG loaded: {len(dag_data.get('nodes', {}))} nodes, {len(dag_data.get('edges', []))} edges")
    
    # Enhance with operator type classification
    enhanced_dag = enhance_dag_with_operator_types(dag_data)
    
    # Convert to OperatorGraph
    operator_graph = OperatorGraph()
    
    # Process nodes
    for node_id, node_info in enhanced_dag.get('nodes', {}).items():
        # Extract function name and map to operator type
        function_name = node_info.get('function_name', 'unknown')
        op_type = map_function_to_operator_type(function_name)
        hardware_type = map_operator_to_hardware_type(op_type)
        
        # Create tensor descriptors from node info
        inputs = []
        outputs = []
        
        # Try to extract tensor information if available
        if 'inputs' in node_info:
            for inp in node_info['inputs']:
                if isinstance(inp, dict) and 'shape' in inp:
                    inputs.append(TensorDesc(inp['shape'], inp.get('dtype','float32')))
                else:
                    inputs.append(TensorDesc([1,1], 'float32'))  # Default
        
        if 'outputs' in node_info:
            for out in node_info['outputs']:
                if isinstance(out, dict) and 'shape' in out:
                    outputs.append(TensorDesc(out['shape'], out.get('dtype','float32')))
                else:
                    outputs.append(TensorDesc([1,1],'float32'))  # Default
        
        # Default tensors if none specified
        if not inputs:
            inputs = [TensorDesc([1,1],'float32')]
        if not outputs:
            outputs = [TensorDesc([1,1],'float32')]
        
        # Create operator node
        operator_node = OperatorNode(
            id=str(node_id),
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            metadata={
                'function_name': function_name,
                'hardware_type': hardware_type,
                'original_node_id': node_id,
                **node_info
            }
        )
        
        operator_graph.nodes[str(node_id)] = operator_node
    
    # Process edges
    for edge in enhanced_dag.get('edges', []):
        if len(edge) >= 2:
            operator_graph.edges.append((str(edge[0]), str(edge[1])))
    
    print(f"Enhanced OperatorGraph created with {len(operator_graph.nodes)} nodes")
    
    # Print operator statistics
    stats = get_operator_statistics(enhanced_dag)
    print(f"Operator Type Distribution:")
    for op_type, count in sorted(stats.items()):
        print(f"   - {op_type}: {count}")
    
    return operator_graph

def analyze_neural_rendering_dag(dag_path: str) -> dict:
    """
    Comprehensive analysis of a neural rendering execution DAG.
    
    Args:
        dag_path: Path to the execution DAG pickle file
        
    Returns:
        Analysis results with operator statistics and characteristics
    """
    print(f"Analyzing Neural Rendering DAG: {dag_path}")
    print("=" * 60)
    
    # Load enhanced DAG
    operator_graph = load_execution_dag_enhanced(dag_path)
    
    # Collect analysis data
    analysis = {
        'total_operators': len(operator_graph.nodes),
        'total_edges': len(operator_graph.edges),
        'operator_types': {},
        'hardware_mapping': {},
        'critical_path': [],
        'parallelization_opportunities': []
    }
    
    # Analyze operator types and hardware mapping
    for node_id, node_data in operator_graph.nodes.items():
        op_type = node_data.op_type
        hw_type = node_data.metadata.get('hardware_type', 'unknown')
        
        analysis['operator_types'][op_type] = analysis['operator_types'].get(op_type, 0) + 1
        analysis['hardware_mapping'][hw_type] = analysis['hardware_mapping'].get(hw_type, 0) + 1
    
    # Find critical path (simplified - longest path through DAG)
    try:
        if hasattr(operator_graph, 'nx_graph'):
            if nx.is_directed_acyclic_graph(operator_graph.nx_graph):
                topo_sorted = list(nx.topological_sort(operator_graph.nx_graph))
                analysis['critical_path'] = topo_sorted[:10]  # First 10 nodes in topo order
        else:
            analysis['critical_path'] = list(operator_graph.nodes.keys())[:10]
    except:
        analysis['critical_path'] = []
    
    # Identify parallelization opportunities
    encoding_ops = [nid for nid, node in operator_graph.nodes.items() 
                   if 'ENCODING' in node.op_type]
    computation_ops = [nid for nid, node in operator_graph.nodes.items() 
                      if 'COMPUTATION' in node.op_type]
    
    analysis['parallelization_opportunities'] = {
        'parallel_encoding': len(encoding_ops),
        'parallel_computation': len(computation_ops),
        'encoding_nodes': encoding_ops[:5],  # Sample
        'computation_nodes': computation_ops[:5]  # Sample
    }
    
    # Print summary
    print(f"Analysis Summary:")
    print(f"   Total Operators: {analysis['total_operators']}")
    print(f"   Total Edges: {analysis['total_edges']}")
    print(f"   \nOperator Type Distribution:")
    for op_type, count in sorted(analysis['operator_types'].items()):
        percentage = (count / analysis['total_operators']) * 100
        print(f"      {op_type}: {count} ({percentage:.1f}%)")
    
    print(f"   \nHardware Module Distribution:")
    for hw_type, count in sorted(analysis['hardware_mapping'].items()):
        percentage = (count / analysis['total_operators']) * 100
        print(f"      {hw_type}: {count} ({percentage:.1f}%)")
    
    print(f"   \nParallelization Opportunities:")
    print(f"      Parallel Encoding Ops: {analysis['parallelization_opportunities']['parallel_encoding']}")
    print(f"      Parallel Computation Ops: {analysis['parallelization_opportunities']['parallel_computation']}")
    
    return analysis

if __name__ == "__main__":
    # Test enhanced DAG loading
    if len(sys.argv) > 1:
        dag_path = sys.argv[1]
    else:
        dag_path = "execution_dag.pkl"
    
    if Path(dag_path).exists():
        try:
            # Load and analyze DAG
            analysis = analyze_neural_rendering_dag(dag_path)
            
            # Test integration with RenderSim pipeline
            print(f"\nTesting RenderSim Integration:")
            print("=" * 40)
            
            operator_graph = load_execution_dag_enhanced(dag_path)
            print(f"Successfully loaded enhanced DAG")
            print("Ready for RenderSim mapping and scheduling")
            
        except Exception as e:
            print(f"Error processing DAG: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"DAG file not found: {dag_path}")
        print(f"Run neural rendering instrumentation first to generate execution_dag.pkl") 