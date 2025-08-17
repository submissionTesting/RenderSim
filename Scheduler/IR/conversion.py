"""Conversion utilities between Python IR dataclasses and C++ pybind11 objects (rendersim_cpp).
If the C++ module isn't available (e.g., before building), fall back to pure-Python dataclasses.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from . import TensorDesc, OperatorNode, OperatorGraph, MappedIR, MappedIRNode

try:
    import rendersim_cpp  # type: ignore
except ImportError:  # pragma: no cover
    rendersim_cpp = None  # type: ignore


def tensor_to_cpp(t: TensorDesc):
    if rendersim_cpp is None:
        return t  # fallback
    shape=list(t.shape)
    if len(shape)<2:
        shape.append(1)
    td = rendersim_cpp.TensorDesc()
    td.shape = shape
    td.dtype = t.dtype
    return td


def node_to_cpp(n: OperatorNode):
    if rendersim_cpp is None:
        return n
    cpp_n = rendersim_cpp.OperatorNode()
    cpp_n.id = n.id
    # Determine op_type for hardware mapping
    hw_type = None
    try:
        hw_type = n.metadata.get('hardware_type') if hasattr(n, 'metadata') else None
    except Exception:
        hw_type = None
    if hw_type:
        cpp_n.op_type = str(hw_type)
    else:
        # Map taxonomy to hardware types
        t = (n.op_type or '').upper()
        if t == 'SAMPLING':
            cpp_n.op_type = 'VOLUME_RENDERING'
        elif t == 'ENCODING' or t == 'POSITIONAL_ENCODING':
            cpp_n.op_type = 'POSITIONAL_ENCODE'
        elif t == 'BLENDING':
            cpp_n.op_type = 'VOLUME_RENDERING'
        elif t == 'FIELD_COMPUTATION' or t == 'MLP_COMPUTATION':
            cpp_n.op_type = 'FIELD_COMPUTATION'
        else:
            cpp_n.op_type = t or 'FIELD_COMPUTATION'
    cpp_n.inputs = [tensor_to_cpp(t) for t in n.inputs]
    cpp_n.outputs = [tensor_to_cpp(t) for t in n.outputs]
    cpp_n.call_count = n.call_count
    return cpp_n


def graph_to_cpp(g: OperatorGraph):
    if rendersim_cpp is None:
        return g
    cpp_g = rendersim_cpp.OperatorGraph()
    id_to_idx = {}
    nodes_list = []
    # Preserve insertion order for stable indexing
    for idx, (nid, n) in enumerate(g.nodes.items()):
        nodes_list.append(node_to_cpp(n))
        id_to_idx[nid] = idx
    cpp_g.nodes = nodes_list  # assign to trigger vector conversion
    edges_list = []
    for src, dst in g.edges:
        if src in id_to_idx and dst in id_to_idx:
            edges_list.append((id_to_idx[src], id_to_idx[dst]))
    cpp_g.edges = edges_list  # assign to trigger vector conversion
    return cpp_g


def python_to_cpp_mapped_ir(mapped_ir: MappedIR):
    """Convert Python MappedIR to C++ MappedIR for compatibility with schedulers"""
    if rendersim_cpp is None:
        return mapped_ir  # fallback
    
    cpp_mapped_ir = rendersim_cpp.MappedIR()
    
    # Convert nodes
    for node_id, mapped_node in mapped_ir.nodes.items():
        cpp_op_node = node_to_cpp(mapped_node.op_node)
        cpp_mapped_node = rendersim_cpp.MappedIRNode()
        cpp_mapped_node.op_node = cpp_op_node
        cpp_mapped_node.hw_unit = mapped_node.hw_unit
        cpp_mapped_ir.nodes[node_id] = cpp_mapped_node
    
    # Convert edges
    for src, dst in mapped_ir.edges:
        cpp_mapped_ir.edges.append((src, dst))
    
    return cpp_mapped_ir 