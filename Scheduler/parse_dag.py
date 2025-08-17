from __future__ import annotations

"""Utility to load an execution_dag.pkl (NetworkX DiGraph) produced by Nerfstudio
instrumentation into RenderSim's internal OperatorGraph structure.
"""

from pathlib import Path
import pickle
from typing import Any, Dict

import networkx as nx  # type: ignore

from .IR import OperatorGraph, OperatorNode, TensorDesc


def _tensor_size_to_shape(size: int) -> list[int]:
    """Heuristic: we only know total elements. Represent as [size]."""
    return [size]


def _extract_node_attrs(gx_node_attrs: Dict[str, Any]) -> OperatorNode:
    op_type = gx_node_attrs.get("label", gx_node_attrs.get("name", "unknown"))
    in_size = gx_node_attrs.get("input_a_size", 0) + gx_node_attrs.get("input_w_size", 0)
    out_size = gx_node_attrs.get("output_size", 0)
    inputs = [TensorDesc(shape=_tensor_size_to_shape(in_size))] if in_size else []
    outputs = [TensorDesc(shape=_tensor_size_to_shape(out_size))] if out_size else []
    call_count = gx_node_attrs.get("count", 1)
    return OperatorNode(
        id=str(gx_node_attrs.get("id", "")),
        op_type=str(op_type),
        inputs=inputs,
        outputs=outputs,
        call_count=call_count,
        metadata=gx_node_attrs,
    )


def load_execution_dag(pkl_path: str | Path) -> OperatorGraph:
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        dag: nx.DiGraph = pickle.load(f)  # noqa: S301

    op_graph = OperatorGraph()
    # the nodes may not have stable ids; create sequential ids
    node_id_map: Dict[Any, str] = {}
    for idx, (node_key, attrs) in enumerate(dag.nodes(data=True)):
        node_id = f"n{idx}"
        node_id_map[node_key] = node_id
        op_graph.nodes[node_id] = _extract_node_attrs({**attrs, "id": node_id})

    for src, dst in dag.edges():
        op_graph.edges.append((node_id_map[src], node_id_map[dst]))

    return op_graph 