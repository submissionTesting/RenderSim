"""IR definitions for Scheduler (Python stub).
Will later be replaced by C++/pybind11 equivalents for performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json

__all__ = [
    "TensorDesc",
    "OperatorNode",
    "OperatorGraph",
    "MappedIRNode",
    "MappedIR",
    "OperatorScheduledIRNode",
    "OperatorScheduledIR",
    "SystemScheduleEntry",
    "SystemSchedule",
]

@dataclass
class TensorDesc:
    shape: List[int]
    dtype: str = "float32"
    def bytes(self) -> int:
        elem_sz = 4 if self.dtype in {"float32", "int32"} else 1
        n_elem = 1
        for d in self.shape:
            n_elem *= d
        return n_elem * elem_sz

@dataclass
class OperatorNode:
    id: str
    op_type: str
    inputs: List[TensorDesc]
    outputs: List[TensorDesc]
    call_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperatorGraph:
    nodes: Dict[str, OperatorNode] = field(default_factory=dict)
    edges: List[tuple[str, str]] = field(default_factory=list)
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def from_json(cls, s: str) -> "OperatorGraph":
        import json as _j
        data = _j.loads(s)
        og = cls()
        for nid, n in data["nodes"].items():
            og.nodes[nid] = OperatorNode(
                id=n["id"],
                op_type=n["op_type"],
                inputs=[TensorDesc(td['shape'], td.get('dtype','float32')) for td in n["inputs"]],
                outputs=[TensorDesc(td['shape'], td.get('dtype','float32')) for td in n["outputs"]],
                call_count=n.get("call_count", 1),
                metadata=n.get("metadata", {}),
            )
        og.edges = [tuple(e) for e in data["edges"]]
        return og

@dataclass
class MappedIRNode:
    op_node: OperatorNode
    hw_unit: str
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MappedIR:
    nodes: Dict[str, MappedIRNode] = field(default_factory=dict)
    edges: List[tuple[str, str]] = field(default_factory=list)
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

@dataclass
class OperatorScheduledIRNode:
    mapped_node: MappedIRNode
    start_cycle: int
    duration: int
    resources: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperatorScheduledIR:
    nodes: Dict[str, OperatorScheduledIRNode] = field(default_factory=dict)
    edges: List[tuple[str, str]] = field(default_factory=list)
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

@dataclass
class SystemScheduleEntry:
    op_id: str
    hw_unit: str
    start_cycle: int
    duration: int

@dataclass
class SystemSchedule:
    entries: List[SystemScheduleEntry] = field(default_factory=list)
    total_cycles: Optional[int] = None
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) 