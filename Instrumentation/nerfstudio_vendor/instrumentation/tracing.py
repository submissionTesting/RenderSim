from __future__ import annotations
import atexit
import importlib
import json
import os
import re
from functools import wraps
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------------------------------------------------------
# Global tracing state
# ----------------------------------------------------------------------------

execution_dag = nx.DiGraph()
_call_stack: List[str] = []
_call_counts = {}
_config_loaded = False
_INDEX_DUPLICATE_CALLS = False

# Stage and sequence tracking
_CURRENT_STAGE: Optional[str] = None
_IN_GET_OUTPUTS_DEPTH: int = 0
_LAST_NODE_BY_STAGE: dict[str, str] = {}
_SEQ_ADD_EDGES: bool = False
_SEQ_EDGE_ATTR: str = "sequence"
# Cross-stage transition (coarse -> fine)
_CROSS_STAGE_ADD_EDGE: bool = False
_CROSS_STAGE_EDGE_ATTR: str = "stage_transition"
# Trunk→Color data edge inside a stage (helps explain 283 = 256+27)
_ADD_COLOR_DATA_EDGE: bool = True
_DATA_COLOR_EDGE_ATTR: str = "data_concat"
_LAST_TRUNK_BY_STAGE: dict[str, str] = {}
_LAST_DENSITY_HEAD_BY_STAGE: dict[str, str] = {}
_ADD_DENSITY_WEIGHTS_EDGE: bool = True
_DENSITY_WEIGHTS_EDGE_ATTR: str = "density_to_weights"

# Plot configuration (overridable via config file)
_PLOT_SHOW_NODE_LABELS: bool = False
_PLOT_SHOW_EDGE_LABELS: bool = False
_PLOT_NODE_LABEL_TEMPLATE: str = "{func_name}"
_PLOT_EDGE_LABEL_ATTR: str = "weight"
_PLOT_FIGSIZE = (24, 16)
_PLOT_DPI = 200
_PLOT_LAYOUT = "spring"  # one of: spring, dot, layered
_PLOT_RANKDIR = "TB"     # for graphviz dot: TB, LR, BT, RL
_PLOT_LAYOUT_K = 0.3
_PLOT_NODE_SIZE = 80
_PLOT_EDGE_WIDTH = 0.4
_PLOT_NODE_COLOR = "#1f77b4"
_PLOT_EDGE_COLOR = "#222222"
_PLOT_FONT_SIZE = 8
_PLOT_AXES_FACECOLOR: Optional[str] = None
_PLOT_AGGREGATE_BY_FUNC: bool = False
_PLOT_SCALE_NODE_SIZE_BY_COUNT: bool = False
_PLOT_NODE_SIZE_MIN: int = 200
_PLOT_NODE_SIZE_MAX: int = 1200
_PLOT_WRITE_VECTOR_COPY: bool = False
_PLOT_VECTOR_COPY_FORMAT: str = "svg"  # svg or pdf
_PLOT_RENDER_SEQUENCE_EDGES: bool = False

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _load_json_with_comments(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    # Strip // line comments
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.MULTILINE)
    # Strip /* ... */ block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return json.loads(text)


def _shape_of(obj) -> Optional[str]:
    try:
        if hasattr(obj, "shape"):
            shp = tuple(getattr(obj, "shape"))
            if len(shp) > 0:
                return str(shp)
    except Exception:
        pass
    return None


def _collect_shapes_from(value) -> List[str]:
    shapes: List[str] = []
    # Direct tensor-like
    s = _shape_of(value)
    if s:
        shapes.append(s)
        return shapes
    # Common containers
    try:
        if isinstance(value, (list, tuple)):
            for it in value:
                if len(shapes) >= 2:
                    break
                s = _shape_of(it)
                if s:
                    shapes.append(s)
        elif isinstance(value, dict):
            for it in value.values():
                if len(shapes) >= 2:
                    break
                s = _shape_of(it)
                if s:
                    shapes.append(s)
    except Exception:
        pass
    return shapes


def _make_node_label(func_name: str, args, kwargs, result) -> tuple[str, str, str]:
    in_shapes = []
    try:
        # positional
        for a in args:
            in_shapes.extend(_collect_shapes_from(a))
        # keyword
        for v in kwargs.values():
            in_shapes.extend(_collect_shapes_from(v))
    except Exception:
        pass
    out_shapes = []
    try:
        if result is None:
            out_shapes = ["None"]
        elif isinstance(result, (list, tuple, dict)):
            out_shapes.extend(_collect_shapes_from(result))
        else:
            s = _shape_of(result)
            out_shapes.append(s if s else "?")
    except Exception:
        out_shapes = ["?"]
    in_sig = ",".join(in_shapes) if in_shapes else "no_tensors"
    out_sig = ",".join([s for s in out_shapes if s]) if out_shapes else "no_tensors"
    node_id = f"{func_name}[{in_sig}->{out_sig}]"
    return node_id, in_sig, out_sig


# ----------------------------------------------------------------------------
# Tracing decorator
# ----------------------------------------------------------------------------

def _trace_wrapper(func, fq_name: str):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        base = fq_name
        # Provisional push to capture edges from children
        _call_stack.append(base)
        # Stage management and get_outputs scope tracking
        entered_get_outputs = False
        try:
            global _IN_GET_OUTPUTS_DEPTH, _CURRENT_STAGE, _LAST_NODE_BY_STAGE, _LAST_TRUNK_BY_STAGE
            if base == "nerfstudio.models.vanilla_nerf.NeRFModel.get_outputs":
                _IN_GET_OUTPUTS_DEPTH += 1
                entered_get_outputs = True
                if _IN_GET_OUTPUTS_DEPTH == 1:
                    _CURRENT_STAGE = None
                    _LAST_NODE_BY_STAGE.clear()
                    _LAST_TRUNK_BY_STAGE.clear()
            # Determine stage for this call based on sampler events
            if _IN_GET_OUTPUTS_DEPTH > 0:
                if base == "nerfstudio.model_components.ray_samplers.UniformSampler.generate_ray_samples":
                    _CURRENT_STAGE = "coarse"
                elif base == "nerfstudio.model_components.ray_samplers.PDFSampler.generate_ray_samples":
                    _CURRENT_STAGE = "fine"
            # Capture stage at call time for node attribution
            stage_for_this_call = _CURRENT_STAGE if _IN_GET_OUTPUTS_DEPTH > 0 else None

            result = func(*args, **kwargs)
            node_id, in_sig, out_sig = _make_node_label(base, args, kwargs, result)
            final_id = node_id
            if _INDEX_DUPLICATE_CALLS:
                k = _call_counts.get(node_id, 0) + 1
                _call_counts[node_id] = k
                final_id = f"{node_id}#{k}"
            # Relabel provisional if present
            try:
                if base in execution_dag and final_id != base:
                    nx.relabel_nodes(execution_dag, {base: final_id}, copy=False)
            except Exception:
                pass
            # Create/update node
            if final_id not in execution_dag:
                execution_dag.add_node(
                    final_id,
                    func_name=base,
                    input_shapes=in_sig,
                    output_shapes=out_sig,
                    stage=stage_for_this_call,
                    count=1,
                )
            else:
                nd = execution_dag.nodes[final_id]
                nd["count"] = int(nd.get("count", 0)) + 1
                nd.setdefault("func_name", base)
                nd.setdefault("input_shapes", in_sig)
                nd.setdefault("output_shapes", out_sig)
                if stage_for_this_call is not None:
                    nd.setdefault("stage", stage_for_this_call)
            # Edge from caller to callee
            if len(_call_stack) >= 2:
                caller = _call_stack[-2]
                if caller != final_id:
                    if execution_dag.has_edge(caller, final_id):
                        execution_dag[caller][final_id]["weight"] = execution_dag[caller][final_id].get("weight", 0) + 1
                    else:
                        execution_dag.add_edge(caller, final_id, weight=1)
            # Optional sequence edge within the same stage to capture linear ordering
            if _SEQ_ADD_EDGES and stage_for_this_call:
                prev = _LAST_NODE_BY_STAGE.get(stage_for_this_call)
                if prev and prev != final_id:
                    if execution_dag.has_edge(prev, final_id):
                        execution_dag[prev][final_id][_SEQ_EDGE_ATTR] = execution_dag[prev][final_id].get(_SEQ_EDGE_ATTR, 0) + 1
                    else:
                        execution_dag.add_edge(prev, final_id, **{_SEQ_EDGE_ATTR: 1})
                _LAST_NODE_BY_STAGE[stage_for_this_call] = final_id
            # Always track the last node per stage even if sequence edges are not being added
            elif stage_for_this_call:
                _LAST_NODE_BY_STAGE[stage_for_this_call] = final_id
            # Track trunk MLP (… ,63)->(…,256) per stage
            try:
                if stage_for_this_call and base.endswith("field_components.mlp.MLP.forward"):
                    is_trunk = (", 63)" in in_sig) and (", 256)" in out_sig)
                    is_color = (", 283)" in in_sig) and (", 128)" in out_sig)
                    if is_trunk:
                        _LAST_TRUNK_BY_STAGE[stage_for_this_call] = final_id
                    # Add trunk→color data edge to explain concat 256+27 → 283
                    if is_color and _ADD_COLOR_DATA_EDGE:
                        trunk = _LAST_TRUNK_BY_STAGE.get(stage_for_this_call)
                        # Fallback: if trunk was not cached (edge case), search the latest trunk node in this stage
                        if not trunk:
                            try:
                                for nid in reversed(list(execution_dag.nodes())):
                                    nd = execution_dag.nodes[nid]
                                    if (nd or {}).get("stage") != stage_for_this_call:
                                        continue
                                    fn = str((nd or {}).get("func_name", ""))
                                    ins = str((nd or {}).get("input_shapes", ""))
                                    outs = str((nd or {}).get("output_shapes", ""))
                                    if fn.endswith("field_components.mlp.MLP.forward") and (", 63)" in ins) and (", 256)" in outs):
                                        trunk = nid
                                        break
                            except Exception:
                                pass
                        if trunk and trunk != final_id:
                            if execution_dag.has_edge(trunk, final_id):
                                execution_dag[trunk][final_id][_DATA_COLOR_EDGE_ATTR] = execution_dag[trunk][final_id].get(_DATA_COLOR_EDGE_ATTR, 0) + 1
                            else:
                                execution_dag.add_edge(trunk, final_id, **{_DATA_COLOR_EDGE_ATTR: 1})
                # Track density head and add density→weights data edge
                if stage_for_this_call and base.endswith("field_components.field_heads.DensityFieldHead.forward"):
                    _LAST_DENSITY_HEAD_BY_STAGE[stage_for_this_call] = final_id
                if (
                    stage_for_this_call
                    and _ADD_DENSITY_WEIGHTS_EDGE
                    and base.endswith("cameras.rays.RaySamples.get_weights")
                ):
                    dens = _LAST_DENSITY_HEAD_BY_STAGE.get(stage_for_this_call)
                    if dens and dens != final_id:
                        if execution_dag.has_edge(dens, final_id):
                            execution_dag[dens][final_id][_DENSITY_WEIGHTS_EDGE_ATTR] = execution_dag[dens][final_id].get(_DENSITY_WEIGHTS_EDGE_ATTR, 0) + 1
                        else:
                            execution_dag.add_edge(dens, final_id, **{_DENSITY_WEIGHTS_EDGE_ATTR: 1})
            except Exception:
                pass
            # Optional cross-stage edge: link last coarse to first fine within current get_outputs
            if (
                _CROSS_STAGE_ADD_EDGE
                and stage_for_this_call == "fine"
                and _LAST_NODE_BY_STAGE.get("coarse")
            ):
                coarse_last = _LAST_NODE_BY_STAGE.get("coarse")
                if coarse_last and coarse_last != final_id:
                    if execution_dag.has_edge(coarse_last, final_id):
                        execution_dag[coarse_last][final_id][_CROSS_STAGE_EDGE_ATTR] = execution_dag[coarse_last][final_id].get(_CROSS_STAGE_EDGE_ATTR, 0) + 1
                    else:
                        execution_dag.add_edge(coarse_last, final_id, **{_CROSS_STAGE_EDGE_ATTR: 1})
            return result
        finally:
            if _call_stack:
                _call_stack.pop()
            if entered_get_outputs:
                _IN_GET_OUTPUTS_DEPTH -= 1
                if _IN_GET_OUTPUTS_DEPTH <= 0:
                    _IN_GET_OUTPUTS_DEPTH = 0
                    _CURRENT_STAGE = None
                    _LAST_NODE_BY_STAGE.clear()
                    _LAST_TRUNK_BY_STAGE.clear()
                    _LAST_DENSITY_HEAD_BY_STAGE.clear()
    return _wrapped


def _apply_patch(module_path: str, class_name: Optional[str], func_name: str, wrapper_factory):
    mod = importlib.import_module(module_path)
    if class_name:
        cls = getattr(mod, class_name)
        attr = getattr(cls, func_name)
        # Preserve descriptor type
        import inspect as _inspect
        orig = _inspect.getattr_static(cls, func_name)
        if isinstance(orig, classmethod):
            wrapped = wrapper_factory(attr.__func__)
            setattr(cls, func_name, classmethod(wrapped))
        elif isinstance(orig, staticmethod):
            wrapped = wrapper_factory(orig.__func__)
            setattr(cls, func_name, staticmethod(wrapped))
        else:
            setattr(cls, func_name, wrapper_factory(attr))
    else:
        func = getattr(mod, func_name)
        setattr(mod, func_name, wrapper_factory(func))


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def load_trace_config(config_path: str, unique_calls: bool = False) -> None:
    global _config_loaded, _INDEX_DUPLICATE_CALLS
    global _PLOT_SHOW_NODE_LABELS, _PLOT_SHOW_EDGE_LABELS, _PLOT_NODE_LABEL_TEMPLATE
    global _PLOT_FIGSIZE, _PLOT_DPI, _PLOT_LAYOUT, _PLOT_RANKDIR, _PLOT_LAYOUT_K, _PLOT_NODE_SIZE, _PLOT_EDGE_WIDTH, _PLOT_EDGE_LABEL_ATTR
    global _PLOT_NODE_COLOR, _PLOT_EDGE_COLOR, _PLOT_FONT_SIZE, _PLOT_AXES_FACECOLOR
    global _PLOT_AGGREGATE_BY_FUNC, _PLOT_SCALE_NODE_SIZE_BY_COUNT, _PLOT_NODE_SIZE_MIN, _PLOT_NODE_SIZE_MAX
    global _PLOT_WRITE_VECTOR_COPY, _PLOT_VECTOR_COPY_FORMAT
    global _PLOT_RENDER_SEQUENCE_EDGES
    global _SEQ_ADD_EDGES, _SEQ_EDGE_ATTR
    global _CROSS_STAGE_ADD_EDGE, _CROSS_STAGE_EDGE_ATTR
    global _ADD_COLOR_DATA_EDGE, _DATA_COLOR_EDGE_ATTR
    global _ADD_DENSITY_WEIGHTS_EDGE, _DENSITY_WEIGHTS_EDGE_ATTR
    if _config_loaded:
        return
    cfg = _load_json_with_comments(config_path)
    _INDEX_DUPLICATE_CALLS = bool(cfg.get("index_duplicate_calls", unique_calls))
    # Plot options
    plot = dict(cfg.get("plot", {}))
    _PLOT_SHOW_NODE_LABELS = bool(plot.get("show_node_labels", _PLOT_SHOW_NODE_LABELS))
    _PLOT_SHOW_EDGE_LABELS = bool(plot.get("show_edge_labels", _PLOT_SHOW_EDGE_LABELS))
    _PLOT_NODE_LABEL_TEMPLATE = str(plot.get("node_label_template", _PLOT_NODE_LABEL_TEMPLATE))
    _PLOT_EDGE_LABEL_ATTR = str(plot.get("edge_label_attr", _PLOT_EDGE_LABEL_ATTR))
    if isinstance(plot.get("figsize"), (list, tuple)) and len(plot.get("figsize")) == 2:
        _PLOT_FIGSIZE = (float(plot["figsize"][0]), float(plot["figsize"][1]))
    _PLOT_DPI = int(plot.get("dpi", _PLOT_DPI))
    _PLOT_LAYOUT = str(plot.get("layout", _PLOT_LAYOUT)).lower()
    _PLOT_RANKDIR = str(plot.get("rankdir", _PLOT_RANKDIR))
    _PLOT_LAYOUT_K = float(plot.get("layout_k", _PLOT_LAYOUT_K))
    _PLOT_NODE_SIZE = int(plot.get("node_size", _PLOT_NODE_SIZE))
    _PLOT_EDGE_WIDTH = float(plot.get("edge_width", _PLOT_EDGE_WIDTH))
    _PLOT_NODE_COLOR = str(plot.get("node_color", _PLOT_NODE_COLOR))
    _PLOT_EDGE_COLOR = str(plot.get("edge_color", _PLOT_EDGE_COLOR))
    _PLOT_FONT_SIZE = int(plot.get("font_size", _PLOT_FONT_SIZE))
    _PLOT_AXES_FACECOLOR = plot.get("axes_facecolor", _PLOT_AXES_FACECOLOR)
    _PLOT_AGGREGATE_BY_FUNC = bool(plot.get("aggregate_by_func", _PLOT_AGGREGATE_BY_FUNC))
    _PLOT_SCALE_NODE_SIZE_BY_COUNT = bool(plot.get("scale_node_size_by_count", _PLOT_SCALE_NODE_SIZE_BY_COUNT))
    _PLOT_NODE_SIZE_MIN = int(plot.get("node_size_min", _PLOT_NODE_SIZE_MIN))
    _PLOT_NODE_SIZE_MAX = int(plot.get("node_size_max", _PLOT_NODE_SIZE_MAX))
    _PLOT_WRITE_VECTOR_COPY = bool(plot.get("write_vector_copy", _PLOT_WRITE_VECTOR_COPY))
    _PLOT_VECTOR_COPY_FORMAT = str(plot.get("vector_copy_format", _PLOT_VECTOR_COPY_FORMAT))
    _PLOT_RENDER_SEQUENCE_EDGES = bool(plot.get("render_sequence_edges", _PLOT_RENDER_SEQUENCE_EDGES))

    # Sequence options
    seq = dict(cfg.get("sequence", {}))
    _SEQ_ADD_EDGES = bool(seq.get("add_sequence_edges", _SEQ_ADD_EDGES))
    _SEQ_EDGE_ATTR = str(seq.get("sequence_edge_attr", _SEQ_EDGE_ATTR))

    # Stage options
    stage_cfg = dict(cfg.get("stage", {}))
    _CROSS_STAGE_ADD_EDGE = bool(stage_cfg.get("link_coarse_to_fine", _CROSS_STAGE_ADD_EDGE))
    _CROSS_STAGE_EDGE_ATTR = str(stage_cfg.get("cross_stage_edge_attr", _CROSS_STAGE_EDGE_ATTR))
    _ADD_COLOR_DATA_EDGE = bool(stage_cfg.get("add_concat_edge", _ADD_COLOR_DATA_EDGE))
    _DATA_COLOR_EDGE_ATTR = str(stage_cfg.get("concat_edge_attr", _DATA_COLOR_EDGE_ATTR))
    _ADD_DENSITY_WEIGHTS_EDGE = bool(stage_cfg.get("add_density_to_weights_edge", _ADD_DENSITY_WEIGHTS_EDGE))
    _DENSITY_WEIGHTS_EDGE_ATTR = str(stage_cfg.get("density_to_weights_edge_attr", _DENSITY_WEIGHTS_EDGE_ATTR))

    functions = list(cfg.get("functions_to_trace", []))
    # Clear previous
    execution_dag.clear()
    _call_stack.clear()
    _call_counts.clear()
    # Apply wrappers
    for fq in functions:
        parts = fq.split(".")
        if len(parts) < 2:
            continue
        func_name = parts[-1]
        if len(parts) >= 3:
            module_path = ".".join(parts[:-2])
            class_name = parts[-2]
        else:
            module_path = ".".join(parts[:-1])
            class_name = None
        try:
            _apply_patch(module_path, class_name, func_name, lambda fn: _trace_wrapper(fn, fq))
        except Exception as e:
            # Non-fatal: continue applying others
            print(f"[Tracing] Failed to patch {fq}: {e}")
    _config_loaded = True


def save_dag(filename: str) -> None:
    try:
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with p.open("wb") as f:
            import copy
            pickle.dump(copy.deepcopy(execution_dag), f)
        print(f"[Tracing] DAG saved to {p}")
    except Exception as e:
        print(f"[Tracing] Failed to save DAG: {e}")


def load_dag(filename: str) -> None:
    """Load a pickled DAG into the global execution_dag."""
    try:
        import pickle
        p = Path(filename)
        with p.open("rb") as f:
            g = pickle.load(f)
        if isinstance(g, nx.DiGraph):
            # Replace in place to preserve references
            execution_dag.clear()
            execution_dag.add_nodes_from(g.nodes(data=True))
            execution_dag.add_edges_from(g.edges(data=True))
        else:
            # Convert to DiGraph if needed
            execution_dag.clear()
            execution_dag.add_nodes_from(g.nodes(data=True))
            execution_dag.add_edges_from(g.edges(data=True))
        print(f"[Tracing] DAG loaded from {p}")
    except Exception as e:
        print(f"[Tracing] Failed to load DAG: {e}")


def _compute_layered_pos(G: nx.DiGraph):
    # Try DAG layering first; fall back to BFS-level layering for cyclic graphs
    try:
        from networkx.algorithms.dag import topological_generations
        if nx.is_directed_acyclic_graph(G):
            layers = list(topological_generations(G))
            pos = {}
            y = 0.0
            y_step = -1.0
            for layer in layers:
                layer = list(layer)
                n = len(layer)
                if n == 0:
                    y += y_step
                    continue
                x_positions = [i - (n - 1) / 2.0 for i in range(n)]
                for x, node in zip(x_positions, layer):
                    pos[node] = (x, y)
                y += y_step
            for node in G.nodes:
                if node not in pos:
                    pos[node] = (0.0, y)
                    y += y_step
            return pos
    except Exception:
        pass
    # BFS-level layering for cyclic graphs
    from collections import deque, defaultdict
    indeg = {n: G.in_degree(n) for n in G.nodes}
    roots = [n for n, d in indeg.items() if d == 0]
    if not roots:
        # pick nodes with max out-degree if no clear roots
        max_out = max((G.out_degree(n) for n in G.nodes), default=0)
        roots = [n for n in G.nodes if G.out_degree(n) == max_out]
        if not roots:
            roots = list(G.nodes)[:1]
    visited = set()
    layers: dict[int, list] = defaultdict(list)
    q = deque((r, 0) for r in roots)
    for r in roots:
        visited.add(r)
    while q:
        node, depth = q.popleft()
        layers[depth].append(node)
        for succ in G.successors(node):
            if succ not in visited:
                visited.add(succ)
                q.append((succ, depth + 1))
    # Assign remaining nodes
    max_depth = max(layers.keys(), default=0)
    for n in G.nodes:
        if n not in visited:
            max_depth += 1
            layers[max_depth].append(n)
    pos = {}
    y = 0.0
    y_step = -1.0
    for depth in sorted(layers.keys()):
        layer = layers[depth]
        n = len(layer)
        xs = [i - (n - 1) / 2.0 for i in range(n)]
        for x, node in zip(xs, layer):
            pos[node] = (x, y)
        y += y_step
    return pos


def _build_plot_graph() -> nx.DiGraph:
    if not _PLOT_AGGREGATE_BY_FUNC:
        return execution_dag
    agg = nx.DiGraph()
    # Aggregate node metrics
    func_to_count: dict[str, int] = {}
    func_to_nodes: dict[str, int] = {}
    for _, data in execution_dag.nodes(data=True):
        fn = data.get("func_name", "unknown")
        c = int(data.get("count", 1))
        func_to_count[fn] = func_to_count.get(fn, 0) + c
        func_to_nodes[fn] = func_to_nodes.get(fn, 0) + 1
    for fn, total in func_to_count.items():
        agg.add_node(fn, func_name=fn, count=total, nodes=func_to_nodes.get(fn, 1))
    # Aggregate edges
    edge_w: dict[tuple[str, str], int] = {}
    for u, v, d in execution_dag.edges(data=True):
        ufn = execution_dag.nodes[u].get("func_name", "unknown")
        vfn = execution_dag.nodes[v].get("func_name", "unknown")
        w = int(d.get("weight", 1))
        edge_w[(ufn, vfn)] = edge_w.get((ufn, vfn), 0) + w
    for (ufn, vfn), w in edge_w.items():
        if ufn not in agg or vfn not in agg:
            continue
        agg.add_edge(ufn, vfn, weight=w)
    return agg


def plot_dag(filename: str) -> None:
    try:
        plt.figure(figsize=_PLOT_FIGSIZE, dpi=_PLOT_DPI)
        Gplot = _build_plot_graph()
        # Choose layout
        pos = None
        if _PLOT_LAYOUT == "dot":
            try:
                try:
                    from networkx.drawing.nx_pydot import graphviz_layout
                except Exception:
                    from networkx.drawing.nx_agraph import graphviz_layout
                pos = graphviz_layout(Gplot, prog="dot")
            except Exception as e:
                print(f"[Tracing] Graphviz layout failed ({e}); falling back to layered.")
                pos = _compute_layered_pos(Gplot)
        elif _PLOT_LAYOUT == "layered":
            pos = _compute_layered_pos(Gplot)
        else:
            pos = nx.spring_layout(Gplot, seed=0, k=_PLOT_LAYOUT_K)
        # Node labels
        labels = None
        if _PLOT_SHOW_NODE_LABELS:
            labels = {}
            for n, data in Gplot.nodes(data=True):
                try:
                    labels[n] = _PLOT_NODE_LABEL_TEMPLATE.format(node_id=n, **data)
                except Exception:
                    labels[n] = data.get("func_name", str(n))
        if _PLOT_AXES_FACECOLOR:
            try:
                plt.gca().set_facecolor(_PLOT_AXES_FACECOLOR)
            except Exception:
                pass
        node_sizes = _PLOT_NODE_SIZE
        if _PLOT_SCALE_NODE_SIZE_BY_COUNT:
            vals = [int(d.get("count", 1)) for _, d in Gplot.nodes(data=True)]
            vmin, vmax = (min(vals), max(vals)) if vals else (1, 1)
            if vmax == vmin:
                node_sizes = [_PLOT_NODE_SIZE for _ in vals]
            else:
                import numpy as _np
                node_sizes = list(_np.interp(vals, [vmin, vmax], [_PLOT_NODE_SIZE_MIN, _PLOT_NODE_SIZE_MAX]))
        nx.draw_networkx(
            Gplot,
            pos=pos,
            with_labels=_PLOT_SHOW_NODE_LABELS,
            labels=labels,
            node_size=node_sizes,
            width=_PLOT_EDGE_WIDTH,
            font_size=_PLOT_FONT_SIZE,
            node_color=_PLOT_NODE_COLOR,
            edge_color=_PLOT_EDGE_COLOR,
        )
        # Edge labels
        if _PLOT_SHOW_EDGE_LABELS:
            edge_labels = {(u, v): (d.get(_PLOT_EDGE_LABEL_ATTR, "") if isinstance(d, dict) else "") for u, v, d in Gplot.edges(data=True)}
            nx.draw_networkx_edge_labels(Gplot, pos=pos, edge_labels=edge_labels, font_size=_PLOT_FONT_SIZE, font_color=_PLOT_EDGE_COLOR)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        # Optional vector copy to avoid raster blurring on zoom
        if _PLOT_WRITE_VECTOR_COPY:
            try:
                from pathlib import Path as _Path
                vec_path = str(_Path(filename).with_suffix(f".{_PLOT_VECTOR_COPY_FORMAT}"))
                plt.savefig(vec_path, bbox_inches="tight")
                print(f"[Tracing] DAG vector copy plotted to {vec_path}")
            except Exception as e:
                print(f"[Tracing] Failed to write vector copy: {e}")
        plt.close()
        print(f"[Tracing] DAG plotted to {filename}")
    except Exception as e:
        print(f"[Tracing] Failed to plot DAG: {e}")


# ----------------------------------------------------------------------------
# DOT export
# ----------------------------------------------------------------------------

def export_dot(filename: str) -> None:
    """Export the current execution DAG to Graphviz DOT format.

    The exported graph respects aggregation and label settings from the
    plotting configuration for consistency.
    """
    try:
        Gplot = _build_plot_graph()

        def _esc(s: str) -> str:
            try:
                return str(s).replace("\\", "\\\\").replace("\"", "\\\"")
            except Exception:
                return str(s)

        lines: List[str] = []
        lines.append("digraph G {")
        lines.append(f"  rankdir={_PLOT_RANKDIR};")
        lines.append("  graph [ranksep=\"1.5\", nodesep=\"0.5\", pad=\"0.5\", splines=true];")
        lines.append("  node [shape=box, style=filled, fillcolor=\"#e7f0ff\", color=\"#1f77b4\", fontname=\"Arial\", fontsize=12];")
        lines.append("  edge [color=\"#111111\", penwidth=1.2, fontname=\"Arial\", fontsize=10, fontcolor=\"#111111\"];")

        # Group by weakly connected components (expected ~157), then nest by stage within each component
        try:
            components = list(nx.connected_components(Gplot.to_undirected()))
        except Exception:
            components = [set(Gplot.nodes())]

        # Pre-compute stage per node
        node_stage: dict[str, str] = {}
        for n, data in Gplot.nodes(data=True):
            st = (data or {}).get("stage")
            node_stage[n] = st if st in ("coarse", "fine") else "other"

        for idx, comp in enumerate(components):
            comp_nodes = set(comp)
            if idx > 0:
                lines.append("")
            lines.append(f"  subgraph cluster_{idx} {{")
            lines.append(f"    label=\"component {idx}\";")
            lines.append("    color=\"#aaaaaa\";")

            # Split component nodes by stage
            coarse_nodes = {n for n in comp_nodes if node_stage.get(n) == "coarse"}
            fine_nodes = {n for n in comp_nodes if node_stage.get(n) == "fine"}
            other_nodes = comp_nodes - coarse_nodes - fine_nodes

            def emit_stage_subgraph(stage_name: str, nodes: set[str]) -> None:
                if not nodes:
                    return
                lines.append(f"    subgraph cluster_{idx}_{stage_name} {{")
                lines.append(f"      label=\"{stage_name}\";")
                lines.append("      color=\"#cccccc\";")
                for n in nodes:
                    data = Gplot.nodes[n]
                    try:
                        label = _PLOT_NODE_LABEL_TEMPLATE.format(node_id=n, **data)
                    except Exception:
                        label = data.get("func_name", str(n))
                    # Ensure shapes are visible even if template lacks them
                    try:
                        in_s = data.get("input_shapes", "?")
                        out_s = data.get("output_shapes", "?")
                        shapes_line = f"{in_s}->{out_s}"
                        if ("input_shapes" not in _PLOT_NODE_LABEL_TEMPLATE) and ("output_shapes" not in _PLOT_NODE_LABEL_TEMPLATE) and (shapes_line not in label):
                            label = f"{label}\n{shapes_line}"
                    except Exception:
                        pass
                    lines.append(f"      \"{_esc(n)}\" [label=\"{_esc(label)}\"];")
                # Edges internal to this stage subgraph: prefer sequence edges to avoid call-graph cycles
                if _PLOT_SHOW_EDGE_LABELS:
                    for u, v, d in Gplot.edges(data=True):
                        if u in nodes and v in nodes and isinstance(d, dict) and d.get(_SEQ_EDGE_ATTR) is not None:
                            lbl = d.get(_PLOT_EDGE_LABEL_ATTR, "")
                            if lbl is None or lbl == "":
                                lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\";")
                            else:
                                lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\" [label=\"{_esc(lbl)}\"];")
                else:
                    for u, v, d in Gplot.edges(data=True):
                        if u in nodes and v in nodes and isinstance(d, dict) and d.get(_SEQ_EDGE_ATTR) is not None:
                            lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\";")
                # Trunk→Color data edges
                if _ADD_COLOR_DATA_EDGE:
                    for u, v, d in Gplot.edges(data=True):
                        if u in nodes and v in nodes and isinstance(d, dict) and d.get(_DATA_COLOR_EDGE_ATTR) is not None:
                            lbl = d.get(_PLOT_EDGE_LABEL_ATTR, "")
                            if lbl is None or lbl == "":
                                lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\";")
                            else:
                                lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\" [label=\"{_esc(lbl)}\"];")
                # Density→Weights data edges
                if _ADD_DENSITY_WEIGHTS_EDGE:
                    for u, v, d in Gplot.edges(data=True):
                        if u in nodes and v in nodes and isinstance(d, dict) and d.get(_DENSITY_WEIGHTS_EDGE_ATTR) is not None:
                            lbl = d.get(_PLOT_EDGE_LABEL_ATTR, "")
                            if lbl is None or lbl == "":
                                lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\";")
                            else:
                                lines.append(f"      \"{_esc(u)}\" -> \"{_esc(v)}\" [label=\"{_esc(lbl)}\"];")
                lines.append("    }")

            # Emit stages in order: coarse then fine (per requirement)
            emit_stage_subgraph("coarse", coarse_nodes)
            emit_stage_subgraph("fine", fine_nodes)

            # Emit any non-staged nodes at the component level
            if other_nodes:
                for n in other_nodes:
                    data = Gplot.nodes[n]
                    try:
                        label = _PLOT_NODE_LABEL_TEMPLATE.format(node_id=n, **data)
                    except Exception:
                        label = data.get("func_name", str(n))
                    # Ensure shapes are visible even if template lacks them
                    try:
                        in_s = data.get("input_shapes", "?")
                        out_s = data.get("output_shapes", "?")
                        shapes_line = f"{in_s}->{out_s}"
                        if ("input_shapes" not in _PLOT_NODE_LABEL_TEMPLATE) and ("output_shapes" not in _PLOT_NODE_LABEL_TEMPLATE) and (shapes_line not in label):
                            label = f"{label}\n{shapes_line}"
                    except Exception:
                        pass
                    lines.append(f"    \"{_esc(n)}\" [label=\"{_esc(label)}\"];")

            # Cross-stage edges within this component (including edges touching other_nodes)
            if _PLOT_SHOW_EDGE_LABELS:
                for u, v, d in Gplot.edges(data=True):
                    if u in comp_nodes and v in comp_nodes:
                        # Skip edges already rendered inside stage subgraphs (both endpoints in same stage)
                        if (u in coarse_nodes and v in coarse_nodes) or (u in fine_nodes and v in fine_nodes):
                            continue
                        lbl = d.get(_PLOT_EDGE_LABEL_ATTR, "") if isinstance(d, dict) else ""
                        if lbl is None or lbl == "":
                            lines.append(f"    \"{_esc(u)}\" -> \"{_esc(v)}\";")
                        else:
                            lines.append(f"    \"{_esc(u)}\" -> \"{_esc(v)}\" [label=\"{_esc(lbl)}\"];")
            else:
                for u, v in Gplot.edges():
                    if u in comp_nodes and v in comp_nodes:
                        if (u in coarse_nodes and v in coarse_nodes) or (u in fine_nodes and v in fine_nodes):
                            continue
                        lines.append(f"    \"{_esc(u)}\" -> \"{_esc(v)}\";")

            lines.append("  }")

        lines.append("}")

        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Tracing] DAG exported to DOT: {p}")
    except Exception as e:
        print(f"[Tracing] Failed to export DOT: {e}")


# Auto-save on exit so ns-eval doesn't need to do anything fancy
@atexit.register
def _auto_save_on_exit():
    try:
        save_dag("execution_dag.pkl")
    except Exception:
        pass