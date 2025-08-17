#!/usr/bin/env python3
"""
RenderSim CLI - Map Command Implementation
"""

import sys
import json
from pathlib import Path

# Add RenderSim to PYTHONPATH (both Python packages and C++ binding)
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "build" / "Scheduler" / "cpp"))
# Import RenderSim C++ core
import rendersim_cpp as rs

# Keep Python DAG parsers for now
from Scheduler.parse_dag import load_execution_dag
from Scheduler.parse_dag_enhanced import load_execution_dag_enhanced, load_execution_dag_with_operators
# Helper for converting Python operator graph to C++ structure


# -----------------------------------------------------------------------------
# Utility: convert Python OperatorGraph (networkx) to rendersim_cpp.OperatorGraph
# -----------------------------------------------------------------------------

def pygraph_to_cpp(py_graph):
    cpp_g = rs.OperatorGraph()
    # Support dict-like (id->node) and iterable containers (list/set of node objects)
    nodes_container = getattr(py_graph, 'nodes', None)
    if nodes_container is None:
        return cpp_g
    try:
        iterator = nodes_container.items()
        is_kv = True
    except Exception:
        iterator = enumerate(nodes_container)
        is_kv = False
    for key, node in iterator:
        cpp_node = rs.OperatorNode()
        # Determine id and op_type robustly
        if is_kv:
            nid = key
            cpp_node.id = getattr(node, 'id', str(nid))
            cpp_node.op_type = getattr(node, 'op_type', getattr(node, 'type', 'UNKNOWN'))
        else:
            nid = getattr(node, 'id', str(key))
            cpp_node.id = nid
            cpp_node.op_type = getattr(node, 'op_type', getattr(node, 'type', 'UNKNOWN'))
        cpp_g.nodes.append(cpp_node)
    # Edges (optional). If py_graph has edges as iterable of (u,v), try to copy them.
    edges = getattr(py_graph, 'edges', None)
    if edges is not None:
        try:
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    cpp_g.edges.append((str(e[0]), str(e[1])))
        except Exception:
            pass
    return cpp_g


def run_map_command(args, verbose=False):
    """
    Map operators from execution DAG to hardware configuration.
    
    Args:
        args: Argparse namespace with execution_dag, hardware_config, output
        verbose: Enable verbose output
    """
    try:
        if verbose:
            print(f"Loading execution DAG from {args.execution_dag}")
        
        # Load execution DAG
        dag_path = Path(args.execution_dag)
        if not dag_path.exists():
            raise FileNotFoundError(f"Execution DAG file not found: {dag_path}")
        
        # Load optimization hints if present
        hints = {}
        try:
            hints_path = Path('optimization_hints.json')
            if hints_path.exists():
                with hints_path.open() as hf:
                    obj = json.load(hf)
                    hints = obj.get('hints', {}) or {}
                if verbose:
                    print(f"   Loaded {len(hints)} optimization hints")
        except Exception as e:
            if verbose:
                print(f"   Failed to load optimization_hints.json ({e})")

        # Aggregate hint flags by operator category for robust propagation
        def _agg_flags():
            agg = {
                'encoding_hash_active': False,
                'field_low_bit': False,
                'sampling_active_ratio': None,  # leave None if not available
            }
            # Try to infer activity ratios from any keys that include 'weights' if present
            for k, v in hints.items():
                key = str(k)
                if 'HashEncoding' in key or 'MLPWithHashEncoding' in key:
                    if isinstance(v, dict) and v.get('hash_index_activity'):
                        agg['encoding_hash_active'] = True
                if 'MLP' in key:
                    if isinstance(v, dict) and v.get('low_bit_observed'):
                        agg['field_low_bit'] = True
                if 'weights' in key and isinstance(v, dict) and 'active_samples_ratio' in v:
                    try:
                        r = float(v['active_samples_ratio'])
                        # take the min ratio observed to be conservative
                        if agg['sampling_active_ratio'] is None:
                            agg['sampling_active_ratio'] = r
                        else:
                            agg['sampling_active_ratio'] = min(agg['sampling_active_ratio'], r)
                    except Exception:
                        pass
            return agg
        agg_flags = _agg_flags()

        # Use parser based on flag preference
        if getattr(args, 'basic_parser', False):
            if verbose:
                print("   Skipping /Operators transform (--basic-parser)")
            try:
                operator_graph = load_execution_dag_enhanced(dag_path)
                if verbose:
                    print("   Using enhanced DAG parser with operator taxonomy")
            except Exception:
                operator_graph = load_execution_dag(dag_path)
        else:
            # Prefer full /Operators transformation
            try:
                operator_graph = load_execution_dag_with_operators(dag_path)
                if verbose:
                    print("   Using complete /Operators transformation with realistic characteristics")
            except Exception as e:
                if verbose:
                    print(f"   /Operators transformation failed ({e}), trying enhanced parser")
                try:
                    operator_graph = load_execution_dag_enhanced(dag_path)
                    if verbose:
                        print("   Using enhanced DAG parser with operator taxonomy")
                except Exception as e2:
                    if verbose:
                        print(f"   Enhanced parser also failed ({e2}), falling back to basic parser")
                    operator_graph = load_execution_dag(dag_path)
        
        if verbose:
            print(f"   Loaded operator graph with {len(operator_graph.nodes)} nodes and {len(operator_graph.edges)} edges")
        
        # Load hardware configuration (prefer Python loader for robustness)
        if verbose:
            print(f"   Loading hardware configuration from {args.hardware_config}")
        config_path = Path(args.hardware_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Hardware config file not found: {config_path}")
        try:
            # Attempt C++ loader
            hw_config_cpp = rs.load_hw_config_from_json(str(config_path))
        except Exception:
            hw_config_cpp = None
        from Scheduler.mapping.hw_config import load_hw_config as load_hw_config_py
        hw_config_py = load_hw_config_py(config_path)
        if verbose:
            print(f"   Loaded hardware configuration with {len(hw_config_py.units)} hardware units")
            print(f"   Accelerator: {hw_config_py.accelerator_name}")
 
        # Try C++ mapping first; fall back to Python mapping if empty or failure
        cpp_mapped_ir = None
        try:
            cpp_graph = pygraph_to_cpp(operator_graph)
            if verbose:
                print(f"   Running operator-to-hardware mapping in C++...")
            cpp_mapped_ir = rs.map_operator_graph(cpp_graph, hw_config_cpp) if hw_config_cpp else None
            # Heuristic: if container has no items, treat as empty
            empty = True
            if cpp_mapped_ir is not None:
                try:
                    it = cpp_mapped_ir.nodes.items(); empty = True
                    for _k,_v in it:
                        empty = False; break
                except Exception:
                    try:
                        it2 = iter(cpp_mapped_ir.nodes)
                        empty = next(it2, None) is None
                    except Exception:
                        empty = True
            if verbose and cpp_mapped_ir is not None:
                print(f"   C++ mapper returned empty={empty}")
            if cpp_mapped_ir is None or empty:
                cpp_mapped_ir = None
        except Exception:
            cpp_mapped_ir = None
  
        # Determine output file
        output_file = args.output if args.output else "mapped_ir.json"
        output_path = Path(output_file)
         
        if verbose:
            print(f"Saving mapped IR to {output_path}")
         
        # Save mapped IR as JSON (C++ or Python fallback)
        with output_path.open('w') as f:
            if cpp_mapped_ir is not None:
                # Serialize C++ mapped IR
                nodes_json = {}
                # Build shape map from operator_graph (Python objects)
                shapes_by_id = {}
                try:
                    for _nid, _node in getattr(operator_graph, 'nodes', {}).items() if hasattr(operator_graph, 'nodes') and isinstance(operator_graph.nodes, dict) else enumerate(getattr(operator_graph, 'nodes', [])):
                        try:
                            oid = getattr(_node, 'id', str(_nid))
                            in_shapes = []
                            out_shape = None
                            if hasattr(_node, 'get_input_tensor_shapes'):
                                try:
                                    in_shapes = _node.get_input_tensor_shapes() or []
                                except Exception:
                                    in_shapes = []
                            if hasattr(_node, 'get_output_tensor_shape'):
                                try:
                                    out_shape = _node.get_output_tensor_shape()
                                except Exception:
                                    out_shape = None
                            shapes_by_id[oid] = (in_shapes, out_shape)
                        except Exception:
                            continue
                except Exception:
                    shapes_by_id = {}
                nodes_obj = getattr(cpp_mapped_ir, 'nodes', {})
                try:
                    iter_nodes = nodes_obj.items()
                except Exception:
                    iter_nodes = enumerate(nodes_obj)
                for nid, mnode in iter_nodes:
                    node_id = getattr(getattr(mnode, 'op_node', mnode), 'id', str(nid))
                    op_type = getattr(getattr(mnode, 'op_node', mnode), 'op_type', 'UNKNOWN')
                    call_count = getattr(getattr(mnode, 'op_node', mnode), 'call_count', 1)
                    hw_unit = getattr(mnode, 'hw_unit', getattr(mnode, 'hardware_unit', 'UNKNOWN'))
                    attrs = dict(getattr(mnode, 'attrs', {}))
                    # Propagate aggregated hint flags into attrs based on op_type
                    if op_type in ("ENCODING", "HASH_ENCODE"):
                        attrs.setdefault("hash_index_activity", str(agg_flags['encoding_hash_active']).lower())
                        if agg_flags['encoding_hash_active']:
                            attrs.setdefault("locality_score", "0.75")
                    elif op_type == "FIELD_COMPUTATION":
                        attrs.setdefault("low_bit_observed", str(agg_flags['field_low_bit']).lower())
                        if agg_flags['field_low_bit']:
                            attrs.setdefault("precision_bits", "8")
                    elif op_type in ("SAMPLING", "VOLUME_RENDERING", "BLENDING"):
                        if agg_flags['sampling_active_ratio'] is not None:
                            attrs.setdefault("active_samples_ratio", str(agg_flags['sampling_active_ratio']))
                    # Populate shapes from operator_graph if available
                    in_descs = []
                    out_descs = []
                    try:
                        ins, out = shapes_by_id.get(node_id, ([], None))
                        if ins:
                            for shp in ins:
                                if isinstance(shp, (list, tuple)):
                                    in_descs.append({"shape": list(shp), "dtype": "float32"})
                        if out is not None and isinstance(out, (list, tuple)):
                            out_descs.append({"shape": list(out), "dtype": "float32"})
                    except Exception:
                        pass
                    nodes_json[node_id] = {
                        "op_node": {
                            "id": node_id,
                            "op_type": op_type,
                            "inputs": in_descs,
                            "outputs": out_descs,
                            "call_count": call_count,
                        },
                        "hw_unit": hw_unit,
                        "attrs": attrs,
                    }
                # Start with edges from C++ mapper, then augment zero‑in‑degree FIELD nodes by (B,N) shape
                edges_json = list(getattr(cpp_mapped_ir, 'edges', []))
                try:
                    # Recompute indegree
                    indeg = {nid: 0 for nid in nodes_json}
                    for u, v in edges_json:
                        if v in indeg:
                            indeg[v] += 1
                    def _bn(shape_list):
                        try:
                            if shape_list and isinstance(shape_list[0].get('shape'), (list, tuple)):
                                shp = shape_list[0]['shape']
                                if len(shp) >= 2:
                                    return (int(shp[0]), int(shp[1]))
                        except Exception:
                            return None
                        return None
                    enc_by_bn = {}
                    samp_by_bn = {}
                    # Build shape keys from inputs/outputs stored in nodes_json (if available later stages fill them)
                    # Here inputs/outputs may be empty in C++ path; fall back to attrs hints if present
                    for nid2, info2 in nodes_json.items():
                        t = info2["op_node"].get("op_type")
                        ins = info2["op_node"].get("inputs", [])
                        outs = info2["op_node"].get("outputs", [])
                        key = None
                        if outs and isinstance(outs[0].get('shape'), (list, tuple)):
                            shp = outs[0]['shape']
                            if len(shp) >= 2:
                                key = (int(shp[0]), int(shp[1]))
                        if key is None:
                            key = _bn(ins)
                        if not key:
                            continue
                        if t == 'ENCODING':
                            enc_by_bn.setdefault(key, []).append(nid2)
                        elif t == 'SAMPLING':
                            samp_by_bn.setdefault(key, []).append(nid2)
                    added2 = 0
                    for nid2, info2 in nodes_json.items():
                        if info2["op_node"].get("op_type") != 'FIELD_COMPUTATION' or indeg.get(nid2, 0) > 0:
                            continue
                        bn = _bn(info2["op_node"].get('inputs', []))
                        if not bn:
                            continue
                        src = None
                        if enc_by_bn.get(bn):
                            src = enc_by_bn[bn][0]
                        elif samp_by_bn.get(bn):
                            src = samp_by_bn[bn][0]
                        if src is not None:
                            edges_json.append((src, nid2))
                            added2 += 1
                    if verbose and added2:
                        print(f"   Added {added2} (B,N)-matched ENCODING/SAMPLING->FIELD edges (C++ mapping path)")
                    # Also link zero in-degree BLENDING to nearest FIELD_COMPUTATION by (B,N)
                    indeg = {nid: 0 for nid in nodes_json}
                    for u, v in edges_json:
                        if v in indeg:
                            indeg[v] += 1
                    fc_by_bn = {}
                    for nid2, info2 in nodes_json.items():
                        if info2["op_node"].get("op_type") != 'FIELD_COMPUTATION':
                            continue
                        ins = info2["op_node"].get("inputs", [])
                        bn = _bn(ins)
                        if bn:
                            fc_by_bn.setdefault(bn, []).append(nid2)
                    added3 = 0
                    for nid2, info2 in nodes_json.items():
                        if info2["op_node"].get("op_type") != 'BLENDING' or indeg.get(nid2, 0) > 0:
                            continue
                        ins = info2["op_node"].get("inputs", [])
                        bn = _bn(ins)
                        if not bn:
                            continue
                        cands = fc_by_bn.get(bn)
                        if cands:
                            edges_json.append((cands[0], nid2))
                            added3 += 1
                    if verbose and added3:
                        print(f"   Added {added3} FIELD->BLENDING links by (B,N) (C++ mapping path)")
                except Exception:
                    pass
                out_data = {
                    "mapped_ir": {
                        "nodes": nodes_json,
                        "edges": edges_json,
                    },
                    "accelerator_name": hw_config_py.accelerator_name,
                    "operator_count": len(nodes_json),
                    "hardware_unit_count": len(hw_config_py.units),
                }
                json.dump(out_data, f, indent=2)
            else:
                # Python fallback: map by operator taxonomy to unit type, round-robin per type
                units_by_type = {}
                for u in hw_config_py.units:
                    units_by_type.setdefault(u.type, []).append(u.id)
                rr_index = {t:0 for t in units_by_type}

                def _taxonomy_from_operator(node_obj) -> str:
                    cls = type(node_obj).__name__
                    # Prefer explicit taxonomy on the node if available
                    try:
                        explicit = getattr(node_obj, 'op_type', None)
                        if not explicit:
                            explicit = getattr(node_obj, 'type', None)
                        if isinstance(explicit, str) and explicit:
                            # Normalize explicit type to canonical categories first
                            e = explicit.upper()
                            if e in ("SAMPLING", "ENCODING", "FIELD_COMPUTATION", "VOLUME_RENDERING"):
                                return e
                            if e == "BLENDING":
                                return "VOLUME_RENDERING"
                            # Heuristic normalization using explicit text
                            if any(tok in explicit for tok in ['Sampler', 'Sample', 'Frustum']):
                                return 'SAMPLING'
                            if any(tok in explicit for tok in ['Encoding', 'Encoder', 'Hash', 'RFF', 'Positional', 'Fourier']):
                                return 'ENCODING'
                            if any(tok in explicit for tok in ['Render', 'Renderer', 'Blend', 'Volume', 'RGB', 'Alpha', 'Composite', 'Depth', 'Semantic', 'Uncertainty', 'Normals']):
                                return 'VOLUME_RENDERING'
                            if any(tok in explicit for tok in ['MLP', 'Network', 'Field', 'Computation', 'Density', 'Color']):
                                return 'FIELD_COMPUTATION'
                            # If explicit provided but unrecognized, fall through to class-name heuristics
                            pass
                    except Exception:
                        pass
                    # Sampling
                    if any(term in cls for term in ['Sampler', 'Sample', 'FrustumCulling', 'FrustrumCulling']):
                        return 'SAMPLING'
                    # Encoding
                    if any(term in cls for term in ['Encoding', 'Encoder', 'Hash', 'RFF', 'Positional', 'Fourier']):
                        return 'ENCODING'
                    # Field Computation
                    if any(term in cls for term in ['MLP', 'Network', 'Field', 'Computation', 'Density', 'Color']):
                        return 'FIELD_COMPUTATION'
                    # Blending / Volume rendering
                    if any(term in cls for term in ['Render', 'Renderer', 'Blend', 'Volume', 'RGB', 'Alpha', 'Composite', 'Depth', 'Semantic', 'Uncertainty', 'Normals']):
                        return 'BLENDING'
                    return 'FIELD_COMPUTATION'

                def pick_unit(op_type: str) -> str:
                    t = op_type
                    if t not in units_by_type:
                        if 'SAMPLING' in t:
                            t = 'SAMPLING'
                        elif 'BLENDING' in t or 'VOLUME_RENDERING' in t:
                            t = 'VOLUME_RENDERING'
                        elif 'FIELD' in t:
                            t = 'FIELD_COMPUTATION'
                        elif 'ENCODING' in t:
                            t = 'ENCODING'
                    ids = units_by_type.get(t)
                    if not ids:
                        # fallback: first available type
                        any_ids = next(iter(units_by_type.values()))
                        return any_ids[0]
                    idx = rr_index[t] % len(ids)
                    rr_index[t] += 1
                    return ids[idx]
                nodes_json = {}
                edges_json = list(operator_graph.edges)
                for nid, node in operator_graph.nodes.items():
                    taxonomy_op_type = _taxonomy_from_operator(node)
                    unit_id = pick_unit(taxonomy_op_type)
                    # Build attrs and inject hints
                    attrs = {"mapped_by": "python_fallback"}
                    if taxonomy_op_type in ("ENCODING"):
                        attrs["hash_index_activity"] = str(agg_flags['encoding_hash_active']).lower()
                        if agg_flags['encoding_hash_active']:
                            attrs["locality_score"] = "0.75"
                    elif taxonomy_op_type == "FIELD_COMPUTATION":
                        attrs["low_bit_observed"] = str(agg_flags['field_low_bit']).lower()
                        if agg_flags['field_low_bit']:
                            attrs["precision_bits"] = "8"
                    elif taxonomy_op_type in ("SAMPLING", "BLENDING"):
                        if agg_flags['sampling_active_ratio'] is not None:
                            attrs["active_samples_ratio"] = str(agg_flags['sampling_active_ratio'])
                    nodes_json[nid] = {
                        "op_node": {
                            "id": node.id,
                            "op_type": taxonomy_op_type,
                            "inputs": [td.__dict__ for td in node.inputs],
                            "outputs": [td.__dict__ for td in node.outputs],
                            "call_count": node.call_count,
                        },
                        "hw_unit": unit_id,
                        "attrs": attrs,
                    }
                # Heuristic: add missing ENCODING->FIELD_COMPUTATION edges by matching shapes when FC has no preds
                try:
                    indeg = {nid: 0 for nid in nodes_json}
                    for u, v in edges_json:
                        if v in indeg:
                            indeg[v] += 1
                    # Build map from encoding output shapes to node ids
                    enc_by_shape = {}
                    for nid, info in nodes_json.items():
                        if info["op_node"]["op_type"] in ("ENCODING",):
                            outs = info["op_node"].get("outputs", [])
                            if outs:
                                shp = tuple(outs[0].get("shape", []))
                                enc_by_shape.setdefault(shp, []).append(nid)
                    added = 0
                    for nid, info in nodes_json.items():
                        if info["op_node"]["op_type"] == "FIELD_COMPUTATION" and indeg.get(nid, 0) == 0:
                            ins = info["op_node"].get("inputs", [])
                            if ins:
                                shp_in = tuple(ins[0].get("shape", []))
                                cands = enc_by_shape.get(shp_in, [])
                                if cands:
                                    # pick the first candidate
                                    edges_json.append((cands[0], nid))
                                    added += 1
                    if verbose and added:
                        print(f"   Added {added} shape-matched ENCODING->FIELD_COMPUTATION edges")
                except Exception:
                    pass

                # Additional fallback: connect zero in-degree FIELD_COMPUTATION to ENCODING or SAMPLING by (B,N) match
                try:
                    # Recompute indegree
                    indeg = {nid: 0 for nid in nodes_json}
                    for u, v in edges_json:
                        if v in indeg:
                            indeg[v] += 1
                    def _bn(shape_list):
                        try:
                            if shape_list and isinstance(shape_list[0].get('shape'), (list, tuple)):
                                shp = shape_list[0]['shape']
                                if len(shp) >= 2:
                                    return (int(shp[0]), int(shp[1]))
                        except Exception:
                            return None
                        return None
                    enc_by_bn = {}
                    samp_by_bn = {}
                    for nid, info in nodes_json.items():
                        t = info["op_node"]["op_type"]
                        ins = info["op_node"].get("inputs", [])
                        outs = info["op_node"].get("outputs", [])
                        key = None
                        # Prefer outputs for source producers
                        if outs and isinstance(outs[0].get('shape'), (list, tuple)):
                            shp = outs[0]['shape']
                            if len(shp) >= 2:
                                key = (int(shp[0]), int(shp[1]))
                        if key is None:
                            key = _bn(ins)
                        if not key:
                            continue
                        if t == 'ENCODING':
                            enc_by_bn.setdefault(key, []).append(nid)
                        elif t == 'SAMPLING':
                            samp_by_bn.setdefault(key, []).append(nid)
                    added2 = 0
                    for nid, info in nodes_json.items():
                        if info["op_node"]["op_type"] != 'FIELD_COMPUTATION' or indeg.get(nid, 0) > 0:
                            continue
                        bn = _bn(info["op_node"].get('inputs', []))
                        if not bn:
                            continue
                        src = None
                        if enc_by_bn.get(bn):
                            src = enc_by_bn[bn][0]
                        elif samp_by_bn.get(bn):
                            src = samp_by_bn[bn][0]
                        if src is not None:
                            edges_json.append((src, nid))
                            added2 += 1
                    if verbose and added2:
                        print(f"   Added {added2} (B,N)-matched ENCODING/SAMPLING->FIELD edges")
                except Exception:
                    pass
                out_data = {
                    "mapped_ir": {
                        "nodes": nodes_json,
                        "edges": edges_json,
                    },
                    "accelerator_name": hw_config_py.accelerator_name,
                    "operator_count": len(nodes_json),
                    "hardware_unit_count": len(hw_config_py.units),
                }
                json.dump(out_data, f, indent=2)
         
        if verbose:
            print(f"   ✅ Mapping results saved successfully")
         
            # Show mapping summary
            print(f"\nMapping Summary:")
            type_counts = {}
            if cpp_mapped_ir is not None:
                # Summarize from C++ mapping result
                try:
                    for nid, mnode in getattr(cpp_mapped_ir, 'nodes', {}).items():
                        unit_type = None
                        for u in hw_config_py.units:
                            if u.id == getattr(mnode, 'hw_unit', None):
                                unit_type = u.type
                                break
                        type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
                except Exception:
                    pass
            else:
                # Summarize from Python fallback/operator_graph by unit id chosen
                try:
                    # Load what we just wrote to disk for a robust summary
                    with output_path.open('r') as _rf:
                        _mapped = json.load(_rf)
                    for _nid, ninfo in (_mapped.get('mapped_ir', {}).get('nodes', {}) or {}).items():
                        hw_unit = (ninfo or {}).get('hw_unit')
                        # Map unit id back to unit type
                        unit_type = None
                        for u in hw_config_py.units:
                            if u.id == hw_unit:
                                unit_type = u.type
                                break
                        type_counts[unit_type] = type_counts.get(unit_type, 0) + 1
                except Exception:
                    pass

            for hw_type, count in type_counts.items():
                print(f"   {hw_type}: {count} operators")
        
        print(f"Mapping completed successfully")
        return 0
        
    except Exception as e:
        print(f"Mapping failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
