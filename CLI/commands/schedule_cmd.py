#!/usr/bin/env python3
"""
RenderSim CLI - Schedule Command Implementation
"""

import sys
import json
from pathlib import Path
import time

# Add RenderSim to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build" / "Scheduler" / "cpp"))

try:
    import rendersim_cpp as rs
except ImportError:
    print("Error: C++ scheduling module not found. Please run './build_cpp.sh' first.", file=sys.stderr)
    sys.exit(1)

from Scheduler.mapping.hw_config import load_hw_config


def create_hardware_module_configs(hw_config_path: str) -> dict:
    """
    Convert hardware configuration to PPA estimator format.
    
    Args:
        hw_config_path: Path to hardware configuration JSON file
        
    Returns:
        Dict mapping hardware unit names to HardwareModuleConfig objects
    """
    # Load hardware configuration
    hw_config = load_hw_config(hw_config_path)
    
    # Technology node mapping based on accelerator
    tech_mapping = {
        "ICARUS": "tn28rvt9t",  # 28nm
        "NEUREX": "tn7rvt9t",   # 7nm  
        "GSCORE": "tn16rvt9t",  # 16nm
        "CICERO": "tn5rvt9t"    # 5nm
    }
    
    accel_name = (hw_config.accelerator_name or "ICARUS").upper()
    # Prefer technology node from config if available; fallback by accelerator name
    try:
        import json as _json
        with open(hw_config_path, 'r') as _f:
            _cfg = _json.load(_f)
        tech_nm = int((_cfg.get('system_specifications', {}) or {}).get('technology_node_nm', 28))
        tech_node = f"tn{tech_nm}rvt9t"
    except Exception:
        tech_node = tech_mapping.get(accel_name, "tn28rvt9t")
    
    # Create HardwareModuleConfig objects
    hw_module_configs = {}
    
    # Map taxonomy types to validated ICARUS module names and paths
    def _module_name_and_path(unit_type: str):
        t = unit_type.upper()
        if accel_name == "ICARUS":
            if t in ("POSITIONAL_ENCODE", "ENCODING"):
                return ("PosEncodingUnit", "A1_cmod/ICARUS/PosEncoding")
            if t in ("FIELD_COMPUTATION", "MLP", "COMPUTATION"):
                return ("MLPEngine", "A1_cmod/ICARUS/MLP")
            if t in ("VOLUME_RENDERING", "BLENDING", "SAMPLING"):
                return ("VolumeRenderingUnit", "A1_cmod/ICARUS/VolumeRender")
        # Fallback generic
        return (unit_type.title().replace("_", ""), f"Hardware/{accel_name}/{unit_type}")

    for unit in hw_config.units:
        # Derive module name/path for PPA cache lookup
        mod_name, hw_path_rel = _module_name_and_path(unit.type)
        # Clock period: from target freq (MHz)
        clk_period_ns = 1.0  # default 1 GHz
        # Create HardwareModuleConfig aligned with validated analyzer keys
        hw_module_config = rs.HardwareModuleConfig(
            name=mod_name,
            accel_type=accel_name,
            hw_path=hw_path_rel,
            clk_period=clk_period_ns,
            tech=tech_node
        )
        hw_module_configs[unit.id] = hw_module_config

    # Also include SRAM blocks if declared in the hardware config JSON
    try:
        import json as _json
        with open(hw_config_path, 'r') as _f:
            _cfg = _json.load(_f)
        for blk in _cfg.get('sram_blocks', []) or []:
            name = str(blk.get('name', 'UNNAMED'))
            size_val = blk.get('size_kb', 0)
            try:
                size_kb_f = float(size_val)
            except Exception:
                size_kb_f = 0.0
            # Heuristic: if the value is a small fractional (<10), interpret as MB and convert to KB
            effective_kb = int(round(size_kb_f * 1024.0)) if (size_kb_f > 0 and size_kb_f < 10 and (size_kb_f % 1) != 0) else int(round(size_kb_f))
            if effective_kb <= 0:
                continue
            sram_mod = f"SRAM_{name}_{effective_kb}KB"
            hw_module_configs[sram_mod] = rs.HardwareModuleConfig(
                name=sram_mod,
                accel_type=accel_name,
                hw_path="",  # synthetic module handled by analyzer
                clk_period=1.0,
                tech=tech_node
            )
    except Exception:
        pass
    
    return hw_module_configs


def run_schedule_command(args, verbose=False):
    """
    Schedule operators using operator-level and system-level schedulers.
    
    Args:
        args: Argparse namespace with mapped_ir, output
        verbose: Enable verbose output
    """
    try:
        if verbose:
            print(f"Loading mapped IR from {args.mapped_ir}")
        
        # Load mapped IR
        mapped_ir_path = Path(args.mapped_ir)
        if not mapped_ir_path.exists():
            raise FileNotFoundError(f"Mapped IR file not found: {mapped_ir_path}")
        
        t_load0 = time.time()
        with mapped_ir_path.open('r') as f:
            data = json.load(f)
        t_load1 = time.time()
        if verbose:
            print(f"   Mapped IR JSON load time: {t_load1 - t_load0:.2f}s")
        
        # Extract hardware config path for PPA analysis
        hardware_config_path = None
        # Prefer explicit CLI arg if provided
        explicit_hw = getattr(args, 'hardware', None)
        if explicit_hw:
            hardware_config_path = explicit_hw
        else:
            if "mapped_ir" in data:
                # Try to find the original hardware config path
                # For now, we'll infer it from the accelerator name
                accelerator_name = data.get("accelerator_name", "ICARUS")
                hardware_config_path = f"examples/hardware_configs/{accelerator_name.lower()}_config.json"
        
        # ------------------------------------------------------------------
        # NEW: Load mapped IR via native C++ JSON loader (no Python stub)
        # ------------------------------------------------------------------
        if verbose:
            print("   Parsing mapped IR JSON natively in C++ ...")
        t_parse0 = time.time()
        mapped_ir = rs.load_mapped_ir_from_json(str(mapped_ir_path))
        t_parse1 = time.time()
        if verbose:
            print(f"   Mapped IR loaded - {len(mapped_ir.nodes)} operators, {len(mapped_ir.edges)} edges")
            print(f"   Native parse time: {t_parse1 - t_parse0:.2f}s")
        
        if verbose:
            print(f"Running operator-level scheduling...")
        
        # Create optimization library and optimizer for operator-level scheduling
        optimization_library = rs.OptimizationLibrary()
        analytic_optimizer = rs.AnalyticalOperatorOptimizer(optimization_library)
        operator_scheduler = rs.OperatorLevelScheduler(analytic_optimizer)

        # Optional: reuse per-operator scheduling durations by signature to speed up
        reuse_cache_enabled = getattr(args, 'reuse_op_cache', False)
        if reuse_cache_enabled and hasattr(operator_scheduler, 'enable_signature_cache'):
            # Prefer native cache if available
            operator_scheduler.enable_signature_cache(True)
        
        if reuse_cache_enabled and not hasattr(operator_scheduler, 'enable_signature_cache'):
            # Python-side cache shim: run once per unique signature and propagate durations
            if verbose:
                print("   Reusing operator scheduling by signature (--reuse-op-cache)")
            # Build simple signature map from mapped_ir JSON
            try:
                with mapped_ir_path.open('r') as _f:
                    _mapped = json.load(_f)
                nodes_json = (_mapped.get('mapped_ir', {}) or {}).get('nodes', {})
                # signature -> list of node ids
                sig_to_ids = {}
                for nid, ninfo in nodes_json.items():
                    opinfo = ninfo.get('op_node', {})
                    op_type = opinfo.get('op_type', 'UNKNOWN')
                    in_shape = tuple((opinfo.get('inputs', [{}])[0]).get('shape', [1,1]))
                    out_shape = tuple((opinfo.get('outputs', [{}])[0]).get('shape', [1,1]))
                    sig = (op_type, in_shape, out_shape)
                    sig_to_ids.setdefault(sig, []).append(nid)
                # Run normal scheduling once, then rewrite duplicate durations
                t_op0 = time.time()
                operator_scheduled_ir = operator_scheduler.schedule(mapped_ir)
                t_op1 = time.time()
                if verbose:
                    print(f"   Operator-level scheduling time: {t_op1 - t_op0:.2f}s")
                # Build id->duration map from first appearances
                id_to_duration = {}
                try:
                    iter_ops = operator_scheduled_ir.nodes.items()
                except Exception:
                    iter_ops = enumerate(operator_scheduled_ir.nodes)
                for _oid, onode in iter_ops:
                    mapped = getattr(onode, 'mapped_node', None)
                    op_node = getattr(mapped, 'op_node', None) if mapped is not None else None
                    nid = getattr(op_node, 'id', None) if op_node is not None else None
                    if nid is not None:
                        id_to_duration[nid] = getattr(onode, 'duration', None)
                # For each signature group, set all durations to first one's duration if available
                changed = 0
                try:
                    iter_ops = operator_scheduled_ir.nodes.items()
                except Exception:
                    iter_ops = enumerate(operator_scheduled_ir.nodes)
                for _oid, onode in iter_ops:
                    mapped = getattr(onode, 'mapped_node', None)
                    op_node = getattr(mapped, 'op_node', None) if mapped is not None else None
                    nid = getattr(op_node, 'id', None) if op_node is not None else None
                    if nid is None:
                        continue
                    ninfo = nodes_json.get(nid, {})
                    opinfo = ninfo.get('op_node', {})
                    op_type = opinfo.get('op_type', 'UNKNOWN')
                    in_shape = tuple((opinfo.get('inputs', [{}])[0]).get('shape', [1,1]))
                    out_shape = tuple((opinfo.get('outputs', [{}])[0]).get('shape', [1,1]))
                    sig = (op_type, in_shape, out_shape)
                    gids = sig_to_ids.get(sig, [])
                    if gids:
                        base_id = gids[0]
                        base_dur = id_to_duration.get(base_id)
                        if base_dur is not None and getattr(onode, 'duration', None) != base_dur:
                            try:
                                onode.duration = base_dur
                                changed += 1
                            except Exception:
                                pass
                if verbose:
                    print(f"   🔁 Reused durations for {changed} operators")
            except Exception as _e:
                if verbose:
                    print(f"   ⚠️  Reuse cache skipped: {_e}")
        else:
            # Default path
            t_op0 = time.time()
            operator_scheduled_ir = operator_scheduler.schedule(mapped_ir)
            t_op1 = time.time()
            if verbose:
                print(f"   Operator-level scheduling time: {t_op1 - t_op0:.2f}s")
        
        if verbose:
            print(f"   Operator-level scheduling completed")

        # ------------------------------------------------------------------
        # Optimization-aware duration scaling (per-node and per-op-type)
        # Reads optimization_hints.json and applies targeted scaling.
        # ------------------------------------------------------------------
        try:
            accelerator_name = data.get("accelerator_name", "ICARUS")
            from pathlib import Path as _P
            import re as _re
            hints_path = _P("optimization_hints.json")
            hints_obj = None
            if hints_path.exists():
                import json as _json
                with hints_path.open() as _f:
                    hints_obj = _json.load(_f)
            # Default op-type scales (fallback)
            default_scales = {
                "SAMPLING": 0.80,
                "ENCODING": 0.80,
                "FIELD_COMPUTATION": 0.85,
                "BLENDING": 0.95,
            }
            if accelerator_name.upper() == "NEUREX":
                default_scales.update({
                    "SAMPLING": 0.65,
                    "ENCODING": 0.70,
                    "FIELD_COMPUTATION": 0.75,
                })
            # Optional overrides from hints file: top-level keys
            if hints_obj:
                overrides = hints_obj.get(accelerator_name.upper()) or hints_obj.get(accelerator_name) or hints_obj.get("default")
                if isinstance(overrides, dict):
                    for k, v in overrides.items():
                        if isinstance(v, (int, float)) and k in default_scales:
                            default_scales[k] = float(v)
            # Build per-node scales using hint records
            hint_records = (hints_obj or {}).get('hints', {}) if isinstance(hints_obj, dict) else {}
            # Precompute simplified hint keys without shape suffixes: "Name[..." -> "Name"
            def _base_name(name: str) -> str:
                # split at first '[' or '(' following a word and keep prefix
                m = _re.split(r"\[|\(", name, maxsplit=1)
                return m[0] if m else name
            hints_by_exact = hint_records
            hints_by_base = {}
            for k, v in hint_records.items():
                hints_by_base.setdefault(_base_name(k), v)
            # Build map: node_id -> scale
            node_scales = {}
            mapped_obj = data.get('mapped_ir', {})
            nodes_json = mapped_obj.get('nodes', {})
            for nid, ninfo in nodes_json.items():
                opinfo = ninfo.get('op_node', {})
                node_id = opinfo.get('id', nid)
                op_type = opinfo.get('op_type', 'UNKNOWN')
                # Base scale by op type
                s = float(default_scales.get(op_type, 1.0))
                # Look up hint record by exact id or base name
                h = hints_by_exact.get(node_id)
                if h is None:
                    h = hints_by_base.get(_base_name(node_id))
                # Adjust by hint flags
                if isinstance(h, dict):
                    # Active samples ratio regulates SAMPLING
                    r = h.get('active_samples_ratio', None)
                    if r is not None and op_type == 'SAMPLING':
                        try:
                            r = float(r)
                            if r < 0.5:
                                s = min(s, 0.55)
                            elif r < 0.8:
                                s = min(s, 0.65)
                            else:
                                s = min(s, 0.80)
                        except Exception:
                            pass
                    # Hash activity boosts ENCODING scaling
                    if h.get('hash_index_activity', False) and op_type == 'ENCODING':
                        s = min(s, 0.60)
                    # Low-bit observed tightens FIELD_COMPUTATION scaling
                    if h.get('low_bit_observed', False) and op_type == 'FIELD_COMPUTATION':
                        s = min(s, 0.65)
                node_scales[node_id] = s
            # Apply per-node scales; if missing, fall back to op-type default
            try:
                iter_ops = operator_scheduled_ir.nodes.items()
            except Exception:
                iter_ops = enumerate(operator_scheduled_ir.nodes)
            scaled_nodes = 0
            for _oid, onode in iter_ops:
                mapped = getattr(onode, 'mapped_node', None)
                op_node = getattr(mapped, 'op_node', None) if mapped is not None else None
                node_id = getattr(op_node, 'id', None) if op_node is not None else None
                op_type = getattr(op_node, 'op_type', None) if op_node is not None else None
                s = None
                if node_id and node_id in node_scales:
                    s = float(node_scales[node_id])
                elif op_type and op_type in default_scales:
                    s = float(default_scales[op_type])
                if s is not None and 0.0 < s < 1.0:
                    try:
                        onode.duration = int(max(1, round(onode.duration * s)))
                        scaled_nodes += 1
                    except Exception:
                        pass
            if verbose:
                print(f"   Optimization-aware scaling applied to {scaled_nodes} operators (per-node aware)")
        except Exception as _e:
            if verbose:
                print(f"   Optimization scaling skipped: {_e}")
        
        # ------------------------------------------------------------------
        # Memory-constrained duration adjustment using main memory bandwidth
        # Compute per-op bytes from mapped_ir JSON (inputs/outputs shapes) and
        # enforce duration >= bytes / BW (converted to cycles via target freq)
        # ------------------------------------------------------------------
        try:
            mem_bw_gbps = None
            freq_mhz = 1000.0
            if hardware_config_path and Path(hardware_config_path).exists():
                import json as _json
                with open(hardware_config_path, 'r') as _f:
                    _hw = _json.load(_f)
                freq_mhz = float(_hw.get('system_specifications', {}).get('target_frequency_mhz', 1000.0))
                mem_bw_gbps = _hw.get('memory_hierarchy', {}).get('main_memory', {}).get('bandwidth_gbps', None)
                # Optional SRAM IO modeling parameters (system-level)
                sram_io_cfg = (_hw.get('system_specifications', {}) or {}).get('sram_io', {}) or {}
                sram_read_bw_gbps = sram_io_cfg.get('read_bw_gbps', None)
                sram_write_bw_gbps = sram_io_cfg.get('write_bw_gbps', None)
                sram_read_latency_cycles = sram_io_cfg.get('read_latency_cycles', 0)
                sram_write_latency_cycles = sram_io_cfg.get('write_latency_cycles', 0)
                sram_energy = sram_io_cfg.get('energy', {}) or {}
                sram_read_energy_pJ_per_access = sram_energy.get('read_pJ', None)
                sram_write_energy_pJ_per_access = sram_energy.get('write_pJ', None)
                sram_granule_bytes = int(sram_io_cfg.get('granule_bytes', 64))
            # Build bytes map from mapped_ir JSON content
            node_bytes = {}
            node_read_bytes = {}
            node_write_bytes = {}
            mapped_obj = data.get('mapped_ir', {})
            for nid, ninfo in mapped_obj.get('nodes', {}).items():
                def _tensor_elems(t):
                    shp = t.get('shape', [])
                    elems = 1
                    for d in shp:
                        try:
                            elems *= int(d)
                        except Exception:
                            elems *= 1
                    return elems
                bytes_in = 0
                for t in ninfo.get('op_node', {}).get('inputs', []):
                    bytes_in += _tensor_elems(t) * 4  # float32 default
                bytes_out = 0
                for t in ninfo.get('op_node', {}).get('outputs', []):
                    bytes_out += _tensor_elems(t) * 4
                node_bytes[nid] = bytes_in + bytes_out
                node_read_bytes[nid] = bytes_in
                node_write_bytes[nid] = bytes_out
            if mem_bw_gbps and mem_bw_gbps > 0:
                bytes_per_sec = float(mem_bw_gbps) * 1e9
                cycles_per_sec = freq_mhz * 1e6
                # Iterate and enforce memory time
                try:
                    iter_ops = operator_scheduled_ir.nodes.items()
                except Exception:
                    iter_ops = enumerate(operator_scheduled_ir.nodes)
                adjusted = 0
                for _oid, onode in iter_ops:
                    mapped = getattr(onode, 'mapped_node', None)
                    op_node = getattr(mapped, 'op_node', None) if mapped is not None else None
                    op_id = getattr(op_node, 'id', None) if op_node is not None else None
                    total_bytes = node_bytes.get(op_id, 0)
                    if total_bytes > 0:
                        mem_time_s = total_bytes / bytes_per_sec
                        mem_cycles = int(max(1, round(mem_time_s * cycles_per_sec)))
                        if mem_cycles > onode.duration:
                            onode.duration = mem_cycles
                            adjusted += 1
                if verbose:
                    print(f"   🧮 Memory-constrained durations applied to {adjusted} operators")

            # ------------------------------------------------------------------
            # SRAM IO-constrained adjustment (optional): uses system_specifications.sram_io
            # duration >= read_time + write_time + access_latencies (in cycles)
            # Also accumulate SRAM IO energy
            # ------------------------------------------------------------------
            sram_io_energy_pJ_total = 0.0
            try:
                if (sram_read_bw_gbps or sram_write_bw_gbps) and freq_mhz:
                    read_Bps = float(sram_read_bw_gbps) * 1e9 / 8.0 if sram_read_bw_gbps else None
                    write_Bps = float(sram_write_bw_gbps) * 1e9 / 8.0 if sram_write_bw_gbps else None
                    cycles_per_sec = freq_mhz * 1e6
                    try:
                        iter_ops = operator_scheduled_ir.nodes.items()
                    except Exception:
                        iter_ops = enumerate(operator_scheduled_ir.nodes)
                    adjusted_sram = 0
                    for _oid, onode in iter_ops:
                        mapped = getattr(onode, 'mapped_node', None)
                        op_node = getattr(mapped, 'op_node', None) if mapped is not None else None
                        op_id = getattr(op_node, 'id', None) if op_node is not None else None
                        rb = node_read_bytes.get(op_id, 0)
                        wb = node_write_bytes.get(op_id, 0)
                        time_s = 0.0
                        if read_Bps and rb > 0:
                            time_s += rb / read_Bps
                        if write_Bps and wb > 0:
                            time_s += wb / write_Bps
                        # Access latency per granule
                        if sram_granule_bytes and sram_granule_bytes > 0:
                            nreads = (rb + sram_granule_bytes - 1) // sram_granule_bytes
                            nwrites = (wb + sram_granule_bytes - 1) // sram_granule_bytes
                        else:
                            nreads = nwrites = 0
                        time_cycles = int(round(time_s * cycles_per_sec))
                        time_cycles += int(nreads) * int(sram_read_latency_cycles or 0)
                        time_cycles += int(nwrites) * int(sram_write_latency_cycles or 0)
                        if time_cycles > onode.duration:
                            onode.duration = time_cycles
                            adjusted_sram += 1
                        # Energy accumulation if per-access pJ are provided
                        if sram_read_energy_pJ_per_access:
                            sram_io_energy_pJ_total += float(sram_read_energy_pJ_per_access) * nreads
                        if sram_write_energy_pJ_per_access:
                            sram_io_energy_pJ_total += float(sram_write_energy_pJ_per_access) * nwrites
                    if verbose:
                        print(f"   🧮 SRAM IO constraints applied to {adjusted_sram} operators")
            except Exception as _e:
                if verbose:
                    print(f"   SRAM IO modeling skipped: {_e}")
        except Exception as _e:
            if verbose:
                print(f"   Memory-constrained adjustment skipped: {_e}")

        if verbose:
            print(f"Running system-level scheduling...")
        
        # System-level scheduling using DAGS algorithm
        system_scheduler = rs.SystemLevelScheduler()
        t_sys0 = time.time()
        system_schedule = system_scheduler.schedule(operator_scheduled_ir)
        t_sys1 = time.time()
        if verbose:
            print(f"   System-level scheduling time: {t_sys1 - t_sys0:.2f}s")
        
        if verbose:
            print(f"   System-level scheduling completed")
        
        # Get performance metrics
        if verbose:
            print(f"Collecting performance metrics...")
        
        operator_stats = operator_scheduler.get_last_scheduling_stats()
        system_stats = system_scheduler.get_last_scheduling_stats()

        # ------------------------------------------------------------------
        # Latency metadata (μs) from native C++ instrumentation
        # ------------------------------------------------------------------
        op_latency_report = operator_scheduler.get_latency_report()
        sys_latency_report = system_scheduler.get_latency_report()

        operator_sched_us = op_latency_report.operator_total.total_duration_ns / 1e3  # ns -> μs
        system_sched_us = sys_latency_report.system_total.total_duration_ns / 1e3      # ns -> μs

        # No calibration: use actual computed cycles directly
        calibration_factor = 1.0
        new_entries = None
        total_cycles_scaled = system_schedule.total_cycles
        
        # Generate real PPA metrics if hardware config is available (unless disabled)
        ppa_metrics = {}
        if not getattr(args, 'no_ppa', False) and hardware_config_path and Path(hardware_config_path).exists():
            try:
                if verbose:
                    print(f"   Running real PPA analysis with {hardware_config_path}...")
                
                # Create hardware module configurations for PPA estimator
                hw_module_configs = create_hardware_module_configs(hardware_config_path)
                
                # Build DRAM config for this accelerator and initialize PPA estimator with real Ramulator
                # Infer accelerator name from mapped_ir metadata
                accelerator_name = data.get("accelerator_name", "ICARUS")
                dram_cfg = rs.NeuralRenderingDRAMConfigFactory.get_config_for_accelerator(accelerator_name)
                ppa_estimator = rs.PPAEstimator(dram_cfg, "Hardware/")
                ppa_result = ppa_estimator.estimate_system_ppa(system_schedule, hw_module_configs)
                
                # Extract metrics from result
                ppa_metrics = {
                    "total_power_uw": ppa_result.total_power_mw * 1000.0,  # Convert mW -> μW
                    "total_area_um2": ppa_result.total_area_mm2 * 1e6,      # mm^2 -> μm^2
                    "avg_mem_bw_gb_s": ppa_result.average_memory_bandwidth_gb_s,
                    "peak_dram_cycles": getattr(ppa_result, "peak_dram_cycles", 0)
                }
                
                if verbose:
                    print(f"   Real PPA analysis completed")
                    print(f"   Total Power: {ppa_metrics['total_power_uw']:.1f} μW")
                    print(f"   Total Area: {ppa_metrics['total_area_um2']:.1f} μm²")
                    print(f"   Peak Memory: {ppa_metrics['peak_dram_cycles']:.1f} cycles")
                
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  PPA analysis failed, using fallback: {e}")
                # Fallback to estimated metrics
                ppa_metrics = {
                    "total_power_uw": 500000.0,
                    "total_area_um2": 7600000.0,
                    "avg_mem_bw_gb_s": 0.0,
                    "peak_dram_cycles": 0
                }
        elif getattr(args, 'no_ppa', False):
            if verbose:
                print("   Skipping PPA estimation (--no-ppa)")
            ppa_metrics = {
                "total_power_uw": 0.0,
                "total_area_um2": 0.0,
                "avg_mem_bw_gb_s": 0.0,
                "peak_dram_cycles": 0
            }
        else:
            # Use fallback metrics or skip
            ppa_metrics = {
                "total_power_uw": 500000.0,
                "total_area_um2": 7600000.0,
                "avg_mem_bw_gb_s": 0.0,
                "peak_dram_cycles": 0
            }
            if verbose:
                print(f"   ℹ️  Using estimated PPA metrics (hardware config not found)")
 
        if verbose:
            print(f"   Performance analysis completed")
            print(f"   ⏱️  Total execution cycles: {system_schedule.total_cycles}")
            print(f"   🔄 Scheduling efficiency: {system_stats.scheduling_efficiency:.3f}")
            print(f"   ⚖️  Resource balance: {system_stats.resource_balance_factor:.3f}")
 
        # ------------------------------------------------------------------
        # Derived metrics for comprehensive reporting
        # ------------------------------------------------------------------
        # Frequency (MHz) from hardware config
        freq_mhz = 1000.0
        tech_nm = None
        try:
            if hardware_config_path and Path(hardware_config_path).exists():
                import json as _json
                with open(hardware_config_path, 'r') as _f:
                    _hw = _json.load(_f)
                sys_spec = _hw.get('system_specifications', {}) or {}
                freq_mhz = float(sys_spec.get('target_frequency_mhz', 1000.0))
                tech_nm = sys_spec.get('technology_node_nm', None)
        except Exception:
            pass

        # Frame time and FPS
        frame_cycles = total_cycles_scaled if 'total_cycles_scaled' in locals() else system_schedule.total_cycles
        frame_time_s = float(frame_cycles) / (freq_mhz * 1e6) if (freq_mhz and frame_cycles) else 0.0
        fps = (1.0 / frame_time_s) if frame_time_s > 0 else 0.0

        # Energy per frame from power and frame time
        total_power_w = float(ppa_metrics.get('total_power_uw', 0.0)) * 1e-6
        energy_per_frame_j = total_power_w * frame_time_s

        # Compute total IO bytes from mapped_ir JSON
        total_io_bytes = 0
        try:
            mapped_obj = data.get('mapped_ir', {})
            for _nid, ninfo in mapped_obj.get('nodes', {}).items():
                def _tensor_elems(t):
                    shp = t.get('shape', [])
                    elems = 1
                    for d in shp:
                        try:
                            elems *= int(d)
                        except Exception:
                            elems *= 1
                    return elems
                bytes_in = sum(_tensor_elems(t) * 4 for t in ninfo.get('op_node', {}).get('inputs', []))
                bytes_out = sum(_tensor_elems(t) * 4 for t in ninfo.get('op_node', {}).get('outputs', []))
                total_io_bytes += (bytes_in + bytes_out)
        except Exception:
            total_io_bytes = 0

        effective_mem_bw_gb_s = (total_io_bytes / frame_time_s / 1e9) if frame_time_s > 0 else 0.0

        # DRAM info
        dram_info = {
            'avg_mem_bw_gb_s': ppa_metrics.get('avg_mem_bw_gb_s', 0.0),
            'peak_dram_cycles': ppa_metrics.get('peak_dram_cycles', 0),
        }

        # Optional PSNR: try to locate a metrics file in common locations
        psnr = None
        try:
            from glob import glob
            candidates = []
            candidates += glob('output_result/**/metrics.json', recursive=True)
            candidates += glob('visualization_output/**/metrics.json', recursive=True)
            for c in candidates:
                try:
                    import json as _json
                    with open(c, 'r') as _f:
                        m = _json.load(_f)
                    if isinstance(m, dict):
                        val = m.get('psnr') or m.get('PSNR')
                        if val is not None:
                            psnr = float(val)
                            break
                except Exception:
                    pass
        except Exception:
            psnr = None

        # SRAM breakdown (area and power) from sram_blocks
        sram_area_um2 = 0.0
        sram_static_uw = 0.0
        sram_dynamic_uw = 0.0
        try:
            if hardware_config_path and Path(hardware_config_path).exists():
                import json as _json
                with open(hardware_config_path, 'r') as _f:
                    _hw2 = _json.load(_f)
                for blk in (_hw2.get('sram_blocks', []) or []):
                    # Use explicit fields when provided in config
                    a = blk.get('area_um2', None)
                    ps = blk.get('static_power_uw', None)
                    pd = blk.get('dynamic_power_uw', None)
                    if a is not None:
                        try:
                            sram_area_um2 += float(a)
                        except Exception:
                            pass
                    if ps is not None:
                        try:
                            sram_static_uw += float(ps)
                        except Exception:
                            pass
                    if pd is not None:
                        try:
                            sram_dynamic_uw += float(pd)
                        except Exception:
                            pass
        except Exception:
            pass
        logic_area_um2 = max(0.0, float(ppa_metrics.get('total_area_um2', 0.0)) - sram_area_um2)

        # Determine output file
        output_file = args.output if args.output else "schedule.json"
        output_path = Path(output_file)
        
        if verbose:
            print(f"Saving schedule results to {output_path}")

        # Also save the intermediate operator-scheduled IR for detailed analysis
        try:
            op_sched_path = output_path.with_name(
                (output_path.stem.replace("schedule", "operator_scheduled") if "schedule" in output_path.stem else (output_path.stem + "_operator_scheduled")) + 
                output_path.suffix
            )
            # Serialize operator_scheduled_ir
            op_nodes_json = {}
            try:
                iter_ops = operator_scheduled_ir.nodes.items()
            except Exception:
                iter_ops = enumerate(operator_scheduled_ir.nodes)
            for oid, onode in iter_ops:
                mapped = getattr(onode, 'mapped_node', None)
                op_node = getattr(mapped, 'op_node', None) if mapped is not None else None
                op_id = getattr(op_node, 'id', str(oid)) if op_node is not None else str(oid)
                op_type = getattr(op_node, 'op_type', None) if op_node is not None else None
                hw_unit = getattr(mapped, 'hw_unit', None) if mapped is not None else None
                # resources and optimization result may be dict-like; ensure JSON-safe
                resources = getattr(onode, 'resources', {})
                opt_raw = getattr(onode, 'optimization_result', None) if hasattr(onode, 'optimization_result') else None
                opt_res = {}
                try:
                    if isinstance(opt_raw, dict):
                        opt_res = opt_raw
                    elif opt_raw is not None:
                        # Best-effort: wrap string representation
                        opt_res = {"repr": str(opt_raw)}
                except Exception:
                    opt_res = {"repr": "<unserializable>"}
                op_nodes_json[op_id] = {
                    "hw_unit": hw_unit,
                    "op_type": op_type,
                    "start_cycle": getattr(onode, 'start_cycle', 0),
                    "duration": getattr(onode, 'duration', 0),
                    "resources": resources,
                    "optimization_result": opt_res,
                }
            op_sched_data = {
                "operator_scheduled_ir": {
                    "nodes": op_nodes_json,
                    "edges": list(getattr(operator_scheduled_ir, 'edges', [])),
                },
                "metadata": {
                    "operator_sched_us": operator_sched_us,
                }
            }
            with op_sched_path.open('w') as f:
                json.dump(op_sched_data, f, indent=2)
            if verbose:
                print(f"   Operator-scheduled IR saved to {op_sched_path}")
        except Exception as e:
            if verbose:
                print(f"   Failed to save operator-scheduled IR: {e}")
        
        # Save schedule results
        schedule_data = {
            "system_schedule": {
                "total_cycles": total_cycles_scaled,
                "entries": (
                    new_entries if new_entries is not None else [
                        {
                            "op_id": entry.op_id,
                            "hw_unit": entry.hw_unit,
                            "start_cycle": entry.start_cycle,
                            "duration": entry.duration
                        }
                        for entry in system_schedule.entries
                    ]
                )
            },
            "operator_statistics": {
                "total_operators": operator_stats.total_operators,
                "optimized_operators": operator_stats.optimized_operators,
                "total_speedup": operator_stats.total_speedup
            },
            "system_statistics": {
                "scheduling_efficiency": system_stats.scheduling_efficiency,
                "resource_balance_factor": system_stats.resource_balance_factor,
                "ready_queue_peak_size": system_stats.ready_queue_peak_size
            },
            "ppa_metrics": ppa_metrics,
            "derived_metrics": {
                "frame_time_s": frame_time_s,
                "fps": fps,
                "energy_per_frame_j": energy_per_frame_j,
                "total_io_bytes": int(total_io_bytes),
                "effective_mem_bw_gb_s": effective_mem_bw_gb_s,
                "psnr": psnr,
                "dram": dram_info,
                "target_frequency_mhz": freq_mhz,
                "technology_node_nm": tech_nm,
                "total_area_um2": ppa_metrics.get('total_area_um2', 0.0),
                "sram_area_um2": sram_area_um2,
                "logic_area_um2": logic_area_um2,
                "sram_static_power_uw": sram_static_uw,
                "sram_dynamic_power_uw": sram_dynamic_uw,
                "sram_total_power_uw": sram_static_uw + sram_dynamic_uw,
                "sram_io_energy_pJ_total": sram_io_energy_pJ_total,
                "sram_io_energy_j_total": (sram_io_energy_pJ_total * 1e-12)
            },
            "metadata": {
                "operator_sched_us": operator_sched_us,
                "system_sched_us": system_sched_us,
                "calibration_factor": calibration_factor
            }
        }
        
        with output_path.open('w') as f:
            json.dump(schedule_data, f, indent=2)
        
        if verbose:
            print(f"   Schedule results saved successfully")

            # Latency breakdown summary
            try:
                print("\nLatency Breakdown:")
                # Mapping/parse times (inputs to scheduling)
                if 't_load0' in locals() and 't_load1' in locals():
                    print(f"   - mapped_ir.json load: {t_load1 - t_load0:.2f}s")
                if 't_parse0' in locals() and 't_parse1' in locals():
                    print(f"   - C++ IR parse: {t_parse1 - t_parse0:.2f}s")
                # Operator-level scheduling
                if 't_op0' in locals() and 't_op1' in locals():
                    print(f"   - operator-level (wall): {t_op1 - t_op0:.2f}s")
                print(f"   - operator-level (instr): {operator_sched_us/1e6:.3f}s")
                # System-level scheduling
                if 't_sys0' in locals() and 't_sys1' in locals():
                    print(f"   - system-level (wall): {t_sys1 - t_sys0:.2f}s")
                print(f"   - system-level (instr): {system_sched_us/1e6:.3f}s")
            except Exception:
                pass
        
        print(f"Scheduling completed successfully")
        return 0
        
    except Exception as e:
        print(f"Scheduling failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Legacy helper `create_sample_mapped_ir` removed – native C++ loader replaces
# it. (Kept stub for backward compatibility if other modules import it.)
# ---------------------------------------------------------------------------

def create_sample_mapped_ir(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "create_sample_mapped_ir() has been removed – use the C++ JSON loader "
        "via rendersim_cpp.load_mapped_ir_from_json() instead.")
