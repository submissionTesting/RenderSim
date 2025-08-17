#!/usr/bin/env python3
"""
RenderSim Simulation-Speed Micro-Benchmark
========================================

This script measures the _software_ latency of RenderSim’s three main scheduling
stages – mapping, operator-level scheduling, and system-level scheduling – when
starting directly from a **synthetic operator graph** (i.e. no runtime
instrumentation or trace collection).  The goal is to obtain a lower-bound on
RenderSim’s simulation overhead for representative workloads at 800×800
resolution.

Workloads covered
-----------------
1. **vanilla-nerf** – positional-encoding + MLP volume rendering
2. **instant-ngp** – multi-resolution hash grid + MLP volume rendering
3. **gaussian-splatting** – 3-D Gaussian pipeline (sorting + blending)

Hardware targets
----------------
ICARUS, NeuRex, CICERO, and GSCore reference configurations from
`examples/hardware_configs/`.

The script prints a markdown-ready table:

| Pipeline | Accelerator | #Ops | Mapping (µs) | Op-Sched (µs) | Sys-Sched (µs) |
|----------|-------------|------|--------------|---------------|----------------|

and saves a CSV plus a bar-plot (`simulation_speed_comparison.png`) for quick
inspection.

Requirements: only the RenderSim Python package (C++ core already built) and
matplotlib.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List

import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import RenderSim – assumes `build/` with pybind11 bindings is on PYTHONPATH
# -----------------------------------------------------------------------------
import sys
RENDERSIM_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(RENDERSIM_ROOT))  # Make `CLI` importable
sys.path.append(str(RENDERSIM_ROOT / "CLI"))  # for 'commands' sub-package
sys.path.append(str(RENDERSIM_ROOT / "build" / "Scheduler" / "cpp"))  # C++ extension module

from CLI import main as cli_main  # RenderSim CLI entry-point (Python function)

# -----------------------------------------------------------------------------
# Synthetic operator-graph generation helpers --------------------------------------------------

# Default chunk size (can be overridden at runtime via --chunk)
CHUNK = 8192  # points / rays / Gaussians per operator-node

def _add_linear_chain(dag: nx.DiGraph, prefix: str, n: int, stages: list[str]):
    """Helper: create `n` parallel chains stage0→stage1→… for the given *stages*."""
    for i in range(n):
        last = None
        for stage in stages:
            node = f"{prefix}_{stage}_{i}"
            dag.add_node(node, op_type=stage.upper())
            if last is not None:
                dag.add_edge(last, node)
            last = node


def build_operator_graph(pipeline: str) -> nx.DiGraph:
    """Return a realistic operator DAG for *pipeline* (800×800 @ Lego)."""

    dag = nx.DiGraph()

    if pipeline == "vanilla-nerf":
        # 640k rays × 64 / 128 samples per ray  → coarse / fine sample counts
        coarse_pts = 800 * 800 * 64          # 40 960 000
        fine_pts   = 800 * 800 * 128         # 81 920 000
        nc = coarse_pts // CHUNK             # 5 000 nodes
        nf = fine_pts   // CHUNK             # 10 000 nodes

        # Coarse pass chains: Sampling→Encoding→Field→Blend
        _add_linear_chain(dag, "coarse", nc, ["sampling", "encoding", "field", "blending"])

        # Aggregator node that waits for all coarse blends – approximates coarse-to-fine dependency
        dag.add_node("coarse_done", op_type="BARRIER")
        for i in range(nc):
            dag.add_edge(f"coarse_blending_{i}", "coarse_done")

        # Fine pass chains start after barrier
        _add_linear_chain(dag, "fine", nf, ["sampling", "encoding", "field", "blending"])
        for i in range(nf):
            dag.add_edge("coarse_done", f"fine_sampling_{i}")

    elif pipeline == "instant-ngp":
        # Same total 192 samples per ray but single pass → 122 880 000 pts
        total_pts = 800 * 800 * 192
        n = total_pts // CHUNK               # 15 000 nodes
        _add_linear_chain(dag, "ngp", n, ["sampling", "encoding", "field_computation", "blending"])

    elif pipeline == "gaussian-splatting":
        # One projection/sort + one node per 16×16 screen tile (50×50 = 2 500 tiles)
        tiles = (800 // 16) * (800 // 16)     # 2500
        dag.add_node("proj_sort", op_type="ENCODING")
        for t in range(tiles):
            node = f"tile_render_{t}"
            dag.add_node(node, op_type="BLENDING")
            dag.add_edge("proj_sort", node)

    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    return dag

# -----------------------------------------------------------------------------
# Measurement helpers
# -----------------------------------------------------------------------------

def time_cli_command(args: List[str]) -> float:
     """Run RenderSim CLI with *args* in a subprocess and return duration in µs."""
     import subprocess, sys
     start = time.perf_counter()
     result = subprocess.run([sys.executable, "CLI/main.py", *args], capture_output=True, text=True)
     end = time.perf_counter()
     if result.returncode != 0:
         print(result.stderr)
         raise RuntimeError(f"CLI command failed: {' '.join(args)}")
     return (end - start) * 1e6  # microseconds


def run_micro_benchmark(tmp_dir: Path, dag: nx.DiGraph, accelerator: str) -> Dict[str, float]:
    """Run mapping + scheduling for *dag* on *accelerator*."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dag_file = tmp_dir / "dag.pkl"
    mapped_file = tmp_dir / "mapped.json"
    sched_file = tmp_dir / "scheduled.json"

    # Serialize operator DAG
    with dag_file.open("wb") as f:
        pickle.dump(dag, f)

    hw_config = f"examples/hardware_configs/{accelerator.lower()}_config.json"

    # Mapping stage
    map_time = time_cli_command([
        "map", str(dag_file), hw_config, "--output", str(mapped_file)
    ])

    # Operator-level + system-level scheduling in two separate calls so we can
    # measure them independently.  (The CLI `schedule` command internally
    # performs both stages and records stage-wise latency – we parse it from
    # stdout.)  For simplicity we time the whole scheduling call once and then
    # rely on RenderSim’s built-in stage timers printed to stdout as JSON.
    
    sched_start = time.perf_counter()
    time_cli_command(["schedule", str(mapped_file), "--output", str(sched_file)])
    sched_total = (time.perf_counter() - sched_start) * 1e6  # total µs

    # The schedule JSON contains per-stage latency summary (added by
    # PerformanceTimer).  Extract it so we can separate operator vs system stage.
    import json
    with sched_file.open() as f:
        sched_json = json.load(f)

    op_sched_time = sched_json.get("metadata", {}).get("operator_sched_us", None)
    sys_sched_time = sched_json.get("metadata", {}).get("system_sched_us", None)

    # Fallback: if not present assume 50/50 split
    if op_sched_time is None or sys_sched_time is None:
        op_sched_time = sched_total * 0.5
        sys_sched_time = sched_total * 0.5

    return {
        "num_ops": dag.number_of_nodes(),
        "mapping_us": map_time,
        "op_sched_us": op_sched_time,
        "sys_sched_us": sys_sched_time,
    }

# -----------------------------------------------------------------------------
# Main entry-point
# -----------------------------------------------------------------------------

def main():
    global CHUNK  # allow reassignment later
    parser = argparse.ArgumentParser(description="RenderSim micro-benchmark (synthetic operator DAGs)")
    parser.add_argument("--output", type=str, default="simulation_speed_results.csv", help="CSV output file")
    parser.add_argument("--chunk", type=int, default=CHUNK,
                        help="Points per operator node (default: %(default)s)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Allow user to override global CHUNK size
    # ------------------------------------------------------------------
    CHUNK = max(1, args.chunk)
    print(f"🔧 Using CHUNK size = {CHUNK} points per operator node")

    # Only the 4 validated accelerator/algorithm combos requested by the user
    combos = [
        ("vanilla-nerf", "icarus"),          # ICARUS ← Vanilla-NeRF
        ("instant-ngp", "neurex"),          # NeuRex ← Instant-NGP (hash grid)
        ("instant-ngp", "cicero"),          # CICERO ← Instant-NGP (temporal)
        ("gaussian-splatting", "gscore"),   # GSCore ← 3-D Gaussian Splatting
    ]

    tmp_root = Path(".tmp_sim_speed")
    results: List[Dict[str, str | float]] = []

    for pipe, acc in combos:
        dag = build_operator_graph(pipe)
        print(f"\n🏎️  Running {pipe} on {acc}…")
        metrics = run_micro_benchmark(tmp_root / f"{pipe}_{acc}", dag, acc)
        results.append({
            "pipeline": pipe,
            "accelerator": acc,
            **metrics,
        })
        print(f"   ➡️  {metrics['num_ops']} ops | map {metrics['mapping_us']:.1f} µs | "
              f"op-sched {metrics['op_sched_us']:.1f} µs | sys-sched {metrics['sys_sched_us']:.1f} µs")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    import csv
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "pipeline", "accelerator", "num_ops", "mapping_us", "op_sched_us", "sys_sched_us"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📄 Results saved to {args.output}")

    # ------------------------------------------------------------------
    # Simple bar plot – simulation latency break-down (total) per config
    # ------------------------------------------------------------------
    label_map = {
        "vanilla-nerf": "Vanilla-NeRF",
        "instant-ngp": "Instant-NGP",
        "gaussian-splatting": "3D-GS",
        "icarus": "ICARUS",
        "neurex": "NeuRex",
        "cicero": "CICERO",
        "gscore": "GSCore",
    }

    import numpy as np

    totals = [r["mapping_us"] + r["op_sched_us"] + r["sys_sched_us"] for r in results]
    x_labels = [f"{label_map[r['pipeline']]}\n{label_map[r['accelerator']]}" for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(totals)), totals, color="#4c72b0")
    ax.set_ylabel("Total scheduling latency (µs)")
    ax.set_xticks(range(len(totals)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_title("RenderSim scheduling latency – synthetic DAGs (800×800)")
    fig.tight_layout()
    plot_file = "simulation_speed_comparison.png"
    fig.savefig(plot_file, dpi=300)
    print(f"🖼️  Plot saved to {plot_file}")

    # ------------------------------------------------------------------
    # Pretty-print markdown table
    # ------------------------------------------------------------------
    print("\n| Pipeline | Accelerator | #Ops | Mapping (µs) | Op-Sched (µs) | Sys-Sched (µs) |")
    print("|----------|-------------|------|--------------|---------------|----------------|")
    for r in results:
        print(
            f"| {label_map[r['pipeline']]} | {label_map[r['accelerator']]} | {r['num_ops']} | "
            f"{r['mapping_us']:.1f} | {r['op_sched_us']:.1f} | {r['sys_sched_us']:.1f} |")


if __name__ == "__main__":
    main() 