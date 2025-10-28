#!/usr/bin/env python3
"""
Extract a subgraph cluster from a DOT file and optionally render it to PNG/SVG.

Default behavior: write only the subgraph DOT. Use flags to render figures:

Usage:
  # DOT-only (default)
  python plot_dot_subgraph.py \
    --dot /path/to/execution_dag_grouped.dot \
    --cluster-index 0 \
    --out-prefix /path/to/execution_dag_component0

  # Render PNG (in addition to DOT)
  python plot_dot_subgraph.py ... --png

  # Render SVG (in addition to DOT)
  python plot_dot_subgraph.py ... --svg

  # Render both
  python plot_dot_subgraph.py ... --png --svg

Cluster selector accepts: index (0,1,2), name (coarse|fine|other), or name:index (e.g., coarse:0)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_header(lines: List[str]) -> Tuple[int, List[str]]:
    start_body = None
    for idx, ln in enumerate(lines):
        if "subgraph" in ln:
            start_body = idx
            break
    if start_body is not None:
        return start_body, lines[:start_body]
    edge_idx = 0
    for idx, ln in enumerate(lines):
        if "edge [" in ln:
            edge_idx = idx
            break
    return edge_idx + 1, lines[: edge_idx + 1]


def discover_stage_order(lines: List[str]) -> List[str]:
    present = []
    for name in ("coarse", "fine", "other"):
        key = f"subgraph cluster_{name} {{"
        if any(ln.strip().startswith(key) for ln in lines):
            present.append(name)
    return present


def list_stage_components(lines: List[str], stage: str) -> List[int]:
    idxs: List[int] = []
    prefix = f"subgraph cluster_{stage}_"
    for ln in lines:
        s = ln.strip()
        if s.startswith(prefix) and s.endswith("{"):
            try:
                # Format: subgraph cluster_<stage>_<idx> {
                tail = s[len(prefix):].split()[0]
                idx = int(tail[:-1]) if tail.endswith("{") else int(tail)
            except Exception:
                parts = s[len(prefix):].split(" ", 1)[0]
                try:
                    idx = int(parts.rstrip("{"))
                except Exception:
                    continue
            if idx not in idxs:
                idxs.append(idx)
    return sorted(idxs)


def resolve_cluster_label(lines: List[str], token: str) -> str:
    # Accept forms: number; stage; stage:index
    if ":" in token:
        st, idx = token.split(":", 1)
        st = st.strip()
        idx = idx.strip()
        if idx.isdigit():
            return f"{st}_{int(idx)}"
        raise ValueError(f"Invalid stage component selector: {token}")

    stage_order = discover_stage_order(lines)
    if token.isdigit() and stage_order:
        idx = int(token)
        if 0 <= idx < len(stage_order):
            return stage_order[idx]
        raise ValueError(f"Stage clusters present ({stage_order}); index {idx} is out of range")
    return token


def extract_cluster(lines: List[str], cluster_label: str) -> Tuple[int, int]:
    # Support top-level stage clusters (cluster_coarse) and nested (cluster_coarse_0)
    key = f"subgraph cluster_{cluster_label} {{"
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key):
            start = i
            break
    if start is None:
        raise ValueError(f"Cluster cluster_{cluster_label} not found")
    brace = 0
    end = None
    for i in range(start, len(lines)):
        brace += lines[i].count("{")
        brace -= lines[i].count("}")
        if brace == 0:
            end = i
            break
    if end is None:
        raise ValueError(f"Cluster cluster_{cluster_label} has no closing brace")
    return start, end


def build_cluster_dot(original_lines: List[str], cluster_label: str) -> List[str]:
    _body_start, header = find_header(original_lines)
    cstart, cend = extract_cluster(original_lines, cluster_label)
    out: List[str] = []
    has_digraph = any(ln.strip().startswith("digraph ") for ln in header)
    if not has_digraph:
        out.append("digraph G {")
    out.extend(header)
    out.append("")
    out.extend(original_lines[cstart : cend + 1])
    out.append("")
    out.append("}")
    return out


def try_graphviz_render(dot_path: Path, out_prefix: Path, *, enable_png: bool, enable_svg: bool) -> bool:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return False
    try:
        rendered_any = False
        if enable_png:
            subprocess.run([dot_bin, "-Tpng", "-o", str(out_prefix.with_suffix(".png")), str(dot_path)], check=True)
            rendered_any = True
        if enable_svg:
            subprocess.run([dot_bin, "-Tsvg", "-o", str(out_prefix.with_suffix(".svg")), str(dot_path)], check=True)
            rendered_any = True
        return rendered_any
    except subprocess.CalledProcessError:
        return False


def fallback_networkx_render(dot_path: Path, out_prefix: Path, *, enable_png: bool, enable_svg: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from networkx.drawing.nx_pydot import read_dot

    G = read_dot(str(dot_path))
    pos = nx.spring_layout(G, seed=0, k=0.3)
    plt.figure(figsize=(24, 16), dpi=200)
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=300, width=0.6, font_size=8)
    plt.tight_layout()
    if enable_png:
        plt.savefig(str(out_prefix.with_suffix('.png')), bbox_inches="tight")
    if enable_svg:
        plt.savefig(str(out_prefix.with_suffix('.svg')), bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a specific subgraph cluster from DOT")
    parser.add_argument("--dot", required=True, type=Path, help="Path to grouped DOT file")
    parser.add_argument("--cluster-index", required=True, type=str, help="Cluster selector: index (0,1,2), name (coarse|fine|other), or name:index (e.g., coarse:0)")
    parser.add_argument("--out-prefix", required=True, type=Path, help="Output path prefix (DOT always; PNG/SVG optional)")
    parser.add_argument("--png", action="store_true", help="Render PNG in addition to DOT")
    parser.add_argument("--svg", action="store_true", help="Render SVG in addition to DOT")
    args = parser.parse_args()

    lines = read_lines(args.dot)
    label = resolve_cluster_label(lines, args.cluster_index)
    # If a bare stage label was provided and nested components exist, pick the first component for that stage
    if label in ("coarse", "fine", "other"):
        comps = list_stage_components(lines, label)
        if comps:
            label = f"{label}_{comps[0]}"
    sub_lines = build_cluster_dot(lines, label)

    out_dot = args.out_prefix.with_suffix(".dot")
    write_lines(out_dot, sub_lines)
    print(f"[plot-dot-subgraph] Wrote subgraph DOT: {out_dot}")

    # Only render figures if requested
    if args.png or args.svg:
        if try_graphviz_render(out_dot, args.out_prefix, enable_png=args.png, enable_svg=args.svg):
            if args.png:
                print(f"[plot-dot-subgraph] Rendered with graphviz: {args.out_prefix.with_suffix('.png')}")
            if args.svg:
                print(f"[plot-dot-subgraph] Rendered with graphviz: {args.out_prefix.with_suffix('.svg')}")
        else:
            print("[plot-dot-subgraph] graphviz 'dot' not found or failed; using networkx fallback")
            fallback_networkx_render(out_dot, args.out_prefix, enable_png=args.png, enable_svg=args.svg)
            if args.png:
                print(f"[plot-dot-subgraph] Rendered with networkx: {args.out_prefix.with_suffix('.png')}")
            if args.svg:
                print(f"[plot-dot-subgraph] Rendered with networkx: {args.out_prefix.with_suffix('.svg')}")


if __name__ == "__main__":
    main() 