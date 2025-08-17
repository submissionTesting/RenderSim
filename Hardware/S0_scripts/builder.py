#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
builder.py ─ Run HLS / FC / PWR according to linker.json
Flow: clean → build → report → parse results
Path: A1_cmod/<Stage>/<Module[_vX]>/
Each flow only deletes its own build directory:
    HLS  → A2_hls/.../build_hls
    FC   → A4_fusion/.../build_fc
    PWR  → A5_pwr/.../build_pwr
For each make task (clean / build / report), time is measured and written to timing.log
"""

from __future__ import print_function
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Directory Settings ───────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]       # Hardware/
CMOD_DIR     = ROOT / "A1_cmod"
HLS_OUT_DIR  = ROOT / "A2_hls"
FC_OUT_DIR   = ROOT / "A4_fusion"
PWR_OUT_DIR  = ROOT / "A5_pwr"

SCRIPT_DIR   = Path(__file__).parent
LINKER_JSON  = SCRIPT_DIR / "linker.json"
REPORT_SH    = SCRIPT_DIR / "report_module.sh"

# Make target mapping
CLEAN       = {"hls": "hlsclean", "fc": "fclean",  "pwr": "pwrclean"}
BUILD       = {"hls": "hls",      "fc": "fc",      "pwr": "pwr"}
OUT_DIR     = {"hls": HLS_OUT_DIR, "fc": FC_OUT_DIR, "pwr": PWR_OUT_DIR}
BUILD_NAME  = {"hls": "build_hls", "fc": "build_fc", "pwr": "build_pwr"}

# Log files
PPA_LOG     = SCRIPT_DIR / "PPA.log"
TIMING_LOG  = SCRIPT_DIR / "timing.log"
FAILED_LOG  = SCRIPT_DIR / "failed.log"

# ── Utility Functions ───────────────────────────────────────────────
def load_db():
    """Read linker.json"""
    with LINKER_JSON.open(encoding="utf-8") as fp:
        return json.load(fp)

def flatten(accel: str, db: dict):
    """
    Expand module list for a single accelerator.
    Returns {"Stage/Module_vX": "Module_vX", ...}
    """
    out = {}
    entry = db.get(accel)
    if not isinstance(entry, dict):
        return out
    for stage, mods in entry.items():
        if isinstance(mods, list):
            for m in mods:
                out[f"{stage}/{m}"] = m
    return out

def tee(cmd: str, logfile: Path, timeout: int, label: str, step: str):
    """Run shell command and write output to log in real time"""
    logfile.parent.mkdir(exist_ok=True)
    with logfile.open("a", encoding="utf-8") as lg:
        lg.write(f"\n==== {datetime.now()} :: {label} :: {step} ====\nCMD: {cmd}\n\n")
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        out = ""
        try:
            for line in proc.stdout:
                print(line, end="")
                lg.write(line)
                out += line
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            lg.write(f"[TIMEOUT >{timeout}s]\n")
            return 1, out
        lg.write(f"[RET {proc.returncode}]\n")
        return proc.returncode, out

def log_fail(label: str, step: str, msg: str):
    """Record failure message to failed.log"""
    tail = "\n".join(msg.splitlines()[-5:])
    with FAILED_LOG.open("a") as fp:
        fp.write(f"[Error] {label} :: {step}\n{tail}\n\n")

def parse_report(txt: str):
    """Parse area / latency / throughput / II from report_module.sh output"""
    m = {}
    ta = re.search(
        r"TOTAL AREA \(After Assignment\):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        txt
    )
    if ta:
        m["TOTAL AREA"], m["Seq AREA"], m["Comb AREA"] = map(float, ta.groups())
    ca = re.search(r"Cell Area \(netlist\):\s+([\d.]+)", txt)
    if ca:
        m["CellArea"] = float(ca.group(1))
    for ln in txt.splitlines():
        if ln.strip().startswith("Design Total:"):
            nums = [int(x) for x in ln.split() if x.isdigit()]
            if len(nums) >= 3:
                m["Latency"], m["Throughput"], m["II"] = nums[1], nums[2], nums[-1]
            break
    return m

def one_step(flow: str, env: str, log: Path, label: str):
    """Run clean / build / report for a single module and measure time"""

    # Only remove the old build directory for the corresponding flow
    try:
        stage, mod = label.split("/")
        build_dir = OUT_DIR[flow] / stage / mod / BUILD_NAME[flow]
        if build_dir.exists():
            shutil.rmtree(build_dir)
    except Exception:
        pass

    # clean
    t0 = time.time()
    rc, out = tee(f"make {CLEAN[flow]} {env}", log, 120, label, "clean")
    t_clean = time.time() - t0
    with TIMING_LOG.open("a") as tf:
        tf.write(f"{flow} {label} clean {t_clean:.2f}s\n")
    if rc:
        log_fail(label, "clean", out)

    # build
    t0 = time.time()
    rc2, out2 = tee(f"make {BUILD[flow]} {env}", log, 900, label, "build")
    t_build = time.time() - t0
    with TIMING_LOG.open("a") as tf:
        tf.write(f"{flow} {label} build {t_build:.2f}s\n")
    if rc2:
        log_fail(label, "build", out2)

    # report
    t0 = time.time()
    rc3, rpt = tee(f"{REPORT_SH} {label}", log, 120, label, "report")
    t_report = time.time() - t0
    with TIMING_LOG.open("a") as tf:
        tf.write(f"{flow} {label} report {t_report:.2f}s\n")
    if rc3 or not rpt.strip():
        log_fail(label, "report", rpt)
        return None
    return parse_report(rpt)

# ── Main Process ───────────────────────────────────────────────
def run(flow: str):
    ap = argparse.ArgumentParser(description=f"{flow.upper()} flow runner")
    ap.add_argument("target", nargs="?", help="Accelerator name, e.g. CICERO, NEUREX, ICARUS")
    ap.add_argument("--all", action="store_true", help="Run all accelerators")
    ap.add_argument("--module", help="Stage/Module_vX, e.g. Encode/PEU")
    args = ap.parse_args()

    db = load_db()

    # Decide target list
    if args.all:
        accels = list(db.keys())
    elif args.target:
        accels = [args.target]
    elif args.module:
        accels = [p for p in db if args.module in flatten(p, db)]
        if not accels:
            sys.exit(f"Module {args.module} not found, please check linker.json")
        if len(accels) > 1:
            print(f"Multiple accelerators {accels} contain {args.module}, all will be executed")
    else:
        sys.exit("Please specify <accelerator>, --all or --module")

    # log filename
    name = (args.module.replace('/', '_') if args.module
            else (args.target or 'all'))

    # reset logs
    PPA_LOG.write_text("")
    TIMING_LOG.write_text("")
    FAILED_LOG.write_text("")

    build_log = SCRIPT_DIR / f"{name}_{flow}.log"

    with PPA_LOG.open('a') as fres:
        for proj in accels:
            mapping = flatten(proj, db)
            targets = [args.module] if args.module else mapping.keys()

            for sm in targets:
                mod = mapping.get(sm)
                if not mod:
                    continue
                stage = sm.split('/')[0]

                src_dir = CMOD_DIR / stage / mod
                if not src_dir.exists():
                    print(f"[WARN] Path does not exist: {src_dir}, skipping")
                    continue

                inc = (CMOD_DIR / stage / 'include').resolve()
                exist = os.environ.get('INCLUDE_PATH_CAT', '')
                new_inc = f"{exist} {inc}".strip()
                os.environ['INCLUDE_PATH_CAT'] = new_inc

                env = (
                    f'PROJ_PATH="{stage}/{mod}" '
                    'HLS_BUILD_NAME=build_hls '
                    'FC_BUILD_NAME=build_fc '
                    'CLK_PERIOD=1.0 '
                    'TECH_NODE=tn28rvt9t '
                    f'INCLUDE_PATH_CAT="{new_inc}"'
                )
                label = f"{stage}/{mod}"
                print(f"\n========== {label} : {flow.upper()} ==========")

                result = one_step(flow, env, build_log, label)
                if result:
                    fres.write(
                        f"{label} " +   # <--- Write stage/module at head
                        ' '.join(f"{k}:{v}" for k, v in result.items()) + "\n"
                    )

# Default to HLS; for FC / PWR use run_fc.py / run_pwr.py externally
if __name__ == '__main__':
    run('hls')

