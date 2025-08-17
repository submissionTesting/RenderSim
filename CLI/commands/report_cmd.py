#!/usr/bin/env python3
"""
RenderSim CLI - Report Command Implementation
"""

import sys
import json
from pathlib import Path

# Add RenderSim to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build" / "Scheduler" / "cpp"))

try:
    import rendersim_cpp as rs
except ImportError:
    print("Error: C++ module not found. Please run './build_cpp.sh' first.", file=sys.stderr)
    sys.exit(1)


def run_report_command(args, verbose=False):
    """
    Generate PPA analysis reports from schedule data.
    
    Args:
        args: Argparse namespace with schedule, output, format
        verbose: Enable verbose output
    """
    try:
        if verbose:
            print(f"Loading schedule data from {args.schedule}")
        
        # Load schedule data
        schedule_path = Path(args.schedule)
        if not schedule_path.exists():
            raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
        
        with schedule_path.open('r') as f:
            schedule_data = json.load(f)
        
        if verbose:
            print(f"   Loaded schedule data")
            if "system_schedule" in schedule_data:
                total_cycles = schedule_data["system_schedule"].get("total_cycles", 0)
                num_ops = len(schedule_data["system_schedule"].get("entries", []))
                print(f"   Total cycles: {total_cycles}")
                print(f"   Operations: {num_ops}")
        
        # Determine output file
        output_file = args.output if args.output else f"report.{args.format}"
        output_path = Path(output_file)
        
        if verbose:
            print(f"Generating {args.format.upper()} report...")
        
        if args.format == 'html':
            generate_html_report(schedule_data, output_path, verbose)
        elif args.format == 'json':
            generate_json_report(schedule_data, output_path, verbose)
        elif args.format == 'text':
            generate_text_report(schedule_data, output_path, verbose)
        
        if verbose:
            print(f"   Report generated successfully")
        
        print(f"Report generated: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Report generation failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def generate_html_report(schedule_data, output_path, verbose=False):
    """Generate an HTML report with comprehensive analysis"""
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RenderSim PPA Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .schedule-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .schedule-table th, .schedule-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .schedule-table th {{
            background-color: #3498db;
            color: white;
        }}
        .schedule-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RenderSim PPA Analysis Report</h1>
        <p><em>Auto-embedded visuals are included when available.</em></p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{schedule_data.get('system_schedule', {}).get('total_cycles', 0)}</div>
                <div class="metric-label">Total Execution Cycles</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{schedule_data.get('system_statistics', {}).get('scheduling_efficiency', 0):.3f}</div>
                <div class="metric-label">Scheduling Efficiency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{schedule_data.get('system_statistics', {}).get('resource_balance_factor', 0):.3f}</div>
                <div class="metric-label">Resource Balance Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{schedule_data.get('operator_statistics', {}).get('total_speedup', 0):.2f}x</div>
                <div class="metric-label">Total Speedup</div>
            </div>
        </div>
        
        <h2>Operator Statistics</h2>
        <ul>
            <li><strong>Total Operators:</strong> {schedule_data.get('operator_statistics', {}).get('total_operators', 0)}</li>
            <li><strong>Optimized Operators:</strong> {schedule_data.get('operator_statistics', {}).get('optimized_operators', 0)}</li>
            <li><strong>Ready Queue Peak Size:</strong> {schedule_data.get('system_statistics', {}).get('ready_queue_peak_size', 0)}</li>
        </ul>
        
        <h2>PPA Metrics</h2>
        <ul>
            <li><strong>Total Power:</strong> {schedule_data.get('ppa_metrics', {}).get('total_power_uw', 0):.1f} μW</li>
            <li><strong>Total Area:</strong> {schedule_data.get('ppa_metrics', {}).get('total_area_um2', 0):.1f} μm²</li>
            <li><strong>Peak Memory:</strong> {schedule_data.get('ppa_metrics', {}).get('peak_memory_kb', 0):.1f} KB</li>
        </ul>
        
        <h2>Gantt Chart</h2>
        <div>
            __GANTT_IMG__
        </div>

        <h2>Execution Schedule</h2>
        <table class="schedule-table">
            <thead>
                <tr>
                    <th>Operator ID</th>
                    <th>Hardware Unit</th>
                    <th>Start Cycle</th>
                    <th>Duration</th>
                    <th>End Cycle</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add schedule entries
    entries = schedule_data.get('system_schedule', {}).get('entries', [])
    for entry in entries:
        end_cycle = entry['start_cycle'] + entry['duration']
        html_template += f"""
                <tr>
                    <td>{entry['op_id']}</td>
                    <td>{entry['hw_unit']}</td>
                    <td>{entry['start_cycle']}</td>
                    <td>{entry['duration']}</td>
                    <td>{end_cycle}</td>
                </tr>
"""
    
    html_template += f"""
            </tbody>
        </table>
        
        <div class="timestamp">
            Generated by RenderSim v1.0.0 at {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    # Try to embed gantt chart image based on common naming convention
    try:
        sched_stem = Path(output_path).stem.replace('report','').strip('_')
        # heuristic names: nerf_800 or ngp_800; prefer matching visualization_output files
        viz_dir = Path('visualization_output')
        gantt_img = None
        for cand in viz_dir.glob(f"{sched_stem}*_gantt_chart.png"):
            gantt_img = cand
            break
        if gantt_img is None:
            # fallback exact matches for known names
            for cand in viz_dir.glob("*_gantt_chart.png"):
                gantt_img = cand
                break
        if gantt_img and gantt_img.exists():
            img_tag = f"<img src='{gantt_img.as_posix()}' alt='Gantt Chart' style='max-width:100%; border:1px solid #ddd; border-radius:8px;'/>"
            html_template = html_template.replace("__GANTT_IMG__", img_tag)
        else:
            html_template = html_template.replace("__GANTT_IMG__", "<p>No Gantt image available.</p>")
    except Exception:
        html_template = html_template.replace("__GANTT_IMG__", "<p>No Gantt image available.</p>")

    # Now format the rest of the template with f-string interpolation
    html_content = f"{html_template}"

    with output_path.open('w') as f:
        f.write(html_content)


def generate_json_report(schedule_data, output_path, verbose=False):
    """Generate a JSON report with detailed metrics"""
    
    report_data = {
        "report_metadata": {
            "generator": "RenderSim v1.0.0",
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "format": "json"
        },
        "summary": {
            "total_cycles": schedule_data.get('system_schedule', {}).get('total_cycles', 0),
            "scheduling_efficiency": schedule_data.get('system_statistics', {}).get('scheduling_efficiency', 0),
            "resource_balance_factor": schedule_data.get('system_statistics', {}).get('resource_balance_factor', 0),
            "total_speedup": schedule_data.get('operator_statistics', {}).get('total_speedup', 0)
        },
        "detailed_metrics": schedule_data
    }
    
    with output_path.open('w') as f:
        json.dump(report_data, f, indent=2)


def generate_text_report(schedule_data, output_path, verbose=False):
    """Generate a text report with key metrics"""
    
    text_content = f"""
RenderSim PPA Analysis Report
=============================

Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Metrics:
- Total Execution Cycles: {schedule_data.get('system_schedule', {}).get('total_cycles', 0)}
- Scheduling Efficiency: {schedule_data.get('system_statistics', {}).get('scheduling_efficiency', 0):.3f}
- Resource Balance Factor: {schedule_data.get('system_statistics', {}).get('resource_balance_factor', 0):.3f}
- Total Speedup: {schedule_data.get('operator_statistics', {}).get('total_speedup', 0):.2f}x

Operator Statistics:
- Total Operators: {schedule_data.get('operator_statistics', {}).get('total_operators', 0)}
- Optimized Operators: {schedule_data.get('operator_statistics', {}).get('optimized_operators', 0)}
- Ready Queue Peak Size: {schedule_data.get('system_statistics', {}).get('ready_queue_peak_size', 0)}

PPA Metrics:
- Total Power: {schedule_data.get('ppa_metrics', {}).get('total_power_uw', 0):.1f} μW
- Total Area: {schedule_data.get('ppa_metrics', {}).get('total_area_um2', 0):.1f} μm²
- Peak Memory: {schedule_data.get('ppa_metrics', {}).get('peak_memory_kb', 0):.1f} KB

Execution Schedule:
"""
    
    entries = schedule_data.get('system_schedule', {}).get('entries', [])
    text_content += f"{'Op ID':<8} {'HW Unit':<20} {'Start':<8} {'Duration':<8} {'End':<8}\n"
    text_content += "-" * 60 + "\n"
    
    for entry in entries:
        end_cycle = entry['start_cycle'] + entry['duration']
        text_content += f"{entry['op_id']:<8} {entry['hw_unit']:<20} {entry['start_cycle']:<8} {entry['duration']:<8} {end_cycle:<8}\n"
    
    with output_path.open('w') as f:
        f.write(text_content)
