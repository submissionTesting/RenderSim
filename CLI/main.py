#!/usr/bin/env python3
"""
RenderSim CLI - Main Entry Point
"""

import argparse
import sys
import os
from pathlib import Path

# Import command implementations
try:
    # Prefer absolute imports when CLI is imported as a package
    from CLI.commands.map_cmd import run_map_command
    from CLI.commands.schedule_cmd import run_schedule_command
    from CLI.commands.report_cmd import run_report_command
    from CLI.commands.analyze_cmd import run_analyze_command
except ImportError:
    # Fallback for running as a script: python CLI/main.py
    sys.path.insert(0, os.path.dirname(__file__))
    from commands.map_cmd import run_map_command
    from commands.schedule_cmd import run_schedule_command
    from commands.report_cmd import run_report_command
    from commands.analyze_cmd import run_analyze_command

def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='render_sim',
        description='RenderSim - Neural Rendering Accelerator Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  render_sim analyze execution_dag.pkl results/ --hardware icarus_config.json
  render_sim map execution_dag.pkl hardware_config.json -o mapped.json
  render_sim schedule mapped.json -o schedule.json  
  render_sim report schedule.json -o report.html

For more help on a specific command:
  render_sim <command> --help
        """
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output')
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # Analyze command (unified pipeline)
    analyze_parser = subparsers.add_parser('analyze', help='Complete end-to-end analysis from DAG to report')
    analyze_parser.add_argument('dag_file', help='Path to execution DAG file (.pkl)')
    analyze_parser.add_argument('output_dir', help='Output directory for all results')
    analyze_parser.add_argument('--hardware', help='Hardware configuration file (default: icarus_config.json)')
    analyze_parser.add_argument('--no-visuals', action='store_true', help='Skip copying/embedding visualization images')
    analyze_parser.add_argument('--report-format', choices=['html', 'json', 'text'], default='html', help='Report format to generate')
    analyze_parser.add_argument('--basic-parser', action='store_true', help='Skip full /Operators transform; use enhanced/basic parser')
    analyze_parser.add_argument('--no-ppa', action='store_true', help='Skip PPA estimation during scheduling')
    analyze_parser.add_argument('--reuse-op-cache', action='store_true', help='Reuse cached operator scheduling durations by signature')
    # Optional: plot DOT subgraph into visuals
    analyze_parser.add_argument('--plot-dot-subgraph', action='store_true', help='Also extract and render a DOT subgraph cluster into the output visuals directory')
    analyze_parser.add_argument('--dot', dest='dot_path', default=None, help='Path to grouped DOT file (e.g., execution_dag_grouped.dot)')
    analyze_parser.add_argument('--cluster-index', dest='cluster_index', default=None, help='Cluster selector: index (0,1,2), name (coarse|fine|other), or name:index (e.g., coarse:0)')
    analyze_parser.add_argument('--subgraph-out-prefix', dest='subgraph_out_prefix', default='execution_dag_component', help='Output filename prefix for subgraph images (within visuals/)')
    
    # Map command
    map_parser = subparsers.add_parser('map', help='Map operators to hardware units')
    map_parser.add_argument('execution_dag', help='Path to execution DAG file (.pkl)')
    map_parser.add_argument('hardware_config', help='Path to hardware configuration file (.json)')
    map_parser.add_argument('-o', '--output', help='Output file for mapped IR (default: mapped_ir.json)')
    map_parser.add_argument('--basic-parser', action='store_true', help='Skip full /Operators transform; use enhanced/basic parser')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Schedule operators on hardware units')
    schedule_parser.add_argument('mapped_ir', help='Path to mapped IR file (.json)')
    schedule_parser.add_argument('-o', '--output', help='Output file for execution schedule (default: schedule.json)')
    schedule_parser.add_argument('--hardware', help='Hardware configuration file (.json). If omitted, attempts to infer from mapped IR.')
    schedule_parser.add_argument('--no-ppa', action='store_true', help='Skip PPA estimation to speed up run')
    schedule_parser.add_argument('--reuse-op-cache', action='store_true', help='Reuse cached operator scheduling durations by signature')
    
    # Report command  
    report_parser = subparsers.add_parser('report', help='Generate PPA analysis reports')
    report_parser.add_argument('schedule', help='Path to execution schedule file (.json)')
    report_parser.add_argument('-o', '--output', help='Output file for report (default: report.html)')
    report_parser.add_argument('--format', choices=['html', 'json', 'text'], default='html')
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        if args.command == 'analyze':
            return run_analyze_command(args, verbose=args.verbose)
        elif args.command == 'map':
            return run_map_command(args, verbose=args.verbose)
        elif args.command == 'schedule':
            return run_schedule_command(args, verbose=args.verbose)
        elif args.command == 'report':
            return run_report_command(args, verbose=args.verbose)
        else:
            parser.error(f"Unknown command: {args.command}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
