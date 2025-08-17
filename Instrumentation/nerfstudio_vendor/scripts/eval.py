#!/usr/bin/env python
"""
Vendor copy of Nerfstudio eval entrypoint (simplified header).
Refer to the Nerfstudio repository for the authoritative version.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os
import tyro

# Automatic tracing integration (see Nerfstudio's eval.py for details)
tracing_mod = None

@dataclass
class ComputePSNR:
	load_config: Path
	output_path: Path = Path("output.json")
	render_output_path: Optional[Path] = None
	eval_image_indices: Optional[Tuple[int, ...]] = None
	enable_trace: bool = False
	trace_config_path: Optional[Path] = None

	def main(self) -> None:
		# Placeholder: use the upstream script in Nerfstudio for real evaluation logic
		print("This is a vendor reference. Use Nerfstudio's ns-eval entrypoint.")


def entrypoint():
	"""Entrypoint for use with pyproject scripts."""
	tyro.extras.set_accent_color("bright_yellow")
	tyro.cli(ComputePSNR).main()

if __name__ == "__main__":
	entrypoint() 