# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Train a radiance field with nerfstudio - WITH INSTRUMENTATION SUPPORT.

This version adds tracing capabilities to ns-train for collecting execution DAGs during training.

Additional flags:
    --enable-trace: Enable execution tracing
    --trace-config-path: Path to trace configuration JSON
    --trace-output-path: Where to save the execution DAG (default: outputs/.../execution_dag.pkl)
    --trace-iterations: List of iterations to trace (e.g., [100, 500, 1000])

For real captures, we recommend using the [bright_yellow]nerfacto[/bright_yellow] model.

Nerfstudio allows for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    ns-train -h
    ns-train nerfacto -h nerfstudio-data
    ns-train nerfacto nerfstudio-data -h

In each of these examples, the -h applies to the previous subcommand (ns-train, nerfacto, and nerfstudio-data).

In the first example, we get the help menu for the ns-train script.
In the second example, we get the help menu for the nerfacto model.
In the third example, we get the help menu for the nerfstudio-data dataparser.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    ns-train nerfacto -h nerfstudio-data

need to come directly after the nerfacto subcommand, since these optional arguments only belong to the nerfacto
subcommand:

    ns-train nerfacto {nerfacto optional args} nerfstudio-data
"""

from __future__ import annotations

import random
import socket
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.available_devices import get_available_devices
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process
    
    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    
    # Check if tracing is enabled
    enable_trace = getattr(config, 'enable_trace', False)
    trace_iterations = getattr(config, 'trace_iterations', [])
    trace_config_path = getattr(config, 'trace_config_path', None)
    trace_output_path = getattr(config, 'trace_output_path', None)
    
    if enable_trace and local_rank == 0:
        CONSOLE.log("[yellow]Tracing enabled for training[/yellow]")
        if trace_iterations:
            CONSOLE.log(f"Will trace at iterations: {trace_iterations}")
        
        # Setup tracing if enabled
        try:
            from nerfstudio.instrumentation import tracing
            
            # Configure tracing
            if trace_config_path:
                tracing.setup_tracing(trace_config_path)
            else:
                # Use default config
                default_config = Path(__file__).parent.parent / "instrumentation" / "trace_config.json"
                if default_config.exists():
                    tracing.setup_tracing(str(default_config))
                else:
                    CONSOLE.log("[red]Warning: No trace config found, using default settings[/red]")
            
            # Set output path for trace
            if trace_output_path:
                tracing.set_trace_output_path(trace_output_path)
            
        except ImportError:
            CONSOLE.log("[red]Warning: Tracing module not found. Install instrumentation dependencies.[/red]")
            enable_trace = False
    
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    
    # Add tracing hooks if enabled
    if enable_trace and local_rank == 0:
        original_train_iteration = trainer.train_iteration
        
        def traced_train_iteration(step: int):
            """Wrapper to trace specific training iterations"""
            should_trace = (not trace_iterations) or (step in trace_iterations)
            
            if should_trace:
                # Enable tracing for this iteration
                try:
                    from nerfstudio.instrumentation import tracing
                    tracing.start_iteration_trace(step)
                    CONSOLE.log(f"[cyan]Tracing iteration {step}[/cyan]")
                except:
                    pass
            
            # Run the actual training iteration
            result = original_train_iteration(step)
            
            if should_trace:
                # Save trace for this iteration
                try:
                    from nerfstudio.instrumentation import tracing
                    output_dir = trainer.base_dir / "traces"
                    output_dir.mkdir(exist_ok=True)
                    trace_file = output_dir / f"execution_dag_iter_{step}.pkl"
                    tracing.save_iteration_trace(str(trace_file))
                    CONSOLE.log(f"[green]Saved trace to {trace_file}[/green]")
                except:
                    pass
            
            return result
        
        # Replace the train_iteration method
        trainer.train_iteration = traced_train_iteration
    
    trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta,
    device_type: Literal["cpu", "cuda", "mps"],
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and calls the main function"""
    if device_type == "cuda" or device_type == "mps":
        torch.cuda.set_device(local_rank)

    global_rank = machine_rank * num_devices_per_machine + local_rank
    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        world_size=world_size,
        rank=global_rank,
        init_method=dist_url,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    comms.LOCAL_PROCESS_GROUP = dist.new_group(
        ranks=list(range(machine_rank * num_devices_per_machine, (machine_rank + 1) * num_devices_per_machine))
    )

    return main_func(local_rank, world_size, config, global_rank)


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs
        config (TrainerConfig, optional): config file specifying training regimen
        timeout (timedelta, optional): timeout of the distributed workers
        device_type: type of device to use for training
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one device, no need to spawn distributed workers
        main_func(local_rank=0, world_size=world_size, config=config)
        profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple devices, spawn distributed workers
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto only supported for single machine jobs"
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                assert process is not None
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig, 
         enable_trace: bool = False,
         trace_config_path: Optional[Path] = None,
         trace_output_path: Optional[Path] = None,
         trace_iterations: Optional[List[int]] = None) -> None:
    """Main function.
    
    Args:
        config: Training configuration
        enable_trace: Enable execution tracing
        trace_config_path: Path to trace configuration JSON
        trace_output_path: Where to save the execution DAG
        trace_iterations: List of iterations to trace (empty = trace all)
    """

    # Check if the specified device type is available
    available_device_types = get_available_devices()
    if config.machine.device_type not in available_device_types:
        raise RuntimeError(
            f"Specified device type '{config.machine.device_type}' is not available. "
            f"Available device types: {available_device_types}. "
            "Please specify a valid device type using the CLI option: --machine.device_type [cuda|mps|cpu]"
        )

    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)
    
    # Add tracing configuration to config
    config.enable_trace = enable_trace
    config.trace_config_path = trace_config_path
    config.trace_output_path = trace_output_path
    config.trace_iterations = trace_iterations or []

    config.set_timestamp()

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    
    # Parse CLI arguments with additional tracing options
    from dataclasses import dataclass
    from typing import List, Optional
    
    @dataclass
    class TrainWithTracing:
        """Extended training configuration with tracing support"""
        config: AnnotatedBaseConfigUnion
        enable_trace: bool = False
        trace_config_path: Optional[Path] = None
        trace_output_path: Optional[Path] = None
        trace_iterations: Optional[List[int]] = None
    
    args = tyro.cli(
        TrainWithTracing,
        description=convert_markup_to_ansi(__doc__),
    )
    
    main(
        config=args.config,
        enable_trace=args.enable_trace,
        trace_config_path=args.trace_config_path,
        trace_output_path=args.trace_output_path,
        trace_iterations=args.trace_iterations,
    )


if __name__ == "__main__":
    entrypoint()
