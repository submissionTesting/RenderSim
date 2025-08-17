from __future__ import annotations

"""Hardware configuration parsing utilities.
Updated to handle the actual hardware configuration format used in examples/hardware_configs/
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import json

__all__ = ["HWUnit", "HWConfig", "load_hw_config"]


@dataclass
class HWUnit:
    id: str
    type: str
    throughput: float  # ops per second
    memory_kb: int = 0
    latency_cycles: int = 1
    area_um2: float = 0.0
    power_uw: float = 0.0
    extra: Dict = field(default_factory=dict)


@dataclass
class HWConfig:
    units: List[HWUnit]
    accelerator_name: str = "Unknown"
    description: str = ""

    def units_by_type(self) -> Dict[str, List[HWUnit]]:
        d: Dict[str, List[HWUnit]] = {}
        for u in self.units:
            d.setdefault(u.type, []).append(u)
        return d


def load_hw_config(path: str | Path) -> HWConfig:
    """Load hardware configuration from the actual format used in examples/hardware_configs/"""
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    
    # Handle the new format with hardware_modules
    units = []
    if "hardware_modules" in data:
        for module_name, module_data in data["hardware_modules"].items():
            # Extract key information
            module_type = module_data.get("module_type", "GENERIC").upper()
            count = module_data.get("count", 1)
            
            # Extract performance metrics
            performance = module_data.get("performance", {})
            latency_cycles = performance.get("latency_cycles", 1)
            throughput_ops_per_cycle = performance.get("throughput_ops_per_cycle", 1.0)
            max_frequency_mhz = performance.get("max_frequency_mhz", 1000)
            
            # Calculate throughput in ops/second
            throughput = throughput_ops_per_cycle * max_frequency_mhz * 1e6
            
            # Extract resources
            resources = module_data.get("resources", {})
            memory_kb = resources.get("memory_kb", 0)
            area_um2 = resources.get("area_um2", 0.0)
            power_uw = resources.get("power_uw", 0.0)
            
            # Create hardware units (one for each count)
            for i in range(count):
                unit_id = f"{module_name}_{i}" if count > 1 else module_name
                units.append(HWUnit(
                    id=unit_id,
                    type=module_type,
                    throughput=throughput,
                    memory_kb=memory_kb,
                    latency_cycles=latency_cycles,
                    area_um2=area_um2,
                    power_uw=power_uw,
                    extra={
                        "specifications": module_data.get("specifications", {}),
                        "implementation": module_data.get("implementation", ""),
                        "module_name": module_name
                    }
                ))
    
    # Fallback to old format for compatibility
    elif "hw_units" in data:
        units = [HWUnit(**u) for u in data["hw_units"]]
    
    return HWConfig(
        units=units,
        accelerator_name=data.get("accelerator_name", "Unknown"),
        description=data.get("description", "")
    ) 