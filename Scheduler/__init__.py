# Scheduler package root

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "IR",
    "mapping",
    "op_sched",
    "sys_sched",
]

if TYPE_CHECKING:
    from .IR import OperatorGraph  # noqa
    from .mapping import MappingEngine  # noqa
    from .op_sched import OperatorLevelScheduler  # noqa
    from .sys_sched import SystemLevelScheduler  # noqa

def __getattr__(name: str) -> ModuleType:  # pragma: no cover
    if name in __all__:
        return import_module(f"RenderSim.Scheduler.{name}")
    raise AttributeError(name) 