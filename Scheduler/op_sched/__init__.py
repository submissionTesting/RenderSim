# operator-level scheduler subpackage

from __future__ import annotations

"""OperatorLevelScheduler applies operator-specific optimizations and timing.
Currently a stub.
"""

from dataclasses import dataclass

from RenderSim.Scheduler.IR import MappedIR, OperatorScheduledIR

__all__ = ["OperatorLevelScheduler"]


@dataclass
class OperatorLevelScheduler:
    def run(self, mapped_ir: MappedIR) -> OperatorScheduledIR:
        # TODO: implement optimisation-aware scheduling
        return OperatorScheduledIR() 