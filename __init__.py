# RenderSim root package

"""RenderSim
A modular simulator for neural rendering accelerators.
This package is organised into:
- Operators: taxonomy-aligned operator abstractions
- Scheduler: mapping / operator-level / system-level schedulers
- DataModels: core data-structures and IR definitions
- Utils: helper utilities shared across packages
"""

__all__ = [
    "Operators",
    "Scheduler",
    "Utils",
] 