"""Prototype block-graph satellite digital twin.

This package is intentionally isolated from the public ADCS API.  It is a design
experiment for the eventual 3.0 simulation architecture.
"""

from .prototype import (
    AsyncEventEngine,
    FixedStepEngine,
    ModelConfig,
    SimulationResult,
    Timing,
    build_satellite,
    run_cpu_monte_carlo,
)
from .block_engine import (
    AsyncEventEngine as HierarchicalAsyncEngine,
    Block,
    CompositeBlock,
    FixedStepEngine as HierarchicalFixedEngine,
    Model,
    compile_fixed,
)
from .satellite_model import build_model

__all__ = [
    "AsyncEventEngine",
    "FixedStepEngine",
    "ModelConfig",
    "SimulationResult",
    "Timing",
    "build_satellite",
    "run_cpu_monte_carlo",
    "Block",
    "CompositeBlock",
    "Model",
    "compile_fixed",
    "HierarchicalFixedEngine",
    "HierarchicalAsyncEngine",
    "build_model",
]
