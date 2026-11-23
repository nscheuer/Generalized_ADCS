"""Helpers for converting backend outputs into public result objects."""

from .resultsformat import ResultsFormat
from GenADCS.backends.shared.simulationresult import SimulationResult

__all__ = ["ResultsFormat", "SimulationResult"]
