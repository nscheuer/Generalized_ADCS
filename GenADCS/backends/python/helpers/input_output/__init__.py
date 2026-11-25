"""Helpers for converting backend outputs into public result objects."""

from GenADCS.backends.shared.simulationresult import SimulationResult

from .resultsformat import ResultsFormat

__all__ = ["ResultsFormat", "SimulationResult"]
