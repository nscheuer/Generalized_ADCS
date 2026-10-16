"""Orbit propagation and compatibility imports for moved environment models."""

import importlib as _importlib
import sys as _sys

# Existing callers can keep importing these modules while their implementations
# live under ADCS.environment. No implementation files remain in ADCS.orbits.
density_model = _importlib.import_module("ADCS.environment.atmosphere.density_model")
ephemeris = _importlib.import_module("ADCS.environment.ephemeris.de421")
_sys.modules[__name__ + ".density_model"] = density_model
_sys.modules[__name__ + ".ephemeris"] = ephemeris

from .helpers.orbit_factory import create_random_circular_orbit, create_random_circular_os

__all__ = ["create_random_circular_orbit", "create_random_circular_os"]
