"""Engine-agnostic support for the 3-D visualization layer.

Everything here must stay importable with the core dependencies only, and
must not import a rendering engine.
"""

from .fov import SensorFOV, sensor_fov

__all__ = ["SensorFOV", "sensor_fov"]
