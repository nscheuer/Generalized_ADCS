__all__ = ["SensorFOV", "sensor_fov"]

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ADCS.satellite_hardware.sensors import (
    EarthHorizonSensor,
    Sensor,
    StarTracker,
    StarTrackerQuaternion,
    SunPair,
    SunSensor,
)


@dataclass(frozen=True)
class SensorFOV:
    r"""
    Field of view of a directional sensor in one convention.

    Sensor classes store their field of view inconsistently: the star trackers
    take a *full* cone angle, the Earth horizon sensor a *half* cone angle, and
    the coarse Sun sensors none at all. Visualization and other consumers use
    this normalized form instead.

    :param boresight_body: Unit boresight in the body frame, shape ``(3,)``.
    :type boresight_body: numpy.ndarray
    :param fov_half_rad: Half-cone angle about the boresight [rad].
    :type fov_half_rad: float
    :param assumed: ``True`` if the angle is implied by the measurement model
        rather than stored on the sensor.
    :type assumed: bool
    :param bidirectional: ``True`` if the sensor also sees the cone about
        ``-boresight_body``.
    :type bidirectional: bool
    """

    boresight_body: np.ndarray
    fov_half_rad: float
    assumed: bool = False
    bidirectional: bool = False


def sensor_fov(sensor: Sensor) -> Optional[SensorFOV]:
    r"""
    Normalize a sensor's field of view to a half-cone angle in radians.

    - :class:`StarTracker`, :class:`StarTrackerQuaternion`: ``fov`` is a full
      angle, halved here.
    - :class:`EarthHorizonSensor`: ``fov`` is already a half angle.
    - :class:`SunSensor`: cosine response clipped at zero, i.e. a hemisphere
      (``pi/2``) about ``axis``; flagged ``assumed``.
    - :class:`SunPair`: dual-hemisphere response about ``±axis``; flagged
      ``assumed`` and ``bidirectional``.

    Sensors without a pointing direction (gyros, magnetometers, GPS) return
    ``None``.

    :param sensor: Any sensor instance.
    :type sensor: :class:`~ADCS.satellite_hardware.sensors.Sensor`
    :return: The normalized field of view, or ``None``.
    :rtype: SensorFOV | None
    """
    if isinstance(sensor, (StarTracker, StarTrackerQuaternion)):
        return SensorFOV(np.array(sensor.boresight, dtype=float), 0.5 * float(sensor.fov))
    if isinstance(sensor, EarthHorizonSensor):
        return SensorFOV(np.array(sensor.boresight, dtype=float), float(sensor.fov))
    if isinstance(sensor, SunPair):
        return SensorFOV(np.array(sensor.axis, dtype=float), 0.5 * np.pi, assumed=True, bidirectional=True)
    if isinstance(sensor, SunSensor):
        return SensorFOV(np.array(sensor.axis, dtype=float), 0.5 * np.pi, assumed=True)
    return None
