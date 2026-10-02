"""Quantities a warm start needs, read straight off a measurement vector.

:func:`vector_observations` turns the attitude sensors' readings into pairs
of unit directions, one in the body frame and one in the inertial frame,
each with an angular standard deviation, ready for
:mod:`~ADCS.estimators.attitude_determination`. Single-axis sensors
(magnetometers, sun sensors, sun pairs) are combined into one direction per
kind by weighted least squares; vector sensors (Earth horizon, star tracker)
are taken as they are. :func:`angular_rate_from_readings` does the same for
the gyros, and :func:`wheel_momentum_from_readings` collects the reaction
wheel readings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "VectorObservation",
    "angular_rate_from_readings",
    "vector_observations",
    "wheel_momentum_from_readings",
]

_DIRECTION_KINDS = ("nadir", "star")
_SCALAR_KINDS = ("magnetic_field", "sun")
_RANK_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class VectorObservation:
    """One direction seen in the body frame and known in the inertial frame.

    :param kind: What was observed (``"magnetic_field"``, ``"sun"``, ``"nadir"``, ``"star"``).
    :param body: Unit direction in the body frame.
    :param reference: Unit direction in the inertial frame.
    :param sigma: Angular standard deviation of ``body`` in radians, the same in
        every direction perpendicular to it.
    :param sources: Names of the measurement-stack entries that produced it.
    """

    kind: str
    body: np.ndarray
    reference: np.ndarray
    sigma: float
    sources: tuple[str, ...]


def _stack(satellite: Any):
    stack = getattr(satellite, "measurement_stack", None)
    if stack is None:
        raise TypeError("readings can only be interpreted for an EstimatedSatellite, which owns the measurement layout")
    return stack


def _readings(stack: Any, readings: Any) -> np.ndarray:
    values = np.asarray(readings, dtype=float).reshape(-1)
    if values.shape != (stack.raw_size,):
        raise ValueError(f"readings must have shape ({stack.raw_size},), got {values.shape}")
    return values


def _noise_std(sensor: Any, length: int) -> np.ndarray:
    std = np.atleast_1d(np.asarray(sensor.noise.std_noise, dtype=float))
    return np.broadcast_to(std, (length,)).copy()


def _unit_reference(reference: Any, source: str) -> np.ndarray:
    """Validate and normalize one inertial reference direction."""
    try:
        vector = np.asarray(reference, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reference direction for {source} must be a numeric 3-vector") from exc
    if vector.shape != (3,):
        raise ValueError(
            f"reference direction for {source} must have shape (3,), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"reference direction for {source} must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"reference direction for {source} must have a finite, non-zero norm")
    return vector / norm


def _weighted_least_squares(rows: list[tuple[np.ndarray, float, float]]):
    """Solve axis_i . v = value_i for v; return v, its covariance, or None if not solvable.

    Perfect sensors (zero noise) get equal weights and a zero covariance;
    mixed sets give the perfect sensors a weight far above the noisy ones,
    without ever dividing by zero.
    """
    if len(rows) < 3:
        return None
    axes = np.vstack([row[0] for row in rows])
    values = np.array([row[1] for row in rows])
    sigmas = np.array([row[2] for row in rows])
    positive = sigmas[sigmas > 0.0]
    if positive.size == 0:
        weights = np.ones(len(rows))
    else:
        sigmas = np.where(sigmas > 0.0, sigmas, 1.0e-6 * positive.min())
        weights = 1.0 / sigmas**2
    scaled = axes * np.sqrt(weights)[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[-1] <= _RANK_TOLERANCE * singular[0]:
        return None  # the axes do not span three dimensions
    information = axes.T @ (weights[:, None] * axes)
    covariance = np.linalg.inv(information)
    solution = covariance @ (axes.T @ (weights * values))
    if positive.size == 0:
        covariance = np.zeros((3, 3))
    return solution, covariance


def _angular_sigma(vector: np.ndarray, covariance: np.ndarray, floor: float) -> float:
    """Per-axis standard deviation of the direction of ``vector``, in radians."""
    norm = float(np.linalg.norm(vector))
    unit = vector / norm
    tangent = np.eye(3) - np.outer(unit, unit)
    variance = 0.5 * float(np.trace(tangent @ covariance @ tangent)) / norm**2
    return max(float(np.sqrt(max(variance, 0.0))), floor)


def vector_observations(
    satellite: Any,
    readings: Any,
    orbital_state: Any,
    *,
    star_references: Mapping[Any, Any] | None = None,
    lit_threshold_sigmas: float = 3.0,
    sigma_floor: float = 1.0e-6,
) -> list[VectorObservation]:
    """Direction pairs from the attitude sensors' readings.

    Single-axis sensors of one kind (three magnetometers, a set of sun sensors)
    are combined by weighted least squares into one direction per kind; at
    least three non-coplanar axes with finite readings are needed, otherwise
    that kind is skipped. Sun sensors count only when lit (reading above
    ``lit_threshold_sigmas`` noise standard deviations); in eclipse their
    readings are ``NaN`` and no sun direction results. Vector sensors give one
    observation each. A vector star tracker needs its star's inertial
    direction in ``star_references`` (keyed by measurement-stack entry name or
    sensor index) because the reading does not say which star it saw.

    :param satellite: The :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        whose measurement layout the readings follow.
    :param readings: Raw measurement vector, in the measurement stack's order.
    :param orbital_state: Orbital state at the time of the readings (reference directions).
    :param star_references: Inertial unit directions for vector star trackers.
    :param lit_threshold_sigmas: A sun sensor is lit when its reading exceeds this many noise sigmas.
    :param sigma_floor: Smallest angular standard deviation reported, in radians;
        perfect (noise-free) sensors get this value. The default, 1e-6 rad, is
        well below any real sensor and keeps the weights of a mixed set within
        what the attitude solvers handle accurately.
    :return: One :class:`VectorObservation` per observed direction; possibly empty.
    """
    stack = _stack(satellite)
    values = _readings(stack, readings)
    references = dict(star_references or {})
    groups: dict[str, dict[str, Any]] = {}
    observations: list[VectorObservation] = []
    for entry in stack.entries:
        if entry.sensor_index is None:
            continue
        sensor = entry.source
        kind = getattr(sensor, "observation_kind", None)
        if kind is None or kind not in _DIRECTION_KINDS + _SCALAR_KINDS:
            continue
        reading = values[entry.raw_slice]
        if not np.all(np.isfinite(reading)):
            continue
        noise = _noise_std(sensor, reading.size)
        if kind in _DIRECTION_KINDS:
            norm = float(np.linalg.norm(reading))
            if norm == 0.0:
                continue
            reference = references.get(entry.name, references.get(entry.sensor_index))
            if reference is None:
                reference = sensor.reference_direction(orbital_state)
            if reference is None:
                continue
            reference = _unit_reference(reference, entry.name)
            observations.append(VectorObservation(
                kind=kind, body=reading / norm, reference=reference,
                sigma=max(float(np.sqrt(np.mean(noise**2))) / norm, sigma_floor), sources=(entry.name,),
            ))
            continue
        # single-axis sensor: one row of a small least-squares problem
        value = float(reading[0])
        sigma = float(noise[0])
        axis = np.asarray(sensor.axis, dtype=float)
        if kind == "sun":
            efficiency = getattr(sensor, "efficiency", 1.0)
            if isinstance(efficiency, (tuple, list, np.ndarray)):  # sun pair: signed, two efficiencies
                gain = float(efficiency[0]) if value > 0.0 else float(efficiency[1])
            else:  # sun sensor: one-sided, only lit diodes carry direction information
                if value <= max(0.0, lit_threshold_sigmas * sigma):
                    continue
                gain = float(efficiency)
            value, sigma = value / gain, sigma / abs(gain)
        group = groups.setdefault(kind, {"rows": [], "sources": [], "reference": None})
        group["rows"].append((axis, value, sigma))
        group["sources"].append(entry.name)
        if group["reference"] is None:
            group["reference"] = sensor.reference_direction(orbital_state)
    for kind, group in groups.items():
        solved = _weighted_least_squares(group["rows"])
        if solved is None or group["reference"] is None:
            continue
        vector, covariance = solved
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        reference = _unit_reference(group["reference"], f"{kind} sensor group")
        observations.append(VectorObservation(
            kind=kind, body=vector / norm, reference=reference,
            sigma=_angular_sigma(vector, covariance, sigma_floor), sources=tuple(group["sources"]),
        ))
    return observations


def angular_rate_from_readings(satellite: Any, readings: Any):
    """Body angular rate from the gyro readings, assuming zero gyro bias.

    :return: ``(rate, covariance)`` in rad/s and (rad/s)^2, by weighted least
        squares over the gyros with finite readings.
    :raises ValueError: when fewer than three non-coplanar gyro axes have readings.
    """
    stack = _stack(satellite)
    values = _readings(stack, readings)
    rows = []
    for entry in stack.entries:
        if entry.sensor_index is None or getattr(entry.source, "observation_kind", None) != "angular_rate":
            continue
        reading = values[entry.raw_slice]
        if reading.size != 1 or not np.isfinite(reading[0]):
            continue
        rows.append((np.asarray(entry.source.axis, dtype=float), float(reading[0]), float(_noise_std(entry.source, 1)[0])))
    solved = _weighted_least_squares(rows)
    if solved is None:
        raise ValueError(
            "the angular rate is not observable from the gyro readings (three non-coplanar "
            "gyro axes with finite readings are needed); pass the initial rate explicitly"
        )
    rate, covariance = solved
    return rate, 0.5 * (covariance + covariance.T)


def wheel_momentum_from_readings(satellite: Any, readings: Any) -> np.ndarray:
    """The reaction-wheel momentum readings, in the satellite's wheel order."""
    stack = _stack(satellite)
    values = _readings(stack, readings)
    momenta = [float(values[entry.raw_slice][0]) for entry in stack.entries if entry.wheel_index is not None]
    if not np.all(np.isfinite(momenta)):
        raise ValueError("wheel momentum readings must be finite")
    return np.asarray(momenta, dtype=float)
