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
from scipy.linalg import block_diag

from ADCS.estimators.attitude_determination import AttitudeSolution, quest, triad
from ADCS.state import EstimatorState, State, _quaternion_tangent_scale

__all__ = [
    "VectorObservation",
    "WarmStart",
    "angular_rate_from_readings",
    "attitude_from_readings",
    "initial_state_from_readings",
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
            reference = np.asarray(reference, dtype=float)
            observations.append(VectorObservation(
                kind=kind, body=reading / norm, reference=reference / np.linalg.norm(reference),
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
        reference = np.asarray(group["reference"], dtype=float)
        observations.append(VectorObservation(
            kind=kind, body=vector / norm, reference=reference / np.linalg.norm(reference),
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


# ----------------------------------------------------------------------------
# attitude and complete initial state

_METHODS = ("auto", "quest", "triad", "tracker")
_ATTITUDE_SIGMA_FLOOR = 1.0e-6  # rad; the same floor vector_observations uses


@dataclass(frozen=True)
class WarmStart:
    """How to build an estimator's initial state from its first readings.

    Pass an instance in place of the ``state`` of any attitude estimator, or
    to :meth:`~ADCS.estimators.attitude_estimators.AttitudeEstimator.from_readings`.
    The attitude is solved from the direction observations (see
    :func:`attitude_from_readings`), the angular rate is read off the gyros
    assuming zero bias, and the wheel momenta off the wheels. Biases and
    disturbance parameters start at zero unless given. Every ``*_std`` is an
    initial standard deviation, a single number or one per element; the
    attitude, rate and wheel-momentum ones default to what the sensors'
    noise implies, the bias and disturbance ones must be given whenever the
    satellite estimates such parameters, and a value you supply yourself
    (a rate, wheel momenta) needs its standard deviation too.

    :param method: ``"auto"`` (a quaternion star tracker when one has a reading,
        otherwise QUEST), ``"quest"``, ``"triad"`` (the two most precise
        directions) or ``"tracker"``.
    :param rate: ``"gyro"`` or the initial angular rate in rad/s.
    :param wheel_momentum: ``"readings"`` or the initial wheel momenta.
    :param act_bias: Initial actuator biases (default zeros).
    :param sens_bias: Initial sensor biases (default zeros).
    :param dist_param: Initial disturbance parameters (default zeros).
    :param attitude_std: Attitude standard deviation in rad, per body axis;
        default from the observations.
    :param rate_std: Rate standard deviation in rad/s; default from the gyro noise.
    :param wheel_momentum_std: Wheel-momentum standard deviation; default from
        the wheels' momentum-measurement noise.
    :param act_bias_std: Actuator-bias standard deviations.
    :param sens_bias_std: Sensor-bias standard deviations.
    :param dist_param_std: Disturbance-parameter standard deviations.
    :param process_noise: The per-step process covariance the filter adds
        (``int_cov``), in the filter's covariance coordinates; default zeros.
    :param star_references: Inertial directions for vector star trackers, see
        :func:`vector_observations`.
    :param max_attitude_std: Cap, in rad, for an attitude axis the observations
        barely determine (nearly parallel directions); the default, pi, means
        "unknown" and leaves the filter's first corrections to settle it.
    """

    method: str = "auto"
    rate: Any = "gyro"
    wheel_momentum: Any = "readings"
    act_bias: Any = None
    sens_bias: Any = None
    dist_param: Any = None
    attitude_std: Any = None
    rate_std: Any = None
    wheel_momentum_std: Any = None
    act_bias_std: Any = None
    sens_bias_std: Any = None
    dist_param_std: Any = None
    process_noise: Any = None
    star_references: Mapping[Any, Any] | None = None
    max_attitude_std: float = np.pi

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}")
        if isinstance(self.rate, str) and self.rate != "gyro":
            raise ValueError(f"rate must be 'gyro' or a 3-vector, got {self.rate!r}")
        if isinstance(self.wheel_momentum, str) and self.wheel_momentum != "readings":
            raise ValueError(f"wheel_momentum must be 'readings' or one value per wheel, got {self.wheel_momentum!r}")
        if not np.isfinite(self.max_attitude_std) or self.max_attitude_std <= 0.0:
            raise ValueError("max_attitude_std must be positive")


def _tracker_attitude(stack: Any, values: np.ndarray) -> AttitudeSolution | None:
    """The most precise quaternion star tracker reading, as an attitude solution."""
    best = None
    for entry in stack.entries:
        if entry.sensor_index is None or getattr(entry.source, "observation_kind", None) != "attitude":
            continue
        reading = values[entry.raw_slice]
        if reading.size != 4 or not np.all(np.isfinite(reading)):
            continue
        norm = float(np.linalg.norm(reading))
        if norm == 0.0:
            continue
        # Noise of s on each quaternion component moves the attitude by about 2 s rad.
        sigma = max(2.0 * float(np.sqrt(np.mean(_noise_std(entry.source, 4) ** 2))), _ATTITUDE_SIGMA_FLOOR)
        if best is None or sigma < best[0]:
            best = (sigma, reading / norm)
    if best is None:
        return None
    sigma, quaternion = best
    return AttitudeSolution(quaternion=quaternion, covariance=sigma**2 * np.eye(3), loss=0.0, method="tracker")


def attitude_from_readings(
    satellite: Any,
    readings: Any,
    orbital_state: Any,
    *,
    method: str = "auto",
    star_references: Mapping[Any, Any] | None = None,
) -> AttitudeSolution:
    """The attitude at the time of one raw measurement vector.

    ``"quest"`` solves Wahba's problem over every direction observation
    (:func:`vector_observations`), ``"triad"`` uses the two most precise
    ones, ``"tracker"`` takes the reading of a quaternion star tracker, and
    ``"auto"`` takes the tracker when one has a finite reading and QUEST
    otherwise. The solution's covariance is the small-angle attitude-error
    covariance in the body frame, in rad^2.

    :raises ValueError: when the readings do not determine the attitude
        (fewer than two usable directions, for example in eclipse with only
        magnetometers and sun sensors, or parallel directions); the message
        says to pass the initial attitude explicitly.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    stack = _stack(satellite)
    values = _readings(stack, readings)
    if method in ("auto", "tracker"):
        tracker = _tracker_attitude(stack, values)
        if tracker is not None:
            return tracker
        if method == "tracker":
            raise ValueError(
                "no quaternion star tracker has a finite reading; choose another method "
                "or pass the initial attitude explicitly"
            )
    observations = vector_observations(satellite, values, orbital_state, star_references=star_references)
    if len(observations) < 2:
        seen = ", ".join(observation.kind for observation in observations) or "none"
        raise ValueError(
            f"the attitude cannot be determined from these readings: {len(observations)} usable "
            f"direction(s) ({seen}) where two non-parallel ones are needed (in eclipse the sun "
            "sensors give none); pass the initial attitude explicitly"
        )
    try:
        if method == "triad":
            first, second = sorted(observations, key=lambda observation: observation.sigma)[:2]
            return triad(
                first.body, second.body, first.reference, second.reference,
                sigma_1=first.sigma, sigma_2=second.sigma,
            )
        body = np.vstack([observation.body for observation in observations])
        reference = np.vstack([observation.reference for observation in observations])
        sigma = np.array([observation.sigma for observation in observations])
        return quest(body, reference, sigma)
    except ValueError as error:
        raise ValueError(f"{error}; pass the initial attitude explicitly") from None


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must have {size} element(s), got shape {np.shape(value)}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _diagonal(std: Any, size: int, name: str) -> np.ndarray:
    """Diagonal covariance from a scalar standard deviation or one per element."""
    values = np.asarray(std, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(size, values[0])
    if values.shape != (size,):
        raise ValueError(f"{name} must be a single number or {size} value(s), got shape {np.shape(std)}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return np.diag(values**2)


def _parameter_block(value: Any, std: Any, length: int, name: str, description: str):
    """Values and covariance of one augmented block (biases, disturbance parameters)."""
    if length == 0:
        if value is not None and np.asarray(value, dtype=float).size != 0:
            raise ValueError(f"{name} was given, but the satellite estimates no {description}")
        return np.zeros(0), np.zeros((0, 0))
    values = np.zeros(length) if value is None else _vector(value, length, name)
    if std is None:
        raise ValueError(
            f"{name}_std is required: the satellite estimates {length} {description} and the "
            "readings say nothing about their initial uncertainty"
        )
    return values, _diagonal(std, length, f"{name}_std")


def _capped(covariance: np.ndarray, cap: float) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    eigenvalues = np.clip(eigenvalues, 0.0, cap**2)
    return (vectors * eigenvalues) @ vectors.T


def _process_noise(state: EstimatorState, value: Any, coordinates: str, quaternion_mode: str) -> np.ndarray:
    tangent_size, full_size = state.tangent_size, state.full_size
    target = tangent_size if coordinates == "tangent" else full_size
    if value is None:
        return np.zeros((target, target))
    if hasattr(value, "as_matrix"):
        value = value.as_matrix()
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (target, target):
        return matrix
    if coordinates == "full" and matrix.shape == (tangent_size, tangent_size):
        return state.covariance_to_full(matrix, quaternion_mode=quaternion_mode)
    raise ValueError(
        f"process_noise must be a {target}x{target} matrix in the filter's covariance coordinates"
        + (f" (a {tangent_size}x{tangent_size} tangent matrix is also accepted)" if coordinates == "full" else "")
        + f", got shape {matrix.shape}"
    )


def _initial_state(
    satellite: Any,
    readings: Any,
    orbital_state: Any,
    warm_start: WarmStart,
    *,
    covariance_coordinates: str,
    quaternion_mode: str,
) -> tuple[EstimatorState, AttitudeSolution]:
    if covariance_coordinates not in ("tangent", "full"):
        raise ValueError("covariance_coordinates must be 'tangent' or 'full'")
    stack = _stack(satellite)
    values = _readings(stack, readings)
    solution = attitude_from_readings(
        satellite, values, orbital_state, method=warm_start.method, star_references=warm_start.star_references,
    )
    if warm_start.attitude_std is not None:
        attitude_covariance = _diagonal(warm_start.attitude_std, 3, "attitude_std")
    else:
        attitude_covariance = _capped(solution.covariance, warm_start.max_attitude_std)

    if isinstance(warm_start.rate, str):
        rate, rate_covariance = angular_rate_from_readings(satellite, values)
        if warm_start.rate_std is not None:
            rate_covariance = _diagonal(warm_start.rate_std, 3, "rate_std")
    else:
        rate = _vector(warm_start.rate, 3, "rate")
        if warm_start.rate_std is None:
            raise ValueError("rate_std is required when the initial rate is given")
        rate_covariance = _diagonal(warm_start.rate_std, 3, "rate_std")

    wheels = list(satellite.rw_actuators)
    if isinstance(warm_start.wheel_momentum, str):
        momentum = wheel_momentum_from_readings(satellite, values)
        if momentum.size != len(wheels):
            raise ValueError(f"the readings hold {momentum.size} wheel momenta for {len(wheels)} wheels")
        if warm_start.wheel_momentum_std is not None:
            momentum_covariance = _diagonal(warm_start.wheel_momentum_std, len(wheels), "wheel_momentum_std")
        else:
            std = [float(np.sqrt(np.mean(np.atleast_1d(np.asarray(wheel.h_meas_noise.std_noise, dtype=float)) ** 2))) for wheel in wheels]
            momentum_covariance = _diagonal(std if wheels else [], len(wheels), "wheel_momentum_std")
    else:
        momentum = _vector(warm_start.wheel_momentum, len(wheels), "wheel_momentum")
        if warm_start.wheel_momentum_std is None:
            raise ValueError("wheel_momentum_std is required when the initial wheel momenta are given")
        momentum_covariance = _diagonal(warm_start.wheel_momentum_std, len(wheels), "wheel_momentum_std")

    act_bias, act_bias_covariance = _parameter_block(
        warm_start.act_bias, warm_start.act_bias_std, satellite.act_bias_len, "act_bias", "actuator biases",
    )
    sens_bias, sens_bias_covariance = _parameter_block(
        warm_start.sens_bias, warm_start.sens_bias_std, satellite.att_sens_bias_len, "sens_bias", "sensor biases",
    )
    dist_param, dist_param_covariance = _parameter_block(
        warm_start.dist_param, warm_start.dist_param_std, satellite.dist_param_len, "dist_param", "disturbance parameters",
    )

    # The attitude block lives in the filter's chart: a small rotation of
    # theta rad is a chart vector of theta / (2 s), s the chart's scale.
    chart_covariance = attitude_covariance / (2.0 * _quaternion_tangent_scale(quaternion_mode)) ** 2
    tangent_covariance = block_diag(
        rate_covariance, chart_covariance, momentum_covariance,
        act_bias_covariance, sens_bias_covariance, dist_param_covariance,
    )
    fields = dict(w=rate, q=solution.quaternion, h=momentum, act_bias=act_bias, sens_bias=sens_bias, dist_param=dist_param)
    state = EstimatorState(**fields, cov=tangent_covariance)
    process_noise = _process_noise(state, warm_start.process_noise, covariance_coordinates, quaternion_mode)
    if covariance_coordinates == "full":
        covariance = state.covariance_to_full(tangent_covariance, quaternion_mode=quaternion_mode)
    else:
        covariance = tangent_covariance
    return EstimatorState(**fields, cov=covariance, int_cov=process_noise), solution


def initial_state_from_readings(
    satellite: Any,
    readings: Any,
    orbital_state: Any,
    warm_start: WarmStart | None = None,
    *,
    covariance_coordinates: str = "tangent",
    quaternion_mode: str = State.DEFAULT_QUATERNION_MODE,
) -> EstimatorState:
    """A complete initial estimator state from one raw measurement vector.

    The values come from the readings as the :class:`WarmStart` recipe says
    (its defaults: attitude by ``"auto"``, rate from the gyros, wheel momenta
    from the wheels, zero biases and disturbance parameters). The covariance
    is block diagonal: the attitude solution's covariance, the gyro
    least-squares covariance, the wheels' momentum-measurement noise, and
    the given bias and disturbance standard deviations. Its attitude block
    is expressed in the ``quaternion_mode`` chart, and for
    ``covariance_coordinates="full"`` (the EKF) the whole matrix is projected
    onto the four quaternion components.

    :param satellite: The :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        the estimator uses.
    :param readings: Raw measurement vector, in the measurement stack's order.
    :param orbital_state: Orbital state at the time of the readings.
    :param warm_start: The recipe; ``None`` for the defaults.
    :param covariance_coordinates: ``"tangent"`` (MEKF, UKF, SRUKF) or ``"full"`` (EKF).
    :param quaternion_mode: Chart of the attitude block; the estimator's ``quaternion_mode``.
    :raises ValueError: when the readings do not determine the attitude or the
        rate, or the recipe lacks a required standard deviation.
    """
    state, _ = _initial_state(
        satellite, readings, orbital_state, warm_start or WarmStart(),
        covariance_coordinates=covariance_coordinates, quaternion_mode=quaternion_mode,
    )
    return state
