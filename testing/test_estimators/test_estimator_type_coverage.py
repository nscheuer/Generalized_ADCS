"""Every sensor, actuator, disturbance and augmentation type, across every filter.

The numeric knobs (charts, covariance form, alpha/beta/kappa, dt, initial
error, covariance scale) are swept elsewhere. This file sweeps the *type
catalogue* instead, so that adding a sensor, actuator or disturbance without
estimator support fails here rather than in flight:

  sensors       Gyro, MTM, SunSensor, SunPair, StarTracker,
                StarTrackerQuaternion, EarthHorizonSensor, GPS
  actuators     MTQ, RW
  disturbances  GG, Dipole, Prop, Torque, General (via a subclass), plus the
                geometry-configured Drag and SRP through the flight factory
  augmentation  sensor bias, actuator bias, disturbance parameter

Each cell asserts something behavioural: the estimate improves, a parameter is
recovered, or a specific documented error is raised. Never merely "it ran".
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import (
    EKF, MEKF, UKF, SRUKF, AugmentedEKF, AugmentedMEKF, AugmentedUKF, AugmentedSRUKF,
)
from ADCS.estimators.process_model import propagate_state
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware import disturbances as D
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import AnisotropicNoise, Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite
from ADCS.satellite_hardware.sensors import (
    GPS, MTM, EarthHorizonSensor, Gyro, StarTracker, StarTrackerQuaternion,
    SunPair, SunSensor,
)
from ADCS.state import EstimatorState


EPHEM = Ephemeris()
PLAIN = [EKF, MEKF, UKF, SRUKF]
AUGMENTED = [AugmentedEKF, AugmentedMEKF, AugmentedUKF, AugmentedSRUKF]
ALL_FILTERS = PLAIN + AUGMENTED
FULL_COV = {EKF, AugmentedEKF}

TRUTH_W = np.array([1.0e-3, 1.0e-3, -2.0e-3])
TRUTH_Q = np.array([0.2588, 0.0, 0.9659, 0.0])
J_0 = np.diag([3.4, 2.9, 1.3])
AXES = np.eye(3)


def orbital_state(j2000: float = 0.22) -> Orbital_State:
    return Orbital_State(
        ephem=EPHEM, J2000=j2000,
        R=np.array([5000.0, 0.0, 5000.0]), V=np.array([0.0, -7.5, 0.0]),
        fast=True,
    )


# --------------------------------------------------------------------------- builders

def _bias(n: int, value=None, rate: float = 1.0e-7) -> Bias:
    """A bias block. The random-walk rate must be non-zero for the augmented
    filters to be able to move the estimate at all: a zero-rate bias is frozen."""
    value = np.zeros(n) if value is None else np.asarray(value, float).reshape(n)
    return Bias(bias=value, std_bias=np.full(n, rate))


def _tracker() -> StarTrackerQuaternion:
    """A quaternion star tracker stubbed to read the true attitude.

    The stub is the house pattern and keeps the reference absolute and
    noiseless so a convergence assertion is about the filter, not about star
    availability. The measurement stack calls it positionally and
    ``Sensor.reading`` by keyword, so it accepts both.
    """
    t = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))

    def clean(*args, **kwargs):
        st = kwargs.get("x", kwargs.get("state"))
        if st is None and args:
            st = args[0]
        return np.asarray(st.q, float).copy()

    t.clean_reading = clean
    return t


def sensor_group(kind: str, *, bias_value=None, estimate_bias: bool = False) -> list:
    """One group of attitude sensors of a single type."""
    want_bias = estimate_bias or bias_value is not None

    def per_axis(i, n=1):
        if not want_bias:
            return None
        v = None if bias_value is None else np.atleast_1d(np.asarray(bias_value, float))[i:i + n]
        return _bias(n, v)

    def whole(n):
        if not want_bias:
            return None
        return _bias(n, None if bias_value is None else np.asarray(bias_value, float)[:n])

    if kind == "Gyro":
        return [Gyro(a, noise=Noise(std_noise=5.0e-7), bias=per_axis(i),
                     estimate_bias=estimate_bias) for i, a in enumerate(AXES)]
    if kind == "MTM":
        return [MTM(a, noise=Noise(std_noise=5.0e-8), bias=per_axis(i),
                    estimate_bias=estimate_bias) for i, a in enumerate(AXES)]
    if kind == "SunSensor":
        return [SunSensor(a, efficiency=0.3, noise=Noise(std_noise=1.0e-3),
                          bias=per_axis(i), estimate_bias=estimate_bias)
                for i, a in enumerate(AXES)]
    if kind == "SunPair":
        return [SunPair(a, efficiency=0.3, noise=Noise(std_noise=1.0e-3),
                        bias=per_axis(i), estimate_bias=estimate_bias)
                for i, a in enumerate(AXES)]
    if kind == "StarTracker":
        return [StarTracker(anisotropic_noise=AnisotropicNoise(std_cross=1.0e-4, std_roll=3.0e-4),
                            bias=whole(3), estimate_bias=estimate_bias)]
    if kind == "StarTrackerQuaternion":
        return [_tracker()]
    if kind == "EarthHorizonSensor":
        return [EarthHorizonSensor(noise=Noise(std_noise=np.full(3, 1.0e-3)),
                                   bias=whole(3), estimate_bias=estimate_bias)]
    if kind == "GPS":
        return [GPS(noise=Noise(std_noise=np.full(6, 1.0e-2)))]
    raise AssertionError(f"unknown sensor kind {kind}")


ATTITUDE_SENSORS = ["Gyro", "MTM", "SunSensor", "SunPair", "StarTracker",
                    "StarTrackerQuaternion", "EarthHorizonSensor"]
ALL_SENSORS = ATTITUDE_SENSORS + ["GPS"]


class ConstantGeneral(D.General_Disturbance):
    """The smallest legal General_Disturbance: a fixed body-frame torque."""

    def __init__(self, torque=(1.0e-7, 0.0, 0.0)):
        super().__init__()
        self._torque = np.asarray(torque, float)

    def torque(self, *args, **kwargs):
        return self._torque.copy()

    def torque_qjac(self, *args, **kwargs):
        return np.zeros((4, 3))


def disturbance(kind: str, *, estimate: bool = False):
    if kind == "GG":
        return D.GG_Disturbance()
    if kind == "Dipole":
        return D.Dipole_Disturbance(np.array([0.4, -0.3, 0.2]), estimate_dist=estimate)
    if kind == "Prop":
        return D.Prop_Disturbance(np.array([1.0e-6, 2.0e-6, -1.0e-6]), estimate_dist=estimate)
    if kind == "Torque":
        return D.Torque_Disturbance(np.array([1.0e-6, 0.0, 0.0]), estimate_dist=estimate)
    if kind == "General":
        assert not estimate, "estimating a General_Disturbance is the subclass's job"
        return ConstantGeneral()
    raise AssertionError(f"unknown disturbance kind {kind}")


SIMPLE_DISTURBANCES = ["GG", "Dipole", "Prop", "Torque", "General"]
ESTIMABLE_DISTURBANCES = ["Dipole", "Torque", "Prop"]


def actuator_group(kind: str, *, estimate_bias: bool = False) -> list:
    if kind == "MTQ":
        return [MTQ(a, max_torque=0.05, noise=Noise(std_noise=1.0e-6),
                    bias=_bias(1) if estimate_bias else None,
                    estimate_bias=estimate_bias) for a in AXES]
    if kind == "RW":
        return [RW(axis=AXES[0], max_torque=0.02, J=1.0e-3, h=0.0, h_max=1.0)]
    raise AssertionError(f"unknown actuator kind {kind}")


def make_state(cls, est_sat, *, att_err_deg: float = 4.0, cov_scale: float = 1.0):
    """Initial estimate offset from truth, sized for whatever blocks the satellite has."""
    n_rw = len(est_sat.rw_actuators)
    n_act = est_sat.act_bias_len
    n_sens = est_sat.att_sens_bias_len
    n_dist = est_sat.dist_param_len
    full = cls in FULL_COV
    dim = 3 + (4 if full else 3) + n_rw + n_act + n_sens + n_dist

    blocks = [np.full(3, 1.0e-6 * cov_scale)]
    att = np.full(4, 1.0e-2 * cov_scale) if full else np.full(3, 1.0e-2 * cov_scale)
    if full:
        att[0] = 0.0
    blocks.append(att)
    blocks += [np.full(n_rw, 1.0e-6 * cov_scale), np.full(n_act, 1.0e-4 * cov_scale),
               np.full(n_sens, 1.0e-4 * cov_scale), np.full(n_dist, 1.0e-1 * cov_scale)]
    cov = np.diag(np.concatenate(blocks))

    half = np.deg2rad(att_err_deg) / 2.0
    axis = np.array([0.3, -0.5, np.sqrt(1.0 - 0.09 - 0.25)])
    dq = np.concatenate(([np.cos(half)], np.sin(half) * axis))
    w0, x0 = TRUTH_Q[0], TRUTH_Q[1:]
    w1, x1 = dq[0], dq[1:]
    q0 = np.concatenate(([w0 * w1 - x0 @ x1], w0 * x1 + w1 * x0 + np.cross(x0, x1)))

    kw = {}
    if n_rw:
        kw["h"] = np.zeros(n_rw)
    if n_act:
        kw["act_bias"] = np.zeros(n_act)
    if n_sens:
        kw["sens_bias"] = np.zeros(n_sens)
    if n_dist:
        kw["dist_param"] = np.zeros(n_dist)
    return EstimatorState(w=TRUTH_W.copy(), q=q0, cov=cov,
                          int_cov=np.zeros((dim, dim)), **kw)


def truth_state(sat) -> EstimatorState:
    """Truth sized for whatever blocks ``sat`` declares (augmented blocks zeroed)."""
    kw = {}
    if len(sat.rw_actuators):
        kw["h"] = np.zeros(len(sat.rw_actuators))
    for name, length in (("act_bias", getattr(sat, "act_bias_len", 0)),
                         ("sens_bias", getattr(sat, "att_sens_bias_len", 0)),
                         ("dist_param", getattr(sat, "dist_param_len", 0))):
        if length:
            kw[name] = np.zeros(length)
    return EstimatorState(w=TRUTH_W.copy(), q=TRUTH_Q.copy(), **kw).normalized()


def physical_psd(cls, est_sat) -> np.ndarray:
    """Continuous PSD for the physical state only.

    ``unmodeled_dynamics_psd`` is sized to the physical state; augmented blocks
    take their process noise from each Bias's ``std_bias`` random-walk rate.
    """
    att = 4 if cls in FULL_COV else 3
    psd = np.concatenate([np.full(3, 1.0e-16), np.full(att, 1.0e-8),
                          np.full(len(est_sat.rw_actuators), 1.0e-16)])
    if cls in FULL_COV:
        psd[3] = 0.0
    return psd


def truth_state_for(sat, truth) -> EstimatorState:
    """Re-express a propagated truth state in the block layout ``sat`` expects."""
    base = truth_state(sat)
    base.w = truth.w.copy()
    base.q = truth.q.copy()
    if len(sat.rw_actuators) and hasattr(truth, "h"):
        base.h = np.asarray(truth.h, float).copy()
    return base


def attitude_error_deg(a, b) -> float:
    return float(2.0 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(a.q, b.q)))))))


def run(estimator, truth_sat, est_sat, *, steps: int, dt: float = 1.0, meas_sat=None,
        use_readings: bool = False):
    """Closed loop on noiseless measurements; returns the attitude error history.

    ``use_readings`` draws the measurements from the truth satellite's own
    sensor readings, which is the only way a real sensor bias reaches the
    filter; the model-side ``measurement_stack.predict`` is bias-free.
    """
    meas_sat = est_sat if meas_sat is None else meas_sat
    stack = meas_sat.measurement_stack
    control = np.zeros(est_sat.control_len)
    truth = truth_state(est_sat)
    os_prev = orbital_state()
    errors = [attitude_error_deg(truth, estimator.state)]
    for k in range(steps):
        os_k = orbital_state(0.22 + (k + 1) * dt / 86400.0)
        truth = propagate_state(truth, truth_sat, control, dt, os_prev, os_k,
                                midpoint_orbital_state=os_k)
        z = (meas_sat.noiseless_sensor_readings(truth, os_k) if use_readings
             else stack.predict(truth_state_for(meas_sat, truth), os_k))
        estimator.predict(control, os_prev, os_k, dt=dt, midpoint_orbital_state=os_k)
        estimator.correct(z, os_k)
        errors.append(attitude_error_deg(truth, estimator.state))
        os_prev = os_k
    return np.array(errors)


def assert_healthy(estimator, size: int) -> None:
    est = estimator.state
    cov = est.covariance.as_matrix()
    assert np.all(np.isfinite(est.as_estimator_array())), "state went non-finite"
    assert np.all(np.isfinite(cov)), "covariance went non-finite"
    assert cov.shape == (size, size)
    assert np.allclose(cov, cov.T, atol=1e-10), "covariance lost symmetry"
    assert np.linalg.eigvalsh(cov).min() > -1.0e-9, "covariance lost positive semidefiniteness"
    assert np.isclose(np.linalg.norm(est.q), 1.0), "quaternion lost normalization"


def one_cycle(cls, est_sat, *, control=None, att_err_deg: float = 4.0):
    """One predict + correct on the estimator's own model; returns the estimator."""
    state = make_state(cls, est_sat, att_err_deg=att_err_deg)
    estimator = cls(est_sat, state, dt=1.0)
    os0 = orbital_state()
    u = np.zeros(est_sat.control_len) if control is None else control
    estimator.predict(u, os0, os0, midpoint_orbital_state=os0)
    estimator.correct(est_sat.measurement_stack.predict(truth_state(est_sat), os0), os0)
    assert_healthy(estimator, state.covariance.dimension)
    return estimator


# =========================================================================== sensors

@pytest.mark.parametrize("cls", ALL_FILTERS)
@pytest.mark.parametrize("kind", ALL_SENSORS)
def test_every_sensor_type_is_accepted_by_every_filter(cls, kind):
    """Each sensor type must survive a predict+correct in every filter.

    GPS carries no attitude information, so it must be ignored cleanly rather
    than contributing a bogus residual.
    """
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group(kind))
    estimator = one_cycle(cls, est_sat)
    z = est_sat.measurement_stack.predict(truth_state(est_sat), orbital_state())
    if kind == "GPS":
        assert np.asarray(z).size == 0, "GPS should contribute nothing to the attitude stack"
        assert estimator.diagnostics["innovation"].size == 0
    else:
        assert np.asarray(z).size > 0


@pytest.mark.parametrize("cls", ALL_FILTERS)
@pytest.mark.parametrize("kind", ["StarTrackerQuaternion", "Gyro", "MTM", "SunPair"])
def test_absolute_reference_sensors_reduce_attitude_error(cls, kind):
    """With an absolute attitude reference available the error must shrink."""
    def sensors():
        return sensor_group(kind) + ([] if kind == "StarTrackerQuaternion" else [_tracker()])
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensors())
    truth_sat = Satellite(J_0=J_0, sensors=sensors())
    estimator = cls(est_sat, make_state(cls, est_sat, att_err_deg=6.0), dt=1.0)
    errors = run(estimator, truth_sat, est_sat, steps=12)
    assert errors[-1] < 0.5 * errors[0], f"{kind}: {errors[0]:.3f} -> {errors[-1]:.3f} deg"


@pytest.mark.parametrize("cls", ALL_FILTERS)
@pytest.mark.parametrize("pair", [("Gyro", "StarTrackerQuaternion"), ("MTM", "SunPair"),
                                  ("Gyro", "MTM"), ("SunPair", "EarthHorizonSensor")])
def test_sensor_combinations(cls, pair):
    """Mixed sensor stacks must compose without dimension or masking errors."""
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group(pair[0]) + sensor_group(pair[1]))
    estimator = one_cycle(cls, est_sat)
    assert estimator.diagnostics["innovation"].size > 0


# ========================================================================= actuators

@pytest.mark.parametrize("cls", ALL_FILTERS)
@pytest.mark.parametrize("kind", ["MTQ", "RW"])
def test_every_actuator_type_is_accepted_by_every_filter(cls, kind):
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                                 actuators=actuator_group(kind))
    state = make_state(cls, est_sat)
    assert state.covariance.dimension == (3 + (4 if cls in FULL_COV else 3)
                                          + len(est_sat.rw_actuators))
    estimator = cls(est_sat, state, dt=1.0)
    before = estimator.state
    os0 = orbital_state()
    estimator.predict(np.full(est_sat.control_len, 1.0e-3), os0, os0, midpoint_orbital_state=os0)
    assert not np.allclose(before.w, estimator.state.w), "control had no effect on the estimate"
    estimator.correct(est_sat.measurement_stack.predict(truth_state(est_sat), os0), os0)
    assert_healthy(estimator, state.covariance.dimension)


@pytest.mark.parametrize("cls", ALL_FILTERS)
def test_both_actuator_types_together(cls):
    """MTQ + RW: the wheel adds a momentum state and the stack must stay consistent."""
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                                 actuators=actuator_group("MTQ") + actuator_group("RW"))
    assert len(est_sat.rw_actuators) == 1
    one_cycle(cls, est_sat, control=np.full(est_sat.control_len, 1.0e-3))


# ====================================================================== disturbances

@pytest.mark.parametrize("cls", ALL_FILTERS)
@pytest.mark.parametrize("kind", SIMPLE_DISTURBANCES)
def test_every_disturbance_type_is_accepted_by_every_filter(cls, kind):
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                                 disturbances=[disturbance(kind)])
    truth_sat = Satellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                          disturbances=[disturbance(kind)])
    estimator = cls(est_sat, make_state(cls, est_sat, att_err_deg=6.0), dt=1.0)
    errors = run(estimator, truth_sat, est_sat, steps=10)
    assert np.all(np.isfinite(errors))
    assert errors[-1] < errors[0], f"{kind}: error grew {errors[0]:.3f} -> {errors[-1]:.3f} deg"


@pytest.mark.parametrize("cls", ALL_FILTERS)
def test_geometry_disturbances_from_the_flight_factory(cls):
    """Drag and SRP need a GeometryConfig, so exercise them via a real satellite."""
    from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
    est_sat = create_beavercube2_cubesat(estimated=True, include_biases=False)
    kinds = {type(d).__name__ for d in est_sat.disturbances}
    assert {"Drag_Disturbance", "SRP_Disturbance", "GG_Disturbance"} <= kinds
    one_cycle(cls, est_sat)


def test_general_disturbance_base_is_abstract_but_subclasses_work():
    with pytest.raises(TypeError, match="abstract"):
        D.General_Disturbance()
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                                 disturbances=[ConstantGeneral()])
    one_cycle(MEKF, est_sat)


# ======================================================================= augmentation

@pytest.mark.parametrize("cls", AUGMENTED)
@pytest.mark.parametrize("kind", ["Gyro", "MTM"])
def test_augmented_filters_recover_a_sensor_bias(cls, kind):
    """The estimated bias must move toward the true bias, not away from it."""
    true_bias = np.array([6.0e-4, -4.0e-4, 5.0e-4])
    truth_sat = EstimatedSatellite(
        J_0=J_0, sensors=sensor_group(kind, bias_value=true_bias) + [_tracker()])
    est_sat = EstimatedSatellite(
        J_0=J_0, sensors=sensor_group(kind, estimate_bias=True) + [_tracker()])
    assert est_sat.att_sens_bias_len == 3
    state = make_state(cls, est_sat, att_err_deg=3.0)
    estimator = cls(est_sat, state, dt=1.0, unmodeled_dynamics_psd=physical_psd(cls, est_sat))
    run(estimator, truth_sat, est_sat, steps=150, meas_sat=truth_sat, use_readings=True)
    final = np.asarray(estimator.state.sens_bias, float)
    assert np.linalg.norm(final - true_bias) < np.linalg.norm(true_bias), (
        f"bias estimate {final} no closer to truth {true_bias} than zero was")


@pytest.mark.parametrize("cls", AUGMENTED)
def test_augmented_filters_accept_an_actuator_bias_block(cls):
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                                 actuators=actuator_group("MTQ", estimate_bias=True))
    assert est_sat.act_bias_len == 3
    one_cycle(cls, est_sat, control=np.full(est_sat.control_len, 1.0e-3))


@pytest.mark.parametrize("cls", AUGMENTED)
@pytest.mark.parametrize("kind", ESTIMABLE_DISTURBANCES)
def test_augmented_filters_accept_every_estimable_disturbance(cls, kind):
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro") + [_tracker()],
                                 disturbances=[disturbance(kind, estimate=True)])
    assert est_sat.dist_param_len == 3, f"{kind} registered {est_sat.dist_param_len} parameters"
    one_cycle(cls, est_sat)


@pytest.mark.parametrize("cls", PLAIN)
def test_plain_filters_reject_an_augmented_state(cls):
    """Only the Augmented subclasses may carry estimated parameter blocks."""
    est_sat = EstimatedSatellite(J_0=J_0, sensors=sensor_group("Gyro", estimate_bias=True) + [_tracker()])
    with pytest.raises(NotImplementedError):
        cls(est_sat, make_state(cls, est_sat), dt=1.0)


@pytest.mark.parametrize("estimate_bias", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_all_four_bias_quadrants_are_legal(has_bias, estimate_bias):
    """bias present/absent crossed with estimated/not must all be constructible."""
    value = np.array([6.0e-4, -4.0e-4, 5.0e-4]) if has_bias else None
    est_sat = EstimatedSatellite(
        J_0=J_0, sensors=sensor_group("Gyro", bias_value=value, estimate_bias=estimate_bias) + [_tracker()])
    assert est_sat.att_sens_bias_len == (3 if estimate_bias else 0)
    one_cycle(AugmentedMEKF if estimate_bias else MEKF, est_sat)


def test_quaternion_star_tracker_rejects_additive_bias_estimation():
    """A four-coefficient quaternion bias is not an additive vector; reject it loudly.

    The check lives in the measurement stack, so it fires when the stack is
    first built rather than at satellite assembly: the satellite happily
    registers a 4-wide sensor-bias block first. Either place is acceptable for
    this contract as long as nothing downstream ever runs.
    """
    with pytest.raises(ValueError, match="cannot be estimated"):
        satellite = EstimatedSatellite(J_0=J_0, sensors=[
            StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)),
                                  bias=_bias(4), estimate_bias=True)])
        satellite.measurement_stack


# ======================================================================== combinations

def _full_flight_sensors():
    return sensor_group("MTM") + sensor_group("Gyro") + sensor_group("SunPair") + [_tracker()]


def _full_flight_actuators():
    return actuator_group("MTQ") + actuator_group("RW")


def _full_flight_disturbances():
    return [disturbance("GG"), disturbance("Dipole"), disturbance("Torque")]


MEKF_INTERACTION = pytest.mark.xfail(
    strict=True,
    reason="MEKF loses most of its correction only in the full combination "
           "(MTM+gyro+sun pair+tracker, MTQ+RW, three disturbances): 6.0 -> 5.2 deg "
           "in 15 steps where the EKF reaches 1.3 and the UKF 0.001; every element "
           "passes alone. Cause not yet identified; see the estimator audit.",
)


@pytest.mark.parametrize("cls", [
    EKF, pytest.param(MEKF, marks=MEKF_INTERACTION), UKF, SRUKF,
    AugmentedEKF, pytest.param(AugmentedMEKF, marks=MEKF_INTERACTION), AugmentedUKF, AugmentedSRUKF,
])
def test_full_flight_configuration(cls):
    """A realistic bus: 3 sensor types, 2 actuator types, 3 disturbance types."""
    est_sat = EstimatedSatellite(J_0=J_0, sensors=_full_flight_sensors(),
                                 actuators=_full_flight_actuators(),
                                 disturbances=_full_flight_disturbances())
    truth_sat = Satellite(J_0=J_0, sensors=_full_flight_sensors(),
                          actuators=_full_flight_actuators(),
                          disturbances=_full_flight_disturbances())
    estimator = cls(est_sat, make_state(cls, est_sat, att_err_deg=6.0), dt=1.0)
    errors = run(estimator, truth_sat, est_sat, steps=15)
    assert errors[-1] < 0.6 * errors[0], f"{errors[0]:.3f} -> {errors[-1]:.3f} deg"


@pytest.mark.parametrize("cls", AUGMENTED)
def test_sensor_bias_and_disturbance_parameter_together(cls):
    """Two augmented block types at once must not collide in the covariance layout."""
    true_bias = np.array([6.0e-4, -4.0e-4, 5.0e-4])
    est_sat = EstimatedSatellite(
        J_0=J_0, sensors=sensor_group("Gyro", estimate_bias=True) + [_tracker()],
        disturbances=[disturbance("Dipole", estimate=True)])
    truth_sat = EstimatedSatellite(
        J_0=J_0, sensors=sensor_group("Gyro", bias_value=true_bias) + [_tracker()],
        disturbances=[disturbance("Dipole")])
    assert est_sat.att_sens_bias_len == 3 and est_sat.dist_param_len == 3
    state = make_state(cls, est_sat, att_err_deg=3.0)
    estimator = cls(est_sat, state, dt=1.0, unmodeled_dynamics_psd=physical_psd(cls, est_sat))
    assert state.covariance.dimension == (3 + (4 if cls in FULL_COV else 3) + 6)
    run(estimator, truth_sat, est_sat, steps=20, meas_sat=truth_sat, use_readings=True)
    assert_healthy(estimator, state.covariance.dimension)
    coordinates = estimator.covariance_coordinates
    sens = estimator.state.slice("sensor_bias", coordinates=coordinates)
    dist = estimator.state.slice("disturbance_parameter", coordinates=coordinates)
    assert sens.start != dist.start
    cov = estimator.state.covariance.as_matrix()
    assert np.isfinite(cov[sens, sens]).all() and np.isfinite(cov[dist, dist]).all()
