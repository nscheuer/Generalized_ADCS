"""Regression tests for the estimator audit fixes.

Each test here pins one defect found in the September 2026 estimator audit:
the fix must keep it green, and the pre-fix code must fail it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import MEKF, AugmentedMEKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware import disturbances as D
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import EarthHorizonSensor, Gyro, StarTrackerQuaternion
from ADCS.state import EstimatorState


EPHEM = Ephemeris()


def _orbital_state(j2000: float, along_track_s: float = 0.0) -> Orbital_State:
    """Circular-orbit sample ``along_track_s`` seconds past the reference point."""
    angle = along_track_s * 7.5 / 7000.0
    R = 7000.0 * np.array([np.cos(angle), np.sin(angle), 0.0])
    V = 7.5 * np.array([-np.sin(angle), np.cos(angle), 0.0])
    return Orbital_State(ephem=EPHEM, J2000=j2000, R=R, V=V, fast=True)


def _tracker() -> StarTrackerQuaternion:
    tracker = StarTrackerQuaternion(noise=Noise(std_noise=np.full(4, 1.0e-3)))

    def clean(*args, **kwargs):
        state = kwargs.get("x", kwargs.get("state"))
        if state is None and args:
            state = args[0]
        return np.asarray(state.q, float).copy()

    tracker.clean_reading = clean
    return tracker


def _satellite(disturbances=()) -> EstimatedSatellite:
    return EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [_tracker()],
        disturbances=list(disturbances),
    )


def _state(size: int = 6, **blocks) -> EstimatorState:
    return EstimatorState(
        w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1],
        cov=np.eye(size) * 1.0e-3, int_cov=np.zeros((size, size)), **blocks,
    ).normalized()


# --- update() adapter: midpoint --------------------------------------------

def test_update_adapter_propagates_with_an_averaged_midpoint():
    """update() must integrate with the endpoint-averaged orbital state.

    Gravity gradient makes the dynamics depend on where the satellite is, so a
    60 s step exposes which orbital state the integrator stages saw.
    """
    satellite = _satellite(disturbances=[D.GG_Disturbance()])
    os_start = _orbital_state(0.22)
    os_end = _orbital_state(0.22 + 60.0 / 86400.0, along_track_s=60.0)
    truth = _state()
    z_start = satellite.measurement_stack.predict(truth, os_start)
    z_end = satellite.measurement_stack.predict(truth, os_end)
    control = np.empty(0)

    adapter = MEKF(satellite, _state(), dt=60.0)
    adapter.update(control, z_start, os_start)
    adapter.update(control, z_end, os_end)

    explicit = MEKF(satellite, _state(), dt=60.0)
    explicit.correct(z_start, os_start)
    explicit.predict(control, os_start, os_end)  # midpoint defaults to the average
    explicit.correct(z_end, os_end)

    biased = MEKF(satellite, _state(), dt=60.0)
    biased.correct(z_start, os_start)
    biased.predict(control, os_start, os_end, midpoint_orbital_state=os_end)
    biased.correct(z_end, os_end)

    np.testing.assert_allclose(
        adapter.state.as_estimator_array(), explicit.state.as_estimator_array(), rtol=0.0, atol=1.0e-15,
    )
    assert not np.allclose(
        explicit.state.as_estimator_array(), biased.state.as_estimator_array(), rtol=0.0, atol=0.0,
    ), "the end-state midpoint must be distinguishable, or this test proves nothing"


# --- disturbances reachable through the augmented filters -------------------

def test_prop_disturbance_is_estimable_like_torque_disturbance():
    nominal = np.array([1.0e-6, 2.0e-6, -1.0e-6])
    disturbance = D.Prop_Disturbance(nominal.copy(), estimate_dist=True)
    assert disturbance.estimated_vector_length == 3
    np.testing.assert_allclose(disturbance.main_param, nominal)
    np.testing.assert_allclose(disturbance.torque_valjac(), np.eye(3))

    disturbance.main_param = [3.0e-6, 0.0, 0.0]
    np.testing.assert_allclose(disturbance.current_torque, [3.0e-6, 0.0, 0.0])

    satellite = _satellite(disturbances=[disturbance])
    assert satellite.dist_param_len == 3
    estimator = AugmentedMEKF(satellite, _state(9, dist_param=np.zeros(3)), dt=1.0)
    os0 = _orbital_state(0.22)
    predicted = estimator.predict(np.empty(0), os0, os0)  # used to raise NotImplementedError
    assert predicted.covariance.dimension == 9


def test_general_disturbance_refuses_direct_instantiation():
    with pytest.raises(TypeError, match="abstract"):
        D.General_Disturbance()
    with pytest.raises(TypeError, match="abstract"):
        D.General_Disturbance(estimate_dist=True, estimated_vector_length=3)

    class Constant(D.General_Disturbance):
        def torque(self, *args, **kwargs):
            return np.array([1.0e-7, 0.0, 0.0])

        def torque_qjac(self, *args, **kwargs):
            return np.zeros((4, 3))

    satellite = _satellite(disturbances=[Constant()])  # subclasses keep working
    os0 = _orbital_state(0.22)
    estimator = MEKF(satellite, _state(), dt=1.0)
    estimator.predict(np.empty(0), os0, os0)


# --- int_cov is not consumed by this filter family --------------------------


def test_quaternion_star_tracker_bias_estimation_is_rejected_at_assembly():
    tracker = StarTrackerQuaternion(bias=Bias(bias=np.zeros(4), std_bias=np.full(4, 1.0e-9)), estimate_bias=True)
    with pytest.raises(ValueError, match="cannot be estimated"):
        EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[tracker])


# --- guard rails --------------------------------------------------------------

def test_two_argument_step_rejects_a_legacy_control_measurement_call():
    """With 7 controls and a 7-row stack the old step(control, measurements)
    used to be corrected silently with the control as the measurement."""
    from ADCS.satellite_hardware.actuators import MTQ
    satellite = EstimatedSatellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        sensors=[Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)] + [_tracker()],
        actuators=[MTQ(np.eye(3)[i % 3], max_torque=0.05, noise=Noise(std_noise=1.0e-6)) for i in range(7)],
    )
    estimator = MEKF(satellite, _state(), dt=1.0)
    with pytest.raises(TypeError, match="Orbital_State"):
        estimator.step(np.full(7, 0.3), np.zeros(7))
    with pytest.raises(TypeError, match="update requires"):
        estimator.update(sensors=np.zeros(7), os=_orbital_state(0.22))


def test_predict_warns_when_dt_disagrees_with_the_orbital_state_gap():
    satellite = _satellite()
    os_start = _orbital_state(0.22)
    os_end = _orbital_state(0.22 + 100.0 / 86400.0, along_track_s=100.0)
    estimator = MEKF(satellite, _state(), dt=10.0)
    with pytest.warns(UserWarning, match="apart"):
        estimator.predict(np.empty(0), os_start, os_end)  # 10 s step over a 100 s gap
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MEKF(satellite, _state(), dt=100.0).predict(np.empty(0), os_start, os_end)  # matched
        MEKF(satellite, _state(), dt=10.0).predict(np.empty(0), os_start, os_start)  # static orbit


# --- augmented actuator bias reaches the propagated mean ---------------------

def test_augmented_filters_recover_a_magnetorquer_bias():
    """The deterministic propagation used to drop the held actuator bias while
    the Jacobian kept it, so the augmented filters wound the bias up 7-10x."""
    from ADCS.satellite_hardware.actuators import MTQ

    true_bias = np.array([0.05, -0.03, 0.02])

    def actuators(*, bias=None, estimate=False):
        return [MTQ(axis, max_torque=0.5, noise=Noise(std_noise=1.0e-9),
                    bias=None if bias is None and not estimate else
                    Bias(bias=np.array([0.0 if bias is None else bias[i]]), std_bias=np.array([1.0e-7])),
                    estimate_bias=estimate) for i, axis in enumerate(np.eye(3))]

    def sensors():
        return [Gyro(axis, noise=Noise(std_noise=1.0e-6)) for axis in np.eye(3)] + [_tracker()]

    truth_sat = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=sensors(), actuators=actuators(bias=true_bias))
    est_sat = EstimatedSatellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=sensors(), actuators=actuators(estimate=True))
    assert est_sat.act_bias_len == 3

    from ADCS.estimators.process_model import propagate_state

    from ADCS.estimators.attitude_estimators import AugmentedSRUKF, AugmentedUKF

    for cls in (AugmentedMEKF, AugmentedUKF, AugmentedSRUKF):
        size = 6 + 3
        cov = np.diag([1.0e-6] * 3 + [1.0e-4] * 3 + [1.0e-2] * 3)
        state = EstimatorState(w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1], act_bias=np.zeros(3),
                               cov=cov, int_cov=np.zeros((size, size))).normalized()
        estimator = cls(est_sat, state, dt=10.0, unmodeled_dynamics_psd=np.array([1.0e-16] * 3 + [1.0e-9] * 3))
        truth = EstimatorState(w=[0.02, -0.015, 0.01], q=[0.9, 0.2, -0.3, 0.1]).normalized()
        os_prev = _orbital_state(0.22)
        rng = np.random.default_rng(3)
        errors = []
        for k in range(40):
            os_k = _orbital_state(0.22 + (k + 1) * 10.0 / 86400.0, along_track_s=(k + 1) * 10.0)
            control = 0.2 * rng.standard_normal(3)  # excite the bias through a varying command
            truth = propagate_state(truth, truth_sat, control, 10.0, os_prev, os_k)
            estimator.predict(control, os_prev, os_k)
            estimator.correct(truth_sat.noiseless_sensor_readings(truth, os_k), os_k)
            errors.append(np.linalg.norm(estimator.state.act_bias - true_bias) / np.linalg.norm(true_bias))
            os_prev = os_k
        assert errors[-1] < 0.25, f"{cls.__name__}: relative bias error {errors[0]:.2f} -> {errors[-1]:.2f}"
        assert errors[-1] < errors[0]


# --- integration: controller base and legacy stubs -----------------------------

def test_controller_base_stores_the_estimated_satellite():
    """The remote controller service reads controller.est_sat; only two
    subclasses used to set it, so 13 of 15 controllers failed remotely."""
    from ADCS.controller.controller import Controller

    class Minimal(Controller):
        def find_u(self, *args, **kwargs):
            return np.zeros(0)

    satellite = _satellite()
    assert Minimal(satellite).est_sat is satellite


def test_legacy_estimator_stubs_warn_on_import():
    import importlib
    import sys
    import warnings

    for name in ("ADCS.estimators.old_attitude_estimators.attitude_UAKF",
                 "ADCS.estimators.old_attitude_estimators.attitude_SRUAKF"):
        sys.modules.pop(name, None)
        with pytest.warns(DeprecationWarning, match="compatibility alias"):
            importlib.import_module(name)
