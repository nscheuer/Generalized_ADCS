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
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Gyro, StarTrackerQuaternion
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

def test_nonzero_int_cov_with_zero_psd_warns_at_construction():
    satellite = _satellite()
    trap = EstimatorState(
        w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=np.eye(6) * 1.0e-3, int_cov=np.eye(6) * 1.0e-8,
    )
    with pytest.warns(UserWarning, match="unmodeled_dynamics_psd"):
        MEKF(satellite, trap, dt=1.0)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MEKF(satellite, trap, dt=1.0, unmodeled_dynamics_psd=1.0e-9)  # PSD given: fine
        MEKF(satellite, _state(), dt=1.0)  # zero int_cov: fine
