"""The RK4 midpoint is the average of the step's start and end orbital states.

Two call sites passed the *end* orbital state as the midpoint instead: the
``update()`` one-call adapter and ``simulate()``. The propagator then
evaluated the environment (magnetic field, sun, atmosphere) one half-step
late for the two middle stages, which cost about 34 arcsec of attitude and
3e-5 rad/s of rate per 10 s step. Leaving the keyword out lets
``noiseless_rk4`` average the two states, as the staged ``step()`` path
already did.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators import MEKF, AugmentedMEKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
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


def test_update_adapter_propagates_with_an_averaged_midpoint():
    """update() must integrate with the endpoint-averaged orbital state.

    Gravity gradient makes the dynamics depend on where the satellite is, so a
    60 s step exposes which orbital state the integrator stages saw.
    """
    satellite = _satellite(disturbances=[D.GG_Disturbance()])
    os_start = _orbital_state(0.22)
    os_end = _orbital_state(0.22 + 60.0 * TimeConstants.sec2cent, along_track_s=60.0)
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


def test_simulate_lets_the_estimator_average_the_step_midpoint():
    """simulate() used to pass the end-of-step orbital state as the RK4 midpoint."""
    import ADCS
    from ADCS.satellite_hardware.satellite import Satellite
    from ADCS.state import State

    class Recording:
        def __init__(self):
            self.midpoints = []
            self._state = _state()

        @property
        def state(self):
            return self._state

        def predict(self, u, os_start, os_end, midpoint_orbital_state=None, **kwargs):
            self.midpoints.append(midpoint_orbital_state)
            return self._state

        def step(self, measurements, os):
            return self._state

        def update(self, *args, **kwargs):
            return self._state

    recorder = Recording()
    satellite = Satellite(J_0=np.diag([0.5, 0.8, 1.2]), sensors=[Gyro(axis, noise=Noise(std_noise=1.0e-4)) for axis in np.eye(3)])
    ADCS.simulate(x=State(w=np.array([0.01, 0.0, 0.0]), q=[1.0, 0.0, 0.0, 0.0]), satellite=satellite,
                  est_satellite=_satellite(), estimator=recorder, os0=_orbital_state(0.22), dt=10.0, tf=30.0)
    assert len(recorder.midpoints) >= 2
    assert all(midpoint is None for midpoint in recorder.midpoints)  # the propagator averages start and end itself
