"""Per-step actuator error updates: the plant draws noise and walks biases once per step.

Before this change nothing in the simulation loop ever re-sampled an
actuator's ``Noise`` or advanced its ``Bias`` random walk (the dynamics run
with ``update_noise=False``/``update_bias=False`` so the sample is held over
the integration step, which is right, but no one drew the sample), so the
noise stayed at its initial zero value for the whole run. Wheel momentum
measurements had the same problem: ``h_meas_noise`` was never redrawn, so the
readings were exactly ``h``.
"""

from __future__ import annotations

import numpy as np

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, ErrorMode, Noise
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.state import State

# The mode the plant integrates with: errors applied but held for the step.
HOLD = ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=False)
CLEAN = ErrorMode(add_bias=True, add_noise=False, update_bias=False, update_noise=False)


def _orbital_state(j2000: float = 0.22) -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=j2000, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def _wheel(**kwargs) -> RW:
    return RW(axis=np.array([0.0, 0.0, 1.0]), max_torque=0.02, J=1.0e-3, h=kwargs.pop("h", 0.0), h_max=1.0, **kwargs)


def test_update_errors_draws_a_fresh_noise_sample_and_walks_the_bias():
    np.random.seed(11)
    mtq = MTQ(np.array([1.0, 0.0, 0.0]), max_torque=1.0, bias=Bias(bias=0.0, std_bias=1.0e-3), noise=Noise(noise=0.0, std_noise=0.01))
    assert mtq.noise.get_noise() == 0.0 and mtq.bias.get_bias(0.22) == 0.0
    mtq.update_errors(0.22)  # first call: a sample is drawn, the walk is only time-stamped
    first = mtq.noise.get_noise()
    assert first != 0.0
    assert mtq.bias.get_bias(0.22) == 0.0
    mtq.update_errors(0.22 + 10.0 * TimeConstants.sec2cent)
    assert mtq.noise.get_noise() != first
    assert mtq.bias.get_bias(0.22) != 0.0  # walked over 10 s


def test_update_errors_is_a_no_op_for_a_perfect_actuator():
    mtq = MTQ(np.array([1.0, 0.0, 0.0]), max_torque=1.0)
    mtq.update_errors(0.22)
    mtq.update_errors(0.23)
    assert mtq.noise.get_noise() == 0.0 and mtq.bias.get_bias(0.23) == 0.0


def test_rw_update_errors_invalidates_the_held_command():
    np.random.seed(5)
    rw = _wheel(noise=Noise(noise=0.0, std_noise=1.0e-4))
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    orbital_state = _orbital_state()
    rw.update_errors(orbital_state.J2000)
    before = np.array(rw.torque(u=1.0e-3, x=state, os=orbital_state, dmode=HOLD))
    np.testing.assert_allclose(rw.torque(u=1.0e-3, x=state, os=orbital_state, dmode=HOLD), before)  # held within the step
    rw.update_errors(orbital_state.J2000)  # a new sample at the same epoch and command
    after = np.array(rw.torque(u=1.0e-3, x=state, os=orbital_state, dmode=HOLD))
    assert not np.allclose(after, before), "the torque was served from the stale cached realization"
    reaction = float(rw.storage_torque(u=1.0e-3, x=state, os=orbital_state, dmode=HOLD))
    assert np.isclose(float(np.dot(after, rw.axis)), -reaction)


def test_satellite_update_actuator_errors_reaches_every_actuator():
    np.random.seed(3)
    satellite = Satellite(
        J_0=np.diag([0.5, 0.8, 1.2]),
        actuators=[MTQ(axis, max_torque=1.0, noise=Noise(noise=0.0, std_noise=0.01)) for axis in np.eye(3)]
        + [_wheel(noise=Noise(noise=0.0, std_noise=1.0e-4))],
    )
    satellite.update_actuator_errors(0.22)
    assert all(float(np.max(np.abs(actuator.noise.get_noise()))) != 0.0 for actuator in satellite.actuators)


def test_rw_momentum_measurement_noise_is_drawn_per_step():
    np.random.seed(9)
    rw = _wheel(h=0.1, h_meas_noise=Noise(noise=0.0, std_noise=1.0e-3))
    assert float(rw.measure_momentum()) == 0.1  # nothing drawn yet
    rw.update_errors(0.22)
    first = float(rw.measure_momentum())
    assert first != 0.1
    assert float(rw.measure_momentum()) == first  # held within the step
    rw.update_errors(0.22 + 10.0 * TimeConstants.sec2cent)
    assert float(rw.measure_momentum()) != first  # fresh sample for the next step
    assert float(rw.measure_momentum(dmode=CLEAN)) == 0.1
    assert float(rw.measure_momentum_noiseless()) == 0.1
    held = float(rw.measure_momentum())
    redraw = ErrorMode(add_bias=True, add_noise=True, update_bias=False, update_noise=True)
    fresh = float(rw.measure_momentum(dmode=redraw))  # explicit redraw replaces the held sample
    assert fresh != held and float(rw.measure_momentum()) == fresh
    satellite = Satellite(J_0=np.diag([0.5, 0.8, 1.2]), actuators=[rw])
    satellite.update_actuator_errors(0.23)
    state = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    orbital_state = _orbital_state()
    assert satellite.noiseless_sensor_readings(state, orbital_state)[-1] == 0.1
    assert satellite.sensor_readings(state, orbital_state)[-1] != 0.1
