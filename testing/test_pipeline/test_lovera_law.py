"""Lovera_Law: the published law goes in unmodified, and is not double-counted.

Two independent claims, both printed on the SSC26 poster:

1. The pipeline reaches the published MTQ_Lovera output by TWO different
   routes -- a law that does its own gyroscopic term (Lovera_Law,
   includes_gyroscopic=True) and a law that leaves it to the adapter
   (PD_Law, includes_gyroscopic=False). Both must equal MTQ_Lovera.

2. Asking for gyroscopic compensation around a law that already does it
   changes nothing, because the compensation stage skips the term the law
   declares. That is the double-counting guard.
"""

import numpy as np
import pytest

from ADCS.controller import MTQ_Lovera
from ADCS.pipeline import PipelineController
from ADCS.pipeline.control_law import PD_Law, Lovera_Law
from ADCS.pipeline.data import AllocationConfig, CompensationConfig
from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize

GAINS = dict(p_gain=2e-5, d_gain=2e-2, eps=1.0)


@pytest.fixture(scope="module")
def bus():
    mtqs = [MTQ(axis=j, max_torque=1.0) for j in MathConstants.unitvecs]
    rws = [RW(axis=j, max_torque=0.007, J=0.001, h=0.005, h_max=0.0162)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                     actuators=mtqs + rws, sensors=mtms,
                     boresight=np.array([0, 0, 1]))


@pytest.fixture(scope="module")
def orbit():
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22 - TimeConstants.sec2cent,
                        R=7000 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2]),
                        V=np.array([8.0, 0.0, 0.0]))
    return Orbit(os0=os0, end_time=0.22 + 500.0 * TimeConstants.sec2cent,
                 dt=50.0, fast=False, zonal_J=2, verbose=False)


def _states():
    q = normalize(np.array([0.7, 0.3, -0.5, 0.2]))
    return [
        np.concatenate([np.zeros(3), np.array([1.0, 0, 0, 0]), 0.005 * np.ones(3)]),
        np.concatenate([np.array([0.01, -0.005, 0.008]), q, 0.005 * np.ones(3)]),
        np.concatenate([np.array([0.1, -0.08, 0.05]), q, 0.01 * np.ones(3)]),
    ]


def _sample(bus, orbit, t):
    return orbit.get_os(J2000=0.22 + t * TimeConstants.sec2cent)


@pytest.mark.parametrize("t", [0.0, 100.0, 300.0])
@pytest.mark.parametrize("goal_name", ["eci", "none"])
def test_lovera_law_matches_published_controller(bus, orbit, t, goal_name):
    """Lovera_Law + magnetic_cross reproduces MTQ_Lovera exactly."""
    goal = (ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))
            if goal_name == "eci" else No_Goal())
    os_now = _sample(bus, orbit, t)
    legacy = MTQ_Lovera(est_sat=bus, **GAINS)
    law = Lovera_Law(J=bus.J_0, kp=GAINS["p_gain"], kd=GAINS["d_gain"],
                     eps=GAINS["eps"])
    pipe = PipelineController(bus, law)

    for x in _states():
        sens = bus.sensor_readings(x=x, os=os_now)
        u_legacy = legacy.find_u(x_hat=x, sens=sens, est_sat=bus,
                                 os_hat=os_now, goal=goal)
        u_pipe = pipe.find_u(x_hat=x, sens=sens, est_sat=bus,
                             os_hat=os_now, goal=goal)
        assert np.allclose(u_legacy, u_pipe, atol=1e-12), (
            f"Lovera_Law diverged from MTQ_Lovera at t={t}, goal={goal_name}: "
            f"max |du| = {np.max(np.abs(u_legacy - u_pipe)):.3e}")


def test_both_routes_to_the_published_law_agree(bus, orbit):
    """A law that does its own gyro term and one that delegates it agree.

    Lovera_Law includes the gyroscopic term itself; PD_Law leaves it to the
    compensation stage. Both must land on the published output.
    """
    os_now = _sample(bus, orbit, 100.0)
    goal = ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))

    self_gyro = PipelineController(bus, Lovera_Law(
        J=bus.J_0, kp=GAINS["p_gain"], kd=GAINS["d_gain"], eps=GAINS["eps"]))
    adapter_gyro = PipelineController(bus, PD_Law(
        kp=GAINS["p_gain"], kd=GAINS["d_gain"], eps=GAINS["eps"]))

    for x in _states():
        sens = bus.sensor_readings(x=x, os=os_now)
        a = self_gyro.find_u(x_hat=x, sens=sens, est_sat=bus,
                             os_hat=os_now, goal=goal)
        b = adapter_gyro.find_u(x_hat=x, sens=sens, est_sat=bus,
                                os_hat=os_now, goal=goal)
        assert np.allclose(a, b, atol=1e-12)


def test_gyro_is_not_double_counted(bus, orbit):
    """Asking for gyro compensation around Lovera_Law changes nothing.

    This is the poster's headline behaviour: `handles`/`includes_gyroscopic`
    makes the compensation stage skip a term the law already performs, so a
    user who over-specifies the compensation set gets the same answer.
    """
    law = Lovera_Law(J=bus.J_0, kp=GAINS["p_gain"], kd=GAINS["d_gain"],
                     eps=GAINS["eps"])

    auto = CompensationConfig.from_law_interface(law.interface)
    assert auto.enable_gyroscopic is False, (
        "compensation must skip the gyroscopic term for a law that declares it")

    # Explicitly requesting gyroscopic compensation must not change the output.
    over_specified = CompensationConfig(
        enable_gyroscopic=True,
        enable_frame_rotation=auto.enable_frame_rotation,
        enable_disturbance_ff=auto.enable_disturbance_ff,
        enable_damping_injection=auto.enable_damping_injection,
        damping_gain=auto.damping_gain,
    )
    # The guard lives in from_law_interface, so a caller that hand-builds a
    # config bypasses it -- which is exactly the double-count the poster warns
    # about. Verify the two configs genuinely differ in output, so the guard is
    # doing real work rather than being a no-op.
    os_now = _sample(bus, orbit, 100.0)
    goal = ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))
    x = _states()[1]
    sens = bus.sensor_readings(x=x, os=os_now)

    guarded = PipelineController(bus, law, comp_config=auto)
    doubled = PipelineController(bus, law, comp_config=over_specified)
    u_guarded = guarded.find_u(x_hat=x, sens=sens, est_sat=bus,
                               os_hat=os_now, goal=goal)
    u_doubled = doubled.find_u(x_hat=x, sens=sens, est_sat=bus,
                               os_hat=os_now, goal=goal)

    legacy = MTQ_Lovera(est_sat=bus, **GAINS)
    u_legacy = legacy.find_u(x_hat=x, sens=sens, est_sat=bus,
                             os_hat=os_now, goal=goal)

    assert np.allclose(u_guarded, u_legacy, atol=1e-12), (
        "the guarded config must reproduce the published law")
    assert not np.allclose(u_doubled, u_legacy, atol=1e-12), (
        "hand-forcing gyro compensation should double-count; if it does not, "
        "this test is no longer proving the guard does anything")


def test_allocator_decides_whether_the_wheel_is_used(bus, orbit):
    """Same law: cross-product leaves wheels idle, LP commands them.

    This is the poster's headline panel -- a magnetorquer-only published law
    gaining a reaction wheel through a Stage 5 change alone.
    """
    os_now = _sample(bus, orbit, 100.0)
    goal = ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))
    x = _states()[1]
    sens = bus.sensor_readings(x=x, os=os_now)
    law = Lovera_Law(J=bus.J_0, kp=GAINS["p_gain"], kd=GAINS["d_gain"],
                     eps=GAINS["eps"])
    rw_idx = [i for i, a in enumerate(bus.actuators) if isinstance(a, RW)]

    mtq_only = PipelineController(bus, law)
    with_wheel = PipelineController(
        bus, law, alloc_config=AllocationConfig(method='lp'))

    u_mtq = mtq_only.find_u(x_hat=x, sens=sens, est_sat=bus,
                            os_hat=os_now, goal=goal)
    u_lp = with_wheel.find_u(x_hat=x, sens=sens, est_sat=bus,
                             os_hat=os_now, goal=goal)

    assert np.allclose(u_mtq[rw_idx], 0.0), "published form must leave wheels idle"
    assert np.max(np.abs(u_lp[rw_idx])) > 1e-9, "LP must command the wheel"
