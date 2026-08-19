"""
Regression test: PipelineController(PD_Law + Gyroscopic + MagneticCross)
should produce identical outputs to MTQ_Lovera given the same state.

This validates that the Phase 1 pipeline correctly decomposes the Lovera
controller into its constituent stages.
"""

import numpy as np
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ADCS.controller import MTQ_Lovera
from ADCS.pipeline import PipelineController
from ADCS.pipeline.control_law import PD_Law
from ADCS.CONOPS.goals import ECI_Goal, No_Goal, Nadir_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.state import State


def make_satellite():
    """Create a test satellite with 3 MTQs and 3 RWs."""
    mtq_max = 1.0
    mtqs = [MTQ(axis=j, max_torque=mtq_max) for j in MathConstants.unitvecs]

    rw_max = 0.007
    rw_J = 0.001
    rw_h0 = 0.005
    rw_hmax = 0.0162
    rws = [RW(axis=j, max_torque=rw_max, J=rw_J, h=rw_h0, h_max=rw_hmax)
           for j in MathConstants.unitvecs]

    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]

    sat = Satellite(
        mass=4.0,
        J_0=np.diagflat([3.4, 2.9, 1.3]),
        actuators=mtqs + rws,
        sensors=mtms,
        boresight=np.array([0, 0, 1]),
    )
    return sat


def make_orbit(tf_sec=500.0, dt=50.0):
    """Create a test orbit."""
    ephem = Ephemeris()
    start_J2000 = 0.22 - 1 * TimeConstants.sec2cent
    end_J2000 = 0.22 + tf_sec * TimeConstants.sec2cent
    R0 = 7000 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2])
    V0 = np.array([8.0, 0.0, 0.0])
    os0 = Orbital_State(ephem=ephem, J2000=start_J2000, R=R0, V=V0)
    orbit = Orbit(os0=os0, end_time=end_J2000, dt=dt, zonal_J=2, fast=False)
    return orbit


def _single_step(sat, orbit, goal, x, t, gains, label=""):
    """Compare MTQ_Lovera and PipelineController on a single step."""
    p_gain, d_gain, eps = gains

    # Create both controllers
    lovera = MTQ_Lovera(est_sat=sat, p_gain=p_gain, d_gain=d_gain, eps=eps)
    pd_law = PD_Law(kp=p_gain, kd=d_gain, eps=eps)
    pipeline = PipelineController(est_sat=sat, law=pd_law)

    # Get orbital state
    J2000 = 0.22 + t * TimeConstants.sec2cent
    os_now = orbit.get_os(J2000=J2000)

    # Sensor readings
    sens = sat.sensor_readings(x=x, os=os_now)

    # Compute commands
    u_lovera = lovera.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
    u_pipeline = pipeline.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)

    # Compare
    diff = np.abs(u_lovera - u_pipeline)
    max_diff = np.max(diff)

    status = "PASS" if max_diff < 1e-12 else "FAIL"
    print(f"  [{status}] {label}: max |u_lovera - u_pipeline| = {max_diff:.2e}")
    if max_diff >= 1e-12:
        print(f"    u_lovera:   {u_lovera}")
        print(f"    u_pipeline: {u_pipeline}")
        print(f"    diff:       {diff}")

    return max_diff < 1e-12


def main():
    print("=" * 60)
    print("Pipeline vs MTQ_Lovera Regression Test")
    print("=" * 60)

    sat = make_satellite()
    orbit = make_orbit()
    gains = (0.00002, 0.02, 1.0)

    all_pass = True

    # --- Test 1: ECI goal, identity quaternion, zero angular velocity ---
    print("\nTest 1: ECI goal, q=[1,0,0,0], w=[0,0,0]")
    x0 = State(w=np.zeros(3), q=np.array([1, 0, 0, 0.0]), h=0.005 * np.ones(3))
    goal = ECI_Goal(normalize(np.array([1.0, 0.0, 0.0])))
    all_pass &= _single_step(sat, orbit, goal, x0, t=0.0, gains=gains,
                                  label="identity quat, zero omega")

    # --- Test 2: ECI goal, rotated quaternion, nonzero omega ---
    print("\nTest 2: ECI goal, rotated state")
    q_rot = normalize(np.array([0.7, 0.3, -0.5, 0.2]))
    w_rot = np.array([0.01, -0.005, 0.008])
    x1 = State(w=w_rot, q=q_rot, h=0.005 * np.ones(3))
    goal_eci = ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))
    all_pass &= _single_step(sat, orbit, goal_eci, x1, t=100.0, gains=gains,
                                  label="rotated quat, nonzero omega")

    # --- Test 3: No goal ---
    print("\nTest 3: No_Goal (zero error)")
    all_pass &= _single_step(sat, orbit, No_Goal(), x1, t=100.0, gains=gains,
                                  label="no goal")

    # --- Test 4: Nadir goal ---
    print("\nTest 4: Nadir goal")
    all_pass &= _single_step(sat, orbit, Nadir_Goal(), x1, t=200.0, gains=gains,
                                  label="nadir goal")

    # --- Test 5: Large angular velocity (saturation test) ---
    print("\nTest 5: Large omega (saturation)")
    w_large = np.array([0.1, -0.08, 0.05])
    x2 = State(w=w_large, q=q_rot, h=0.01 * np.ones(3))
    all_pass &= _single_step(sat, orbit, goal_eci, x2, t=50.0, gains=gains,
                                  label="large omega, saturation")

    # --- Test 6: Different time in orbit (different B field) ---
    print("\nTest 6: Different orbital position (t=300s)")
    all_pass &= _single_step(sat, orbit, goal_eci, x1, t=300.0, gains=gains,
                                  label="t=300s, different B field")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED - Pipeline matches MTQ_Lovera exactly!")
    else:
        print("SOME TESTS FAILED - Pipeline does NOT match MTQ_Lovera!")
    print("=" * 60)

    return 0 if all_pass else 1


def test_pipeline_matches_lovera():
    """Pytest entry point: the pipeline must reproduce MTQ_Lovera bit-for-bit.

    Without this, ``main()`` only ran when the file was executed directly, so
    the parity check never fired under pytest or in CI.
    """
    assert main() == 0, "PipelineController output diverged from MTQ_Lovera"


if __name__ == "__main__":
    sys.exit(main())
