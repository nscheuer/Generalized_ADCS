"""Run a 10-trial BeaverCube II horizon-to-horizon ground-contact Monte Carlo."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import ADCS
from ADCS.helpers.math_helpers import dcm_to_quat, normalize, quat_mult
from ADCS.orbits.universal_constants import TimeConstants


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
NAME = "bc2_3mtq_1rw_boston_horizon_to_horizon_mc_10"
NUM_RUNS = 10
DT_S = 2.0
TF_S = 1000.0
BOSTON_LAT_DEG = 42.36
BOSTON_LON_DEG = -71.06
ORBIT_ALTITUDE_KM = 635.0
PASS_CENTER_S = 500.0
# A cross-track miss keeps the maximum elevation near 12.5 deg.  It creates a
# realistic long pass without the near-zenith LOS-rate singularity.
CROSS_TRACK_MISS_DEG = 15.0


def make_pass_orbit(goal: ADCS.goals.Coordinate_Goal) -> ADCS.Orbital_State:
    """Construct a 635 km circular Boston pass, centred within the simulation.

    The satellite is propagated for pre- and post-contact time.  The resulting
    pure-geometric visibility interval is approximately 11.2 min; actual link
    time is evaluated afterward using the 5 degree pointing requirement.
    """
    radius_km = 6378.137 + ORBIT_ALTITUDE_KM
    j2000_peak = 0.22 + PASS_CENTER_S / TimeConstants.cent2sec
    reference = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=j2000_peak,
        R=np.array([radius_km, 0.0, 0.0]),
        V=np.array([0.0, 1.0, 0.0]),
    )
    target_radial = normalize(reference.ecef_to_eci(goal.target_ecef))
    east = normalize(np.cross(np.array([0.0, 0.0, 1.0]), target_radial))
    cross_track = normalize(np.cross(target_radial, east))
    miss = np.deg2rad(CROSS_TRACK_MISS_DEG)
    position_direction = np.cos(miss) * target_radial + np.sin(miss) * cross_track
    peak_state = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=j2000_peak,
        R=radius_km * position_direction,
        V=np.sqrt(398600.4418 / radius_km) * east,
    )
    return peak_state.propagate_orbit_rk4(-PASS_CENTER_S)


def pointing_state(goal: ADCS.goals.Coordinate_Goal, orbit: ADCS.Orbital_State) -> ADCS.State:
    """Pre-point the BC2 +Y boresight at the scheduled target before AOS."""
    target, w_ref_eci = goal.to_ref(orbit)
    body_y_eci = target[1:]
    body_x_eci = normalize(np.cross(body_y_eci, np.array([0.0, 0.0, 1.0])))
    body_z_eci = normalize(np.cross(body_x_eci, body_y_eci))
    rotation_b_to_i = np.column_stack((body_x_eci, body_y_eci, body_z_eci))
    quaternion = dcm_to_quat(rotation_b_to_i)
    return ADCS.State(
        w=rotation_b_to_i.T @ w_ref_eci,
        q=quaternion,
        h=np.zeros(1),
    )


def _mc_config(q_nominal: np.ndarray, w_nominal: np.ndarray) -> ADCS.MCConfig:
    def q_sample(rng: np.random.Generator) -> np.ndarray:
        axis = normalize(rng.standard_normal(3))
        # Large scheduled acquisition error: the controller must acquire the
        # ground station target before AOS rather than starting nearly aligned.
        angle = np.deg2rad(rng.uniform(30.0, 170.0))
        q_error = np.concatenate(([np.cos(angle / 2.0)], axis * np.sin(angle / 2.0)))
        return normalize(quat_mult(q_error, q_nominal))

    def w_sample(rng: np.random.Generator) -> np.ndarray:
        axis = normalize(rng.standard_normal(3))
        return w_nominal + axis * np.deg2rad(rng.uniform(0.10, 1.00))

    return ADCS.MCConfig(
        q=q_sample,
        w=w_sample,
        h=lambda rng: rng.uniform(-0.05e-3, 0.05e-3, size=1),
    )


def main() -> None:
    # This is a controller-only study with the BC2 geometry and disturbance
    # models enabled.  Biases are disabled to isolate allocation/tracking.
    satellite = ADCS.satellite_factory.create_beavercube2_cubesat(
        estimated=False, include_biases=False
    )
    controller = ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=5.0e-5,
        d_gain=2.0e-3,
        c_gain=1.0e-3,
        h_target=np.zeros(3),
    )
    goal = ADCS.goals.Coordinate_Goal(
        lat=BOSTON_LAT_DEG, lon=BOSTON_LON_DEG, alt=0.0
    )
    orbit = make_pass_orbit(goal)
    initial_state = pointing_state(goal, orbit)
    results = ADCS.simulate_mc(
        x=initial_state,
        satellite=satellite,
        controller=controller,
        goal=goal,
        os0=orbit,
        dt=DT_S,
        tf=TF_S,
        mc_config=_mc_config(initial_state.q, initial_state.w),
        num_runs=NUM_RUNS,
        max_workers=4,
        base_seed=20260911,
    )
    path = results.save(NAME, out_dir=OUTPUT_DIR)
    print(f"Saved {len(results)} BeaverCube II runs to {path}")


if __name__ == "__main__":
    main()
