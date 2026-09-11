"""Run the missing matched 6U 3-MTQ + 2-RW disturbed LP campaign."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = PAPER_DIR / "6U_estimator_nodisturbance"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COMMON_DIR))

import ADCS
from ADCS.helpers.math_helpers import normalize
from _6u_mc_common import (
    DT_S,
    KD,
    KP,
    MOMENTUM_GAIN,
    ORBIT_PERIOD_S,
    WHEEL_BASELINE_MOMENTUM,
    WHEEL_MOMENTUM_MAX,
    WHEEL_TORQUE_MAX,
    MTQ_DIPOLE_MAX,
    _disturbances,
    _sensors,
    random_400km_97deg_orbit,
)


OUTPUT_DIR = SCRIPT_DIR / "outputs"
CAMPAIGN_NAME = "6u_degraded_3mtq_2rw_lp_mc_10"
NUM_RUNS = 10


def make_3p2_satellite() -> ADCS.Satellite:
    actuators = [ADCS.MTQ(axis, max_torque=MTQ_DIPOLE_MAX) for axis in np.eye(3)]
    actuators += [
        ADCS.RW(
            axis=axis,
            max_torque=WHEEL_TORQUE_MAX,
            J=1.0e-5,
            h=WHEEL_BASELINE_MOMENTUM,
            h_max=WHEEL_MOMENTUM_MAX,
        )
        for axis in np.eye(3)[:2]
    ]
    disturbances, com = _disturbances()
    return ADCS.Satellite(
        mass=12.0,
        COM=com,
        J_0=np.diag([0.13, 0.10, 0.05]),
        disturbances=disturbances,
        actuators=actuators,
        sensors=_sensors(estimated=False),
        boresight=np.array([0.0, 0.0, 1.0]),
    )


def make_mc_config() -> ADCS.MCConfig:
    return ADCS.MCConfig(
        q=lambda rng: normalize(rng.standard_normal(4)),
        h=lambda _rng: np.full(2, WHEEL_BASELINE_MOMENTUM),
        goal=lambda rng: ADCS.goals.Fixed_Attitude_Goal(normalize(rng.standard_normal(4))),
        orbit=random_400km_97deg_orbit,
    )


def main() -> None:
    satellite = make_3p2_satellite()
    h_target = np.array([WHEEL_BASELINE_MOMENTUM, WHEEL_BASELINE_MOMENTUM, 0.0])
    controller = ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=KP,
        d_gain=KD,
        c_gain=MOMENTUM_GAIN,
        h_target=h_target,
    )
    state = ADCS.State(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.full(2, WHEEL_BASELINE_MOMENTUM),
    )
    results = ADCS.simulate_mc(
        x=state,
        satellite=satellite,
        est_satellite=satellite,
        controller=controller,
        estimator=None,
        goal=ADCS.goals.Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0])),
        os0=random_400km_97deg_orbit(np.random.default_rng(20260911)),
        dt=DT_S,
        tf=ORBIT_PERIOD_S,
        mc_config=make_mc_config(),
        num_runs=NUM_RUNS,
        max_workers=4,
        base_seed=20260911,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = results.save(CAMPAIGN_NAME, out_dir=OUTPUT_DIR)
    print(f"Saved {len(results)} matched 3+2 runs to {path}")


if __name__ == "__main__":
    main()
