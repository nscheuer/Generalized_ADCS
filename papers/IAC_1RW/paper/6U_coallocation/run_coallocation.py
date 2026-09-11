"""Run a disturbed 6U coallocation Monte Carlo sweep over gamma."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PAPER_DIR / "6U_estimator_nodisturbance"))

import ADCS
from ADCS.CONOPS.goals import No_Goal
from ADCS.helpers.math_helpers import normalize
from _6u_mc_common import (
    KD,
    KP,
    MOMENTUM_GAIN,
    WHEEL_MOMENTUM_MAX,
    controller_for,
    make_satellite,
    random_400km_97deg_orbit,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CAMPAIGN_PREFIX = "6u_coallocation_disturbance"
GAMMAS = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
DT_S = 5.0
DEFAULT_DURATION_MIN = 180.0
INITIAL_MOMENTUM_FRACTION = 0.50


def gamma_slug(gamma: float) -> str:
    return f"{gamma:.2f}".replace(".", "p")


class CoallocationController:
    r"""Blend pure pointing and pure desaturation actuator commands.

    ``gamma=0`` gives pure LP pointing and ``gamma=1`` gives the modified
    B-dot/No_Goal desaturation law.  Since both endpoint commands obey the
    actuator limits, their convex combination does too.
    """

    def __init__(self, pointing_controller, desaturation_controller, gamma: float) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.pointing_controller = pointing_controller
        self.desaturation_controller = desaturation_controller
        self.gamma = float(gamma)

    def find_u(self, *, x_hat, sens, est_sat, os_hat, goal):
        u_pointing = self.pointing_controller.find_u_pointing(
            x_hat=x_hat, sens=sens, est_sat=est_sat, os_hat=os_hat, goal=goal
        )
        u_desaturation = self.desaturation_controller.find_u_desaturate(
            x_hat=x_hat, sens=sens, est_sat=est_sat, os_hat=os_hat, goal=No_Goal()
        )
        return (1.0 - self.gamma) * u_pointing + self.gamma * u_desaturation


def make_controller(satellite, gamma: float) -> CoallocationController:
    # c_gain=0 makes this endpoint genuinely pure pointing; otherwise the LP
    # controller also attempts a secondary momentum dump in pointing mode.
    pointing = controller_for(
        "lp", satellite, 1, kp=KP, kd=KD, c_gain=0.0
    )
    desaturation = controller_for(
        "lp", satellite, 1, kp=KP, kd=KD, c_gain=MOMENTUM_GAIN
    )
    return CoallocationController(pointing, desaturation, gamma)


def make_mc_config() -> ADCS.MCConfig:
    return ADCS.MCConfig(
        q=lambda rng: normalize(rng.standard_normal(4)),
        h=lambda _rng: np.array([
            INITIAL_MOMENTUM_FRACTION * WHEEL_MOMENTUM_MAX
        ]),
        goal=lambda rng: ADCS.goals.Fixed_Attitude_Goal(
            normalize(rng.standard_normal(4))
        ),
        orbit=random_400km_97deg_orbit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=DEFAULT_DURATION_MIN)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if args.runs <= 0 or args.minutes <= 0.0:
        raise ValueError("--runs and --minutes must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    satellite = make_satellite(1, estimated=False, disturbances=True)
    satellite.enforce_hard_momentum_limits = True
    state = ADCS.State(
        w=np.zeros(3),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        h=np.array([INITIAL_MOMENTUM_FRACTION * WHEEL_MOMENTUM_MAX]),
    )
    goal = ADCS.goals.Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))
    os0 = random_400km_97deg_orbit(np.random.default_rng(args.seed))

    for gamma in GAMMAS:
        print(f"\nRunning gamma={gamma:.2f} ({args.runs} paired cases) ...")
        results = ADCS.simulate_mc(
            x=state,
            satellite=satellite,
            est_satellite=satellite,
            controller=make_controller(satellite, float(gamma)),
            estimator=None,
            goal=goal,
            os0=os0,
            dt=DT_S,
            tf=args.minutes * 60.0,
            mc_config=make_mc_config(),
            num_runs=args.runs,
            max_workers=args.workers,
            base_seed=args.seed,
        )
        name = (
            f"{CAMPAIGN_PREFIX}_gamma_{gamma_slug(float(gamma))}"
            f"_mc_{args.runs}_{args.minutes:g}min"
        )
        print(f"Saved simulation: {results.save(name, out_dir=OUTPUT_DIR)}")

    print("Run plot_coallocation.py to regenerate the figures from the latest .sim files.")


if __name__ == "__main__":
    main()
