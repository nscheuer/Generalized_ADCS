"""Run a 10-trial 6U 3-MTQ + 3-RW LP Monte Carlo campaign."""

from pathlib import Path
from _6u_mc_common import run_monte_carlo, KP, KD


if __name__ == "__main__":
    run_monte_carlo(number_rw=3, allocator="lp", label="3 MTQ + 3 RW (LP)", use_estimator=True, campaign_prefix="6u_estimator_nodisturbance", output_dir=Path(__file__).resolve().parent / "outputs", default_kp=KP, default_kd=KD)
