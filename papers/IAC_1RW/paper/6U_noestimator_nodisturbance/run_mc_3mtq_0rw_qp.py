"""Run a 10-trial 6U 3-MTQ + 0-RW QP Monte Carlo campaign."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import run_monte_carlo, TUNED_3MTQ_0RW_KP, TUNED_3MTQ_0RW_KD


if __name__ == "__main__":
    run_monte_carlo(number_rw=0, allocator="qp", label="3 MTQ + 0 RW (no estimator, QP)", use_estimator=False, campaign_prefix="6u_noestimator_nodisturbance", output_dir=Path(__file__).resolve().parent / "outputs", default_kp=TUNED_3MTQ_0RW_KP, default_kd=TUNED_3MTQ_0RW_KD)
