"""Run a 10-trial disturbed 6U 3-MTQ + 0-RW QP Monte Carlo campaign."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import run_monte_carlo, TUNED_3MTQ_0RW_KP, TUNED_3MTQ_0RW_KD

if __name__ == "__main__":
    run_monte_carlo(number_rw=0, allocator="qp", label="3 MTQ + 0 RW (disturbed, QP)", use_estimator=True, disturbances=True, campaign_prefix="6u_estimator_disturbance", output_dir=Path(__file__).resolve().parent / "outputs", default_kp=TUNED_3MTQ_0RW_KP, default_kd=TUNED_3MTQ_0RW_KD)
