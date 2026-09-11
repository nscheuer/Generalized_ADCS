"""Run a 10-trial disturbed 6U 3-MTQ + 1-RW LP Monte Carlo campaign."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import run_monte_carlo, KP, KD

if __name__ == "__main__":
    run_monte_carlo(number_rw=1, allocator="lp", label="3 MTQ + 1 RW (disturbed, LP)", use_estimator=True, disturbances=True, campaign_prefix="6u_estimator_disturbance", output_dir=Path(__file__).resolve().parent / "outputs", default_kp=KP, default_kd=KD)
