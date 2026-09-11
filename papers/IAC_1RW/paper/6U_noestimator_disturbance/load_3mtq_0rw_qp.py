"""Load and plot the latest disturbed 3-MTQ + 0-RW QP result without an estimator."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import load_diagnostic

if __name__ == "__main__":
    load_diagnostic(number_rw=0, allocator="qp", label="3 MTQ + 0 RW (no estimator, disturbed, QP)", use_estimator=False, campaign_prefix="6u_noestimator_disturbance", output_dir=Path(__file__).resolve().parent / "outputs")
