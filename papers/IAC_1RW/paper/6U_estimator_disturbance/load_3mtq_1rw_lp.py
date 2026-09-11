"""Load and plot the latest disturbed 3-MTQ + 1-RW LP MC result."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import load_diagnostic

if __name__ == "__main__":
    load_diagnostic(number_rw=1, allocator="lp", label="3 MTQ + 1 RW (disturbed, LP)", campaign_prefix="6u_estimator_disturbance", output_dir=Path(__file__).resolve().parent / "outputs", result_prefix="mc")
