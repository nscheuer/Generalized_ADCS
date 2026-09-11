"""Load and display saved 3-MTQ + 1-RW LP diagnostics without estimator traces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
from _6u_mc_common import load_diagnostic


if __name__ == "__main__":
    load_diagnostic(number_rw=1, allocator="lp", label="3 MTQ + 1 RW (no estimator, LP)", use_estimator=False, campaign_prefix="6u_noestimator_nodisturbance", output_dir=Path(__file__).resolve().parent / "outputs")
