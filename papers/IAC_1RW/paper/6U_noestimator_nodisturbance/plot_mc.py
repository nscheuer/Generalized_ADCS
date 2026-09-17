"""Plot mean +/- 1-sigma attitude convergence for the latest 6U MC results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
import plot_mc as _plot_mc

_plot_mc.CAMPAIGN_PREFIX = "6u_noestimator_nodisturbance"
_plot_mc.OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

if __name__ == "__main__":
    _plot_mc.main()
