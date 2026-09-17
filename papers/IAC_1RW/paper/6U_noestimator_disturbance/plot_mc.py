"""Plot mean +/- 1-sigma attitude convergence for the latest 6U MC results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "6U_estimator_nodisturbance"))
import plot_mc as _plot_mc

_plot_mc.CAMPAIGN_PREFIX = "6u_noestimator_disturbance"
_plot_mc.OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_plot_mc.CONFIGURATIONS = ((0, "qp", "3+0 (QP)"),
                           (1, "lp", "3+1 (LP)"),
                           (3, "lp", "3+3 (LP)"))
_plot_mc.PERCENTILE_TITLE = "6U attitude convergence (100-run Monte Carlo)"
_plot_mc.PERCENTILE_YLABEL = "Median angle error [deg]"


def _latest_quaternion_100_sim(output_dir, number_rw, allocator):
    pattern = (f"{_plot_mc.CAMPAIGN_PREFIX}_quaternion_3mtq_{number_rw}rw_"
               f"{allocator}_mc_100_*.sim")
    candidates = list(output_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No 100-run quaternion result matching {pattern!r}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


_plot_mc._latest_sim = _latest_quaternion_100_sim

if __name__ == "__main__":
    _plot_mc.main()
