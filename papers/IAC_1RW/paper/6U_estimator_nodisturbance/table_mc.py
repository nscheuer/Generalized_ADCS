"""Print the convergence summary table from the latest 6U MC results."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
import ADCS

CAMPAIGN_PREFIX = "6u_estimator_nodisturbance"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CONFIGURATIONS = ((0, "qp", "3+0"), (1, "lp", "3+1"), (3, "lp", "3+3"))


def _angle_error_deg(run) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(run.time_s, dtype=float)
    states = np.asarray([state.q for state in run.state_hist], dtype=float)
    targets = np.asarray(run.target_hist, dtype=float)
    dots = np.clip(np.abs(np.sum(states * targets, axis=1)), 0.0, 1.0)
    return time, np.rad2deg(2.0 * np.arccos(dots))


def _latest_sim(number_rw: int, allocator: str) -> Path:
    pattern = f"{CAMPAIGN_PREFIX}_3mtq_{number_rw}rw_{allocator}_mc_*.sim"
    candidates = list(OUTPUT_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No saved MC result matching {pattern!r} in {OUTPUT_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    print(f"Campaign: {CAMPAIGN_PREFIX}")
    print(f"{'Configuration':<16} {'Final mean error [deg]':>23} {'Below 5°':>12} {'Mean time to 5° [min]':>24}")
    print("-" * 79)
    for number_rw, allocator, label in CONFIGURATIONS:
        results = ADCS.SimulationResults.load(_latest_sim(number_rw, allocator))
        final_errors = []
        convergence_times = []
        for run in results.runs:
            time, angle = _angle_error_deg(run)
            final_errors.append(angle[-1])
            crossing = np.flatnonzero(angle < 5.0)
            if crossing.size:
                convergence_times.append(time[crossing[0]] / 60.0)
        reached = len(convergence_times)
        total = len(results.runs)
        time_text = f"{np.mean(convergence_times):.1f}" if reached else "—"
        print(f"{label:<16} {np.mean(final_errors):>23.3g} {reached:>5}/{total:<6} {time_text:>24}")


if __name__ == "__main__":
    main()
