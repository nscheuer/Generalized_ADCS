"""Generate the final-angle MC summary without retaining MC files in memory.

Each simulation file is processed in a separate child process.  This keeps the
parent process small and lets the operating system reclaim all simulation data
after each file has been summarized.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


MAX_THREADS = 4
CAMPAIGN_PREFIX = "6u_noestimator_disturbance"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SUMMARY_PATH = OUTPUT_DIR / f"{CAMPAIGN_PREFIX}_final_angle_error_bars_mc_summary.json"
FILE_PATTERN = re.compile(
    rf"^{re.escape(CAMPAIGN_PREFIX)}_(quaternion|vector)_3mtq_"
    r"(\d+)rw_(lp|qp)_mc_100_.+\.sim$"
)


def _thread_limited_environment() -> dict[str, str]:
    """Return an environment that limits common numerical libraries to 4 threads."""
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[variable] = str(MAX_THREADS)
    return environment


def _final_angle_errors(path: Path) -> dict:
    """Load one simulation file and return only its final-error statistics."""
    import gc

    import numpy as np

    repo_root = Path(__file__).resolve().parents[4]
    paper_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(paper_dir))

    import ADCS
    from ADCS.helpers.math_helpers import rot_mat

    def final_angle_error_deg(run) -> float:
        q_final = np.asarray(run.state_hist[-1].q, dtype=float)
        target_final = np.asarray(run.target_hist[-1], dtype=float)
        if np.isnan(target_final[0]):
            goal_eci = target_final[1:]
            boresight_eci = rot_mat(q_final) @ np.array([0.0, 0.0, 1.0])
            cosine = np.clip(np.dot(boresight_eci, goal_eci), -1.0, 1.0)
            return float(np.rad2deg(np.arccos(cosine)))
        cosine = np.clip(abs(np.dot(q_final, target_final)), 0.0, 1.0)
        return float(np.rad2deg(2.0 * np.arccos(cosine)))

    results = None
    final_errors = None
    try:
        results = ADCS.SimulationResults.load(path)
        final_errors = np.asarray(
            [final_angle_error_deg(run) for run in results.runs],
            dtype=float,
        )
        if final_errors.size == 0:
            raise ValueError(f"Simulation file contains no runs: {path}")
        sigma = float(np.std(final_errors, ddof=1)) if final_errors.size > 1 else 0.0
        p10, median, p90 = np.percentile(final_errors, (10, 50, 90))
        return {
            "runs": int(final_errors.size),
            "final_angle_error_mean_deg": float(np.mean(final_errors)),
            "final_angle_error_std_dev_deg": sigma,
            "one_sigma_lower_deg": max(0.0, float(np.mean(final_errors)) - sigma),
            "one_sigma_upper_deg": float(np.mean(final_errors)) + sigma,
            "final_angle_error_median_deg": float(median),
            "final_angle_error_p10_deg": float(p10),
            "final_angle_error_p90_deg": float(p90),
        }
    finally:
        # Explicitly release the large object graph before the child exits.
        del final_errors
        del results
        gc.collect()


def _run_child(path: Path) -> int:
    summary = _final_angle_errors(path)
    print(f"SUMMARY_JSON {json.dumps(summary)}", flush=True)
    return 0


def _latest_simulations() -> list[tuple[Path, str, str, int]]:
    """Find one latest MC100 file for every available campaign case."""
    latest: dict[tuple[str, str, int], Path] = {}
    for path in OUTPUT_DIR.glob(f"{CAMPAIGN_PREFIX}_*_mc_100_*.sim"):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        goal_type, number_rw, allocator = match.groups()
        key = (goal_type, allocator, int(number_rw))
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path

    if not latest:
        raise FileNotFoundError(f"No MC100 simulation files found in {OUTPUT_DIR}")

    return sorted(
        (
            path,
            goal_type,
            allocator,
            number_rw,
        )
        for (goal_type, allocator, number_rw), path in latest.items()
    )


def _summarize_one(path: Path) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--single-file", str(path)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        env=_thread_limited_environment(),
        cwd=Path(__file__).resolve().parents[4],
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("SUMMARY_JSON "):
            return json.loads(line.removeprefix("SUMMARY_JSON "))
    raise RuntimeError(f"Child process returned no summary for {path}")


def _write_summary() -> Path:
    rows = []
    simulations = _latest_simulations()
    print(f"Found {len(simulations)} MC100 simulation files.", flush=True)
    for index, (path, goal_type, allocator, number_rw) in enumerate(simulations, start=1):
        print(f"[{index}/{len(simulations)}] Loading {path.name}", flush=True)
        row = _summarize_one(path)
        row.update({
            "goal_type": goal_type,
            "architecture": f"3+{number_rw}",
            "number_reaction_wheels": number_rw,
            "allocator": allocator.upper(),
            "source_simulation": path.name,
        })
        rows.append(row)

    payload = {
        "metric": "final angle error",
        "units": "deg",
        "one_sigma_bounds": (
            "mean ± sample standard deviation; lower bound is clipped to zero degrees"
        ),
        "percentile_bounds": "10th to 90th percentile of final angle errors",
        "results": rows,
    }
    temporary_path = SUMMARY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(SUMMARY_PATH)
    return SUMMARY_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_file is not None:
        return _run_child(args.single_file)
    print(f"Saved {_write_summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
