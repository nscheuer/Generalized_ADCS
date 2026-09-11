"""Run all nine 6U Monte Carlo campaigns consecutively."""

import os
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PAPER_DIR.parents[3]

RUNS = [
    PAPER_DIR / "6U_estimator_nodisturbance" / "run_mc_3mtq_0rw_qp.py",
    PAPER_DIR / "6U_estimator_nodisturbance" / "run_mc_3mtq_1rw_lp.py",
    PAPER_DIR / "6U_estimator_nodisturbance" / "run_mc_3mtq_3rw_lp.py",
    PAPER_DIR / "6U_estimator_disturbance" / "run_mc_3mtq_0rw_qp.py",
    PAPER_DIR / "6U_estimator_disturbance" / "run_mc_3mtq_1rw_lp.py",
    PAPER_DIR / "6U_estimator_disturbance" / "run_mc_3mtq_3rw_lp.py",
    PAPER_DIR / "6U_noestimator_nodisturbance" / "run_mc_3mtq_0rw_qp.py",
    PAPER_DIR / "6U_noestimator_nodisturbance" / "run_mc_3mtq_1rw_lp.py",
    PAPER_DIR / "6U_noestimator_nodisturbance" / "run_mc_3mtq_3rw_lp.py",
]


def main() -> None:
    environment = os.environ.copy()
    # Prevent each campaign from waiting for a plot window before continuing.
    environment.setdefault("MPLBACKEND", "Agg")

    for index, run_file in enumerate(RUNS, start=1):
        print(f"[{index}/{len(RUNS)}] Starting {run_file}", flush=True)
        subprocess.run(
            [sys.executable, str(run_file)],
            cwd=REPO_DIR,
            env=environment,
            check=True,
        )
        print(f"[{index}/{len(RUNS)}] Finished {run_file}", flush=True)


if __name__ == "__main__":
    main()
