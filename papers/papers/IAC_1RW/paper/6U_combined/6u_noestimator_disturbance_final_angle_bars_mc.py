"""Run the 12 disturbed no-estimator 6U MC campaigns and make the bar plot."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PAPER_DIR.parents[3]
CAMPAIGN_DIR = PAPER_DIR / "6U_noestimator_disturbance"
RUNNER = CAMPAIGN_DIR / "run_mc.py"
PLOTTER = CAMPAIGN_DIR / "plot_final_angle_bars.py"

ARCHITECTURES = (0, 1, 3)
ALLOCATORS = ("lp", "qp")
GOAL_TYPES = ("quaternion", "vector")
NUM_RUNS = 100
WORKERS = 20


def main() -> None:
    environment = os.environ.copy()
    # The MC helper also writes diagnostic figures. Keep those child processes
    # non-interactive so all 12 campaigns can run unattended.
    environment.setdefault("MPLBACKEND", "Agg")

    campaigns = [
        (architecture, allocator, goal_type)
        for architecture in ARCHITECTURES
        for allocator in ALLOCATORS
        for goal_type in GOAL_TYPES
    ]
    for index, (architecture, allocator, goal_type) in enumerate(campaigns, start=1):
        description = f"3+{architecture}, {allocator.upper()}, {goal_type}"
        print(f"[{index}/{len(campaigns)}] Starting {description}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--architecture", str(architecture),
                "--allocator", allocator,
                "--goal-type", goal_type,
                "--runs", str(NUM_RUNS),
                "--workers", str(WORKERS),
            ],
            cwd=REPO_DIR,
            env=environment,
            check=True,
        )
        print(f"[{index}/{len(campaigns)}] Finished {description}", flush=True)

    print("All campaigns finished; generating the final-angle bar plot.", flush=True)
    subprocess.run(
        [sys.executable, str(PLOTTER)],
        cwd=REPO_DIR,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
