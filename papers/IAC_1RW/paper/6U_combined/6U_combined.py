"""Run the 24 no-estimator 6U Monte Carlo campaigns (10 trials each)."""

import os
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PAPER_DIR.parents[3]

CAMPAIGN_DIRS = (
    "6U_noestimator_nodisturbance",
    "6U_noestimator_disturbance",
)
ARCHITECTURES = (0, 1, 3)
ALLOCATORS = ("lp", "qp")
GOAL_TYPES = ("quaternion", "vector")

# A campaign tuple is (runner, architecture, allocator, goal type).  Keeping
# this Cartesian product explicit makes it difficult to silently omit one of
# the 2 disturbance × 2 goal × 2 allocator × 3 architecture no-estimator cases.
RUNS = [
    (PAPER_DIR / campaign_dir / "run_mc.py", architecture, allocator, goal_type)
    for campaign_dir in CAMPAIGN_DIRS
    for architecture in ARCHITECTURES
    for allocator in ALLOCATORS
    for goal_type in GOAL_TYPES
]


def main() -> None:
    environment = os.environ.copy()
    # Prevent each campaign from waiting for a plot window before continuing.
    environment.setdefault("MPLBACKEND", "Agg")

    for index, (run_file, architecture, allocator, goal_type) in enumerate(RUNS, start=1):
        description = f"{run_file.parent.name}: 3+{architecture}, {allocator.upper()}, {goal_type}"
        print(f"[{index}/{len(RUNS)}] Starting {description}", flush=True)
        subprocess.run(
            [
                sys.executable, str(run_file),
                "--architecture", str(architecture),
                "--allocator", allocator,
                "--goal-type", goal_type,
                "--runs", "10",
            ],
            cwd=REPO_DIR,
            env=environment,
            check=True,
        )
        print(f"[{index}/{len(RUNS)}] Finished {description}", flush=True)


if __name__ == "__main__":
    main()
