#!/usr/bin/env python
"""Easy-run dispatcher for all SmallSat-2026 paper data-gen tests.

Each paper figure/table has a generator under papers/<paper>/; this runs any of
them at a chosen scale without fiddling with environment variables.

Usage
-----
  python papers/run_paper_tests.py --list
  python papers/run_paper_tests.py p2.4            # fast smoke (default)
  python papers/run_paper_tests.py p1.3 --paper    # full published scale
  python papers/run_paper_tests.py all --fast      # smoke every test

Scales: --fast (quick sanity, few trials / short horizon) | --paper (published).
Each test is run as a subprocess with the venv's Python, so dotted filenames and
worker-spawn both behave exactly as a direct invocation.
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GA = "papers/Generalized_ACS"
PL = "papers/Planner"

# task -> (script, scale_env_var or None, {fast_extra_env}, {paper_extra_env})
TESTS = {
    # Paper 1 (Generalized_ACS)
    "p1.2": (f"{GA}/generate_p1.2_same_pd.py", "PAPER1_SCALE", {}, {}),
    "p1.3": (f"{GA}/generate_p1.3_difflaw_mc.py", "PAPER1_SCALE",
             {"P13_TF": "300"}, {"P13_TF": "2000"}),
    "p1.4": (f"{GA}/generate_p1.4_digest.py", None, {}, {}),   # derived, no compute
    "p1.5": (f"{GA}/generate_fig_failure.py", "PAPER1_SCALE", {}, {}),
    # Paper 2 (Planner)
    "p2.3": (f"{PL}/generate_p2.3_multigoal.py", "PAPER2_SCALE",
             {"P23_CONFIGS": "3+1"}, {"P23_CONFIGS": "3+1,3+3"}),
    "p2.4": (f"{PL}/generate_p2.4_tvlqr_vs_mpc.py", "PAPER2_SCALE", {}, {}),
    "p2.4-tune": (f"{PL}/generate_p2.4_tuning.py", None,
                  {"P24T_TRIALS": "6"}, {"P24T_TRIALS": "25"}),
    "p2.6": (f"{PL}/generate_p2.6_replan.py", None, {}, {}),   # fixed single demo
    "p2.7": (f"{PL}/generate_p2.7_sensitivity.py", "PAPER2_SCALE",
             {"P27_TRIALS": "2"}, {"P27_TRIALS": "6"}),
    "p2.8": (f"{PL}/generate_p2.8_mismatch.py", None,
             {"P28_TF": "200"}, {"P28_TF": "1000"}),
}


def run_one(task, paper_scale):
    script, scale_var, fast_env, paper_env = TESTS[task]
    env = dict(os.environ)
    if scale_var:
        env[scale_var] = "paper" if paper_scale else "fast"
    env.update(paper_env if paper_scale else fast_env)
    print(f"\n=== {task} ({'paper' if paper_scale else 'fast'}) -> {script} ===",
          flush=True)
    return subprocess.call([sys.executable, script], cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser(description="Run paper data-gen tests.")
    ap.add_argument("task", nargs="?", help="task id (e.g. p2.4) or 'all'")
    ap.add_argument("--paper", action="store_true", help="full published scale")
    ap.add_argument("--fast", action="store_true", help="quick smoke (default)")
    ap.add_argument("--list", action="store_true", help="list tasks")
    a = ap.parse_args()

    if a.list or not a.task:
        print("Available tests (run: python papers/run_paper_tests.py <task> [--fast|--paper]):")
        for k, (s, *_ ) in TESTS.items():
            print(f"  {k:11s} {s}")
        print("  all          (every test)")
        return 0

    paper = a.paper and not a.fast
    tasks = list(TESTS) if a.task == "all" else [a.task]
    bad = [t for t in tasks if t not in TESTS]
    if bad:
        print(f"unknown task(s): {bad}; use --list", file=sys.stderr)
        return 2
    rc = 0
    for t in tasks:
        rc |= run_one(t, paper)
    return rc


if __name__ == "__main__":
    sys.exit(main())
