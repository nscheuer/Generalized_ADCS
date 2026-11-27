"""Shared command-line protocol for the per-tool runners.

Every runner is executed in its own virtual environment by ``run_all.py`` and
prints one JSON document on stdout:

    {"tool": ..., "variant": ..., "version": ..., "scenario": ..., "steps": N,
     "samples_s": [...], "final": {"r": [...], "v": [...], "dcm_bi": [[...]],
     "w": [...]}, "notes": [...]}

Timing covers only the propagation (including each tool's per-step state
recording). Object construction and a short warm-up run are excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scenario as S  # noqa: E402


def parse_args(variants=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=sorted(S.SCENARIOS), default="coupled")
    p.add_argument("--steps", type=int, default=S.steps_for(1.0))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=120.0,
                   help="stop repeating once one run takes longer than this (s)")
    p.add_argument("--warmup-steps", type=int, default=64)
    if variants:
        p.add_argument("--variant", choices=variants, required=True)
    return p.parse_args()


def timed_runs(make_run, args):
    """Warm up, then time ``repeats`` runs. ``make_run(steps)`` builds a fresh
    simulation (untimed) and returns a zero-argument callable that propagates
    it and returns the final state dict."""
    make_run(args.warmup_steps)()
    samples, final = [], None
    # At least ``repeats`` runs; fast tools keep going until 2 s of samples
    # have been collected (max 20 runs) so the median is stable. A run longer
    # than ``time_limit`` stops further repeats.
    while len(samples) < args.repeats or (sum(samples) < MIN_TOTAL_S and len(samples) < MAX_RUNS):
        run = make_run(args.steps)
        gc.collect()
        start = time.perf_counter()
        final = run()
        samples.append(time.perf_counter() - start)
        if samples[-1] > args.time_limit:
            break
    return samples, final


MIN_TOTAL_S = 2.0
MAX_RUNS = 20


def emit(**payload):
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
