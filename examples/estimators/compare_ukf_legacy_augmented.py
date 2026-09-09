"""Direct legacy UAKF versus AugmentedUKF comparison runner.

The paired scripts are intentionally left unchanged.  This runner captures their
simulation results, resets the RNG before each run, and reports comparable
runtime, attitude, and angular-rate convergence metrics.
"""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ADCS
PAIRS = (
    ("example_ukf_basic", "example_ukf_basic_augmented"),
    ("example_ukf_bias", "example_ukf_bias_augmented"),
    ("example_ukf_reaction_wheels", "example_ukf_reaction_wheels_augmented"),
    ("03_simple_estimation", "03_simple_estimation_augmented"),
    ("04_complex_estimation", "04_complex_estimation_augmented"),
)


def _angle(q_true: np.ndarray, q_est: np.ndarray) -> float:
    dot = abs(float(np.dot(q_true, q_est)))
    return 2.0 * np.arccos(np.clip(dot / (np.linalg.norm(q_true) * np.linalg.norm(q_est)), 0.0, 1.0))


def _run(path: Path):
    captured = []
    simulate = ADCS.simulate

    def capture(*args, **kwargs):
        result = simulate(*args, **kwargs)
        captured.append(result)
        return result

    ADCS.simulate = capture
    np.random.seed(20260909)
    started = time.perf_counter()
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        ADCS.simulate = simulate
        import matplotlib.pyplot as plt
        plt.close("all")
    if not captured:
        raise RuntimeError(f"{path} did not call ADCS.simulate")
    return captured[-1].first(), time.perf_counter() - started


def _metrics(run):
    attitude = np.array([_angle(t.q, e.q) for t, e in zip(run.state_hist, run.est_state_hist)])
    rate = np.array([np.linalg.norm(t.w - e.w) for t, e in zip(run.state_hist, run.est_state_hist)])
    threshold = max(np.deg2rad(1.0), 1.05 * attitude[-1])
    settled = np.flatnonzero(np.maximum.accumulate(attitude[::-1])[::-1] <= threshold)
    return {
        "attitude_final_deg": np.rad2deg(attitude[-1]),
        "attitude_rms_deg": np.rad2deg(np.sqrt(np.mean(attitude**2))),
        "rate_final": rate[-1],
        "rate_rms": np.sqrt(np.mean(rate**2)),
        "settled_at_s": run.time_s[settled[0]] if len(settled) else float("nan"),
    }


def main() -> None:
    for legacy_name, augmented_name in PAIRS:
        print(f"\n=== {legacy_name} vs {augmented_name} ===")
        results = []
        for name in (legacy_name, augmented_name):
            path = ROOT / "examples" / ("tutorials" if name[:2].isdigit() else "estimators") / f"{name}.py"
            try:
                run, elapsed = _run(path)
                metrics = _metrics(run)
                metrics["elapsed_s"] = elapsed
                results.append((name, metrics))
                print(name, ", ".join(f"{key}={value:.6g}" for key, value in metrics.items()))
            except Exception as error:
                print(f"{name}: FAILED: {type(error).__name__}: {error}")
        if len(results) == 2:
            old = results[0][1]
            new = results[1][1]
            print(f"delta(new-old): runtime={new['elapsed_s'] - old['elapsed_s']:+.6g}s, "
                  f"attitude_rms={new['attitude_rms_deg'] - old['attitude_rms_deg']:+.6g}deg, "
                  f"rate_rms={new['rate_rms'] - old['rate_rms']:+.6g}")


if __name__ == "__main__":
    main()
