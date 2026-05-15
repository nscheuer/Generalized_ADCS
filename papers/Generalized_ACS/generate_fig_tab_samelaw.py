"""Paper 1 -- FIG-SAMELAW + TAB-SAMELAW (Section V-A).

"Same control law, unchanged, run across MTQ-only / 3MTQ+1RW / 3MTQ+3RW."
The framework quaternion-PD controller (MTQ_w_RW_LP) and the *scenario*
(initial attitude, rate, ECI goal -- one fixed seed) are held identical;
only the actuator configuration changes. This is the paper's core promise:
one control law, any hardware.

Emits:
  * output_data/fig_samelaw.png        -> pointing error vs time, 3 configs
  * output_data/tab_samelaw.{tex,csv,md} -> settling time + steady-state
                                            error + final error per config

Single knob: ``PAPER1_SCALE`` only sets tf/dt here (one deterministic run
per config, not Monte Carlo).
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers import metrics as M
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import (
    MC_CONFIGS, scale, simulate, make_config,
)

OUTPUT_DIR = "papers/Generalized_ACS/output_data"
SAMELAW_SEED = 7  # fixed -> identical scenario across all configs


def _make_lp(sat: Satellite, config: Dict[str, Any]):
    return MTQ_w_RW_LP(est_sat=sat, p_gain=0.00005, d_gain=0.001,
                       c_gain=0.001, h_target=np.zeros(3))


def _worker(config: Dict[str, Any]) -> Dict[str, Any]:
    return simulate(config, _make_lp)


def main() -> None:
    s = scale()
    tf, dt = s["tf"], s["dt"]
    print(f"[SAMELAW] one run/config, fixed seed={SAMELAW_SEED}, "
          f"tf={tf}s dt={dt}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows: List[Dict[str, Any]] = []

    for cfg in MC_CONFIGS:
        # num_runs=1 via the proven runner path (sets up the worker slots);
        # fixed seed makes the scenario identical across configs.
        runner = MonteCarloRunner(
            sim_func=_worker,
            config_generator=lambda rid, c=cfg: make_config(
                rid, c, tf, dt, seed=SAMELAW_SEED),
            num_runs=1,
        )
        res = [r for r in runner.run() if r is not None]
        if not res:
            raise RuntimeError(f"SAMELAW run failed for {cfg}.")
        res = res[0]

        t, err = M.run_pointing_error(res)
        ax.plot(t, err, label=cfg, linewidth=1.6)
        rows.append({
            "config": cfg,
            "settling_time_s": M.settling_time(t, err, threshold_deg=5.0),
            "steady_state_err_deg": M.steady_state_error_deg(t, err,
                                                             last_frac=0.1),
            "final_err_deg": float(err[-1]),
        })
        print(f"  {cfg}: settle={rows[-1]['settling_time_s']}s "
              f"ss={rows[-1]['steady_state_err_deg']:.3f}deg")

    ax.axhline(5.0, ls="--", c="r", alpha=0.6, label="5 deg threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pointing error [deg]")
    ax.set_title("FIG-SAMELAW: same PD law, varying actuator config")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_samelaw.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(p, dpi=150)
    print("[FIG-SAMELAW] wrote", p)

    cols = ["config", "settling_time_s", "steady_state_err_deg",
            "final_err_deg"]
    print("[TAB-SAMELAW] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR, "tab_samelaw"),
                         columns=cols), sep="\n  ")


if __name__ == "__main__":
    main()
