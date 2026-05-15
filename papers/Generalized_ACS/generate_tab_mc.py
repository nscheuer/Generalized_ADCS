"""Paper 1 -- TAB-MC + FIG-MC.

Monte Carlo validation across actuator configurations (MTQ-only, 3MTQ+1RW,
3MTQ+3RW): per-config convergence statistics for the framework controller.

Gap-fill: the existing ``generate_mc_*`` scripts emit plots only; this reuses
the shared sim core + the metrics module to emit a numeric table + figure:

  * output_data/tab_mc.{tex,csv,md}   -> TAB-MC
  * output_data/fig_mc.png            -> FIG-MC (final-error distribution)
  * output_data/<CONFIG>_LP_mc_<N>.*  -> raw runs (reusable)

Single knob: ``PAPER1_SCALE=fast`` (default, smoke) | ``paper`` (published).
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers.save_and_load.save_and_load import save_data
from ADCS.helpers import metrics as M
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import (
    MC_CONFIGS, scale, simulate, make_config,
)

OUTPUT_DIR = "papers/Generalized_ACS/output_data"


def _make_lp(sat: Satellite, config: Dict[str, Any]):
    # h_target is a length-3 body-axis RW-momentum target in this framework,
    # independent of wheel count (matches every existing generate_mc_*).
    return MTQ_w_RW_LP(est_sat=sat, p_gain=0.00005, d_gain=0.001,
                       c_gain=0.001, h_target=np.zeros(3))


def _worker(config: Dict[str, Any]) -> Dict[str, Any]:
    return simulate(config, _make_lp)


def main() -> None:
    s = scale()
    n, tf, dt = s["num_runs"], s["tf"], s["dt"]
    print(f"[TAB-MC] num_runs={n} tf={tf}s dt={dt}s")

    rows: List[Dict[str, Any]] = []
    finals: Dict[str, np.ndarray] = {}
    for cfg in MC_CONFIGS:
        runner = MonteCarloRunner(
            sim_func=_worker,
            config_generator=lambda rid, c=cfg: make_config(rid, c, tf, dt),
            num_runs=n,
        )
        raw = runner.run()
        results = [r for r in raw if r is not None]
        if not results:
            raise RuntimeError(f"All {len(raw)} runs failed for {cfg}.")
        if len(results) != len(raw):
            print(f"  [warn] {cfg}: {len(raw) - len(results)} run(s) failed")
        save_data(f"{cfg}_LP_mc_{len(results)}", results, out_dir=OUTPUT_DIR)

        st = M.convergence_stats(results, threshold_deg=5.0)
        finals[cfg] = np.array([M.final_error_deg(r) for r in results])
        rows.append({
            "config": cfg, "n": st["n"],
            "pct_converged": st["pct_converged"],
            "mean_final_deg": st["mean_final"],
            "median_final_deg": st["median_final"],
            "max_final_deg": st["max_final"],
            "mean_settle_s": st["mean_settle"],
        })
        print(f"  {cfg}: {st['pct_converged']:.0f}% <5deg, "
              f"mean {st['mean_final']:.2f} deg")

    cols = ["config", "n", "pct_converged", "mean_final_deg",
            "median_final_deg", "max_final_deg", "mean_settle_s"]
    print("[TAB-MC] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR, "tab_mc"),
                         columns=cols), sep="\n  ")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([finals[c] for c in MC_CONFIGS],
               tick_labels=MC_CONFIGS, showmeans=True)
    ax.axhline(5.0, ls="--", c="r", alpha=0.6, label="5 deg threshold")
    ax.set_ylabel("Final pointing error [deg]")
    ax.set_title(f"FIG-MC: MC final-error distribution (N={n})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_mc.png")
    fig.savefig(p, dpi=150)
    print("[FIG-MC] wrote", p)


if __name__ == "__main__":
    main()
