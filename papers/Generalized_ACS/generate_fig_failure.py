"""Paper 1 -- FIG-FAILURE + FIG-FAILURE-U (Section V-D).

Automatic reallocation under actuator failure. A 3MTQ+3RW spacecraft points
at a fixed ECI target; mid-run one reaction wheel fails (its torque bound is
set to zero). The control law is unchanged -- only the allocation layer sees
the new actuator set -- so pointing degrades gracefully and recovers with
the remaining actuators.

The mid-run failure is injected through the ``config['fail']`` hook already
in ``_paper1_sim.simulate`` (no framework modification).

Emits:
  * output_data/fig_failure.png    -> pointing error: failure, transient,
                                      recovery (failure time marked)
  * output_data/fig_failure_u.png  -> actuator commands showing reallocation
  * output_data/tab_failure.{tex,csv,md} -> steady-state error pre/post
                                            failure + recovery time

Single knob: ``PAPER1_SCALE`` sets tf/dt (one deterministic run).
"""

import os
import sys
from typing import Any, Dict

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers import metrics as M
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import scale, simulate, make_config

OUTPUT_DIR = "papers/Generalized_ACS/output_data"
CONFIG = "3MTQ+3RW"          # acts order: MTQ 0,1,2 then RW 3,4,5
FAILED_ACT_INDEX = 3         # first reaction wheel
FAILURE_SEED = 7


def _make_lp(sat: Satellite, config: Dict[str, Any]):
    return MTQ_w_RW_LP(est_sat=sat, p_gain=0.00005, d_gain=0.001,
                       c_gain=0.001, h_target=np.zeros(3))


def _worker(config: Dict[str, Any]) -> Dict[str, Any]:
    return simulate(config, _make_lp)


def main() -> None:
    s = scale()
    tf, dt = s["tf"], s["dt"]
    t_fail = tf * 0.5
    print(f"[FAILURE] {CONFIG}, RW act#{FAILED_ACT_INDEX} fails at "
          f"t={t_fail}s, tf={tf}s dt={dt}s")

    def gen(rid):
        c = make_config(rid, CONFIG, tf, dt, seed=FAILURE_SEED)
        c["fail"] = {"t": t_fail, "act_index": FAILED_ACT_INDEX}
        return c

    runner = MonteCarloRunner(sim_func=_worker, config_generator=gen,
                              num_runs=1)
    res = [r for r in runner.run() if r is not None]
    if not res:
        raise RuntimeError("FAILURE run produced no result.")
    res = res[0]

    t, err = M.run_pointing_error(res)
    u = np.asarray(res["u"], float)
    pre = t < t_fail
    post = t >= t_fail
    # recovery time = settling (<5 deg, stays) measured on the post-failure
    # segment, reported relative to the failure instant.
    rec = M.settling_time(t[post], err[post], threshold_deg=5.0)
    rec_rel = float(rec - t_fail) if np.isfinite(rec) else float("nan")
    rows = [{
        "config": CONFIG,
        "ss_err_pre_deg": M.steady_state_error_deg(t[pre], err[pre], 0.2)
        if pre.any() else float("nan"),
        "max_err_post_deg": float(np.nanmax(err[post])) if post.any()
        else float("nan"),
        "ss_err_post_deg": M.steady_state_error_deg(t[post], err[post], 0.2)
        if post.any() else float("nan"),
        "recovery_time_s": rec_rel,
    }]
    print(f"  pre ss={rows[0]['ss_err_pre_deg']:.3f} deg | post max="
          f"{rows[0]['max_err_post_deg']:.3f} deg | recovery="
          f"{rows[0]['recovery_time_s']} s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[TAB-FAILURE] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR, "tab_failure"),
                         columns=list(rows[0])), sep="\n  ")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, err, lw=1.6, color="tab:blue")
    ax.axvline(t_fail, ls="--", c="k", alpha=0.7, label="RW failure")
    ax.axhline(5.0, ls="--", c="r", alpha=0.5, label="5 deg threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pointing error [deg]")
    ax.set_title(f"FIG-FAILURE: {CONFIG}, RW fails at t={t_fail:.0f}s")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig1.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "fig_failure.png")
    fig1.savefig(p1, dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    for j in range(u.shape[1]):
        ax2.plot(t, u[:, j], lw=1.0,
                 label=("MTQ" if j < 3 else "RW") + f"{j}")
    ax2.axvline(t_fail, ls="--", c="k", alpha=0.7,
                label=f"RW#{FAILED_ACT_INDEX} fails")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Actuator command")
    ax2.set_title("FIG-FAILURE-U: reallocation after failure")
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, ls="--", alpha=0.5)
    fig2.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "fig_failure_u.png")
    fig2.savefig(p2, dpi=150)
    print("[FIG-FAILURE] wrote", p1, "\n[FIG-FAILURE-U] wrote", p2)


if __name__ == "__main__":
    main()
