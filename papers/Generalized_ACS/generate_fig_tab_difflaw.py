"""Paper 1 -- FIG-DIFFLAW + TAB-DIFFLAW (Section V-B).

Three literature control laws run through the framework on the *same*
hardware and the *same* scenario (one fixed seed); the only variable is the
control law itself:

  * Wie        -- quaternion PD via the framework LP allocation
  * Lovera     -- magnetic PD with adaptive projection
  * Wisniewski -- LTV sliding-mode magnetic control

All three share the framework's goal formulation / compensation /
allocation, so this is the apples-to-apples comparison the paper motivates
(previously each law needed its own actuator interface re-implementation).

Common config: 3MTQ+1RW (MTQ present for the magnetic laws). Emits:
  * output_data/fig_difflaw.png
  * output_data/tab_difflaw.{tex,csv,md}

Single knob: ``PAPER1_SCALE`` sets tf/dt (one deterministic run per law).
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.controller import MTQ_w_RW_LP, MTQ_Lovera, MTQ_Wisniewski
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.helpers import metrics as M
from ADCS.mc.monte_carlo_runner import MonteCarloRunner

from papers.Generalized_ACS._paper1_sim import scale, simulate, make_config

OUTPUT_DIR = "papers/Generalized_ACS/output_data"
COMMON_CONFIG = "3MTQ+1RW"
DIFFLAW_SEED = 7
LAWS = ["Wie", "Lovera", "Wisniewski"]


def _make_controller(sat: Satellite, config: Dict[str, Any]):
    """Dispatch on ``config['law']`` (module-level -> picklable for spawn)."""
    law = config["law"]
    if law == "Wie":
        return MTQ_w_RW_LP(est_sat=sat, p_gain=0.00005, d_gain=0.001,
                           c_gain=0.001, h_target=np.zeros(3))
    if law == "Lovera":
        return MTQ_Lovera(est_sat=sat, p_gain=0.001, d_gain=0.005, eps=1.0)
    if law == "Wisniewski":
        return MTQ_Wisniewski(est_sat=sat,
                              lambda_s=np.diag([0.01, 0.01, 0.01]),
                              lambda_q=np.diag([0.002, 0.002, 0.002]))
    raise ValueError(f"unknown law {law!r}")


def _worker(config: Dict[str, Any]) -> Dict[str, Any]:
    return simulate(config, _make_controller)


def main() -> None:
    s = scale()
    tf, dt = s["tf"], s["dt"]
    print(f"[DIFFLAW] config={COMMON_CONFIG} seed={DIFFLAW_SEED} "
          f"tf={tf}s dt={dt}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows: List[Dict[str, Any]] = []

    for law in LAWS:
        def gen(rid, _law=law):
            c = make_config(rid, COMMON_CONFIG, tf, dt, seed=DIFFLAW_SEED)
            c["law"] = _law
            return c

        runner = MonteCarloRunner(sim_func=_worker, config_generator=gen,
                                  num_runs=1)
        res = [r for r in runner.run() if r is not None]
        if not res:
            raise RuntimeError(f"DIFFLAW run failed for law={law}.")
        t, err = M.run_pointing_error(res[0])
        ax.plot(t, err, label=law, linewidth=1.6)
        rows.append({
            "law": law,
            "settling_time_s": M.settling_time(t, err, threshold_deg=5.0),
            "steady_state_err_deg": M.steady_state_error_deg(t, err, 0.1),
            "final_err_deg": float(err[-1]),
        })
        print(f"  {law}: settle={rows[-1]['settling_time_s']}s "
              f"ss={rows[-1]['steady_state_err_deg']:.3f}deg")

    ax.axhline(5.0, ls="--", c="r", alpha=0.6, label="5 deg threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pointing error [deg]")
    ax.set_title(f"FIG-DIFFLAW: 3 laws, same hardware ({COMMON_CONFIG})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p = os.path.join(OUTPUT_DIR, "fig_difflaw.png")
    fig.savefig(p, dpi=150)
    print("[FIG-DIFFLAW] wrote", p)

    cols = ["law", "settling_time_s", "steady_state_err_deg", "final_err_deg"]
    print("[TAB-DIFFLAW] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR, "tab_difflaw"),
                         columns=cols), sep="\n  ")


if __name__ == "__main__":
    main()
