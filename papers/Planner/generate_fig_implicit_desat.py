"""Paper 2 -- FIG (IV-A "It Works"): pointing with implicit desaturation.

A 3MTQ+1RW spacecraft starts with non-zero reaction-wheel momentum and is
given a fixed pointing goal. The feasibility-aware planner drives pointing
error to zero *and* bleeds the wheel momentum down within the same
trajectory -- no dedicated desaturation mode or scheduling. (Paper Sec IV-A.)

Emits:
  * output/fig_implicit_desat.png        -> pointing error + |h_RW| vs time
  * output/tab_implicit_desat.{tex,csv,md} -> params + final error / momentum

Single deterministic run (cheapest planner smoke). Knob: ``PAPER2_SCALE``.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import ADCS as ADCS
from ADCS.helpers import metrics as M
from papers.Planner import _paper2_sim as P2

OUTPUT_DIR = "papers/Planner/output"
CONFIG = "3+1"
H0 = 0.005           # initial RW momentum to desaturate [N m s]
GOAL_VEC = ADCS.helpers.normalize(np.array([1.0, 1.0, 1.0]))


def main() -> None:
    s = P2.scale()
    tf, dt = s["tf"], s["dt"]
    # state = [w(3), q(4)=identity, h(1)=H0]
    x = np.concatenate([np.zeros(3), [1.0, 0, 0, 0], [H0]])
    print(f"[IV-A] {CONFIG}, h0={H0} N m s, tf={tf}s dt={dt}s (1 run)")

    results = P2.run(CONFIG, goal=ADCS.goals.ECI_Goal(GOAL_VEC),
                     num_runs=1, tf=tf, dt=dt, x=x)
    res = M.from_simulation_results(results)[0]
    t, err = M.run_pointing_error(res)
    h = M.rw_momentum(res)
    h_norm = np.linalg.norm(h, axis=1)

    rows = [{
        "config": CONFIG,
        "h0_Nms": H0,
        "tf_s": tf,
        "final_err_deg": float(err[-1]),
        "final_h_Nms": float(h_norm[-1]),
        "settle_s": M.settling_time(t, err, threshold_deg=5.0),
        "h_reduced_pct": float(100.0 * (1.0 - h_norm[-1] / (h_norm[0] + 1e-18))),
    }]
    print(f"  final err={rows[0]['final_err_deg']:.3f} deg | "
          f"|h| {h_norm[0]:.4f}->{h_norm[-1]:.4f} "
          f"({rows[0]['h_reduced_pct']:.0f}% reduced)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[TAB IV-A] wrote",
          *M.write_table(rows, os.path.join(OUTPUT_DIR,
                                            "tab_implicit_desat"),
                         columns=list(rows[0])), sep="\n  ")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(t, err, color="tab:blue", lw=1.6, label="pointing error")
    ax1.axhline(5.0, ls="--", c="r", alpha=0.5)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Pointing error [deg]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, h_norm, color="tab:green", lw=1.6, label="|h_RW|")
    ax2.set_ylabel("RW momentum |h| [N m s]", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_title("FIG IV-A: simultaneous pointing + implicit desaturation")
    ax1.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUTPUT_DIR, "fig_implicit_desat.png")
    fig.savefig(p, dpi=150)
    print("[FIG IV-A] wrote", p)


if __name__ == "__main__":
    main()
