"""P2.3 coast-window exploration -- how wide do the No_Goal coasts need to be
to hold each goal (small in-window average) while keeping a small visible
edge lift-up?

Single-trajectory multi-goal runs on 3+1, sweeping the coast width C (goal
width G fixed). Goals A,B,C are G s each; the two No_Goal coasts are C s each:
  A:[0,G)  coast:[G,G+C)  B:[G+C,2G+C)  coast:[2G+C,2G+2C)  C:[2G+2C,3G+2C)
(C=100,G=300 reproduces the committed scenario, tf=1000.)

Per goal window it reports:
  hold_mean   = mean pointing error over the window (small = good hold)
  edge_lift   = error in the last EDGE_FRAC of the window minus the window min
                (the visible "lift-up" before the next slew)
  acquired    = min error in window < 5 deg

  python explore_p2.3_windows.py            # sweep, print table
  P23W_RENDER=<C> python explore_p2.3_windows.py   # also render best+worst at C
"""

import os
import sys
import json
import datetime as _dt

import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ADCS as ADCS
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.orbit import Orbit
import _paper2_sim as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
CONFIG, CONV, SEC2CENT = "3+1", 5.0, TimeConstants.sec2cent
G = float(os.environ.get("P23W_GOAL", 300.0))
C_WIDTHS = [float(x) for x in os.environ.get("P23W_COASTS", "100,200,300,400").split(",")]
SEEDS = list(range(1000, 1000 + int(os.environ.get("P23W_SEEDS", "6"))))
EDGE_FRAC = 0.15


def windows_for(C):
    a0, a1 = 0.0, G
    b0, b1 = G + C, 2 * G + C
    c0, c1 = 2 * G + 2 * C, 3 * G + 2 * C
    return [("A", a0, a1), ("B", b0, b1), ("C", c0, c1)], 3 * G + 2 * C


def goallist_for(rng, C):
    A, B, Cc = (normalize(rng.standard_normal(3)) for _ in range(3))
    g = G
    timeline = {0.0: ADCS.goals.ECI_Goal(A), g: ADCS.goals.No_Goal(),
                g + C: ADCS.goals.ECI_Goal(B), 2 * g + C: ADCS.goals.No_Goal(),
                2 * g + 2 * C: ADCS.goals.ECI_Goal(Cc)}
    return ADCS.GoalList(goal_timeline=timeline, time_units="seconds",
                         start_juliantime=0.22)


def _bore(sat):
    try:
        b = sat.get_boresight(None)
    except Exception:
        b = sat.boresight
        b = b.get("default", next(iter(b.values()))) if isinstance(b, dict) else b
    return normalize(np.asarray(b, float).reshape(3))


def run_traj(C, seed, dt=1.0):
    sat = P.make_sat(CONFIG, estimated=False)
    bu = _bore(sat)
    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=sat, planner_settings=P.make_planner_settings(sat))
    rng = np.random.default_rng(seed)
    os0 = P.default_os0()
    wins, tf = windows_for(C)
    orb = Orbit(os0=os0, end_time=os0.J2000 + (tf + 5) * SEC2CENT, dt=dt,
                use_J2=True, fast=False, verbose=False)
    N = int(tf / dt)
    os_seq = [orb.get_os(J2000=os0.J2000 + k * dt * SEC2CENT) for k in range(N + 1)]
    gl = goallist_for(rng, C)
    x = P.x0(1)
    traj = ctrl.calculate_trajectory(os0.J2000, tf, x, os0, gl)
    ctrl.set_active_trajectory(traj)

    t_h, err_h = [], []
    for k in range(N):
        os_k = os_seq[k]
        ag = gl.get_active_goal(os_k.J2000, time_units="centuries")
        tgt = ag.to_ref(os_k)[0]
        e = np.nan
        if tgt.size == 4 and np.isnan(tgt[0]) and np.linalg.norm(tgt[1:4]) > 0:
            e = _angle_deg(_boresight_eci(x[3:7], bu), tgt[1:4])
        t_h.append(k * dt); err_h.append(e)
        u = ctrl.find_u(x_hat=x, sens=sat.sensor_readings(x=x, os=os_k),
                        est_sat=sat, os_hat=os_k, goal=ag)
        out = solve_ivp(fun=sat.dynamics_for_solver, t_span=(0, dt), y0=x,
                        method="RK45", args=(u, os_k, os_seq[k + 1]),
                        rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]; x[3:7] = normalize(x[3:7])
    return np.asarray(t_h), np.asarray(err_h), wins


def window_metrics(t, err, wins):
    out = {}
    for lbl, t0, t1 in wins:
        m = (t >= t0) & (t < t1)
        e = err[m]; e = e[np.isfinite(e)]
        if e.size < 5:
            out[lbl] = dict(hold_mean=np.nan, edge_lift=np.nan, acquired=False,
                            final=np.nan)
            continue
        n_edge = max(1, int(EDGE_FRAC * e.size))
        out[lbl] = dict(hold_mean=float(np.mean(e)),
                        edge_lift=float(np.mean(e[-n_edge:]) - np.min(e)),
                        acquired=bool(np.min(e) < CONV), final=float(e[-1]))
    return out


def main():
    ts = os.environ.get("P23W_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"[P23W] G={G} coasts={C_WIDTHS} seeds={SEEDS}")
    sweep = {}
    store = {}     # (C) -> list of (seed, t, err, wins, metrics, traj_mean)
    for C in C_WIDTHS:
        recs = []
        for sd in SEEDS:
            t, err, wins = run_traj(C, sd)
            mt = window_metrics(t, err, wins)
            tmean = float(np.nanmean([mt[l]["hold_mean"] for l, _, _ in wins]))
            recs.append((sd, t, err, wins, mt, tmean))
        store[C] = recs
        # aggregate over seeds & goals
        hm = np.nanmean([mt[l]["hold_mean"] for _, _, _, _, mt, _ in recs
                         for l, _, _ in wins])
        el = np.nanmean([mt[l]["edge_lift"] for _, _, _, _, mt, _ in recs
                         for l, _, _ in wins])
        acq = 100 * np.mean([mt[l]["acquired"] for _, _, _, _, mt, _ in recs
                             for l, _, _ in wins])
        # per-goal hold means (A,B,C)
        per = {l: float(np.nanmean([mt[l]["hold_mean"] for _, _, _, _, mt, _ in recs]))
               for l, _, _ in wins}
        sweep[C] = dict(hold_mean=float(hm), edge_lift=float(el),
                        acquired_pct=float(acq), tf=windows_for(C)[1], per_goal=per)
        print(f"  C={C:4.0f}s (tf={sweep[C]['tf']:.0f}): hold_mean {hm:6.2f} deg  "
              f"edge_lift {el:6.2f} deg  acquired {acq:5.0f}%  "
              f"per-goal A/B/C {per['A']:.1f}/{per['B']:.1f}/{per['C']:.1f}")

    json.dump({"G": G, "edge_frac": EDGE_FRAC, "seeds": SEEDS, "sweep": sweep},
              open(f"{OUT}/P2.3_window_sweep_{ts}.json", "w"), indent=2)
    print(f"[P23W] wrote {OUT}/P2.3_window_sweep_{ts}.json")

    # optional render of best + worst trajectory at a chosen coast width
    render_C = os.environ.get("P23W_RENDER")
    if render_C is not None:
        C = float(render_C)
        recs = store[C]
        recs_sorted = sorted(recs, key=lambda r: r[5])
        best, worst = recs_sorted[0], recs_sorted[-1]
        wins = best[3]
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, rec, name in ((axs[0], best, "best"), (axs[1], worst, "worst")):
            sd, t, err, wins, mt, tmean = rec
            ax.plot(t, err, lw=1.5, color="C0")
            ax.axhline(CONV, ls=":", c="k", lw=0.8)
            for _, t0, t1 in wins:
                ax.axvspan(t0, t1, color="C1", alpha=0.07)
            for lbl, t0, t1 in wins:
                ax.text((t0 + t1) / 2, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 50,
                        f"goal {lbl}", ha="center", fontsize=9, color="C3")
            ax.set_yscale("log"); ax.set_xlabel("time [s]")
            ax.set_title(f"{name} (seed {sd}, traj-mean {tmean:.1f} deg)")
            ax.grid(True, which="both", alpha=0.3)
        axs[0].set_ylabel("pointing error [deg]")
        fig.suptitle(f"P2.3 multi-goal, coast C={C:.0f}s (shaded = goal windows)")
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_multigoal_bestworst_C{int(C)}.png", dpi=150)
        fig.savefig(f"{OUT}/fig_multigoal_bestworst_C{int(C)}.pdf")
        plt.close(fig)
        print(f"[P23W] rendered best/worst at C={C}: fig_multigoal_bestworst_C{int(C)}.*")


if __name__ == "__main__":
    main()
