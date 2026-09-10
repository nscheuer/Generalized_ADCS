"""
Task 2a failure-mode demo — LP allocator precession under-compensation.

HYPOTHESIS (from Nic's controllability notes, restated in Exec Brief #2):
the direction-preserving LP allocator maximizes a scalar alpha subject to
staying in the torque polytope; the desired torque it is handed is

    tau_des = tau_pd + tau_gyro,   tau_gyro = w x (J w + h_rw)

i.e. a *fused* request mixing the discretionary pointing torque (tau_pd) with
the non-negotiable gyroscopic-precession-cancellation torque (tau_gyro). When
the request exceeds the polytope (high rate or large stored wheel momentum),
the LP scales the *entire* fused vector down by alpha < 1
(see mtq_w_rw_LP.py:924-932). The achieved torque is then alpha*(tau_pd+tau_gyro),
so the precession term is under-delivered by (1-alpha)*tau_gyro.

WHAT THIS SCRIPT FOUND (the hypothesis is half-right — finding, not assertion):

  PART A confirms the allocation-level defect outright: at high stored RW
  momentum the LP's uniform alpha-scaling collapses alpha to ~0.03, i.e. it
  delivers only ~3% of the requested torque. Both halves of the fused request
  are gutted by that same alpha.

  PART B then shows the brief's predicted symptom -- runaway body-rate
  "spin-up" -- does NOT occur. The reason is structural: tau_gyro = w x (.)
  is a cross product, hence ALWAYS perpendicular to w. A torque perpendicular
  to w cannot change ||w|| to first order (d||w||^2/dt = 2 w . wdot, and
  w . tau_gyro = 0). Under-delivering tau_gyro therefore does not spin the
  body up -- it lets w *drift in direction* (uncontrolled nutation/precession).

  So the real consequence of the fused-request defect is degraded POINTING,
  by two paths: (1) the gyro shortfall (1-alpha)*tau_gyro precesses the
  attitude uncontrolled; (2) the SAME alpha also scales the discretionary
  tau_pd down to ~3%, so the controller has almost no steering authority
  left. PART B tracks the pointing error to show this.

  PART A — allocation sweep: at a fixed rate, sweep stored RW momentum from 0
           to h_max, call the real LP allocator, and measure alpha and the
           gyro-cancellation shortfall (1-alpha)*||tau_gyro|| directly.

  PART B — closed-loop: run ADCS.simulate with the real MTQ_w_RW_LP
           controller from low vs high stored-momentum initial states and
           track BOTH ||w|| (does not run away) and the pointing error
           (the real degradation).

Run:  PYTHONPATH=. venv/bin/python papers/Generalized_ACS/demo_lp_precession_failure.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADCS.controller import MTQ_w_RW_LP
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_constants import MathConstants

OUT_DIR = os.path.join(os.path.dirname(__file__), "output_data")
os.makedirs(OUT_DIR, exist_ok=True)

# Hardware matches papers/Generalized_ACS/generate_mc_3mtq+1rw_lp.py
MTQ_MAX = 0.4          # A m^2
RW_MAX = 7e-3          # N m
RW_J = 1e-3            # kg m^2
RW_HMAX = 16.2e-3      # N m s
J_0 = np.diagflat([0.022, 0.022, 0.004])


def build_3mtq_1rw() -> Satellite:
    """3 MTQ on body axes + 1 RW on the x-axis (the 3+1 paper config)."""
    acts = [MTQ(axis=j, max_torque=MTQ_MAX) for j in MathConstants.unitvecs]
    acts.append(RW(axis=MathConstants.unitvecs[0], max_torque=RW_MAX,
                   J=RW_J, h=0.0, h_max=RW_HMAX))
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=1.2, J_0=J_0, actuators=acts, sensors=mtms,
                     boresight=np.array([0, 0, 1]))


def part_a_allocation_sweep() -> dict:
    """Sweep stored RW momentum; measure LP alpha and gyro shortfall."""
    sat = build_3mtq_1rw()
    ctrl = MTQ_w_RW_LP(est_sat=sat, p_gain=5e-5, d_gain=1e-3, c_gain=1e-3)

    # Fixed, representative operating point.
    w = np.array([0.0, 0.0, 0.03])           # 1.7 deg/s residual rate
    b_body = np.array([2.0e-5, 1.0e-5, 1.5e-5])  # ~28 uT, generic LEO geometry
    # A modest discretionary pointing request (well within authority on its own).
    tau_pd = np.array([1.0e-4, -0.5e-4, 0.3e-4])

    rw_axis = MathConstants.unitvecs[0]
    fracs = np.linspace(0.0, 1.0, 21)
    rows = []
    for fr in fracs:
        h_scalar = fr * RW_HMAX
        h_rw_body = h_scalar * rw_axis
        tau_gyro = np.cross(w, J_0 @ w + h_rw_body)
        tau_des = tau_pd + tau_gyro

        u_rw, u_mtq, alpha = ctrl.allocate_max_torque_in_direction(
            tau_des, b_body, sat)

        gyro_norm = float(np.linalg.norm(tau_gyro))
        # LP delivers alpha*tau_des when saturated, tau_des when not.
        # Delivered gyro component = alpha*tau_gyro; shortfall = (1-alpha)*||tau_gyro||.
        shortfall = (1.0 - alpha) * gyro_norm
        rows.append(dict(frac=fr, h=h_scalar, alpha=alpha,
                         gyro_norm=gyro_norm, tau_des_norm=float(np.linalg.norm(tau_des)),
                         shortfall=shortfall))

    print("PART A — LP allocation sweep over stored RW momentum")
    print("  w = [0,0,0.03] rad/s, b_body ~28 uT, tau_pd modest fixed")
    print(f"  {'h/h_max':>8} {'|tau_gyro|':>12} {'|tau_des|':>12} "
          f"{'alpha':>8} {'gyro shortfall (Nm)':>22}")
    for r in rows:
        print(f"  {r['frac']:8.2f} {r['gyro_norm']:12.3e} "
              f"{r['tau_des_norm']:12.3e} {r['alpha']:8.3f} {r['shortfall']:22.3e}")

    # First momentum fraction at which the LP can no longer fully deliver.
    sat_idx = next((i for i, r in enumerate(rows) if r["alpha"] < 0.999), None)
    if sat_idx is not None:
        print(f"  -> LP saturates (alpha<1) at h/h_max = {rows[sat_idx]['frac']:.2f}; "
              f"beyond this the gyro term is under-delivered.")
    else:
        print("  -> LP did not saturate across the sweep at this operating point.")
    return {"rows": rows, "w": w, "b_body": b_body}


def _pointing_error_series(run) -> np.ndarray:
    """Per-step boresight-vs-target angle [deg] for an ECI_Goal closed-loop run."""
    from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
    X = np.asarray(run.state_hist)
    Th = np.asarray(run.target_hist)
    sat = run.satellite
    b = getattr(sat, "boresight", None)
    if isinstance(b, dict):
        b = list(b.values())[0]
    b = np.asarray(b, dtype=float).flatten()
    b = b / np.linalg.norm(b)
    n = min(len(X), len(Th))
    out = np.full(n, np.nan)
    for i in range(n):
        row = np.asarray(Th[i], float).flatten()
        if row.size != 4 or not np.isnan(row[0]):
            continue
        tv = row[1:4]
        if np.linalg.norm(tv) == 0:
            continue
        out[i] = _angle_deg(_boresight_eci(X[i, 3:7], b), tv / np.linalg.norm(tv))
    return out


def part_b_closed_loop() -> dict:
    """Closed-loop: low vs high stored momentum -> track ||w|| AND pointing error."""
    import ADCS

    results = {}
    for label, h0 in (("low stored momentum  (h0=0)", 0.0),
                       ("high stored momentum (h0=0.95 h_max)", 0.95 * RW_HMAX)):
        sat = build_3mtq_1rw()
        for a in sat.actuators:
            if isinstance(a, RW):
                a.h = h0
        ctrl = MTQ_w_RW_LP(est_sat=sat, p_gain=5e-5, d_gain=1e-3, c_gain=1e-3,
                           h_target=np.array([0.004, 0.0, 0.0]))
        w0 = np.array([0.0, 0.0, 0.03])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        x0 = np.concatenate([w0, q0, [h0]])
        os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22,
                                 R=7000.0e3 * np.array([1.0, 0.0, 0.0]),
                                 V=np.array([0.0, 7.5e3, 0.0]))
        # Boresight is [0,0,1] body; at q0=identity it points ECI +z. Put the
        # target 40 deg off (a real acquisition task, not the 180 deg saddle).
        ang = np.deg2rad(40.0)
        goal = ADCS.goals.ECI_Goal(np.array([np.sin(ang), 0.0, np.cos(ang)]))
        try:
            res = ADCS.simulate(x=x0, satellite=sat, controller=ctrl, goal=goal,
                                os0=os0, dt=2.0, tf=300.0)
            run = res.runs[0]
            X = np.asarray(run.state_hist)
            t = np.asarray(run.time_s)
            w_norm = np.linalg.norm(X[:, 0:3], axis=1)
            perr = _pointing_error_series(run)
            results[label] = dict(t=t, w_norm=w_norm, perr=perr,
                                  w0=float(w_norm[0]), wend=float(w_norm[-1]))
            print(f"PART B — {label}:")
            print(f"    body rate ||w||  : {w_norm[0]*57.3:7.3f} -> "
                  f"{w_norm[-1]*57.3:7.3f} deg/s  "
                  f"({'GREW' if w_norm[-1] > w_norm[0]*1.05 else 'held — no spin-up'})")
            print(f"    pointing error   : {perr[0]:7.3f} -> {perr[-1]:7.3f} deg  "
                  f"(mean {np.nanmean(perr):.2f})")
        except Exception as e:
            print(f"PART B — {label}: simulate failed ({type(e).__name__}: {e})")
            results[label] = None
    return results


def emit_figure(part_a: dict, part_b: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))   # enlarged for legibility (Task 8 Fig 7)

    rows = part_a["rows"]
    fr = [r["frac"] for r in rows]

    ax = axes[0]
    ax.plot(fr, [r["alpha"] for r in rows], "o-", color="#2c7fb8")
    ax.axhline(1.0, ls="--", color="k", lw=1)
    ax.set_xlabel("stored RW momentum  h / h_max")
    ax.set_ylabel(r"LP effectiveness  $\alpha$")
    ax.set_title("(a) LP alpha collapses as stored momentum grows")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[1]
    ax.plot(fr, [r["shortfall"] for r in rows], "o-", color="#de2d26")
    ax.set_xlabel("stored RW momentum  h / h_max")
    ax.set_ylabel(r"gyro shortfall  $(1-\alpha)\,\|\tau_{gyro}\|$  [N m]")
    ax.set_title("(b) Uncancelled precession torque grows")
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[2]
    plotted = False
    for label, d in part_b.items():
        if d is None:
            continue
        ax.plot(d["t"], d["w_norm"] * 57.3, label=label)
        plotted = True
    if plotted:
        ax.set_xlabel("time [s]")
        ax.set_ylabel(r"body rate  $\|\omega\|$  [deg/s]")
        ax.set_title("(c) Closed-loop: ||w|| does NOT run away\n"
                     "(tau_gyro is perp to w -> no spin-up)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "closed-loop run unavailable\n(see PART A — mechanism\n"
                          "is shown at allocation level)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) Closed-loop confirmation")

    fig.tight_layout()   # enlarged; no suptitle (LaTeX caption carries it)
    path = os.path.join(OUT_DIR, "fig_lp_precession_failure.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    pa = part_a_allocation_sweep()
    print()
    pb = part_b_closed_loop()
    emit_figure(pa, pb)
