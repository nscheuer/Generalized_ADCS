"""SSC26 poster figure: the pipeline reproduces the published law, then extends it.

Three panels, each answering one question a poster reader will actually ask:

  A. "Did you change the control law?"   -> no: bit-exact parity vs MTQ_Lovera
  B. "What does swapping the allocator cost?" -> torque direction error
  C. "What does it buy?"                 -> fraction of demanded torque delivered

Panels B and C are the measured version of Snippet D's claim that LP preserves
direction while QP recovers magnitude and tilts.

Run:  python papers/SSC26_poster/generate_fig_parity.py
Out:  papers/SSC26_poster/output/fig_ssc26_parity.{png,pdf}
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.controller import MTQ_Lovera
from ADCS.pipeline import PipelineController
from ADCS.pipeline.control_law import PD_Law
from ADCS.pipeline.allocation import allocation_step, assemble_B_tau
from ADCS.pipeline.data import AllocationConfig
from ADCS.CONOPS.goals import ECI_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.state import State

# --- Palette: slots 1-3 of the validated reference categorical palette,
# --- used unmodified. Each panel carries a single series, so identity is
# --- carried by the axis, not by hue.
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SERIES = "#2a78d6"      # slot 1, blue
ACCENT = "#eb6834"      # slot 2, orange -- reference lines only
GOOD = "#1baf7a"        # slot 3, aqua

N_STATES = 200
RNG = np.random.default_rng(20260804)


def make_bus():
    mtqs = [MTQ(axis=j, max_torque=1.0) for j in MathConstants.unitvecs]
    rws = [RW(axis=j, max_torque=0.007, J=0.001, h=0.005, h_max=0.0162)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                     actuators=mtqs + rws, sensors=mtms,
                     boresight=np.array([0, 0, 1]))


def make_orbit():
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22 - TimeConstants.sec2cent,
                        R=7000 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2]),
                        V=np.array([8.0, 0.0, 0.0]))
    return Orbit(os0=os0, end_time=0.22 + 3000.0 * TimeConstants.sec2cent,
                 dt=50.0, zonal_J=2, fast=False, verbose=False)


def random_state():
    q = normalize(RNG.normal(size=4))
    w = RNG.normal(scale=0.02, size=3)
    h = RNG.normal(scale=0.004, size=3)
    return State(w=w, q=q, h=h)


def main():
    sat = make_bus()
    orbit = make_orbit()
    gains = dict(p_gain=2e-5, d_gain=2e-2, eps=1.0)
    goal = ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))

    legacy = MTQ_Lovera(est_sat=sat, **gains)
    pipe = PipelineController(sat, PD_Law(kp=gains["p_gain"],
                                          kd=gains["d_gain"],
                                          eps=gains["eps"]))

    # ---- Panel A: parity -------------------------------------------------
    residuals = []
    for _ in range(N_STATES):
        x = random_state()
        t = RNG.uniform(0.0, 2000.0)
        os_now = orbit.get_os(J2000=0.22 + t * TimeConstants.sec2cent)
        sens = sat.sensor_readings(x=x, os=os_now)
        u_l = legacy.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
        u_p = pipe.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
        residuals.append(np.max(np.abs(u_l - u_p)))
    residuals = np.asarray(residuals)

    # ---- Panels B/C: allocator behaviour vs demand -----------------------
    # LP, QP and wQP are indistinguishable while the demand is inside the
    # actuators' capability -- they all deliver it exactly. The differences
    # the poster claims only exist under SATURATION, so sweep the demanded
    # torque magnitude across the saturation knee rather than picking one
    # level and reporting a degenerate tie.
    # LP and wQP coincide almost exactly in panel C, so LP is drawn wide and
    # wQP dashed on top of it -- otherwise one series silently hides the other.
    methods = [("lp", "LP", SERIES, "-", 3.6),
               ("qp", "QP", ACCENT, "-", 2.0),
               ("qpw", "wQP", GOOD, (0, (5, 2)), 2.0),
               ("magnetic_cross", "cross (MTQ only)", MUTED, (0, (2, 2)), 2.0)]
    demands = np.logspace(-5, -1, 17)      # N*m
    dir_err = {m: np.zeros(len(demands)) for m, *_ in methods}
    mag_frac = {m: np.zeros(len(demands)) for m, *_ in methods}

    groups = pipe.actuator_groups
    n_samp = 60
    states = []
    for _ in range(n_samp):
        x = random_state()
        t = RNG.uniform(0.0, 2000.0)
        os_now = orbit.get_os(J2000=0.22 + t * TimeConstants.sec2cent)
        sens = sat.sensor_readings(x=x, os=os_now)
        sens_c = np.nan_to_num(np.asarray(sens).reshape(-1))
        B_body = pipe.M_read @ sens_c
        d = RNG.normal(size=3)
        states.append((x, B_body, d / np.linalg.norm(d)))

    for di, dmag in enumerate(demands):
        acc_d = {m: [] for m, *_ in methods}
        acc_g = {m: [] for m, *_ in methods}
        for x, B_body, dhat in states:
            tau_des = dmag * dhat
            B_tau, _, _ = assemble_B_tau(groups, B_body)
            for m, *_ in methods:
                res = allocation_step(
                    tau_desired=tau_des, actuator_groups=groups,
                    alloc_config=AllocationConfig(method=m), B_body=B_body,
                    n_actuators=len(sat.actuators), omega=x[0:3],
                    h_rw_body=np.zeros(3),
                )
                tau_ach = res.tau_achieved
                if tau_ach is None:
                    tau_ach = B_tau @ res.u
                n_ach = np.linalg.norm(tau_ach)
                if n_ach < 1e-15:
                    acc_d[m].append(np.nan)
                    acc_g[m].append(0.0)
                    continue
                cos_a = np.clip(np.dot(tau_ach, tau_des) / (n_ach * dmag), -1.0, 1.0)
                acc_d[m].append(np.degrees(np.arccos(cos_a)))
                acc_g[m].append(n_ach / dmag)
        for m, *_ in methods:
            dir_err[m][di] = np.nanmedian(acc_d[m])
            mag_frac[m][di] = np.nanmedian(acc_g[m])

    # ---- Figure ----------------------------------------------------------
    plt.rcParams.update({
        "font.size": 11, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # Panel A -- parity
    ax = axes[0]
    eps = np.finfo(float).eps
    plotted = np.maximum(residuals, eps * 1e-2)
    floor = eps * 1e-2
    n_exact = int((residuals == 0.0).sum())
    ax.scatter(np.arange(len(plotted)), plotted, s=14, color=SERIES,
               alpha=0.75, linewidths=0)
    ax.axhline(eps, color=ACCENT, lw=2, ls="--", zorder=3)
    ax.text(len(plotted) * 0.98, eps * 1.35, "float64 machine epsilon",
            ha="right", va="bottom", fontsize=9, color=ACCENT)
    ax.annotate(f"exactly 0 — bit-identical ({n_exact}/{len(residuals)})",
                xy=(len(plotted) * 0.5, floor), xytext=(len(plotted) * 0.5, floor * 3.0),
                ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, eps * 6)
    ax.set_xlabel(f"random state ({N_STATES} samples)")
    ax.set_ylabel(r"max $|u_{\rm legacy} - u_{\rm pipeline}|$")
    ax.set_title("A · Same law, restructured\nPipeline reproduces MTQ_Lovera",
                 loc="left", fontsize=11.5, color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel B -- direction error vs demand
    ax = axes[1]
    for m, lbl, col, ls, lw in methods:
        ax.plot(demands, dir_err[m], color=col, lw=lw, ls=ls, label=lbl, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"demanded $|\tau|$  [N$\cdot$m]")
    ax.set_ylabel("torque direction error [deg]")
    ax.set_title("B · Cost of the swap\ndirection held vs. tilted",
                 loc="left", fontsize=11.5, color=INK)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel C -- magnitude delivered vs demand
    ax = axes[2]
    for m, lbl, col, ls, lw in methods:
        ax.plot(demands, mag_frac[m], color=col, lw=lw, ls=ls, label=lbl, zorder=2)
    ax.axhline(1.0, color=INK, lw=1.2, ls=":", zorder=1)
    ax.text(demands[-1], 1.03, "demand met", ha="right", va="bottom",
            fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel(r"demanded $|\tau|$  [N$\cdot$m]")
    ax.set_ylabel(r"delivered $|\tau|$ / demanded $|\tau|$")
    ax.set_title("C · What it buys\nfraction of demand delivered",
                 loc="left", fontsize=11.5, color=INK)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_ssc26_parity.{ext}"),
                    dpi=200, bbox_inches="tight")

    # ---- Report the numbers so they can be quoted on the poster ---------
    print(f"Panel A: {N_STATES} random states")
    print(f"  max residual over all states : {residuals.max():.3e}")
    print(f"  median residual              : {np.median(residuals):.3e}")
    print(f"  states with residual < 1e-12 : "
          f"{int((residuals < 1e-12).sum())}/{len(residuals)}")
    print(f"\nPanels B/C: {n_samp} states x {len(demands)} demand levels")
    print(f"  {'demand':>10} " + " ".join(f"{lbl:>18}" for _, lbl, _, _, _ in methods))
    for di, dmag in enumerate(demands):
        cells = " ".join(
            f"{dir_err[m][di]:7.2f}deg/{mag_frac[m][di]:5.2f}" if not np.isnan(dir_err[m][di])
            else f"{'--':>18}"
            for m, *_ in methods)
        print(f"  {dmag:10.2e} {cells}")
    print(f"\nwrote {outdir}/fig_ssc26_parity.png|pdf")


if __name__ == "__main__":
    main()
