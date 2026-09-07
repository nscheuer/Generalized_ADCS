"""Figures 1 (sigma over an orbit) and 2 (altitude scaling), per spec.
Column-width supporting figures; Okabe-Ito + line-style redundancy.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import _get_orbit, EPOCH  # noqa: E402
from ADCS.orbits.universal_constants import TimeConstants  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style
fig_style.apply(10.0)
OUT = os.path.join(HERE, "output_data")
S2C = TimeConstants.sec2cent
T_ORB = 5553.6
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "grey": "#666666", "sky": "#56B4E9"}


def fig1():
    dj = json.load(open(os.path.join(OUT, "D_sigma_duty_20260818_174554.json")))
    ref = {k.split("|")[0]: v for k, v in dj["cells"].items() if "|97|nadir" in k}

    oc = {"raan_deg": 0.0, "phase_deg": 0.0, "inc_deg": 97.0}
    orb = _get_orbit(oc, 1.0, T_ORB + 10)
    ts = np.arange(0.0, T_ORB, 10.0)
    Bh, nh, hh = [], [], []
    for t in ts:
        o = orb.get_os(J2000=EPOCH + t * S2C)
        R = np.asarray(o.R, float); V = np.asarray(o.V, float)
        B = np.asarray(o.B, float)
        Bh.append(B / np.linalg.norm(B))
        nh.append(-R / np.linalg.norm(R))
        h = np.cross(R, V); hh.append(h / np.linalg.norm(h))
    Bh, nh, hh = map(np.asarray, (Bh, nh, hh))
    # D's nadir frame has body-y = ANTI orbit-normal (generate_D line 164),
    # so the 45deg body axis [0,1,1] maps to (nadir - h_orb)/sqrt(2) in ECI.
    m45 = (nh - hh) / np.linalg.norm(nh - hh, axis=1, keepdims=True)

    traces = [("Boresight", "boresight", np.abs(np.sum(nh * Bh, axis=1)), "-", 2.0),
              ("45$^\\circ$", "45deg", np.abs(np.sum(m45 * Bh, axis=1)), "--", 1.5),
              ("Orbit-normal", "orbit_normal", np.abs(np.sum(hh * Bh, axis=1)), ":", 1.6)]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.axhspan(0.3, 1.0, color=fig_style.BAND_RESTORE, alpha=0.14, lw=0)
    ax.axhspan(0.0, 0.1, color=fig_style.BAND_DUMP, alpha=0.16, lw=0)
    ax.axhspan(0.0, 0.02, color=fig_style.BAND_RANKLOSS, alpha=0.55, lw=0)
    ax.text(0.5, 0.86, "Better for rank restoration", fontsize=10, ha="center",
            color="#2A6F97", transform=ax.get_yaxis_transform())
    ax.text(0.72, 0.115, "Better for clean desaturation", fontsize=10, ha="center",
            va="bottom", color="#9A5B00", transform=ax.get_yaxis_transform())

    x = ts / T_ORB
    for name, key, sig, lsty, lw in traces:
        med = float(np.median(sig))
        assert abs(med - ref[key]["median_sigma"]) < 0.06, (name, med, ref[key]["median_sigma"])
        ax.plot(x, sig, color=fig_style.INK, ls=lsty, lw=lw,
                label=f"{name}, $\\tilde\\sigma$ = {ref[key]['median_sigma']:.2f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Orbit fraction")
    ax.set_ylabel(r"$\sigma = |\hat a \cdot \hat B|$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=2.6,
              labelspacing=0.25, borderaxespad=0.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_sigma.{ext}"))
    print("fig1 medians:", {n: round(float(np.median(s)), 3) for n, _, s, _, _ in traces})


def fig2():
    fj = json.load(open(os.path.join(OUT, "F_altitude_20260818_174558.json")))
    rows = sorted(fj["rows_by_case"]["m_res=0.05"], key=lambda r: r["alt_km"])
    alt = np.array([r["alt_km"] for r in rows])
    drag = np.array([r["accum_by_source_Nms"]["Drag_Disturbance"] for r in rows]) * 1e3
    dip = np.array([r["accum_by_source_Nms"]["Dipole_Disturbance"] for r in rows]) * 1e3
    tot = np.array([r["accum_per_orbit_Nms"] for r in rows]) * 1e3
    marg = np.array([r["margin"] for r in rows])

    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True,
                               constrained_layout=True)
    a.plot(alt, drag, "-o", ms=4, color=fig_style.INK, lw=1.6, label="drag")
    a.plot(alt, dip, "--s", ms=3.6, color=fig_style.MUTED, lw=1.5, label="residual dipole")
    a.plot(alt, tot, "-.^", ms=3.6, color=fig_style.INK, lw=1.0, alpha=0.8, label="total")
    a.set_yscale("log")
    a.set_ylabel("secular momentum\n[mN·m·s per orbit]")
    a.legend(loc="upper right", handlelength=2.4, labelspacing=0.3)
    b.plot(alt, marg, "-D", ms=3.8, color=fig_style.INK, lw=1.6)
    b.axhline(1.0, color=fig_style.MUTED, lw=1.0, ls=":")
    b.set_yscale("log")
    b.set_ylabel("momentum margin")
    b.set_xlabel("altitude [km]")
    b.set_ylim(bottom=0.3)
    for ax in (a, b):
        ax.axvline(400, color="#BBBBBB", lw=0.9, ls="--")
        fig_style.log_decades_only(ax, "y")
        ax.grid(True, which="major", axis="y")
    a.set_xticks([300, 400, 500, 600, 700, 800])
    for ax, lab in zip((a, b), "ab"):
        ax.text(0.015, 0.04, f"({lab})", transform=ax.transAxes, fontweight="bold",
                va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_altitude.{ext}"))
    x_bind = float(fj["altitude_unity_margin_km"]["m_res=0.05"])
    print(f"fig2: binding ~{x_bind:.0f} km (caption); margin(400) = {marg[list(alt).index(400)]:.1f}")


if __name__ == "__main__":
    fig1()
    fig2()
