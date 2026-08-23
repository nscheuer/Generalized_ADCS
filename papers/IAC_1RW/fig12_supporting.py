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

    traces = [("boresight", np.abs(np.sum(nh * Bh, axis=1)), OI["blue"], "-"),
              ("45$^\\circ$", np.abs(np.sum(m45 * Bh, axis=1)), OI["orange"], "--"),
              ("orbit-normal", np.abs(np.sum(hh * Bh, axis=1)), OI["green"], "-.")]

    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    ax.axhspan(0.3, 1.0, color=OI["sky"], alpha=0.13, lw=0)
    ax.axhspan(0.0, 0.1, color=OI["verm"], alpha=0.12, lw=0)
    ax.text(0.985, 0.62, "restoration-favourable:\nwheel supplies the along-field axis",
            fontsize=7, ha="right", color=OI["blue"], transform=ax.get_yaxis_transform())
    ax.text(0.985, 0.045, "dump-favourable: MTQs can\ncancel the wheel's reaction",
            fontsize=7, ha="right", color=OI["verm"], transform=ax.get_yaxis_transform())

    x = ts / T_ORB
    for name, sig, col, lsty in traces:
        med = float(np.median(sig))
        jref = ref[{"boresight": "boresight", "45$^\\circ$": "45deg",
                    "orbit-normal": "orbit_normal"}[name]]
        assert abs(med - jref["median_sigma"]) < 0.06, (name, med, jref["median_sigma"])
        ax.plot(x, sig, color=col, ls=lsty, lw=1.5,
                label=f"{name} (median {jref['median_sigma']:.2f})")
    duties = ", ".join(f"{n} {100*ref[k]['restore_duty']:.1f}%" for n, k in
                       (("boresight", "boresight"), ("45$^\\circ$", "45deg"),
                        ("orbit-normal", "orbit_normal")))
    ax.text(0.02, 0.315, f"restoration duty ($\\sigma>0.3$): {duties}",
            fontsize=6.8, transform=ax.transAxes, color="0.25")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("orbit phase  $t/T_{orb}$")
    ax.set_ylabel(r"$\sigma = |\hat a \cdot \hat B(t)|$")
    ax.legend(loc="center left", fontsize=7, framealpha=0.95)
    ax.grid(alpha=0.15, lw=0.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_sigma.{ext}"), dpi=220)
    print("fig1 medians:", {n: round(float(np.median(s)), 3) for n, s, _, _ in traces})


def fig2():
    fj = json.load(open(os.path.join(OUT, "F_altitude_20260818_174558.json")))
    rows = sorted(fj["rows_by_case"]["m_res=0.05"], key=lambda r: r["alt_km"])
    alt = np.array([r["alt_km"] for r in rows])
    drag = np.array([r["accum_by_source_Nms"]["Drag_Disturbance"] for r in rows]) * 1e3
    dip = np.array([r["accum_by_source_Nms"]["Dipole_Disturbance"] for r in rows]) * 1e3
    tot = np.array([r["accum_per_orbit_Nms"] for r in rows]) * 1e3
    marg = np.array([r["margin"] for r in rows])

    fig, (a, b) = plt.subplots(2, 1, figsize=(4.6, 4.8), sharex=True,
                               constrained_layout=True)
    a.plot(alt, drag, "-", color=OI["blue"], lw=1.5, label="drag")
    a.plot(alt, dip, "--", color=OI["orange"], lw=1.5, label="residual dipole")
    a.plot(alt, tot, "-.", color="0.2", lw=1.4, label="total")
    a.set_yscale("log")
    a.set_ylabel("secular momentum\n[mN·m·s / orbit]")
    a.legend(fontsize=7, loc="upper right", framealpha=0.95)
    i = int(np.argmin(np.abs(np.log(drag) - np.log(dip))))
    a.annotate("dipole floor takes over:\naltitude stops helping",
               xy=(470, 1.32), xytext=(545, 0.12), fontsize=6.8,
               arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))

    b.plot(alt, marg, "-", color=OI["green"], lw=1.6, label="dump capacity / accumulation")
    b.axhline(1.0, color=OI["verm"], lw=1.2, ls=":")
    b.set_yscale("log")
    b.set_ylabel("momentum margin")
    b.set_xlabel("altitude [km]")
    b.legend(fontsize=7, loc="lower right", framealpha=0.95)
    # interpolated binding altitude (F sampled from 300 km)
    lo = rows[0]
    # F's own extrapolation (its fit, not a two-point re-derivation here)
    x_bind = float(fj["altitude_unity_margin_km"]["m_res=0.05"])
    b.annotate(f"margin = 1 at ~{x_bind:.0f} km\n(extrapolated below the\n300 km sample)",
               xy=(300, float(lo["margin"])), xytext=(335, 0.35), fontsize=6.8,
               arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    b.set_ylim(bottom=0.2)
    for ax in (a, b):
        ax.axvline(400, color="0.55", lw=0.9, ls="--")
        ax.grid(alpha=0.15, lw=0.4)
    a.text(404, a.get_ylim()[1] * 0.4, "reference", fontsize=6.5, color="0.4",
           rotation=90, va="top")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_altitude.{ext}"), dpi=220)
    print(f"fig2: crossover near {alt[i]} km; binding ~{x_bind:.0f} km; "
          f"margin(400) = {marg[list(alt).index(400)]:.1f}")


if __name__ == "__main__":
    fig1()
    fig2()
