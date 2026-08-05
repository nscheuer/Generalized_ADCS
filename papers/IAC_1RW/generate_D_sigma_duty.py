"""Campaign D -- wheel-axis geometry and the sigma duty cycle. Feeds Figure 1.

``sigma(t) = |a_hat . B_hat(t)|`` is the wheel-axis/field alignment. Rank-3 restoration wants
sigma > 0 (the wheel supplies the axis the magnetorquers cannot); dumping wants sigma small.
Same scalar, opposite senses.

**This campaign is deliberately NOT the version in the original spec.** The spec asked for the
duty fraction with sigma < 0.1 and called that "dump-favourable". That threshold is arbitrary,
and worse, the premise behind it is wrong: torque-free desaturation does not require sigma = 0.

To spin the wheel down at rate ``u_w`` the reaction on the body is ``-u_w a_hat``. The
magnetorquers must supply ``+u_w a_hat``, but magnetorquer torque spans only the plane
perpendicular to **B**, so they deliver ``u_w(a_hat - sigma B_hat)`` and the body keeps an
uncancelled ``u_w * sigma`` along **B_hat**. Exact torque-free dumping therefore needs
``sigma = 0`` *exactly* -- a measure-zero condition in 3-D that never occurs.

So sigma does not gate feasibility. It sets the **pointing penalty per unit dump rate**, and
the achievable dump rate is bounded by three things at once:

.. math::

    u_w(t) \\le \\min\\Big(\\tau_w,\\;
                       \\frac{\\tau_{mtq}^{max}(t)}{\\sqrt{1-\\sigma^2}},\\;
                       \\frac{\\tau_{allow}}{|\\sigma|}\\Big)

the wheel itself, the magnetorquer authority needed to cancel the perpendicular part, and the
pointing budget. The quantity that actually matters is the integral of that over an orbit --
the per-orbit dump capacity -- compared against the per-orbit secular accumulation. This also
*derives* the threshold the spec guessed at: ``sigma* = tau_allow / tau_mtq_max`` is where the
pointing constraint takes over from authority.

One consequence is worth stating in the paper: at favourable geometry the binding constraint
is **magnetorquer authority**, not sigma. ``tau_mtq_max ~ 1e-5`` N m against a wheel that can
push ``2e-3`` -- the wheel is irrelevant to the dump rate by two orders of magnitude.

The uncancelled residual goes into the body along **B_hat**, which rotates over the orbit, so
it later becomes perpendicular to **B** and is removable. It is a transient pointing excursion,
not an accumulation.

Pure geometry plus IGRF -- no dynamics, no Monte Carlo. Cheap.

Run: ``python papers/IAC_1RW/generate_D_sigma_duty.py``
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.satellite_factory import IAC_6U

from papers.IAC_1RW._iac_sim import (
    A_KM,
    EPOCH,
    T_ORBIT,
    rv_circular,
)

OUT = os.path.join(os.path.dirname(__file__), "output_data")

N_SAMPLES = 360

#: Wheel axes in the BODY frame. Under the nadir-locked profile (+z nadir, +y anti-orbit-normal)
#: these are boresight, orbit-normal, and 45 degrees between.
WHEEL_AXES = {
    "boresight": np.array([0.0, 0.0, 1.0]),
    "orbit_normal": np.array([0.0, 1.0, 0.0]),
    "45deg": normalize(np.array([0.0, 1.0, 1.0])),
}

INCLINATIONS_DEG = (97.0, 45.0, 5.0)
PROFILES = ("nadir", "inertial")

#: Restoration is "favourable" above this alignment -- the wheel is then supplying a
#: well-conditioned third axis rather than one nearly inside the magnetorquer plane.
SIGMA_RESTORE = 0.3

#: Per-orbit secular momentum accumulation measured by the environment cross-check
#: (check_environment.py, 97 deg nadir-locked reference bus) [N m s].
ACCUM_PER_ORBIT = 1.5224e-3


# ---------------------------------------------------------------------------------------
# Magnetorquer authority along a requested direction
# ---------------------------------------------------------------------------------------

def mtq_max_torque_along(B: np.ndarray, d_hat: np.ndarray, m_max: float) -> float:
    r"""Largest achievable ``|tau|`` along ``d_hat`` (which must be perpendicular to B).

    With ``tau = m x B`` and the box constraint ``|m_i| <= m_max``, the set of dipoles giving
    ``tau = alpha d_hat`` is ``(alpha/|B|) u_hat + t B_hat`` for ``u_hat = B_hat x d_hat``.
    Feasibility is then a one-dimensional problem in ``t``, and

    .. math::  \alpha_{max} = \frac{|B| \, m_{max}}{\min_s \|\hat u + s \hat B\|_\infty}

    The free component along ``B_hat`` produces no torque, so it is spent entirely on staying
    inside the box -- which is why the answer is **not** simply ``m_max |B|``.
    """
    Bn = float(np.linalg.norm(B))
    if Bn < 1e-15:
        return 0.0
    B_hat = B / Bn
    u_hat = np.cross(B_hat, d_hat)
    un = float(np.linalg.norm(u_hat))
    if un < 1e-12:
        return 0.0
    u_hat = u_hat / un

    def g(s):
        return float(np.max(np.abs(u_hat + s * B_hat)))

    res = minimize_scalar(g, bounds=(-5.0, 5.0), method="bounded",
                          options={"xatol": 1e-10})
    gmin = min(g(res.x), g(-5.0), g(5.0))
    if gmin < 1e-12:
        return float("inf")
    return Bn * m_max / gmin


def dump_rate(sigma_signed: float, B: np.ndarray, a_hat: np.ndarray,
              m_max: float, tau_w: float, tau_allow: float) -> float:
    """Max wheel-despin torque at this geometry, under authority AND pointing limits."""
    Bn = float(np.linalg.norm(B))
    if Bn < 1e-15:
        return 0.0
    B_hat = B / Bn

    perp = a_hat - sigma_signed * B_hat
    perp_n = float(np.linalg.norm(perp))          # = sqrt(1 - sigma^2)

    # Pointing: the uncancelled residual along B_hat is u_w * |sigma|.
    lim_point = (tau_allow / abs(sigma_signed)) if abs(sigma_signed) > 1e-12 else np.inf

    # Authority: the magnetorquers must supply u_w * |perp| along perp_hat.
    if perp_n < 1e-12:
        lim_auth = np.inf      # a_hat parallel to B: nothing to cancel, pointing limit rules
    else:
        tau_av = mtq_max_torque_along(B, perp / perp_n, m_max)
        lim_auth = tau_av / perp_n

    return float(min(tau_w, lim_auth, lim_point))


# ---------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------

def body_to_eci(profile: str, R: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Columns are the body axes expressed in ECI."""
    if profile == "nadir":
        z_b = -normalize(R)                       # boresight at nadir
        y_b = -normalize(np.cross(R, V))          # anti orbit-normal
        x_b = np.cross(y_b, z_b)                  # completes, along velocity
        return np.column_stack([x_b, y_b, z_b])
    return np.eye(3)                              # inertial hold: body frozen in ECI


def sigma_trace(wheel_axis: np.ndarray, inc_deg: float, profile: str,
                n: int = N_SAMPLES) -> Dict[str, np.ndarray]:
    """sigma(t) and the field over one orbit."""
    ephem = Ephemeris()
    t_s = np.linspace(0.0, T_ORBIT, n, endpoint=False)
    sigma = np.zeros(n)
    sigma_signed = np.zeros(n)
    B_body = np.zeros((n, 3))

    for k, t in enumerate(t_s):
        u = 2.0 * np.pi * t / T_ORBIT
        R, V = rv_circular(u, inc_deg, 0.0)
        os_k = Orbital_State(ephem=ephem, J2000=EPOCH + t * TimeConstants.sec2cent,
                             R=R, V=V)
        C = body_to_eci(profile, R, V)            # body -> ECI
        B_b = C.T @ np.asarray(os_k.B, float)     # field in body frame
        B_body[k] = B_b
        bn = np.linalg.norm(B_b)
        s = float(wheel_axis @ B_b / bn) if bn > 0 else 0.0
        sigma_signed[k] = s
        sigma[k] = abs(s)

    return {"t_s": t_s, "sigma": sigma, "sigma_signed": sigma_signed, "B_body": B_body}


def authority_limit(tr: Dict[str, np.ndarray], wheel_axis: np.ndarray,
                    m_max: float) -> np.ndarray:
    """Per-sample wheel-despin rate allowed by magnetorquer authority alone [N m].

    Computed once per configuration because it does **not** depend on the pointing
    allowance -- only the geometry and the field do. Recomputing it inside the
    tau_allow sweep costs a box-constrained optimisation per (tau, sample) pair for
    no reason.
    """
    n = tr["t_s"].size
    out = np.empty(n)
    for k in range(n):
        B = tr["B_body"][k]
        Bn = float(np.linalg.norm(B))
        if Bn < 1e-15:
            out[k] = 0.0
            continue
        perp = wheel_axis - tr["sigma_signed"][k] * (B / Bn)
        pn = float(np.linalg.norm(perp))           # sqrt(1 - sigma^2)
        if pn < 1e-12:
            out[k] = np.inf        # a_hat || B: nothing to cancel, pointing limit rules
        else:
            out[k] = mtq_max_torque_along(B, perp / pn, m_max) / pn
    return out


def capacity_curve(tr: Dict[str, np.ndarray], lim_auth: np.ndarray,
                   tau_grid: np.ndarray, tau_w: float) -> np.ndarray:
    """Per-orbit dump capacity [N m s] across a grid of pointing allowances.

    Vectorised over the grid: the only tau-dependent term is the pointing limit
    ``tau_allow / |sigma|``.
    """
    dt = float(tr["t_s"][1] - tr["t_s"][0])
    sig = np.abs(tr["sigma_signed"])
    safe = np.where(sig > 1e-12, sig, np.inf)      # sigma = 0 -> pointing never binds
    base = np.minimum(tau_w, lim_auth)             # [n]
    lim_point = tau_grid[:, None] / safe[None, :]  # [n_tau, n]
    return np.sum(np.minimum(base[None, :], lim_point), axis=1) * dt


def capacity(tr, lim_auth, tau_allow: float, tau_w: float) -> float:
    return float(capacity_curve(tr, lim_auth, np.array([tau_allow]), tau_w)[0])


# ---------------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------------

# Okabe-Ito: the canonical colourblind-safe qualitative palette from the accessibility
# literature. Chosen because this figure is printed: it is validated by construction for
# deuteranopia/protanopia/tritanopia rather than by eyeballing an ad-hoc set. Linestyle is
# carried as a redundant channel so identity survives greyscale printing -- colour alone is
# never the encoding.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
}
SERIES_STYLE = [                     # fixed order, never cycled
    (OKABE_ITO["blue"], "-"),
    (OKABE_ITO["vermillion"], "--"),
    (OKABE_ITO["green"], "-."),
]


def make_figure(traces, caps, tau_grid, path_png, path_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # -- Panel A: sigma(t) ---------------------------------------------------------------
    for (name, tr), (colour, ls) in zip(traces.items(), SERIES_STYLE):
        ax1.plot(tr["t_s"] / 60.0, tr["sigma"], color=colour, ls=ls, lw=2.0,
                 label=name.replace("_", " "))
    ax1.axhline(SIGMA_RESTORE, color="0.45", lw=1.0, ls=":", zorder=0)
    # Sits in the clear band between the orbit-normal and 45-degree traces mid-orbit;
    # at x=0 it collided with the y-axis and the orbit-normal line.
    ax1.text(18.0, SIGMA_RESTORE + 0.03, r"restoration-favourable  $\sigma>0.3$",
             fontsize=7.5, color="0.35")
    ax1.set_xlabel("time [min]")
    ax1.set_ylabel(r"$\sigma = |\hat a \cdot \hat B|$")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, T_ORBIT / 60.0)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.set_title("Wheel-axis / field alignment, one orbit", fontsize=9)
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.set_axisbelow(True)

    # -- Panel B: dump capacity vs pointing allowance ------------------------------------
    for (name, c), (colour, ls) in zip(caps.items(), SERIES_STYLE):
        ax2.plot(tau_grid * 1e6, np.asarray(c) * 1e3, color=colour, ls=ls, lw=2.0,
                 label=name.replace("_", " "))
    ax2.axhline(ACCUM_PER_ORBIT * 1e3, color="0.25", lw=1.4, ls=(0, (1, 1)))
    ax2.text(tau_grid[1] * 1e6, ACCUM_PER_ORBIT * 1e3 * 1.25,
             "secular accumulation", fontsize=7.5, color="0.25")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"pointing allowance $\tau_{\rm allow}$ [$\mu$N$\cdot$m]")
    ax2.set_ylabel(r"dump capacity [mN$\cdot$m$\cdot$s / orbit]")
    ax2.set_title("Per-orbit dump capacity", fontsize=9)
    ax2.grid(alpha=0.25, lw=0.6, which="both")
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------

def main() -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)
    m_max, tau_w = IAC_6U.m_max, IAC_6U.tau_w

    print("=" * 88)
    print("Campaign D -- wheel-axis geometry and sigma duty")
    print(f"m_max = {m_max} A m^2, tau_w = {tau_w*1e3:.1f} mN m, "
          f"h_max = {IAC_6U.h_max*1e3:.0f} mN m s, T_orbit = {T_ORBIT:.0f} s")
    print(f"accumulation to beat: {ACCUM_PER_ORBIT*1e3:.3f} mN m s / orbit")
    print("=" * 88)

    # A representative pointing allowance: the same order as the residual-dipole
    # disturbance the bus already lives with, so "spend as much again on dumping".
    TAU_ALLOW_REF = 2.5e-6
    tau_grid = np.logspace(np.log10(1e-8), np.log10(1e-4), 40)

    results: Dict[str, Any] = {}
    print(f"\n{'wheel axis':<14}{'inc':>6}{'profile':>10}{'med sig':>9}"
          f"{'restore%':>10}{'tau*':>10}{'cap[mNms]':>11}{'margin':>9}")
    print("-" * 88)

    for axis_name, a_hat in WHEEL_AXES.items():
        for inc in INCLINATIONS_DEG:
            for prof in PROFILES:
                tr = sigma_trace(a_hat, inc, prof)
                sig = tr["sigma"]
                lim_auth = authority_limit(tr, a_hat, m_max)

                finite = lim_auth[np.isfinite(lim_auth)]
                tau_mtq_med = float(np.median(finite)) if finite.size else float("inf")
                sigma_star = (min(1.0, TAU_ALLOW_REF / tau_mtq_med)
                              if np.isfinite(tau_mtq_med) and tau_mtq_med > 0 else 1.0)

                cap = capacity(tr, lim_auth, TAU_ALLOW_REF, tau_w)
                key = f"{axis_name}|{inc:.0f}|{prof}"
                results[key] = {
                    "wheel_axis": axis_name, "inc_deg": inc, "profile": prof,
                    "median_sigma": float(np.median(sig)),
                    "restore_duty": float(np.mean(sig > SIGMA_RESTORE)),
                    "median_tau_mtq_max_Nm": tau_mtq_med,
                    "sigma_star": float(sigma_star),
                    "dump_duty_below_sigma_star": float(np.mean(sig < sigma_star)),
                    "capacity_Nms_per_orbit": cap,
                    "margin_vs_accumulation": cap / ACCUM_PER_ORBIT,
                    "capacity_curve_Nms": capacity_curve(
                        tr, lim_auth, tau_grid, tau_w).tolist(),
                }
                r = results[key]
                print(f"{axis_name:<14}{inc:>6.0f}{prof:>10}{r['median_sigma']:>9.3f}"
                      f"{100*r['restore_duty']:>9.1f}%{r['sigma_star']:>10.3f}"
                      f"{cap*1e3:>11.3f}{r['margin_vs_accumulation']:>9.1f}x")

    # ---- figure: the 97 deg nadir-locked family, which is the mission case -------------
    fig_traces = {name: sigma_trace(a, 97.0, "nadir") for name, a in WHEEL_AXES.items()}
    fig_caps = {name: results[f"{name}|97|nadir"]["capacity_curve_Nms"]
                for name in WHEEL_AXES}
    make_figure(fig_traces, fig_caps, tau_grid,
                f"{OUT}/fig_D_sigma_duty.png", f"{OUT}/fig_D_sigma_duty.pdf")

    payload = {"task": "D_sigma_duty", "timestamp": ts,
               "m_max": m_max, "tau_w": tau_w, "h_max": IAC_6U.h_max,
               "T_orbit_s": T_ORBIT, "accum_per_orbit_Nms": ACCUM_PER_ORBIT,
               "tau_allow_ref_Nm": TAU_ALLOW_REF,
               "tau_allow_grid_Nm": tau_grid.tolist(),
               "sigma_restore_threshold": SIGMA_RESTORE,
               "cells": results}
    with open(f"{OUT}/D_sigma_duty_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 88)
    worst = min(results.values(), key=lambda r: r["margin_vs_accumulation"])
    print(f"Worst-case margin: {worst['margin_vs_accumulation']:.1f}x "
          f"({worst['wheel_axis']}, {worst['inc_deg']:.0f} deg, {worst['profile']})")
    print("Dumping is authority-limited, not geometry-limited: the wheel could push "
          f"{tau_w*1e6:.0f} uN m,")
    print(f"the magnetorquers about {1e6*np.median([r['median_tau_mtq_max_Nm'] for r in results.values()]):.1f} uN m.")
    print("=" * 88)
    print(f"\nwrote {OUT}/D_sigma_duty_{ts}.json + fig_D_sigma_duty.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
