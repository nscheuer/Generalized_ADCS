"""Campaign F -- altitude scaling, and the altitude where the momentum boundary binds.

Campaign D found that per-orbit dump capacity exceeds secular accumulation by 9.7-32x at
400 km, so the momentum boundary sits nowhere near the frontier and the envelope is set by
agility and accuracy instead. That is a real result, but as stated it is unbounded and a
reviewer will take it apart. Two assumptions carry it:

* **The cp-cg offset.** Drag torque on this bus comes *entirely* from the 2 cm offset (a
  uniform box with its COM at the geometric centre has identically zero drag torque), so the
  margin is inversely proportional to that number. 19 cm is not plausible on a 6U, so this
  assumption is safe.
* **Density.** This one is not safe, because altitude moves it by orders of magnitude.

So the deliverable here is a single number: **the altitude at which the margin reaches unity.**
That converts "momentum management does not bind" into "momentum management does not bind above
X km", which is the version that survives review.

The scaling is not one-sided, which is why it needs computing rather than asserting:

===================  ==========================================================
Quantity             Altitude dependence
===================  ==========================================================
Density              falls ~2 orders of magnitude 400 -> 800 km (drag torque)
Geomagnetic field    falls as r^-3, ~30% over the same span
Dump capacity        proportional to magnetorquer authority, so falls with B
Dipole torque        also proportional to B, so accumulation falls too
===================  ==========================================================

Both capacity and part of the accumulation fall with B, so the ratio is not simply "less drag
is better" -- the drag term has to win by enough to overcome the shrinking magnetorquer
authority. Going *down* in altitude, drag rises far faster than authority does, so the margin
collapses; the question is where.

Run: ``python papers/IAC_1RW/generate_F_altitude.py``
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.satellite_factory import create_iac_6u_bus, IAC_6U

from papers.IAC_1RW.generate_D_sigma_duty import (
    authority_limit,
    body_to_eci,
    capacity,
)

OUT = os.path.join(os.path.dirname(__file__), "output_data")
EPOCH = 0.22
INC_DEG = 97.0
N_SAMPLES = 180

#: Altitudes for the reported table [km].
ALTITUDES_KM = (300.0, 400.0, 500.0, 600.0, 800.0)

#: Pointing allowance used for the headline margin, matching Campaign D.
TAU_ALLOW_REF = 2.5e-6

#: Orbits simulated when separating secular drift from cyclic oscillation. A single-orbit
#: mean cannot tell the two apart -- it picks up part of the cyclic swing and overstates the
#: secular rate (by ~13% for the dipole). Fitting a ramp over several orbits does separate
#: them, and the separation matters: the secular part sets the dumping cadence, the cyclic
#: part permanently reserves wheel range that is never available for control.
N_ORBITS_FIT = 6

#: Residual dipole cases: the reference bus and the labelled sensitivity.
M_RES_CASES = (0.05, 0.1)

WHEEL_AXIS = np.asarray(IAC_6U.boresight, float)   # the reference mounting


def period_s(alt_km: float) -> float:
    a = EarthConstants.R_e + alt_km
    return 2.0 * np.pi * np.sqrt(a ** 3 / EarthConstants.mu_e)


def rv_at(alt_km: float, u_rad: float, inc_deg: float = INC_DEG):
    a = EarthConstants.R_e + alt_km
    i, Om = np.deg2rad(inc_deg), 0.0
    v = np.sqrt(EarthConstants.mu_e / a)
    cu, su, cO, sO, ci, si = (np.cos(u_rad), np.sin(u_rad), np.cos(Om),
                              np.sin(Om), np.cos(i), np.sin(i))
    r_hat = np.array([cO * cu - sO * ci * su, sO * cu + cO * ci * su, si * su])
    v_hat = np.array([-cO * su - sO * ci * cu, -sO * su + cO * ci * cu, si * cu])
    return a * r_hat, v * v_hat


def quat_from_cols(C: np.ndarray) -> np.ndarray:
    t = np.trace(C)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (C[2, 1] - C[1, 2]) / s,
                      (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(C)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + C[i, i] - C[j, j] - C[k, k]) * 2
        q = np.zeros(4)
        q[0] = (C[k, j] - C[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (C[j, i] + C[i, j]) / s
        q[k + 1] = (C[k, i] + C[i, k]) / s
    return normalize(q)


def survey(alt_km: float, n: int = N_SAMPLES, m_res: float = None,
           n_orbits: int = N_ORBITS_FIT, profile: str = "nadir") -> Dict[str, Any]:
    """Secular vs cyclic momentum, and dump capacity, at one altitude (nadir-locked)."""
    ephem = Ephemeris()
    sat = create_iac_6u_bus(n_rw=1, m_res=m_res)
    names = [type(d).__name__ for d in sat.disturbances]
    T = period_s(alt_km)
    n = n * n_orbits
    t_s = np.linspace(0.0, n_orbits * T, n, endpoint=False)

    torques = {nm: np.zeros((n, 3)) for nm in names}
    sigma_signed = np.zeros(n)
    B_body = np.zeros((n, 3))
    rho = np.zeros(n)

    for k, t in enumerate(t_s):
        u = 2.0 * np.pi * t / T
        R, V = rv_at(alt_km, u)
        os_k = Orbital_State(ephem=ephem, J2000=EPOCH + t * TimeConstants.sec2cent,
                             R=R, V=V)
        C = body_to_eci(profile, R, V)
        q = quat_from_cols(C)
        x = np.concatenate([np.zeros(3), q, np.zeros(1)])

        Bb = C.T @ np.asarray(os_k.B, float)
        B_body[k] = Bb
        bn = np.linalg.norm(Bb)
        sigma_signed[k] = float(WHEEL_AXIS @ Bb / bn) if bn > 0 else 0.0
        rho[k] = float(getattr(os_k, "rho", np.nan))

        for d, nm in zip(sat.disturbances, names):
            try:
                tau = d.torque(sat=sat, x=x, os=os_k)
            except TypeError:
                tau = d.torque(x=x, os=os_k)
            torques[nm][k] = np.ravel(tau)

    # Separate secular drift from cyclic oscillation by fitting a ramp to the running
    # momentum over several orbits. The slope is what actually accumulates; the residual
    # is a bounded excursion that reserves wheel range but never saturates it.
    dt = float(t_s[1] - t_s[0])
    # Fit against time in ORBITS, centred: seconds against a ones column makes the design
    # matrix badly conditioned (column norms differ by ~1e4) and the solve warns.
    tau_orb = (t_s - t_s.mean()) / T
    A = np.vstack([tau_orb, np.ones_like(tau_orb)]).T
    per_src, cyc_src = {}, {}
    total_rate = np.zeros(3)
    for nm in names:
        run = np.cumsum(torques[nm], axis=0) * dt
        coef, *_ = np.linalg.lstsq(A, run, rcond=None)
        total_rate += coef[0]
        per_src[nm] = float(np.linalg.norm(coef[0]))               # secular, per orbit
        cyc_src[nm] = float(np.linalg.norm(run - A @ coef, axis=1).max())   # cyclic amp
    # The VECTOR sum, not the sum of magnitudes: drag-secular and dipole-secular point in
    # different directions and partially cancel, so the per-source magnitudes do not add.
    accum = float(np.linalg.norm(total_rate))

    # Only the component along the wheel axis saturates the WHEEL. Transverse secular
    # momentum has to be taken by the magnetorquers directly or it shows up as attitude
    # error -- a different failure mode with a different remedy, so it is reported apart.
    accum_along_a = float(abs(total_rate @ WHEEL_AXIS))
    accum_transverse = float(np.linalg.norm(total_rate - (total_rate @ WHEEL_AXIS) * WHEEL_AXIS))

    run_tot = np.cumsum(sum(torques[nm] for nm in names), axis=0) * dt
    coef_tot, *_ = np.linalg.lstsq(A, run_tot, rcond=None)
    cyc_total = float(np.linalg.norm(run_tot - A @ coef_tot, axis=1).max())

    one = slice(0, n // n_orbits)      # one orbit's worth, for the capacity integral
    tr = {"t_s": t_s[one], "sigma_signed": sigma_signed[one], "B_body": B_body[one],
          "sigma": np.abs(sigma_signed[one])}
    lim_auth = authority_limit(tr, WHEEL_AXIS, IAC_6U.m_max)
    cap = capacity(tr, lim_auth, TAU_ALLOW_REF, IAC_6U.tau_w)

    finite = lim_auth[np.isfinite(lim_auth)]
    return {
        "alt_km": alt_km, "T_orbit_s": T,
        "m_res": float(IAC_6U.m_res if m_res is None else m_res),
        "cyclic_by_source_Nms": cyc_src,
        "cyclic_total_Nms": cyc_total,
        "reserved_wheel_frac": cyc_total / IAC_6U.h_max,
        "accum_along_wheel_Nms": accum_along_a,
        "accum_transverse_Nms": accum_transverse,
        "orbits_to_saturate_along_a": (IAC_6U.h_max / accum_along_a
                                       if accum_along_a > 0 else float("inf")),
        "profile": profile,
        "median_rho_kg_m3": float(np.nanmedian(rho)),
        "median_B_T": float(np.median(np.linalg.norm(B_body, axis=1))),
        "median_tau_mtq_max_Nm": float(np.median(finite)) if finite.size else float("inf"),
        "accum_per_orbit_Nms": accum,
        "accum_by_source_Nms": per_src,
        "capacity_per_orbit_Nms": cap,
        "margin": cap / accum if accum > 0 else float("inf"),
        "orbits_to_saturate": IAC_6U.h_max / accum if accum > 0 else float("inf"),
    }


def main() -> int:
    from papers.IAC_1RW._iac_sim import assert_settled_bus
    assert_settled_bus()
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 100)
    print("Campaign F -- altitude scaling and the momentum-boundary altitude")
    print(f"boresight-mounted wheel, nadir-locked, inc {INC_DEG} deg, "
          f"tau_allow = {TAU_ALLOW_REF*1e6:.1f} uN m, {N_ORBITS_FIT} orbits fitted")
    print("=" * 100)

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    unity: Dict[str, Any] = {}

    for m_res in M_RES_CASES:
        tag = f"m_res={m_res}"
        print(f"\n### {tag} A m^2 " + ("(reference bus)" if m_res == 0.05
                                        else "(labelled sensitivity)"))
        print(f"{'alt[km]':>8}{'rho[kg/m3]':>13}{'B[uT]':>8}{'SECULAR':>10}{'drag':>9}"
              f"{'dipole':>9}{'CYCLIC':>9}{'wheel%':>8}{'cap':>9}{'margin':>8}{'orb->sat':>10}")
        print("-" * 100)
        rows = []
        for alt in ALTITUDES_KM:
            r = survey(alt, m_res=m_res)
            rows.append(r)
            sec = r["accum_by_source_Nms"]
            print(f"{alt:>8.0f}{r['median_rho_kg_m3']:>13.3e}{r['median_B_T']*1e6:>8.2f}"
                  f"{r['accum_per_orbit_Nms']*1e3:>10.4f}"
                  f"{sec.get('Drag_Disturbance',0)*1e3:>9.4f}"
                  f"{sec.get('Dipole_Disturbance',0)*1e3:>9.4f}"
                  f"{r['cyclic_total_Nms']*1e3:>9.3f}"
                  f"{100*r['reserved_wheel_frac']:>7.1f}%"
                  f"{r['capacity_per_orbit_Nms']*1e3:>9.2f}"
                  f"{r['margin']:>8.1f}{r['orbits_to_saturate']:>10.1f}")
        all_rows[tag] = rows

        def margin_at(alt, _m=m_res):
            return survey(alt, n=90, m_res=_m, n_orbits=3)["margin"] - 1.0

        lo, hi = 150.0, 400.0
        try:
            m_lo, m_hi = margin_at(lo), margin_at(hi)
            if m_lo * m_hi < 0:
                a_u = brentq(margin_at, lo, hi, xtol=2.0, rtol=1e-3, maxiter=40)
                unity[tag] = a_u
                print(f"  -> margin reaches unity at ~{a_u:.0f} km")
            else:
                unity[tag] = None
                print(f"  -> no unity crossing in [{lo:.0f}, {hi:.0f}] km "
                      f"({m_lo+1:.1f} at {lo:.0f}, {m_hi+1:.1f} at {hi:.0f})")
        except Exception as exc:
            unity[tag] = None
            print(f"  -> unity solve failed: {type(exc).__name__}: {exc}")

    # ---- profile comparison: is Earth observation really the harder momentum problem? ----
    #
    # IV-A claims drag's character flips between mission classes: ram-locked (nadir) keeps the
    # velocity direction body-fixed so the torque is secular, while an inertial hold lets the
    # velocity sweep the body frame once per orbit so much of it averages out. That is the
    # basis for saying EO is a harder momentum problem than inertial staring, and it is cheap
    # to measure rather than assert.
    print("\n" + "=" * 100)
    print("Profile comparison at 400 km (m_res = 0.05): does drag's character flip?")
    print(f"{'profile':<12}{'drag secular':>15}{'drag cyclic':>14}{'sec/cyc':>10}"
          f"{'total secular':>16}{'along a_hat':>13}{'orb->sat(a)':>13}")
    print("-" * 100)
    prof_rows = {}
    for prof in ("nadir", "inertial"):
        r = survey(400.0, m_res=0.05, profile=prof)
        prof_rows[prof] = r
        ds = r["accum_by_source_Nms"].get("Drag_Disturbance", 0.0)
        dc = r["cyclic_by_source_Nms"].get("Drag_Disturbance", 0.0)
        ratio = ds / dc if dc > 1e-12 else float("inf")
        print(f"{prof:<12}{ds*1e3:>15.4f}{dc*1e3:>14.4f}{ratio:>10.2f}"
              f"{r['accum_per_orbit_Nms']*1e3:>16.4f}{r['accum_along_wheel_Nms']*1e3:>13.4f}"
              f"{r['orbits_to_saturate_along_a']:>13.1f}")
    dn = prof_rows["nadir"]["accum_by_source_Nms"].get("Drag_Disturbance", 0.0)
    di = prof_rows["inertial"]["accum_by_source_Nms"].get("Drag_Disturbance", 0.0)
    if di > 0:
        print(f"\n  ram-locked drag secular is {dn/di:.1f}x the inertial-hold value "
              f"-> EO IS the harder momentum problem" if dn > di else
              f"\n  ram-locked drag secular is {dn/di:.2f}x inertial -> claim NOT supported")
    else:
        print("\n  inertial-hold drag secular is ~0 -> the flip is total")

    print("\n" + "=" * 100)
    print("SECULAR sets the dumping cadence; CYCLIC permanently reserves wheel range.")
    print("Summing them into one 'accumulation' column conflates two costs with different")
    print("remedies -- dumping cadence vs wheel sizing.")
    for tag, a_u in unity.items():
        if a_u is not None:
            print(f"  {tag}: momentum boundary binds at ~{a_u:.0f} km")
    print("=" * 100)

    payload = {"task": "F_altitude", "timestamp": ts,
               "inc_deg": INC_DEG, "tau_allow_Nm": TAU_ALLOW_REF,
               "n_orbits_fit": N_ORBITS_FIT,
               "wheel_axis": WHEEL_AXIS.tolist(),
               "h_max_Nms": IAC_6U.h_max, "m_max": IAC_6U.m_max,
               "com_offset_m": IAC_6U.com_offset_m,
               "altitude_unity_margin_km": unity,
               "rows_by_case": all_rows,
               "profile_comparison_400km": prof_rows}
    with open(f"{OUT}/F_altitude_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}/F_altitude_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
