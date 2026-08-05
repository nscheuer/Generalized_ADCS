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


def survey(alt_km: float, n: int = N_SAMPLES) -> Dict[str, Any]:
    """Per-orbit accumulation and dump capacity at one altitude, nadir-locked."""
    ephem = Ephemeris()
    sat = create_iac_6u_bus(n_rw=1)
    names = [type(d).__name__ for d in sat.disturbances]
    T = period_s(alt_km)
    t_s = np.linspace(0.0, T, n, endpoint=False)

    torques = {nm: np.zeros((n, 3)) for nm in names}
    sigma_signed = np.zeros(n)
    B_body = np.zeros((n, 3))
    rho = np.zeros(n)

    for k, t in enumerate(t_s):
        u = 2.0 * np.pi * t / T
        R, V = rv_at(alt_km, u)
        os_k = Orbital_State(ephem=ephem, J2000=EPOCH + t * TimeConstants.sec2cent,
                             R=R, V=V)
        C = body_to_eci("nadir", R, V)
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

    # Secular accumulation = orbit-mean body torque x period (what a body-fixed wheel takes).
    per_src = {nm: float(np.linalg.norm(torques[nm].mean(axis=0)) * T) for nm in names}
    total_vec = sum(torques[nm].mean(axis=0) for nm in names)
    accum = float(np.linalg.norm(total_vec) * T)

    tr = {"t_s": t_s, "sigma_signed": sigma_signed, "B_body": B_body,
          "sigma": np.abs(sigma_signed)}
    lim_auth = authority_limit(tr, WHEEL_AXIS, IAC_6U.m_max)
    cap = capacity(tr, lim_auth, TAU_ALLOW_REF, IAC_6U.tau_w)

    finite = lim_auth[np.isfinite(lim_auth)]
    return {
        "alt_km": alt_km, "T_orbit_s": T,
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 92)
    print("Campaign F -- altitude scaling and the momentum-boundary altitude")
    print(f"boresight-mounted wheel, nadir-locked, inc {INC_DEG} deg, "
          f"tau_allow = {TAU_ALLOW_REF*1e6:.1f} uN m")
    print("=" * 92)
    print(f"\n{'alt[km]':>8}{'T[s]':>8}{'rho[kg/m3]':>13}{'B[uT]':>8}"
          f"{'accum[mNms]':>13}{'cap[mNms]':>11}{'margin':>9}{'orbits->sat':>13}")
    print("-" * 92)

    rows: List[Dict[str, Any]] = []
    for alt in ALTITUDES_KM:
        r = survey(alt)
        rows.append(r)
        print(f"{alt:>8.0f}{r['T_orbit_s']:>8.0f}{r['median_rho_kg_m3']:>13.3e}"
              f"{r['median_B_T']*1e6:>8.2f}{r['accum_per_orbit_Nms']*1e3:>13.4f}"
              f"{r['capacity_per_orbit_Nms']*1e3:>11.3f}{r['margin']:>9.1f}"
              f"{r['orbits_to_saturate']:>13.1f}")

    # ---- the number the claim needs: where does margin reach unity? -------------------
    def margin_at(alt):
        return survey(alt, n=120)["margin"] - 1.0

    print("\n" + "-" * 92)
    lo, hi = 150.0, 400.0
    try:
        m_lo, m_hi = margin_at(lo), margin_at(hi)
        if m_lo * m_hi < 0:
            alt_unity = brentq(margin_at, lo, hi, xtol=2.0, rtol=1e-3, maxiter=40)
            print(f"MOMENTUM BOUNDARY BINDS AT ~{alt_unity:.0f} km "
                  f"(margin = 1 for the boresight mounting).")
            print(f"Above it the architecture is agility- and accuracy-limited, not "
                  f"momentum-limited.")
        else:
            alt_unity = None
            side = "above" if m_lo > 0 else "below"
            print(f"margin does not cross unity in [{lo:.0f}, {hi:.0f}] km "
                  f"(margin stays {side} 1: {m_lo+1:.1f} at {lo:.0f} km, "
                  f"{m_hi+1:.1f} at {hi:.0f} km).")
            print("The momentum boundary does not bind anywhere in the plausible LEO range.")
    except Exception as exc:            # keep the table even if the solve misbehaves
        alt_unity = None
        print(f"unity solve failed: {type(exc).__name__}: {exc}")

    payload = {"task": "F_altitude", "timestamp": ts,
               "inc_deg": INC_DEG, "tau_allow_Nm": TAU_ALLOW_REF,
               "wheel_axis": WHEEL_AXIS.tolist(),
               "h_max_Nms": IAC_6U.h_max, "m_max": IAC_6U.m_max,
               "com_offset_m": IAC_6U.com_offset_m, "m_res": IAC_6U.m_res,
               "altitude_unity_margin_km": alt_unity,
               "rows": rows}
    with open(f"{OUT}/F_altitude_{ts}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("=" * 92)
    print(f"\nwrote {OUT}/F_altitude_{ts}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
