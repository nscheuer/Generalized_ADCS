"""Campaign §2 environment cross-check -- **run this before anything else**.

The campaign spec gives four hand-computed peak disturbance torques and says: "if the simulated
values disagree with these by more than ~2x, stop and reconcile before running anything else."
These hand calculations drive the entire boundary analysis in Section IV, so this script is the
gate on R, A, B, C, D, E and F.

It also answers two questions the spec leaves as assumptions:

* **Is the residual dipole really "cyclic"?** It is the largest disturbance by ~15x, so even a
  small secular fraction would dominate the momentum budget -- plausibly over drag. This
  measures the orbit-mean, in the body frame, which is what a body-fixed wheel actually absorbs.
* **Is SRP torque non-zero once the COM is offset?** For a uniform box with COM at the geometric
  centre both drag and SRP torques vanish identically. Drag is verified elsewhere; this confirms
  SRP behaves the same way and that the offset revives it.

Run: ``python papers/IAC_1RW/check_environment.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.helpers.math_helpers import normalize
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.satellite_factory import create_iac_6u_bus, IAC_6U

MU = EarthConstants.mu_e
R_E = EarthConstants.R_e
ALT_KM = 400.0
A_KM = R_E + ALT_KM
INC_DEG = 97.0
T_ORBIT = 2.0 * np.pi * np.sqrt(A_KM ** 3 / MU)
EPOCH = 0.22

# Campaign §2 predictions [N m]. The spec's column is "Predicted **peak** torque", so the
# comparison below is a peak over attitudes *and* over the orbit -- not the value along one
# attitude profile. That distinction is load-bearing for two of the four:
#
#   * gravity gradient is identically zero for a principal-axis nadir lock (n_hat x J n_hat
#     vanishes when the minor axis points at nadir), so a nadir-locked trace reads 0.000 and
#     tells you nothing about whether the model is right;
#   * drag torque is prop. to (c_hat x V_hat) and so vanishes whenever the cp-cg offset is
#     parallel to ram.
#
# Neither is a model defect; both would read as one if the gate compared against a single
# profile. The per-profile secular analysis is reported separately, further down.
#
# The dipole prediction is quoted in the spec for m_res = 0.1 A m^2 at B = 30 uT. It is scaled
# here to the configured m_res so the gate stays valid under the 0.05 reference / 0.1
# sensitivity decision.
_SPEC_DIPOLE_M_RES = 0.1
PREDICTED = {
    "dipole": 3.0e-6 * (IAC_6U.m_res / _SPEC_DIPOLE_M_RES),
    "drag": 0.2e-6,
    "gg": 0.15e-6,
    "srp": 0.008e-6,
}
TOLERANCE = 2.0  # the spec's "~2x" gate


def rv_circular(u_rad: float, inc_deg: float = INC_DEG, raan_deg: float = 0.0):
    """ECI position/velocity on a circular orbit at argument-of-latitude ``u``."""
    i, Om = np.deg2rad(inc_deg), np.deg2rad(raan_deg)
    v_circ = np.sqrt(MU / A_KM)
    cu, su, cO, sO, ci, si = (np.cos(u_rad), np.sin(u_rad), np.cos(Om),
                              np.sin(Om), np.cos(i), np.sin(i))
    r_hat = np.array([cO * cu - sO * ci * su, sO * cu + cO * ci * su, si * su])
    v_hat = np.array([-cO * su - sO * ci * cu, -sO * su + cO * ci * cu, si * cu])
    return A_KM * r_hat, v_circ * v_hat


def quat_from_dcm_cols(x_b_eci, y_b_eci, z_b_eci):
    """Quaternion (body->ECI) from the three body axes expressed in ECI."""
    C = np.column_stack([x_b_eci, y_b_eci, z_b_eci])
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


def nadir_locked_quat(R, V):
    """Boresight (+z) at nadir, +y along negative orbit normal, +x completing (ram-ish)."""
    z_b = -normalize(R)                       # nadir
    h = np.cross(R, V)
    y_b = -normalize(h)                       # negative orbit normal
    x_b = np.cross(y_b, z_b)
    return quat_from_dcm_cols(x_b, y_b, z_b)


def sample_orbit(n=240, inc_deg=INC_DEG, attitude="nadir", q_fixed=None):
    """Walk one orbit, returning per-disturbance body torques and the orbital states."""
    ephem = Ephemeris()
    sat = create_iac_6u_bus(n_rw=1)
    names = [type(d).__name__ for d in sat.disturbances]

    us = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    out = {nm: np.zeros((n, 3)) for nm in names}
    quats = np.zeros((n, 4))

    for k, u in enumerate(us):
        R, V = rv_circular(u, inc_deg=inc_deg)
        t_s = (u / (2.0 * np.pi)) * T_ORBIT
        os_k = Orbital_State(ephem=ephem, J2000=EPOCH + t_s * TimeConstants.sec2cent,
                             R=R, V=V)
        q = nadir_locked_quat(R, V) if attitude == "nadir" else q_fixed
        quats[k] = q
        x = np.concatenate([np.zeros(3), q, np.zeros(1)])
        for d, nm in zip(sat.disturbances, names):
            try:
                tau = d.torque(sat=sat, x=x, os=os_k)
            except TypeError:
                tau = d.torque(x=x, os=os_k)
            out[nm][k] = np.ravel(tau)
    return out, quats, sat


def peak_over_attitudes(n_orbit=48, n_att=64, seed=0):
    """Peak |tau| per source over a grid of orbit positions x random attitudes.

    This is what the spec's "predicted peak torque" column means. Sampling attitudes as well
    as orbit positions is what makes the gravity-gradient and drag numbers meaningful -- both
    have attitudes at which they vanish exactly.
    """
    ephem = Ephemeris()
    sat = create_iac_6u_bus(n_rw=1)
    names = [type(d).__name__ for d in sat.disturbances]
    rng = np.random.default_rng(seed)
    quats = [normalize(rng.standard_normal(4)) for _ in range(n_att)]
    peaks = {nm: 0.0 for nm in names}

    for u in np.linspace(0.0, 2.0 * np.pi, n_orbit, endpoint=False):
        R, V = rv_circular(u)
        t_s = (u / (2.0 * np.pi)) * T_ORBIT
        os_k = Orbital_State(ephem=ephem, J2000=EPOCH + t_s * TimeConstants.sec2cent,
                             R=R, V=V)
        for q in quats:
            x = np.concatenate([np.zeros(3), q, np.zeros(1)])
            for d, nm in zip(sat.disturbances, names):
                try:
                    tau = d.torque(sat=sat, x=x, os=os_k)
                except TypeError:
                    tau = d.torque(x=x, os=os_k)
                peaks[nm] = max(peaks[nm], float(np.linalg.norm(np.ravel(tau))))
    return peaks


def main() -> int:
    print("=" * 78)
    print("IAC 6U -- Campaign §2 environment cross-check")
    print(f"400 km circular, inc {INC_DEG} deg, T_orbit = {T_ORBIT:.1f} s, "
          f"m_res = {IAC_6U.m_res} A m^2, cp-cg = {IAC_6U.com_offset_m*100:.0f} cm "
          f"on {np.array(IAC_6U.com_offset_dir).astype(int)}")
    print("=" * 78)

    torques, quats, sat = sample_orbit(attitude="nadir")
    key = {"Dipole_Disturbance": "dipole", "Drag_Disturbance": "drag",
           "GG_Disturbance": "gg", "SRP_Disturbance": "srp"}

    print("\nPeak over orbit positions x attitudes -- the spec's 'predicted peak' column.")
    print(f"\n{'source':<10}{'peak [uN m]':>14}{'predicted':>12}{'ratio':>9}   verdict")
    print("-" * 78)
    ok = True
    peaks_att = peak_over_attitudes()
    peaks = {}
    for cls, short in key.items():
        peak = peaks_att[cls]
        peaks[short] = peak
        pred = PREDICTED[short]
        ratio = peak / pred if pred else np.inf
        good = (1.0 / TOLERANCE) <= ratio <= TOLERANCE
        ok &= good
        print(f"{short:<10}{peak*1e6:>14.4f}{pred*1e6:>12.4f}{ratio:>9.2f}   "
              f"{'OK' if good else '** OUT OF TOLERANCE **'}")

    # ---- Is the residual dipole actually cyclic? --------------------------------------
    print("\n" + "-" * 78)
    print("Residual dipole: cyclic or secular?  (body frame -- what a body-fixed wheel sees)")
    print("-" * 78)
    dip = torques["Dipole_Disturbance"]
    mean_body = dip.mean(axis=0)
    peak_dip = np.linalg.norm(dip, axis=1).max()
    frac = float(np.linalg.norm(mean_body) / peak_dip)
    print(f"  orbit-mean |tau|      = {np.linalg.norm(mean_body)*1e6:8.4f} uN m")
    print(f"  peak |tau|            = {peak_dip*1e6:8.4f} uN m")
    print(f"  secular fraction      = {frac:8.3f}   "
          f"({'NOT purely cyclic' if frac > 0.05 else 'effectively cyclic'})")

    # Per-orbit momentum accumulation vs storage, for each source and in total.
    print("\n" + "-" * 78)
    print("Per-orbit secular momentum accumulation vs h_max "
          f"({IAC_6U.h_max*1e3:.0f} mN m s)")
    print("-" * 78)
    total = np.zeros(3)
    for cls, short in key.items():
        acc = torques[cls].mean(axis=0) * T_ORBIT
        total += acc
        print(f"  {short:<8} {np.linalg.norm(acc)*1e3:8.4f} mN m s/orbit"
              f"   = {100*np.linalg.norm(acc)/IAC_6U.h_max:6.2f}% of h_max")
    n_orb = IAC_6U.h_max / max(np.linalg.norm(total), 1e-30)
    print(f"  {'TOTAL':<8} {np.linalg.norm(total)*1e3:8.4f} mN m s/orbit"
          f"   = {100*np.linalg.norm(total)/IAC_6U.h_max:6.2f}% of h_max")
    print(f"\n  -> saturation from rest in ~{n_orb:.1f} orbits.")
    if n_orb > 2.0:
        print("     CONFIRMS the audit: a one-orbit horizon is structurally blind to")
        print("     momentum saturation. Campaign E's saturation-line points must run")
        print("     multi-orbit (or be computed analytically); Campaign C reaches it only")
        print("     because it pre-loads h_0.")

    # ---- Ram-locked vs inertial hold: the §IV-A claim ---------------------------------
    print("\n" + "-" * 78)
    print("Drag secular content: ram-locked (nadir) vs inertial hold   [§IV-A claim]")
    print("-" * 78)
    q_inert = nadir_locked_quat(*rv_circular(0.0))
    t_inert, _, _ = sample_orbit(attitude="fixed", q_fixed=q_inert)
    for label, arr in (("nadir-locked", torques["Drag_Disturbance"]),
                       ("inertial hold", t_inert["Drag_Disturbance"])):
        pk = np.linalg.norm(arr, axis=1).max()
        mn = np.linalg.norm(arr.mean(axis=0))
        print(f"  {label:<15} peak {pk*1e6:7.4f} uN m   orbit-mean {mn*1e6:7.4f} uN m"
              f"   secular fraction {mn/max(pk,1e-30):6.3f}")
    print("  (the claim is that the ram-locked case retains a much larger secular fraction)")

    # ---- SRP is only non-zero because the COM is offset -------------------------------
    print("\n" + "-" * 78)
    print("cp-cg offset check: uniform box with COM at the geometric centre")
    print("-" * 78)
    centred = create_iac_6u_bus(n_rw=1, com_offset_m=0.0)
    ephem = Ephemeris()
    R, V = rv_circular(1.0)
    os_k = Orbital_State(ephem=ephem, J2000=EPOCH, R=R, V=V)
    rng = np.random.default_rng(0)
    worst = {"Drag_Disturbance": 0.0, "SRP_Disturbance": 0.0}
    for _ in range(300):
        q = normalize(rng.standard_normal(4))
        x = np.concatenate([np.zeros(3), q, np.zeros(1)])
        for d in centred.disturbances:
            nm = type(d).__name__
            if nm in worst:
                try:
                    tau = d.torque(sat=centred, x=x, os=os_k)
                except TypeError:
                    tau = d.torque(x=x, os=os_k)
                worst[nm] = max(worst[nm], float(np.linalg.norm(tau)))
    for nm, v in worst.items():
        print(f"  COM at centre: max |tau_{nm.split('_')[0].lower():<4}| over 300 attitudes "
              f"= {v:.3e} N m")
    print(f"  with the {IAC_6U.com_offset_m*100:.0f} cm offset: drag peak "
          f"{peaks['drag']*1e6:.4f} uN m, SRP peak {peaks['srp']*1e6:.4f} uN m")
    print("  -> the offset is the entire source of both torques.")

    print("\n" + "=" * 78)
    if ok:
        print("GATE PASSED -- simulated budget agrees with the hand calculations within 2x.")
    else:
        print("GATE FAILED -- reconcile before running any campaign.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
