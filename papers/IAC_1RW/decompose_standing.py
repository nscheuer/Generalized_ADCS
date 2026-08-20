"""Two on-disk checks feeding the sweep's interpretation (Patrick, 2026-08-20):

CHECK A: decompose the 0.71 deg STANDING error into along-B and transverse parts.
  Error 'direction' = axis n of the boresight-correcting rotation (theta about n).
  Along-field share = (n . B_hat)^2, theta^2-weighted; isotropy baseline 1/3.
  If the plan's residual pools along-field, part of the floor is the corridor bound
  (Theorem 3) appearing in the optimizer -- not removable by any weight ratio --
  and the sweep should show angle-weight returns asymptoting at the along-field
  component instead of going to zero.

CHECK B: is the JOLT direction transverse to B?
  Excursion rotation vector delta (body frame) from q(t0) -> q(t0+75) at each
  replan; alignment = (delta_hat . B_hat_body)^2 vs isotropy 1/3. Consistent
  transverse orientation = bracket-maneuver signature (held loosely per the
  magnitude mismatch).
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import (  # noqa: E402
    _get_orbit, error_series, EPOCH, IAC_6U)
from ADCS.orbits.universal_constants import TimeConstants  # noqa: E402
from ADCS.helpers.math_helpers import quat_inv, quat_mult, normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")
SEC2CENT = TimeConstants.sec2cent


def rot_body_to_eci(q, v):
    w, x, y, z = q
    vv = np.asarray([x, y, z])
    t = 2.0 * np.cross(vv, v)
    return v + w * t + np.cross(vv, t)


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    files = sorted(glob.glob(os.path.join(OUT, "A_trials", "1rw_reduced_planner_seed*.pkl")))
    bore = np.asarray(IAC_6U.boresight, float)

    fracs, th_along, th_perp, th_tot = [], [], [], []
    jolt_align, jolt_mag = [], []
    for p in files:
        with open(p, "rb") as f:
            r = pickle.load(f)
        e = error_series(r)
        if e[-1] > 30:
            continue
        cfg = r["config"]
        t = np.asarray(r["time"], float)
        q = np.asarray(r["state"], float)[:, 3:7]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        tgt = normalize(np.asarray(cfg["goal_vec"], float))
        orb = _get_orbit(cfg, 1.0, float(cfg["tf"]))

        # ---- CHECK A: held-window error-axis decomposition ----
        num = den = 0.0
        tha, thp, tht = [], [], []
        for tm in np.arange(3500.0, 5400.0, 20.0):
            i = int(np.searchsorted(t, tm))
            if i >= len(q):
                continue
            b_eci = rot_body_to_eci(q[i], bore)
            c = float(np.clip(b_eci @ tgt, -1, 1))
            th = np.arccos(c)
            if th < 1e-6:
                continue
            n = np.cross(b_eci, tgt)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            os_k = orb.get_os(J2000=EPOCH + tm * SEC2CENT)
            Bh = normalize(np.asarray(os_k.B, float))
            a2 = float((n @ Bh) ** 2)
            num += th * th * a2
            den += th * th
            tha.append(np.rad2deg(th) * abs(n @ Bh))
            thp.append(np.rad2deg(th) * np.sqrt(max(0.0, 1 - a2)))
            tht.append(np.rad2deg(th))
        if den > 0:
            fracs.append(num / den)
            th_along.append(float(np.median(tha)))
            th_perp.append(float(np.median(thp)))
            th_tot.append(float(np.median(tht)))

        # ---- CHECK B: jolt direction vs B (body frame) ----
        for w0 in range(1000, 5001, 500):
            i0 = int(np.searchsorted(t, w0))
            i1 = int(np.searchsorted(t, w0 + 75))
            if i1 >= len(q):
                continue
            dq = quat_mult(quat_inv(q[i0]), q[i1])
            dq = dq / np.linalg.norm(dq)
            ang = 2.0 * np.arccos(np.clip(abs(dq[0]), 0, 1))
            if ang < np.deg2rad(0.02):
                continue
            axis = dq[1:4] * np.sign(dq[0])
            an = np.linalg.norm(axis)
            if an < 1e-12:
                continue
            axis /= an
            os_k = orb.get_os(J2000=EPOCH + w0 * SEC2CENT)
            # body-frame B: rotate the ECI field by the inverse attitude
            Bb = normalize(rot_body_to_eci(quat_inv(q[i0]), np.asarray(os_k.B, float)))
            jolt_align.append(float((axis @ Bb) ** 2))
            jolt_mag.append(float(np.rad2deg(ang)))

    say(f"converged trials analyzed: {len(fracs)}")
    say("\n== CHECK A: standing-error decomposition (held window, theta^2-weighted) ==")
    say(f"along-B energy fraction: median {np.median(fracs):.3f}  IQR "
        f"[{np.percentile(fracs,25):.3f}, {np.percentile(fracs,75):.3f}]  (isotropy 1/3)")
    say(f"median standing error {np.median(th_tot):.3f} deg = along {np.median(th_along):.3f} "
        f"+ perp {np.median(th_perp):.3f} (medians of per-trial medians; not additive exactly)")
    say("reading: fraction >> 1/3 => corridor floor visible in the PLAN's residual; "
        "sweep should asymptote at the along component. fraction ~ 1/3 => elective "
        "equilibrium, isotropic; 'elective floor' sentence stands as-is.")

    say("\n== CHECK B: jolt direction vs B (pooled excursions) ==")
    ja = np.asarray(jolt_align)
    say(f"n excursions {len(ja)}; (axis.B)^2: median {np.median(ja):.3f}  mean {np.mean(ja):.3f} "
        f" (isotropy 1/3; 0 = pure transverse, 1 = about-B)")
    say(f"fraction of excursions with (axis.B)^2 < 0.1: {np.mean(ja < 0.1):.2f}")
    say(f"jolt rotation magnitude: median {np.median(jolt_mag):.2f} deg")

    with open(os.path.join(OUT, "STANDING_DECOMP.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    say("\nwritten: output_data/STANDING_DECOMP.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
