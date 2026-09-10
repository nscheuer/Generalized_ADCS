"""Investigate the sharp |omega| valley at ~825 s in the 3+1 -> 3+0 RW-failure case.

Reproduces the EXACT deterministic run from failure_case_rw.py (seed 7, kill the wheel
at t=500 s) but logs the per-axis body rate, the field direction, the rate/field and
error-axis/field alignment, and the LP allocation scale -- so we can dissect what the
|omega|-magnitude plot hides at the valley. Does NOT change the test; pure diagnostics.

Output: papers/Generalized_ACS/output_data/fig_failure_valley_diag.png + console analysis.
"""
import os, sys
import numpy as np
ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import failure_case_rw as F   # reuse exact constants + helpers
from failure_case_rw import (LPAllocator, create_3mtq_1rw_config, time_varying_b_field,
                             skewsym, normalize, rot_mat, quat_err, full_att_err_deg,
                             boresight_err_deg, TAU_DIST, BORESIGHT, TF, DT, T_FAIL)


def ang(a, b):
    a = normalize(a); b = normalize(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def run_logged():
    rng = np.random.default_rng(7)                 # identical to failure_case_rw.main
    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    axis = normalize(rng.standard_normal(3)); a = np.radians(3.0)
    q0 = normalize(np.concatenate([[np.cos(a/2)], axis*np.sin(a/2)]))
    w0 = rng.standard_normal(3) * 0.002
    bfield = time_varying_b_field(orbit_period=5800.0)
    cfg = create_3mtq_1rw_config(tf=TF, dt=DT)
    x = np.concatenate([w0, q0, np.zeros(1)]); t = 0.0
    alloc = LPAllocator()
    n_rw, n_mtq = cfg.A_rw.shape[1], cfg.A_mtq_axes.shape[1]
    steps = int(TF/DT) + 1
    log = {k: np.zeros(steps) for k in ("t", "wx", "wy", "wz", "wn", "full", "bore",
                                        "ang_wB", "ang_eB", "alpha", "tdes", "tach")}
    log["wxyz"] = np.zeros((steps, 3))
    for k in range(steps):
        omega, q = x[0:3], normalize(x[3:7]); h_rw = x[7:8]
        b_body = bfield(q, t)
        u_rw_max = cfg.u_rw_max.copy()
        if t >= T_FAIL:
            u_rw_max[0] = 0.0
        q_e = quat_err(q, q_goal)[1:4]
        tau_pd = -cfg.kp*q_e - cfg.kd*omega
        tau_gyro = np.cross(omega, cfg.J @ omega + cfg.A_rw @ h_rw)
        tau_des = tau_pd + tau_gyro
        r = alloc.allocate(tau_des, b_body, cfg.A_rw, cfg.A_mtq_axes, u_rw_max, cfg.u_mtq_max)
        log["t"][k] = t
        log["wxyz"][k] = omega; log["wn"][k] = np.degrees(np.linalg.norm(omega))
        log["wx"][k], log["wy"][k], log["wz"][k] = np.degrees(omega)
        log["full"][k] = full_att_err_deg(q, q_goal); log["bore"][k] = boresight_err_deg(q, q_goal)
        log["ang_wB"][k] = ang(omega, b_body) if np.linalg.norm(omega) > 1e-12 else np.nan
        log["ang_eB"][k] = ang(q_e, b_body) if np.linalg.norm(q_e) > 1e-12 else np.nan
        log["alpha"][k] = getattr(r, "magnitude_ratio", getattr(r, "alpha", np.nan))
        log["tdes"][k] = np.linalg.norm(tau_des)
        log["tach"][k] = np.linalg.norm(getattr(r, "tau_achieved", np.zeros(3)))
        if k == steps - 1:
            break
        u_rw, u_mtq = r.u_rw, r.u_mtq

        def dyn(tl, y):
            w = y[0:3]; quat = normalize(y[3:7]); hrw = y[7:8]
            bl = bfield(quat, t+tl)
            tau_mtq = (-skewsym(bl) @ cfg.A_mtq_axes) @ u_mtq
            tau_rw = cfg.A_rw @ u_rw
            wd = np.linalg.solve(cfg.J, tau_rw + tau_mtq + TAU_DIST - np.cross(w, cfg.J @ w + cfg.A_rw @ hrw))
            W = np.zeros((4, 3)); W[0] = -quat[1:4]; W[1:4] = quat[0]*np.eye(3) + skewsym(quat[1:4])
            return np.concatenate([wd, 0.5*W @ w, -u_rw])
        x = solve_ivp(dyn, [0, DT], x, method="RK45", rtol=1e-8, atol=1e-10).y[:, -1]
        x[3:7] = normalize(x[3:7]); t += DT
    return log


def main():
    L = run_logged()
    t = L["t"]
    # locate the valley: |omega| minimum within the post-failure peak window
    win = (t >= 700) & (t <= 950)
    iv = np.where(win)[0][np.argmin(L["wn"][win])]
    tv = t[iv]
    ipk = np.where(win)[0][np.argmax(L["full"][win])]
    print(f"=== 3+1 -> 3+0 RW-failure valley diagnosis ===")
    print(f"|omega| minimum (valley): t={tv:.0f}s  |w|={L['wn'][iv]:.4f} deg/s")
    print(f"  components at valley: wx={L['wx'][iv]:+.4f} wy={L['wy'][iv]:+.4f} wz={L['wz'][iv]:+.4f} deg/s")
    print(f"full-attitude error peak: t={t[ipk]:.0f}s  full={L['full'][ipk]:.1f} deg (turnaround)")
    print(f"  at valley: full={L['full'][iv]:.1f} bore={L['bore'][iv]:.1f} deg")
    print(f"  angle(omega,B)={L['ang_wB'][iv]:.0f} deg  angle(err-axis,B)={L['ang_eB'][iv]:.0f} deg")
    print(f"  LP alpha(achieved/desired)={L['alpha'][iv]:.3f}  |tau_des|={L['tdes'][iv]:.2e}  |tau_ach|={L['tach'][iv]:.2e}")
    # is the valley exactly the attitude-error turnaround? (sign flip of d(full)/dt)
    dfull = np.gradient(L["full"], t)
    print(f"  d(full)/dt at valley = {dfull[iv]:+.3f} deg/s (≈0 ⇒ error turnaround)")

    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    ax[0].plot(t, L["full"], "C3", label="full attitude"); ax[0].plot(t, L["bore"], "C1", ls="--", label="boresight")
    ax[0].set_ylabel("error [deg]"); ax[0].legend(fontsize=8)
    ax[1].plot(t, L["wx"], label="ωx"); ax[1].plot(t, L["wy"], label="ωy"); ax[1].plot(t, L["wz"], label="ωz")
    ax[1].plot(t, L["wn"], "k", lw=1.6, label="|ω|"); ax[1].set_ylabel("body rate [deg/s]"); ax[1].legend(fontsize=8, ncol=4)
    ax[2].plot(t, L["ang_wB"], "C4", label="∠(ω, B)"); ax[2].plot(t, L["ang_eB"], "C2", ls="--", label="∠(err-axis, B)")
    ax[2].axhline(90, ls=":", c="0.6"); ax[2].set_ylabel("angle [deg]"); ax[2].legend(fontsize=8)
    ax[3].plot(t, L["alpha"], "C0", label="LP α (achieved/desired)"); ax[3].set_ylabel("LP α"); ax[3].legend(fontsize=8)
    for a in ax:
        a.axvline(T_FAIL, color="k", ls=":", lw=1.0); a.axvline(tv, color="r", ls="-", lw=0.8, alpha=0.5)
        a.grid(alpha=0.3); a.set_xlim(400, 1000)
    ax[3].set_xlabel("time [s]")
    fig.tight_layout()
    out = os.path.join(ROOT, "papers/Generalized_ACS/output_data/fig_failure_valley_diag.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
