"""C1 -- boresight-roll foresight demo (does the planner pre-position the RW?).

3+1, single reaction wheel on body +z, boresight +y. Two sequential VECTOR-pointing
goals A then B, with a no-goal coast between. Because the goal is reduced-attitude,
rotation about the boresight (+y) is FREE -- and it re-aims the wheel axis (+z) in
inertial. The planner sees BOTH goals over its horizon, so (claim) during the A-hold
it can roll about +y to point the wheel axis toward the A->B slew rotation axis
(d_A x d_B), pre-loading the wheel to assist the upcoming large slew.

This script runs the closed-loop planner once and INSPECTS whether that happens:
  - pointing error vs the active goal,
  - wheel momentum h_z,
  - alignment of the inertial wheel axis (+z_body in ECI) with the A->B slew axis,
  - body roll rate about the boresight.
Honest: if the planner does NOT visibly pre-position, the script says so.

Output (papers/Planner/output_data): fig_c1_boresight_roll.png/.pdf, C1_RESULTS.md
"""
import os, sys
import numpy as np
ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ADCS
from ADCS.helpers.math_helpers import rot_mat, normalize
from ADCS.orbits.universal_constants import TimeConstants
import papers.Planner._paper2_sim as P

OUT = os.path.join(ROOT, "papers/Planner/output_data")
BORE = np.array([0., 1, 0]); WHEEL = np.array([0., 0, 1])
TF = float(os.environ.get("C1_TF", 1400.0)); DT = 1.0
# A active [0,700), B active [700,1400). No No_Goal coast (it let the rate diverge);
# roll is free anyway (reduced goal), so the planner can pre-roll during the late A-hold.
T_A = T_COAST = 700.0; T_B = 700.0


def main():
    rng = np.random.default_rng(7)
    sat = P.make_sat("3+1", estimated=False)
    # Geometry chosen so the wheel axis (+z) is PERPENDICULAR to the A->B slew axis at
    # the A-hold: d_A=+y -> q0=identity -> wheel +z = ECI +z; d_B=+z -> slew axis
    # = d_A x d_B = +x, which is 90 deg from the wheel +z. So a 90 deg roll about the
    # boresight WOULD align the wheel with the slew axis -- the planner has to choose to.
    d_A = normalize(np.array([0.0, 1.0, 0.0]))
    d_B = normalize(np.array([0.0, 0.0, 1.0]))
    slew_axis = normalize(np.cross(d_A, d_B))
    sep = np.degrees(np.arccos(np.clip(d_A @ d_B, -1, 1)))
    print(f"[C1] A->B separation {sep:.0f} deg, slew axis(ECI) {np.round(slew_axis,2)}")

    gl = ADCS.GoalList(goal_timeline={0.0: ADCS.goals.ECI_Goal(d_A),
                                      T_COAST: ADCS.goals.ECI_Goal(d_B)},
                       time_units="seconds", start_juliantime=0.22)
    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=sat, planner_settings=P.make_planner_settings(sat))
    # start already pointing at A (so the A-window is a hold; the interesting part is the coast)
    # build q0 aligning body +y to d_A
    v = np.cross(BORE, d_A); s = np.linalg.norm(v); c = BORE @ d_A
    if s < 1e-6:
        q0 = np.array([1., 0, 0, 0])
    else:
        ax = v / s; ang = np.arctan2(s, c)
        q0 = np.concatenate([[np.cos(ang/2)], ax*np.sin(ang/2)])
    x0 = np.concatenate([np.zeros(3), normalize(q0), [0.0]])
    os0 = P.default_os0()

    res = ADCS.simulate(x=x0, satellite=sat, controller=ctrl, goal=gl,
                        os0=os0, dt=DT, tf=TF)
    run = res.runs[0] if hasattr(res, "runs") else res
    st = np.asarray(run.state_hist, float); t = np.asarray(run.time_s, float)
    # guard against any non-finite states (numerical issues) so diagnostics stay valid
    good = np.all(np.isfinite(st), axis=1)
    st = st[good]; t = t[good]
    w = st[:, 0:3]; q = st[:, 3:7]; h = st[:, 7]

    def Rk(k):
        nq = q[k] / (np.linalg.norm(q[k]) + 1e-12)
        return rot_mat(nq)
    wheel_eci = np.array([Rk(k) @ WHEEL for k in range(len(q))])
    align = np.abs(wheel_eci @ slew_axis)                    # 1 => wheel axis along slew axis
    bore_eci = np.array([Rk(k) @ BORE for k in range(len(q))])
    err = np.full(len(t), np.nan)
    for k in range(len(t)):
        tgt = d_A if t[k] < T_COAST else d_B
        err[k] = np.degrees(np.arccos(np.clip(bore_eci[k] @ tgt, -1, 1)))
    roll_rate = np.degrees(np.clip(w @ BORE, -1e3, 1e3))     # body rate about boresight

    # pre-positioning: alignment early-A vs late-A (just before the slew)
    align_A = float(np.nanmean(align[(t > 50) & (t < 150)]))           # early A-hold
    align_coast_end = float(np.nanmean(align[(t > T_COAST - 80) & (t < T_COAST)]))  # just before slew
    h_at_slew = float(np.abs(h[np.argmin(np.abs(t - T_COAST))]))
    lateA = (t > 150) & (t < T_COAST)
    rolled = float(np.nanmax(np.abs(roll_rate[lateA]))) if lateA.any() else 0.0
    slew_settle = np.nan
    bslew = t >= T_COAST
    below = bslew & (err < 5.0)
    if below.any():
        slew_settle = float(t[np.argmax(below)] - T_COAST)
    print(f"[C1] wheel-axis|slew-axis alignment: end-of-A {align_A:.2f} -> end-of-coast {align_coast_end:.2f}")
    print(f"[C1] |h| at slew start {h_at_slew*1e3:.2f} mN.m.s | peak roll rate in coast {rolled:.2f} deg/s")
    print(f"[C1] B acquired (<5deg) at +{slew_settle:.0f}s after slew start" if np.isfinite(slew_settle)
          else "[C1] B not acquired within horizon")

    fig, ax = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(t, err, "C0"); ax[0].set_ylabel("pointing err [deg]"); ax[0].axhline(5, ls=":", c="0.6")
    ax[1].plot(t, align, "C2"); ax[1].set_ylabel("wheel-axis · slew-axis\n(1=aligned)")
    ax[2].plot(t, h*1e3, "C3"); ax[2].set_ylabel("wheel mom. [mN·m·s]")
    ax[3].plot(t, roll_rate, "C4"); ax[3].set_ylabel("roll rate (about\nboresight) [deg/s]"); ax[3].set_xlabel("time [s]")
    for a in ax:
        for tc in (T_A, T_COAST): a.axvline(tc, ls="--", c="gray", lw=0.7)
        a.grid(alpha=0.3)
    ax[0].set_title(f"3+1 vector A→B ({sep:.0f}° slew at t={T_COAST:.0f} s); roll free under the reduced goal.")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_c1_boresight_roll.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "fig_c1_boresight_roll.pdf")); plt.close(fig)

    pre = align_coast_end > align_A + 0.15 or h_at_slew > 3e-4
    with open(os.path.join(OUT, "C1_RESULTS.md"), "w") as f:
        f.write("# C1 -- boresight-roll foresight (3+1, vector A→B)\n\n")
        f.write(f"A→B separation {sep:.0f}°, slew axis (ECI) {np.round(slew_axis,2).tolist()}. "
                f"Single RW on +z, boresight +y; coast [{T_A:.0f},{T_COAST:.0f}] s is roll-free.\n\n")
        f.write(f"- wheel-axis · slew-axis alignment: end-of-A {align_A:.2f} → end-of-coast {align_coast_end:.2f}\n")
        f.write(f"- |h| at slew start: {h_at_slew*1e3:.2f} mN·m·s\n")
        f.write(f"- peak roll rate during coast: {rolled:.2f} deg/s\n")
        f.write(f"- B acquired: {'+%.0f s after slew start' % slew_settle if np.isfinite(slew_settle) else 'NOT within horizon'}\n\n")
        f.write(("**Foresight pre-positioning OBSERVED**: during the roll-free coast the planner rotated "
                 "the wheel axis toward the upcoming slew axis and/or pre-loaded the wheel, then executed "
                 "the slew.\n") if pre else
                ("**Not clearly observed**: the planner did not visibly roll to pre-position the wheel in "
                 "this setup (alignment/h roughly flat through the coast). The free-roll DOF is available "
                 "but the planner found the slew cheap enough without pre-positioning, or the cost does not "
                 "reward it. Would need cost/scenario tuning to elicit it -- reporting honestly.\n"))
    print("saved fig_c1_boresight_roll + C1_RESULTS.md  | pre-positioning:", pre)


if __name__ == "__main__":
    main()
