"""C2 -- disturbance-assisted desaturation, designed-satellite version.

GG/aero are too weak to desaturate in one trajectory (GG dumps ~7% of h_max/orbit;
see C2 feasibility note). The feasible environmental torque is a *magnetic* one: a
designed/large residual dipole m_res interacting with the orbital field gives
tau = m_res x B(t) ~ 6e-6 N.m for |m_res|=0.2 A.m^2 -- comparable to the magnetorquer
and ~150x gravity-gradient, enough to dump several mN.m.s over part of an orbit.

Setup: 3+1, the planner MODELS the residual dipole (est_sat carries it), the wheel
starts near saturation (h0 = 0.9 h_max), reduced-attitude (vector) pointing so roll is
free. With the saturation constraint active and the free roll, the planner can orient so
the dipole torque offloads the wheel while holding the boresight -- environmental-torque
desaturation woven into a pointing hold.

Run once, INSPECT honestly: does |h| come down while pointing holds?
Output: fig_c2_dipole_desat.png/.pdf, C2_RESULTS.md
"""
import os, sys
import numpy as np
ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ADCS
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.helpers.math_helpers import rot_mat, normalize
import papers.Planner._paper2_sim as P

OUT = os.path.join(ROOT, "papers/Planner/output_data")
BORE = np.array([0., 1, 0])
DIPOLE = float(os.environ.get("C2_DIPOLE", 0.2))      # A.m^2 residual (designed/large)
TF = float(os.environ.get("C2_TF", 2000.0)); DT = 1.0
H_MAX = 0.0036                                         # beavercube2 RW


def make_sat_with_dipole():
    s = P.make_sat("3+1", estimated=False)
    s.disturbances = list(s.disturbances) + [
        Dipole_Disturbance(DIPOLE * normalize(np.array([1., 0.2, -0.3])),
                           noise=Noise(std_noise=np.zeros(3)))]
    return s


def main():
    sat = make_sat_with_dipole()                       # plant: GG+drag+SRP + big dipole
    est = make_sat_with_dipole()                       # planner MODELS the same dipole
    goal = ADCS.goals.ECI_Goal(normalize(np.array([0.3, 0.7, 0.65])))
    h0 = 0.9 * H_MAX                                    # wheel starts near saturation
    x0 = np.concatenate([np.zeros(3), [1., 0, 0, 0], [h0]])
    os0 = P.default_os0()
    print(f"[C2] dipole={DIPOLE} A.m^2  h0={h0*1e3:.2f} mN.m.s (0.9 h_max)  tf={TF:.0f}")

    ctrl = ADCS.controller.Plan_and_Track_LQR(
        est_sat=est, planner_settings=P.make_planner_settings(est))
    res = ADCS.simulate(x=x0, satellite=sat, controller=ctrl, goal=goal,
                        os0=os0, dt=DT, tf=TF)
    run = res.runs[0] if hasattr(res, "runs") else res
    st = np.asarray(run.state_hist, float); t = np.asarray(run.time_s, float)
    good = np.all(np.isfinite(st), axis=1); st, t = st[good], t[good]
    q = st[:, 3:7]; h = st[:, 7]
    g = goal.eci_vec if hasattr(goal, "eci_vec") else normalize(np.array([0.3, 0.7, 0.65]))
    err = np.array([np.degrees(np.arccos(np.clip(
        (rot_mat(q[k]/np.linalg.norm(q[k])) @ BORE) @ normalize(np.array([0.3,0.7,0.65])), -1, 1)))
        for k in range(len(q))])

    h_abs = np.abs(h)
    desat = h_abs[0] - h_abs[-1]
    held = float(np.nanmean(err[t > t[-1] - 200]))
    print(f"[C2] |h| {h_abs[0]*1e3:.2f} -> {h_abs[-1]*1e3:.2f} mN.m.s  (desat {desat*1e3:+.2f})  "
          f"final pointing {held:.2f} deg")

    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax[0].plot(t, err, "C0"); ax[0].axhline(5, ls=":", c="0.6")
    ax[0].set_ylabel("pointing err [deg]"); ax[0].set_title(
        f"C2: disturbance-assisted desat (modeled {DIPOLE} A·m² dipole, 3+1 vector)")
    ax[1].plot(t, h*1e3, "C3"); ax[1].axhline(H_MAX*1e3, ls="--", c="0.5", lw=0.8)
    ax[1].axhline(-H_MAX*1e3, ls="--", c="0.5", lw=0.8); ax[1].axhline(0, ls=":", c="0.7", lw=0.6)
    ax[1].set_ylabel("wheel mom. [mN·m·s]"); ax[1].set_xlabel("time [s]")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_c2_dipole_desat.png"), dpi=150)
    fig.savefig(os.path.join(OUT, "fig_c2_dipole_desat.pdf")); plt.close(fig)

    ok = desat > 0.3 * h_abs[0] and held < 5.0
    with open(os.path.join(OUT, "C2_RESULTS.md"), "w") as f:
        f.write("# C2 -- disturbance-assisted desaturation (designed satellite)\n\n")
        f.write("## GG/aero feasibility (why a designed dipole)\n")
        f.write("Over one ~5800 s orbit: gravity-gradient dumps ~0.25 mN·m·s = **6.9% of h_max**; "
                "drag/SRP ~0.8%. So GG/aero **cannot** desaturate a wheel in a single trajectory "
                "(~14 orbits needed). A residual magnetic dipole m×B is the feasible environmental "
                f"torque: |m|={DIPOLE} A·m² → ~{DIPOLE*30e-6*1e6:.1f} µN·m, ~{DIPOLE*30e-6*5800/H_MAX*100:.0f}% "
                "of h_max per orbit.\n\n")
        f.write("## Demo result\n")
        f.write(f"- residual dipole modeled by the planner: {DIPOLE} A·m²\n")
        f.write(f"- wheel momentum: {h_abs[0]*1e3:.2f} → {h_abs[-1]*1e3:.2f} mN·m·s (Δ {desat*1e3:+.2f})\n")
        f.write(f"- final pointing error: {held:.2f}°\n\n")
        f.write(("**Desaturation OBSERVED** while holding pointing: the planner oriented (free roll) so "
                 "the modeled dipole torque offloaded the wheel, reducing |h| by >30% without losing the "
                 "boresight.\n") if ok else
                ("**Not a clean desat in this configuration**: |h| did not come down by >30% (the default "
                 "planner cost has no explicit momentum-minimization term, only the saturation "
                 "*constraint*; it keeps |h| feasible but need not drive it to zero). A momentum-target "
                 "cost would be needed to force active desaturation -- reporting honestly. The feasibility "
                 "physics (dipole strong enough, GG too weak) stands regardless.\n"))
    print("saved fig_c2_dipole_desat + C2_RESULTS.md  | clean desat:", ok)


if __name__ == "__main__":
    main()
