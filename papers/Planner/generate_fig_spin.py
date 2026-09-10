"""
Paper 2 -- spinning maneuver (P2.2): the SALTRO planner DISCOVERS a spinning
trajectory to average out a body-fixed disturbance torque from a true
cold start (ω = 0, q = identity).

Physics: a body-fixed constant torque (e.g. off-axis propulsion) becomes a
ROTATING vector in the inertial frame when the body spins about a
PERPENDICULAR axis. Over one rotation the inertial-frame integral of the
in-plane component is zero. If we pin the spin axis = pointing axis, the
planner can hold mean pointing without continuously fighting the
disturbance. Spinning costs zero in the steady-state ω-cost budget as
long as that cost does not penalize roll-about-pointing-axis rotation.

Why cold-start discovery requires the new SALTRO machinery
----------------------------------------------------------
Pre-PR-#26 SALTRO's vec-mode (2-DOF) cost Hessian was indefinite past
~90 deg, which made the backward pass fail outright when the warm-start
trajectory ever tilted off-axis. Adding `cost_hess_gauss_newton = True`
under the PhD-style refactor in PR #26 (vec-pointing-2dof) makes the
Hessian a rank-1 outer product PSD-by-construction at every pointing
angle.

This script depends on three SALTRO changes (merged into PKMN_antispike
locally):
  - PR #26 (feat/vec-pointing-2dof) — vec cost + GN-PSD Hessian.
  - PR #27 (fix/plan-for-prop-disturbance) — apply prop_torque in
    disturbanceTorque (previously declared and bound but never applied
    in dynamics; the planner couldn't see the prop torque at all).
  - The PKMN_antispike spike-removal hooks (unchanged).

Pointing/spinning geometry (verified perpendicular)
----------------------------------------------------
- Pointing axis (body boresight, overridden to RW axis): body **+z**.
- Goal: ECI **+z**.
- Prop torque (body frame, default 40 µN m on +x): body **+x**.
- τ_prop ⊥ ŝ_point by construction, which is what makes the
  "spin averages the disturbance to zero" trick work.

CLI knobs (defensible defaults; tweak if you want a different scenario):
  --prop-torque-N-m x y z       default "4e-5 0 0"   (body-frame T_nom)
  --tf                          default 240.0
  --dt                          default 1.0
  --ang-cost-func-type          default 4   (PR #26 recommends 4 with GN:
                                             (1-c)^2, constant f''=2,
                                             well-conditioned everywhere)
  --no-gauss-newton             disable cost_hess_gauss_newton (ablation)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ADCS
from ADCS.satellite_factory.satellites.create_cubesats import (
    create_beavercube2_cubesat,
)
from ADCS.satellite_hardware.disturbances.prop_disturbance import Prop_Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.controller.saltro.SALTRO_pass_settings import (
    PassConfig, CostConfig, ILQRConfig, AugLagConfig, RegularizationConfig,
)
from ADCS.CONOPS.goallist import GoalList
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import rot_mat
from ADCS.satellite_hardware.actuators import MTQ, RW


def build_3p1_with_prop(prop_body):
    """BeaverCube2 3+1 + body-fixed off-axis Prop_Disturbance.

    Boresight is overridden to body +z (the RW axis) so the natural spin
    axis is also the RW axis and the pointing-direction-locked spin
    averages out a disturbance perpendicular to that axis.
    """
    sat = create_beavercube2_cubesat(estimated=False)
    bz = np.array([0., 0., 1.])
    if isinstance(sat.boresight, dict):
        key = next(iter(sat.boresight))
        sat.boresight = {key: bz}
    else:
        sat.boresight = bz
    sat.disturbances = list(sat.disturbances) + [
        Prop_Disturbance(torque_nominal=np.asarray(prop_body, float).reshape(3),
                         noise=Noise(std_noise=np.zeros(3)))
    ]
    return sat


def make_planner_settings(real_sat, *, dt, gauss_newton, ang_cost_func_type):
    """SALTRO settings for cold-start spin discovery.

    Key choices, in priority order:
      - init_traj.initcontroller = 2  (integrated B-dot warm start; gives the
        iLQR an initial trajectory that has accumulated some attitude error
        under the prop disturbance, breaking the static-equilibrium local
        optimum the zero-controller warm start would land in).
      - ang_cost_func_type = 4  (`(1-c)^2`, constant f''=2, PR #26 verified
        as the cleanest pairing with the GN-PSD Hessian; `c` is the dot
        product `bs · R(q)^T · r̂_eci`).
      - cost_hess_gauss_newton = True  (PR #26: rank-1 outer-product
        Hessian, PSD-by-construction at every pointing angle).
      - use_cost_hess = True  (now safe under GN; before PR #26 it would
        produce indefinite Hessians past 90°).
      - ang_vel = 0  AND  ang_vel_roll_ratio = 0.1  (NO cost on rate
        magnitude; roll-about-pointing-axis cost down-weighted further --
        explicitly tell the optimizer that spinning costs nothing).
      - ang_vel_err_dir_ratio = 0  (no Lyapunov crossterm; that term is
        useful for tracking but actively interferes here).
      - rw_AM_weight = 1e4  (PR #26 calls this out: the GN rank-1 angle
        Hessian provides curvature only along ∂c/∂θ, so RW angular-momentum
        cost is needed to give the optimizer information about the wheel-
        momentum axis).
      - plan_for_prop = 1  (per PR #27, this actually applies prop_torque
        in dynamics now).
    """
    ps = PlannerSettings(est_sat=real_sat)
    ps.init_traj.initcontroller = 2

    ps.passes = [PassConfig()]
    p = ps.passes[0]
    p.dt = float(dt)
    p.ilqr = ILQRConfig()
    p.ilqr.max_iters = 100
    p.aug_lag = AugLagConfig()
    p.aug_lag.max_outer_iters = 100
    p.reg = RegularizationConfig()
    p.reg.reg_init = 1e-3
    p.reg.reg_min = 1e-8

    p.cost = CostConfig(
        angle=1e4, angle_N=1e6,
        ang_vel=0.0, ang_vel_N=0.0,
        ang_vel_mag=0.0, ang_vel_mag_N=0.0,
        ang_vel_err_dir=0.0, ang_vel_err_dir_N=0.0,
        control_mult=1.0,
        mtq_control_weight=1.0,
        rw_control_weight=1.0,
        ang_cost_func_type=int(ang_cost_func_type),
        use_cost_hess=1,
    )
    ps.disturbances.plan_for_prop = 1

    # Wrap CostConfig.to_cpp to set the PR #26 fields the Python wrapper
    # doesn't yet expose: cost_hess_gauss_newton, ang_vel_roll_ratio,
    # ang_vel_err_dir_ratio, rw_AM_weight.
    _orig_to_cpp = p.cost.to_cpp

    def to_cpp_patched():
        cpp = _orig_to_cpp()
        cpp.cost_hess_gauss_newton = bool(gauss_newton)
        cpp.ang_vel_roll_ratio = 0.1
        cpp.ang_vel_err_dir_ratio = 0.0
        cpp.rw_AM_weight = 1e4
        return cpp

    p.cost.to_cpp = to_cpp_patched
    return ps


def plan_trajectory(ps, real_sat, x_0, os_0, goal_list, tf, dt):
    """Inline the wrapper's calculate_trajectory so we can keep the
    partial result when ok=False (the wrapper raises and discards X/U).
    """
    from ADCS.orbits.orbit import Orbit
    from ADCS.helpers.math_helpers import normalize
    from ADCS.controller.helpers.optional_dependencies import get_saltro_module

    t_start = float(os_0.J2000)
    t_end = float(t_start + tf * TimeConstants.sec2cent)
    n_steps = max(1, int(np.ceil(tf / dt)))
    jtime = np.ascontiguousarray(
        np.linspace(t_start, t_end, n_steps + 1, dtype=np.float64))

    sim_orbit = Orbit(os_0, t_end, dt=dt, use_J2=True, fast=True, verbose=False)
    q_goal = np.empty((4, jtime.size), dtype=np.float64)
    boresight = np.empty((3, jtime.size), dtype=np.float64)
    for i, t_k in enumerate(jtime):
        os_at_t = sim_orbit.get_os(float(t_k))
        ag = goal_list.get_active_goal(float(t_k), time_units="centuries")
        tr, _ = ag.to_ref(os_at_t)
        tr = np.asarray(tr, dtype=np.float64).reshape(4)
        q_goal[:, i] = tr if np.isnan(tr[0]) else normalize(tr)
        boresight[:, i] = np.asarray(real_sat.get_boresight(),
                                     dtype=np.float64).reshape(3)

    saltro_py = get_saltro_module()
    cpp_settings = ps.to_cpp()
    cpp_sat = saltro_py.Satellite()
    cpp_sat.setInertia(np.asarray(real_sat.J_COM, dtype=np.float64))
    for act in real_sat.actuators:
        if isinstance(act, MTQ):
            cpp_sat.addMTQ(np.asarray(act.axis, dtype=np.float64), float(act.u_max))
    for act in real_sat.actuators:
        if isinstance(act, RW):
            cpp_sat.addRW(np.asarray(act.axis, dtype=np.float64),
                          float(act.u_max), float(act.J), float(act.h),
                          float(act.h_max))
    r0 = np.asarray(os_0.R, dtype=np.float64).reshape(3) * 1.0e3
    v0 = np.asarray(os_0.V, dtype=np.float64).reshape(3) * 1.0e3
    x0_clean = np.asarray(x_0, dtype=np.float64).reshape(-1)

    ok, Xs, Us, K = saltro_py.trajOpt(
        cpp_settings, cpp_sat, x0_clean, r0, v0,
        np.ascontiguousarray(jtime),
        np.ascontiguousarray(q_goal),
        np.ascontiguousarray(boresight),
    )
    return bool(ok), np.asarray(Xs), np.asarray(Us), jtime, t_start


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prop-torque-N-m", type=float, nargs=3,
                   default=[4.0e-5, 0.0, 0.0], metavar=("x", "y", "z"))
    p.add_argument("--tf", type=float, default=240.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--ang-cost-func-type", type=int, default=4,
                   choices=[0, 1, 2, 3, 4],
                   help="Angle cost shape. 4 = (1-c)^2 (PR #26 default with GN).")
    p.add_argument("--no-gauss-newton", action="store_true",
                   help="Ablation: disable cost_hess_gauss_newton.")
    args = p.parse_args()
    use_gn = not args.no_gauss_newton

    real_sat = build_3p1_with_prop(args.prop_torque_N_m)

    # Vector goal: pin body +z (boresight, RW axis) to ECI +z.
    goal = ADCS.goals.ECI_Goal(np.array([0., 0., 1.]))
    t0_cent = 0.22
    goal_list = GoalList({t0_cent: goal})

    # Cold start: omega = 0, q = identity, h_rw = 0. NO seed spin.
    x_0 = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=t0_cent,
        R=7000.0 * np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]),
        V=np.array([8.0, 0.0, 0.0]))

    ps = make_planner_settings(
        real_sat,
        dt=args.dt,
        gauss_newton=use_gn,
        ang_cost_func_type=args.ang_cost_func_type,
    )

    tau_b = np.asarray(args.prop_torque_N_m, float)
    print(f"SALTRO planner-discovered spin maneuver  (PR #26 + PR #27)")
    print(f"  prop_body = {tau_b}  ({np.linalg.norm(tau_b)*1e6:.1f} uN m)")
    print(f"  goal      = ECI_Goal([0,0,1]) (vec mode)")
    print(f"  tf={args.tf:.0f} s, dt={args.dt:.1f} s")
    print(f"  GN Hessian = {use_gn}, ang_cost_func_type = {args.ang_cost_func_type}")
    print(f"  cold start: omega = 0, q = identity")

    ok, X, U, jtime, t_start = plan_trajectory(
        ps, real_sat, x_0, os0, goal_list, args.tf, args.dt)

    print(f"  -> trajOpt ok = {ok}")
    print(f"  -> X shape {X.shape}, U shape {U.shape}")

    # X is (8, N), U is (4, N) on the C++ side.
    N = X.shape[1]
    t_sec = (jtime - jtime[0]) * 36525.0 * 86400.0
    omega_deg = X[0:3, :].T * (180.0 / np.pi)   # (N, 3)
    q_hist = X[3:7, :].T
    h_rw = X[7:8, :].T if X.shape[0] >= 8 else None

    # Boresight in ECI -- shows pointing performance.
    bz_eci = np.array([rot_mat(q_hist[i]) @ np.array([0, 0, 1])
                       for i in range(N)])
    pointing_err_deg = np.degrees(np.arccos(np.clip(bz_eci[:, 2], -1, 1)))

    # ---- plot --------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    ax = axes[0]
    ax.plot(t_sec, omega_deg[:, 0], label=r"$\omega_x$")
    ax.plot(t_sec, omega_deg[:, 1], label=r"$\omega_y$")
    ax.plot(t_sec, omega_deg[:, 2], label=r"$\omega_z$ (spin / pointing axis)", lw=2.0)
    ax.set_ylabel("body rate (deg/s)")
    ax.set_title(
        f"SALTRO-planned body rate (cold start ω=0; prop {tau_b*1e6} µN·m on +x)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, ls="--", alpha=0.4)

    ax = axes[1]
    ax.plot(t_sec, pointing_err_deg, color="purple")
    ax.set_ylabel("boresight error\nfrom ECI +z (deg)")
    ax.set_title("Pointing -- planner-discovered hold")
    ax.grid(True, ls="--", alpha=0.4)

    if U is not None and U.shape[0] >= 4:
        ax = axes[2]
        ax2 = ax.twinx()
        n_mtq = min(3, U.shape[0] - 1)
        for i, lab in enumerate(("MTQ x", "MTQ y", "MTQ z")[:n_mtq]):
            ax.plot(t_sec, U[i, :], label=lab, alpha=0.75)
        ax2.plot(t_sec, U[n_mtq, :], label="RW z", color="red", lw=2.0)
        ax.set_ylabel(r"MTQ dipole (A·m²)")
        ax2.set_ylabel(r"RW torque (N·m)", color="red")
        ax.set_title("Commanded actuator effort (spinning ⇒ low steady-state effort)")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    if h_rw is not None:
        ax = axes[3]
        ax.plot(t_sec, h_rw[:, 0] * 1000.0)
        ax.axhline(16.2, ls=":", color="grey", lw=0.8, label="h_max (16.2 mN·m·s)")
        ax.axhline(-16.2, ls=":", color="grey", lw=0.8)
        ax.set_ylabel("RW stored momentum\n(mN·m·s)")
        ax.set_xlabel("time (s)")
        ax.set_title("Reaction-wheel momentum (drives the discovered spin)")
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    title = (
        "SALTRO planner-discovered spinning trajectory — BeaverCube 3+1\n"
        "Goal: boresight (body +z) on ECI +z under a body-fixed off-axis "
        f"prop torque ({np.linalg.norm(tau_b)*1e6:.0f} µN·m, body +x).\n"
        "True cold start: ω = 0, q = identity. "
        f"GN={use_gn}, ang_cost_func_type={args.ang_cost_func_type}."
    )
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = os.path.join(os.path.dirname(__file__), "output_data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(
        out_dir,
        "fig_spin_discovered.png" if use_gn else "fig_spin_discovered_no_gn.png",
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ---- summary ----------------------------------------------------------
    last = slice(-max(1, int(30 / max(args.dt, 1e-6))), None)
    om_mean = omega_deg[last, :].mean(axis=0)
    pe_mean = float(pointing_err_deg[last].mean())
    print()
    print(f"steady-state body rate (last 30 s mean):       {om_mean.round(2)} deg/s")
    print(f"steady-state pointing error (last 30 s mean):  {pe_mean:.2f} deg")
    if h_rw is not None:
        print(f"final stored RW momentum:                      {h_rw[-1, 0]*1000:.2f} mN m s")
    if abs(om_mean[2]) > 1.0:
        print(f"  -> non-zero steady-state spin about body +z:  {om_mean[2]:+.2f} deg/s")
        print(f"     (planner DISCOVERED the spinning trajectory from cold start)")
    else:
        print(f"  -> no significant steady-state spin (mean |ω_z| < 1°/s)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
