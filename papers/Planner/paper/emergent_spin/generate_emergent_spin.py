#!/usr/bin/env python
"""
P2.2 -- autonomous spinning maneuver: the regenerable generator for the paper's
fig_spin_pe / fig_spin_omega / fig_spin_h trio.

PROVENANCE
----------
The manuscript's spin trio was exported (papers/EXPORT_INDEX.md, 2026-06-04) from
a saved ADCS_wt run (spin_paper_*.png, 2026-05-29) whose generator lived only on
the SALTRO PKMN_antispike branch as tests/debug/optimizer/alilqr_python/
autonomous_spin.py. This script is that config, verbatim, committed where the
paper figures live so the trio can be regenerated on demand.

Reference metrics -- May-29 canonical run vs 2026-07-02 reproduction on the
PR-stack build (main + PR #26 vec-pointing + PR #28 PD warm start):
    PE_fin     0.12 deg   ->  0.07 deg
    <w_z>_ss   13.8 deg/s ->  13.68 deg/s
    |h|_max    1.70 mN m s -> 1.70 mN m s   (= 0.85 * h_max margin, exact)
    MTQ / RW   0.56x / 0.56x -> 0.56x / 0.56x

SALTRO BUILD REQUIREMENT
------------------------
Needs a saltro_py build containing PR #26 (vec-pointing 2-DOF full-Newton
Hessian, ang_vel_roll_ratio, rw_AM knee) AND PR #28 (PDController warm start,
initcontroller=3). DDP (#62) and rw_momentum_limit_scale (#39) are on SALTRO
main since 2026-06-23. Until #26/#28 merge, point --saltro-build at an
integration build. Default: <Generalized_ADCS>/SALTRO/build.

OUTPUT
------
papers/Planner/output_data/fig_spin_{pe,omega,h}_regen.png by default;
pass --final to overwrite the canonical fig_spin_{pe,omega,h}.png the
manuscript references.

THE RECIPE (see the PKMN_antispike header for the full derivation; every
ingredient matters): full Newton (GN OFF), DDP + psd_clip, penalty_max 1e15,
Bryson-normalized control weights (mtq 1000 / rw 2.5e7), roll-free ang-vel
cost, running angle 3x terminal, rw_momentum_limit_scale 0.85, wmax 60 deg/s,
plain-PD warm start, NO seed. Table 7.3 thesis satellite, 179 deg anti-velocity
acquisition under an overwhelming 3e-4 N m body-x prop torque.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ADCS as _A
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.disturbances.prop_disturbance import Prop_Disturbance
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import rot_mat, normalize
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.controller.saltro.SALTRO_pass_settings import (
    PassConfig, CostConfig, ILQRConfig, AugLagConfig, RegularizationConfig)

# ---- Table 7.3 satellite (verbatim from autonomous_spin.py) -------------------
J_THESIS = np.array([[0.1, 0., 0.00013],
                     [0., 0.05, -0.00021],
                     [0.00013, -0.00021, 0.005]])              # kg*m^2, 20:1 oblate
Q0_THESIS = np.array([-0.232, -0.664, -0.234, -0.671]); Q0_THESIS /= np.linalg.norm(Q0_THESIS)
U_MTQ_MAX = np.array([0.19, 0.57, 0.57])                       # A*m^2
U_RW_MAX = 2e-4                                                # N*m
PROP = np.array([3e-4, 0., 0.])                                # body-X prop torque, N*m
DT = 1.0
TF = 500.0


def build_satellite():
    sat = create_beavercube2_cubesat(estimated=False)
    sat.J_COM = J_THESIS.copy(); sat.J_0 = J_THESIS.copy()
    sat.actuators = [
        MTQ(axis=np.array([1., 0, 0]), max_torque=0.19),
        MTQ(axis=np.array([0, 1., 0]), max_torque=0.57),
        MTQ(axis=np.array([0, 0, 1.]), max_torque=0.57),
        RW(axis=np.array([0, 1., 0]), max_torque=U_RW_MAX, J=2e-6,
           h=np.array([0.0]), h_max=np.array([2e-3])),
    ]
    bz = np.array([0., 0., 1.])                                # boresight = body +z
    sat.boresight = {next(iter(sat.boresight)): bz} if isinstance(sat.boresight, dict) else bz
    sat.disturbances = [d for d in sat.disturbances if not isinstance(d, Prop_Disturbance)] + [
        Prop_Disturbance(torque_nominal=PROP.copy(), noise=Noise(std_noise=np.zeros(3)))]
    return sat


def cs_satellite(saltro_py):
    cs = saltro_py.Satellite(); cs.setInertia(np.asarray(J_THESIS))
    cs.addMTQ(np.array([1., 0, 0]), 0.19)
    cs.addMTQ(np.array([0., 1, 0]), 0.57)
    cs.addMTQ(np.array([0., 0, 1]), 0.57)
    cs.addRW(np.array([0., 1, 0]), U_RW_MAX, 2e-6, 0.0, 2e-3)  # +y wheel, h_max=2 mN*m*s
    return cs


def grids(dt=DT):
    os0 = _A.Orbital_State(ephem=_A.Ephemeris(), J2000=0.22,
                           R=6800. * np.array([0., np.cos(np.deg2rad(51.5)), np.sin(np.deg2rad(51.5))]),
                           V=np.array([7.65, 0., 0.]))
    goal = _A.goals.AntiVelocity_Goal(); t0 = 0.22; gl = GoalList({t0: goal})
    t_end = t0 + TF * TimeConstants.sec2cent; n = int(TF / dt) + 1
    jtime = np.linspace(t0, t_end, n)
    so = Orbit(os0, t_end, dt=dt, zonal_J=2, fast=True, verbose=False)
    qg = np.empty((4, n)); bs = np.empty((3, n))
    for i, tk in enumerate(jtime):
        os_at = so.get_os(float(tk)); ag = gl.get_active_goal(float(tk), time_units="centuries")
        tr, _ = ag.to_ref(os_at); tr = np.asarray(tr).reshape(4)
        qg[:, i] = tr if np.isnan(tr[0]) else normalize(tr)
        bs[:, i] = np.array([0., 0., 1.])
    r0 = np.asarray(os0.R) * 1e3; v0 = np.asarray(os0.V) * 1e3
    return jtime, qg, bs, r0, v0, n


def make_settings(sat):
    ps = PlannerSettings(est_sat=sat)
    ps.constraints.wmax = 60 * np.pi / 180.
    ps.init_traj.initcontroller = 3                            # PD, NO seed (PR #28)
    ps.disturbances.plan_for_prop = 1
    ps.passes = [PassConfig()]; p = ps.passes[0]; p.dt = DT
    p.ilqr = ILQRConfig(); p.ilqr.max_iters = 150
    p.aug_lag = AugLagConfig()
    # The current SALTRO main branch needs additional AL passes for this
    # deliberately difficult anti-ram, no-seed initial condition.
    p.aug_lag.max_outer_iters = 100
    p.aug_lag.penalty_init = 0.01
    p.aug_lag.penalty_scale = 3.0
    p.aug_lag.penalty_max = 1e15
    p.reg = RegularizationConfig(); p.reg.reg_init = 1e-3
    p.reg.use_dynamics_hess = 1                                # DDP (#62, merged)
    p.reg.psd_clip_quu_ddp = 1
    p.cost = CostConfig(angle=3e7, angle_N=1e7,
                        ang_vel=1e2, ang_vel_N=1e2,
                        control_mult=1.0,
                        mtq_control_weight=1000.0,
                        rw_control_weight=2.5e7,
                        ang_cost_func_type=3, use_cost_hess=1)
    orig_cost = p.cost.to_cpp

    def cost_to_cpp():
        c = orig_cost()
        if hasattr(c, "cost_hess_gauss_newton"):
            c.cost_hess_gauss_newton = False                    # FULL NEWTON (PR #26)
        if hasattr(c, "ang_vel_roll_ratio"):
            c.ang_vel_roll_ratio = 0.0
        if hasattr(c, "ang_vel_err_dir_ratio"):
            c.ang_vel_err_dir_ratio = 0.0
        c.rw_AM_weight = 1e4
        # knee at 0.5*h_max — knob renamed RWh_ok_mult -> RWh_knee_frac in SALTRO PR #54
        if hasattr(c, "RWh_knee_frac"):
            c.RWh_knee_frac = 0.5
        else:
            c.RWh_ok_mult = 0.5
        c.rw_stic_weight = 0.0; c.RWh_stiction_mult = 0.05
        return c

    p.cost.to_cpp = cost_to_cpp
    orig_con = ps.constraints.to_cpp

    def con_to_cpp():
        c = orig_con()
        if hasattr(c, "rw_momentum_limit_scale"):
            c.rw_momentum_limit_scale = 0.85                    # #39, merged
        return c

    ps.constraints.to_cpp = con_to_cpp
    return ps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saltro-build", default=os.path.join(ROOT, "SALTRO", "build"),
                    help="dir containing saltro_py (needs PR #26 + PR #28 content)")
    ap.add_argument("--final", action="store_true",
                    help="overwrite the canonical fig_spin_{pe,omega,h}.png "
                         "(default writes *_regen.png alongside)")
    args = ap.parse_args()

    sys.path.append(args.saltro_build)
    import saltro_py as S

    sat = build_satellite()
    ps = make_settings(sat)
    jt, qg, bs, r0, v0, n = grids(DT)
    x0 = np.concatenate([np.zeros(3), Q0_THESIS, [0.0]])       # rest at the antipode, NO seed

    ok, X, U, _K = S.trajOpt(ps.to_cpp(), cs_satellite(S), x0, r0, v0,
                             np.ascontiguousarray(jt), np.ascontiguousarray(qg),
                             np.ascontiguousarray(bs))
    X = np.asarray(X); U = np.asarray(U)

    t = (jt - jt[0]) * 36525.0 * 86400.0
    pe = np.array([float(np.degrees(np.arccos(np.clip(
        (rot_mat(X[3:7, k]) @ np.array([0, 0, 1])) @ (qg[1:4, k] / np.linalg.norm(qg[1:4, k])),
        -1, 1)))) for k in range(X.shape[1])])
    w = X[0:3, :] * 180.0 / np.pi
    h = X[7, :] * 1000.0

    mtq = float(np.max(np.abs(U[0:3, :].T) / U_MTQ_MAX)); rw = float(np.max(np.abs(U[3, :])) / U_RW_MAX)
    print(f"ok={ok}  PE_fin={pe[-1]:.2f}  PE_m30={float(np.mean(pe[-30:])):.2f}  "
          f"<wz>_m30={float(np.mean(w[2, -30:])):.2f} deg/s  |h|max={float(np.abs(h).max()):.2f} mN*m*s  "
          f"mtq={mtq:.2f}x  rw={rw:.2f}x")
    if not ok:
        print("WARNING: trajOpt did not converge; not writing figures.")
        return 1

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if args.final else "_regen"

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, pe)
    ax.set_title("Pointing error"); ax.set_ylabel("deg"); ax.set_xlabel("t (s)")
    ax.grid(alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig_spin_pe{suffix}.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, lab in enumerate(("wx", "wy", "wz")):
        ax.plot(t, w[i], label=lab)
    ax.plot(t, np.linalg.norm(w, axis=0), color="red", label="|w|")
    ax.set_title("Angular velocity"); ax.set_ylabel("deg/s"); ax.set_xlabel("t (s)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig_spin_omega{suffix}.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, h)
    ax.axhline(2.0, ls=":", color="red"); ax.axhline(-2.0, ls=":", color="red")
    ax.set_title("Reaction-wheel stored momentum"); ax.set_ylabel("mN m s"); ax.set_xlabel("t (s)")
    ax.grid(alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig_spin_h{suffix}.png"), dpi=110)
    plt.close(fig)

    print(f"wrote fig_spin_{{pe,omega,h}}{suffix}.png to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
