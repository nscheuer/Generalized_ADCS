"""Shared simulation core for Paper 2 (Planner) gap scripts.

Factored from the proven ``papers/Planner/generate_altro_*`` template so the
gap-fill scripts reuse one identical planner/tracker setup instead of
copy-pasting the (long) ``PlannerSettings`` tuning block. Behaviour matches
the existing scripts: ``Plan_and_Track_LQR`` (ALTRO planner + TVLQR tracker)
driven by ``ADCS.simulate_mc``.

Parametrised over actuator config, goal (single ``Goal`` or multi-goal
``GoalList``), an optional digital-twin mismatch (IV-F: ``est_sat`` !=
true sat) and a baseline controller for planner-vs-PD comparisons. Nothing
in the framework is modified here.
"""

import os
import sys
from typing import Any, Callable, Optional

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

import ADCS as ADCS

# Actuator configs -> satellite factories (estimated=False => true plant).
SAT_FACTORIES: dict[str, Callable[..., Any]] = {
    "3+0": ADCS.satellite_factory.create_beavercube1_cubesat,   # 3 MTQ
    "3+1": ADCS.satellite_factory.create_beavercube2_cubesat,   # 3 MTQ + 1 RW
    "3+3": ADCS.satellite_factory.create_3_3_beavercube2_cubesat,  # 3 MTQ+3RW
}

# ALTRO is orders of magnitude slower than feedback control, so the smoke
# scaffold must be genuinely small. "paper" reproduces the published runs.
SCALES = {
    "fast":  {"tf": 150.0,  "dt": 1.0, "num_runs": 2},
    "paper": {"tf": 1000.0, "dt": 1.0, "num_runs": 100},
}


def scale() -> dict:
    return SCALES[os.environ.get("PAPER2_SCALE", "fast")]


def make_sat(config_key: str, estimated: bool = False):
    return SAT_FACTORIES[config_key](estimated=estimated)


def make_planner_settings(real_sat):
    """The proven PlannerSettings tuning (from generate_altro_3+1_reduced).
    Kept verbatim so convergence behaviour matches the existing scripts."""
    ps = ADCS.controller.plan_and_track.PlannerSettings(
        est_sat=real_sat, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)
    ps.verbosity = False
    ps.cost_main.use_full_cost_hessian = True
    ps.pass1.regularization.use_dynamics_hess = 1
    ps.init_traj.bdot_gain = 500
    ps.pass1.aug_lag.penalty_init = 1e-3
    ps.pass1.aug_lag.penalty_scale = 10
    ps.pass1.convergence.max_outer_iter = 15
    ps.pass1.convergence.max_inner_iter = 40
    ps.pass2.aug_lag.penalty_init = 1e5
    ps.pass2.aug_lag.penalty_scale = 10
    ps.pass2.convergence.max_outer_iter = 8
    ps.pass2.convergence.max_inner_iter = 20
    ps.cost_main = ADCS.controller.plan_and_track.CostWeights(
        angle=1e1, angle_N=1e1, ang_vel=1e5, ang_vel_N=1e5,
        ang_vel_err_dir=1e2, ang_vel_err_dir_N=0.0, ang_vel_mag=0.0,
        ang_vel_mag_N=0.0, control_mult=1.0, ang_cost_func_type=2)
    ps.cost_second = ps.cost_main
    ps.cost_tvlqr = ADCS.controller.plan_and_track.CostWeights(
        angle=1e5, angle_N=1e6, ang_vel=1e6, ang_vel_N=1e8,
        ang_vel_mag=0.0, ang_vel_mag_N=0.0, control_mult=1.0,
        ang_cost_func_type=2)
    return ps


def make_planner_controller(real_sat, settings=None):
    return ADCS.controller.Plan_and_Track_LQR(
        est_sat=real_sat,
        planner_settings=settings or make_planner_settings(real_sat))


def make_baseline_controller(real_sat, kind: str):
    """Reactive baseline for planner-vs-PD comparisons (IV-G / IV-I)."""
    if kind == "lovera":
        return ADCS.controller.MTQ_Lovera(est_sat=real_sat, p_gain=1e-4,
                                          d_gain=1e-3, eps=1.0)
    if kind == "lp":
        return ADCS.controller.MTQ_w_RW_LP(
            est_sat=real_sat, p_gain=5e-5, d_gain=2e-3, c_gain=1e-3,
            h_target=np.zeros(3))
    raise ValueError(f"unknown baseline {kind!r}")


def x0(n_rw: int) -> np.ndarray:
    """Initial state [w(3), q(4)=identity, h(n_rw)=0]."""
    return np.concatenate([np.zeros(3), [1.0, 0, 0, 0], np.zeros(n_rw)])


def default_os0():
    return ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        V=np.array([8, 0, 0]))


def make_random_os(rng: np.random.Generator):
    return ADCS.orbits.create_random_circular_os(
        radius_km=7000.0, J2000=0.22, rng=rng)


def run(config_key: str,
        goal,
        controller=None,
        est_sat=None,
        mc_config: Optional[Any] = None,
        num_runs: Optional[int] = None,
        tf: Optional[float] = None,
        dt: Optional[float] = None,
        x: Optional[np.ndarray] = None,
        base_seed: int = 42):
    """Run ``ADCS.simulate_mc`` and return ``SimulationResults``.

    ``goal`` may be a single ``Goal`` or a ``GoalList`` (multi-goal IV-B).
    ``est_sat`` (optional) injects a digital-twin mismatch (IV-F): the
    controller plans on ``est_sat`` while the plant is the true sat.
    """
    s = scale()
    n = num_runs if num_runs is not None else s["num_runs"]
    tf = tf if tf is not None else s["tf"]
    dt = dt if dt is not None else s["dt"]

    real_sat = make_sat(config_key, estimated=False)
    n_rw = len([a for a in real_sat.actuators
                if a.__class__.__name__ == "RW"])
    ctrl = controller or make_planner_controller(est_sat or real_sat)

    return ADCS.simulate_mc(
        x=x0(n_rw) if x is None else np.asarray(x, dtype=float),
        satellite=real_sat,
        est_satellite=est_sat, controller=ctrl, goal=goal,
        os0=default_os0(), dt=dt, tf=tf,
        mc_config=mc_config, num_runs=n, base_seed=base_seed)
