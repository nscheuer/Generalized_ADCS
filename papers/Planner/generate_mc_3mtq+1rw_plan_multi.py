"""
Monte Carlo: 3MTQ+1RW ALTRO+TVLQR Planner - Multi-Goal Test.

Uses BC2 satellite configuration with trajectory planner.
Same multi-goal structure as LP test:
  Goal1 (0-300s) → No_Goal (300-350s) → Goal2 (350-650s) → No_Goal (650-700s) → Goal3 (700-1000s)
"""
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.planner_subsettings import CostWeights
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.mc.monte_carlo_runner import (
    MonteCarloRunner, claim_worker_slot, release_worker_slot, update_worker_progress
)

BODY_BORESIGHT = np.array([0, 1, 0])

_CACHED_ORBIT = None
_CACHED_ORBIT_KEY = None


def run_single_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY

    slot_id = claim_worker_slot()
    run_id = config["run_id"]

    try:
        tf = config.get("tf", 1000)
        dt = config.get("dt", 2)
        dt_planning = config.get("dt_planning", 1)
        N = int(tf / dt)

        radius_km = config.get("radius_km", 7000.0)
        orbit_key = (slot_id, radius_km, tf, dt_planning)

        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            rng_state = np.random.get_state()
            try:
                np.random.seed(100_000 + int(slot_id))
                _CACHED_ORBIT = create_random_circular_orbit(
                    radius_km=radius_km, dt=dt_planning, tf=tf, use_J2=True, fast=True
                )
                _CACHED_ORBIT.populate_environment(compute_B=True, compute_S=True)
                _CACHED_ORBIT_KEY = orbit_key
            finally:
                np.random.set_state(rng_state)

        orb = _CACHED_ORBIT
        np.random.seed(config["seed"])

        real_sat = create_beavercube2_cubesat(estimated=False)
        rws = real_sat.rw_actuators

        x0 = np.concatenate([config["w0"], config["q0"], config["h0"]])
        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        planner_settings = PlannerSettings(
            est_sat=real_sat, bdot_on=0, dt_tp=50, dt_tvlqr=dt_planning
        )
        planner_settings.verbosity = False
        planner_settings.cost_main.use_full_cost_hessian = True
        planner_settings.pass1.regularization.use_dynamics_hess = 1
        planner_settings.init_traj.bdot_gain = 500
        planner_settings.pass1.aug_lag.penalty_init = 1e-3
        planner_settings.pass1.aug_lag.penalty_scale = 10
        planner_settings.pass1.convergence.max_outer_iter = 15
        planner_settings.pass1.convergence.max_inner_iter = 40
        planner_settings.pass2.aug_lag.penalty_init = 1e5
        planner_settings.pass2.aug_lag.penalty_scale = 10
        planner_settings.pass2.convergence.max_outer_iter = 8
        planner_settings.pass2.convergence.max_inner_iter = 20

        planner_settings.cost_main = CostWeights(
            angle=1e1,
            angle_N=1e1,
            ang_vel=1e5,
            ang_vel_N=1e5,
            ang_vel_err_dir=1e2,
            ang_vel_err_dir_N=0.0,
            ang_vel_mag=0.0,
            ang_vel_mag_N=0.0,
            control_mult=1.0,
            ang_cost_func_type=2,
        )
        planner_settings.cost_second = planner_settings.cost_main
        planner_settings.cost_tvlqr = CostWeights(
            angle=1e5,
            angle_N=1e6,
            ang_vel=1e6,
            ang_vel_N=1e8,
            ang_vel_mag=0.0,
            ang_vel_mag_N=0.0,
            control_mult=1.0,
            ang_cost_func_type=2,
        )

        controller = Plan_and_Track_LQR(est_sat=real_sat, planner_settings=planner_settings)

        # Multi-goal structure matching LP test
        sec2cent = TimeConstants.sec2cent
        t0_j2000 = 0.22
        goals = GoalList({
            t0_j2000: ECI_Goal(config["goal1"]),
            t0_j2000 + 300 * sec2cent: No_Goal(),
            t0_j2000 + 350 * sec2cent: ECI_Goal(config["goal2"]),
            t0_j2000 + 650 * sec2cent: No_Goal(),
            t0_j2000 + 700 * sec2cent: ECI_Goal(config["goal3"]),
        })
        os0 = orb.get_os(t0_j2000)

        try:
            traj = controller.calculate_trajectory(
                t_start=t0_j2000, duration=tf, x_0=x0, os_0=os0, goals=goals, verbose=False
            )
            controller.set_active_trajectory(traj)
            traj_valid = True
        except Exception as e:
            return {"run_id": run_id, "config": config, "error": str(e), "traj_valid": False}

        time_hist = np.zeros(N)
        state_hist = np.zeros((N, len(x0)))
        u_hist = np.zeros((N, len(real_sat.actuators)))
        boresight_hist = np.zeros((N, 3))

        for i, rw in enumerate(rws):
            rw.h = config["h0"][i]

        x = x0.copy()
        t = 0

        for i in range(N):
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, N)

            J2000 = t0_j2000 + t * sec2cent
            os_state = orb.get_os(J2000=J2000)
            sens = real_sat.sensor_readings(x=x, os=os_state)
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat, os_hat=os_state)

            time_hist[i] = t
            state_hist[i, :] = x
            u_hist[i, :] = u
            eci_goal_ref, _ = goals.to_ref(t=J2000, os0=os_state)
            boresight_hist[i, :] = eci_goal_ref

            t += dt
            os_next = orb.get_os(t0_j2000 + t * sec2cent)
            out = solve_ivp(
                real_sat.dynamics_for_solver, (0, dt), x, method="RK45",
                args=(u, os_state, os_next), rtol=1e-7, atol=1e-7
            )
            x = out.y[:, -1]
            x[3:7] = normalize(x[3:7])

        update_worker_progress(slot_id, run_id, N, N)

        return {
            "run_id": run_id, "config": config, "traj_valid": True,
            "time": time_hist, "state": state_hist, "u": u_hist,
            "boresight_goal": boresight_hist, "goal_type": "multi"
        }
    finally:
        release_worker_slot(slot_id)


def generate_mc_config(run_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed=run_id + 1000)
    
    q0 = normalize(rng.standard_normal(4))
    goal1 = normalize(rng.standard_normal(3))
    goal2 = normalize(rng.standard_normal(3))
    goal3 = normalize(rng.standard_normal(3))
    
    return {
        "run_id": run_id,
        "seed": run_id,
        "tf": 1000,
        "dt": 2,
        "dt_planning": 1,
        "radius_km": 7000.0,
        "w0": normalize(rng.standard_normal(3)) * (rng.uniform(0.1, 1.0) * np.pi / 180.0),
        "q0": q0,
        "h0": rng.uniform(-0.0001, 0.0001, size=1),
        "goal1": goal1,
        "goal2": goal2,
        "goal3": goal3,
    }


if __name__ == "__main__":
    RUN_MC = True
    OUTPUT_DIR = "papers/Planner/output_data"
    NUM_RUNS = 100

    if RUN_MC:
        runner = MonteCarloRunner(
            sim_func=run_single_sim,
            config_generator=generate_mc_config,
            num_runs=NUM_RUNS,
            max_workers=2
        )
        full_results = runner.run()
        
        valid = [r for r in full_results if r and r.get("traj_valid", False)]
        print(f"\n--- Monte Carlo Complete: {len(valid)}/{len(full_results)} valid ---")
        save_data(f"3MTQ+1RW_plan_multi_mc_{NUM_RUNS}", full_results, out_dir=OUTPUT_DIR)
        create_close_all_button_window()
    else:
        results = load_data(f"{OUTPUT_DIR}/3MTQ+1RW_plan_multi_mc_{NUM_RUNS}")
        full_results = results[0] if isinstance(results, tuple) else results
        create_close_all_button_window()
