"""Shared simulation core for Paper 1 (Generalized_ACS) gap scripts.

Factored out of the per-config ``generate_mc_*`` template so the gap-fill
scripts (TAB-MC, SAMELAW, DIFFLAW, ALLOC, FAILURE, ...) reuse one identical
worker instead of copy-pasting it six times. Behaviour matches the existing
``generate_mc_*`` scripts (same hardware, gains, orbit, ECI goal); the only
additions are config-parametrized actuators and an optional mid-run actuator
failure hook (used by the FAILURE script).

Nothing here modifies the framework.
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize
from ADCS.mc.monte_carlo_runner import (
    MonteCarloRunner,
    claim_worker_slot,
    release_worker_slot,
    update_worker_progress,
)
from ADCS.helpers.simresults import RunResults, SimulationResults
from ADCS.state import State

# Paper 1 §IV-F / §V actuator configurations.
N_RW = {"3MTQ+0RW": 0, "3MTQ+1RW": 1, "3MTQ+2RW": 2, "3MTQ+3RW": 3,
        "0MTQ+3RW": 3}
MC_CONFIGS = ["3MTQ+0RW", "3MTQ+1RW", "3MTQ+3RW"]

SCALES = {
    "fast":  {"num_runs": 4,   "tf": 120,  "dt": 2},
    "paper": {"num_runs": 100, "tf": 1000, "dt": 2},
}


def scale() -> Dict[str, int]:
    """Resolve the single run-scale knob (env ``PAPER1_SCALE``)."""
    return SCALES[os.environ.get("PAPER1_SCALE", "fast")]


def build_actuators(act_config: str):
    """Return ``(actuators, reaction_wheels)`` for an actuator config."""
    mtq_max, rw_max, rw_hmax = 0.4, 7e-3, 16.2e-3
    n_mtq = 3 if act_config.startswith("3MTQ") else 0
    acts = [MTQ(axis=j, max_torque=mtq_max)
            for j in MathConstants.unitvecs[:n_mtq]]
    rws = [
        RW(axis=j, max_torque=rw_max, J=1e-3, h=0, h_max=rw_hmax)
        for j in MathConstants.unitvecs
    ][: N_RW[act_config]]
    acts.extend(rws)
    return acts, rws


def make_config(run_id: int, act_config: str, tf: int, dt: int,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """One MC config dict. ``seed=run_id`` unless overridden (SAMELAW/DIFFLAW
    pass a fixed seed so the *only* thing that changes is the config/law)."""
    s = run_id if seed is None else seed
    rng = np.random.default_rng(seed=s)
    n_rw = N_RW[act_config]
    return {
        "run_id": run_id, "seed": s, "act_config": act_config,
        "tf": tf, "dt": dt,
        "w0": normalize(rng.standard_normal(3))
        * (rng.uniform(0.1, 2.0) * np.pi / 180.0),
        "q0": normalize(rng.standard_normal(4)),
        "h0": rng.uniform(-0.005, 0.005, size=n_rw),
        "goal_eci_vec": normalize(rng.standard_normal(3)),
        "orbit_R": 7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        "orbit_V": np.array([8, 0, 0]),
    }


_CACHED_ORBIT = None
_CACHED_ORBIT_KEY = None


def simulate(config: Dict[str, Any],
             make_controller: Callable[[Satellite, Dict[str, Any]], Any],
             ) -> Dict[str, Any]:
    """Run one trajectory. ``make_controller(sat, config) -> controller``
    lets each gap script inject its own control law (Wie/Lovera/Wisniewski,
    LP/QP/cQP) while sharing identical dynamics, orbit and goal.

    An optional ``config['fail']`` ``{"t": seconds, "act_index": i}`` zeroes
    an actuator's bounds mid-run (FIG-FAILURE). Returns the standard result
    dict consumed by ``ADCS.helpers.metrics``; ``tau_des``/``tau_cmd`` are
    captured when the controller exposes ``last_tau_des``/``last_tau_cmd``.
    """
    global _CACHED_ORBIT, _CACHED_ORBIT_KEY
    slot_id = claim_worker_slot()
    run_id = config["run_id"]
    try:
        np.random.seed(config["seed"])
        tf, dt, t0 = config["tf"], config["dt"], 0
        steps = int((tf - t0) / dt)

        acts, rws = build_actuators(config["act_config"])
        mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
        real_sat = Satellite(
            mass=1.2, J_0=np.diagflat([0.022, 0.022, 0.004]),
            actuators=acts, sensors=mtms, boresight=np.array([0, 0, 1]),
        )

        h0 = np.asarray(config["h0"], dtype=float)
        x = State(w=config["w0"], q=config["q0"], h=config["h0"])
        for i, rw in enumerate(rws):
            rw.h = h0[i]

        orbit_key = (tuple(config["orbit_R"]), tuple(config["orbit_V"]), tf, dt)
        if _CACHED_ORBIT is None or _CACHED_ORBIT_KEY != orbit_key:
            ephem = Ephemeris()
            os0 = Orbital_State(
                ephem=ephem, J2000=0.22 - 1 * TimeConstants.sec2cent,
                R=config["orbit_R"], V=config["orbit_V"],
            )
            _CACHED_ORBIT = Orbit(
                os0=os0, end_time=0.22 + tf * TimeConstants.sec2cent,
                dt=dt, zonal_J=2, fast=False, verbose=False,   # was use_J2=True;
                # zonal_J=2 is the current spelling of the same J2-only model.
            )
            _CACHED_ORBIT_KEY = orbit_key
        orb = _CACHED_ORBIT

        controller = make_controller(real_sat, config)
        # Full-attitude goal (DIFFLAW Wie cell) if a target quaternion is
        # supplied; otherwise the default vector-pointing ECI goal.
        if config.get("goal_quat") is not None:
            goal = Fixed_Attitude_Goal(np.asarray(config["goal_quat"], float))
        else:
            goal = ECI_Goal(config["goal_eci_vec"])
        fail = config.get("fail")

        time_hist = np.zeros(steps)
        state_hist = np.zeros((steps, len(x.as_array())))
        u_hist = np.zeros((steps, len(acts)))
        bore_hist = np.zeros((steps, 4))
        tau_des_hist = np.full((steps, 3), np.nan)
        tau_cmd_hist = np.full((steps, 3), np.nan)
        alpha_hist = np.full(steps, np.nan)
        t = t0
        sec2cent = TimeConstants.sec2cent

        for i in range(steps):
            if i % 10 == 0:
                update_worker_progress(slot_id, run_id, i, steps)
            if fail and not fail.get("_done") and t >= fail["t"]:
                acts[fail["act_index"]].max_torque = 0.0
                fail["_done"] = True

            J2000 = 0.22 + t * sec2cent
            os_state = orb.get_os(J2000=J2000)
            sens = real_sat.sensor_readings(x=x, os=os_state)
            u = controller.find_u(x_hat=x, sens=sens, est_sat=real_sat,
                                  os_hat=os_state, goal=goal)

            time_hist[i] = t
            state_hist[i, :] = x.as_array()
            u_hist[i, :] = u
            bore_hist[i, :] = goal.to_ref(os0=os_state)[0]
            td = getattr(controller, "last_tau_des", None)
            tc = getattr(controller, "last_tau_cmd", None)
            if td is not None:
                tau_des_hist[i, :] = np.asarray(td, dtype=float).ravel()[:3]
            if tc is not None:
                tau_cmd_hist[i, :] = np.asarray(tc, dtype=float).ravel()[:3]
            a = getattr(controller, "last_alpha", None)
            if a is not None:
                alpha_hist[i] = float(a)

            t += dt
            os_next = orb.get_os(0.22 + (t - t0) * sec2cent)
            out = solve_ivp(
                fun=real_sat.dynamics_for_solver, t_span=(0, dt), y0=x.as_array(),
                method="RK45", args=(u, os_state, os_next),
                rtol=1e-6, atol=1e-6,
            )
            x = State.from_array(out.y[:, -1]).normalized()

        update_worker_progress(slot_id, run_id, steps, steps)
        return {
            "run_id": run_id, "config": config, "time": time_hist,
            "state": state_hist, "u": u_hist, "boresight_goal": bore_hist,
            "tau_des": tau_des_hist, "tau_cmd": tau_cmd_hist, "alpha": alpha_hist,
            "satellite": real_sat,
        }
    finally:
        release_worker_slot(slot_id)


def _table6_make_controller(sat, config):
    from ADCS.controller import MTQ_w_RW_LP, MTQ_w_RW_QP

    cls = MTQ_w_RW_LP if config["allocator"] == "lp" else MTQ_w_RW_QP
    return cls(est_sat=sat, p_gain=5e-5, d_gain=1e-3, c_gain=1e-3,
               h_target=np.zeros(3))


def _table6_worker(config):
    return simulate(config, _table6_make_controller)


def run_table6_cell(act_config: str, task: str, allocator: str,
                    output_name: str) -> None:
    """Run one Table 6 cell and save it as a native ``.sim`` file."""
    if task not in {"vector", "full"}:
        raise ValueError("task must be 'vector' or 'full'")
    if allocator not in {"lp", "qp"}:
        raise ValueError("allocator must be 'lp' or 'qp'")

    s = scale()

    def gen(run_id):
        c = make_config(run_id, act_config, s["tf"], s["dt"], seed=run_id)
        c["allocator"] = allocator
        if task == "full":
            c["goal_quat"] = normalize(
                np.random.default_rng(20_000 + run_id).standard_normal(4))
        return c

    runner = MonteCarloRunner(
        sim_func=_table6_worker,
        config_generator=gen,
        num_runs=s["num_runs"],
        max_workers=4,
    )
    results = [r for r in runner.run() if r is not None]
    if len(results) != s["num_runs"]:
        raise RuntimeError(f"Expected {s['num_runs']} runs, got {len(results)}")

    runs = []
    for result in results:
        state_hist = [State.from_array(row) for row in result["state"]]
        runs.append(RunResults(
            satellite=result["satellite"],
            time_s=result["time"],
            state_hist=state_hist,
            target_hist=result["boresight_goal"],
            control_hist=result["u"],
        ))
    saved = SimulationResults(runs=runs).save(
        output_name,
        out_dir=os.path.join(os.path.dirname(__file__), "section_IV_A", "outputs"),
    )
    print(f"Saved {saved}")
