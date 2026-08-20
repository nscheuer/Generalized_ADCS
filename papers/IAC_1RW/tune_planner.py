"""Planner warm-start + cost-weight tuning study (user-directed, exploratory).

One 'reasonable but slightly difficult' candidate: seed 49 -- a PD-diverged frontier
draw the planner rescues but holds imprecisely (2.80 deg final in the money cell).

Warm start uses the DECOMPOSED pipeline the pybind already exposes
(prepareForAlilqr -> generateInitialTrajectory(U_seed) -> alilqr ->
cleanUpAfterAlilqr), replicating trajOpt's exact orchestration (OldPlanner.cpp:502
is literally before -> alilqr -> after). Seed = previous plan's controls over the
overlap, tail = hold-last or zero; first window / post-failure = heuristic init
(bdotOn selectable). Warm state is captured PARENT-side in set_active_trajectory
because the solve itself runs in a kill-on-overrun forked child (_plan_in_child)
whose in-child mutations die with it.

Campaign isolation: nothing here touches _iac_sim.py or the frozen campaign
makers; simulate() takes the maker as an argument.

Usage:
  --check   pipeline-equivalence smoke (decomposed-cold vs stock trajOpt, short run)
  --sweep   full 12-config sweep on seed 49 (refuses while campaign A runs)
"""
import os
import pickle
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from ADCS.controller import Plan_and_Track_LQR                      # noqa: E402
from ADCS.controller.plan_and_track import CostWeights, PlannerSettings  # noqa: E402
from ADCS.controller.plan_and_track.build_csat import (             # noqa: E402
    reorder_controls_cpp_to_python, reorder_gains_cpp_to_python)
from ADCS.controller import plan_and_track_base as _ptb             # noqa: E402
from ADCS.controller.helpers.trajectory import Trajectory           # noqa: E402
from papers.IAC_1RW._iac_sim import (                               # noqa: E402
    simulate, make_config, error_series, T_ORBIT)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")
SEED = 49
SEC2CENT = _ptb.TimeConstants.sec2cent


class WarmPlanTrack(Plan_and_Track_LQR):
    """Plan_and_Track_LQR with prior-trajectory warm starting via the decomposed
    pybind pipeline. warm_mode: 'off' | 'hold' | 'zero' (tail fill past the prior
    plan's end). init_bdot: heuristic used when there is no usable prior."""

    def __init__(self, *a, warm_mode="off", init_bdot=0, tag="", **kw):
        super().__init__(*a, **kw)
        self._warm_mode = warm_mode
        self._init_bdot = int(init_bdot)
        self._tag = tag
        self._prev = None            # (times_cent, U_cpp (n_u x n_steps-ish))
        self._perm_py_from_cpp = None

    # ---- parent-side capture: survives the fork-per-solve isolation ----
    def set_active_trajectory(self, traj):
        super().set_active_trajectory(traj)
        if self._warm_mode == "off":
            return
        t = np.asarray(traj.times, float)
        U = np.asarray(traj.controls, float)
        if U.shape[0] == len(t) or (U.ndim == 2 and U.shape[0] != len(self.est_sat.actuators)):
            U = U.T                  # normalize to (n_u, n_cols)
        self._prev = (t, self._py_to_cpp(U))

    def clear_warm(self):
        self._prev = None

    def _derive_perm(self):
        n_u = len(self.est_sat.actuators)
        probe = np.arange(n_u, dtype=float).reshape(n_u, 1)
        out = np.asarray(reorder_controls_cpp_to_python(probe, self.est_sat.actuators),
                         float).reshape(n_u)
        self._perm_py_from_cpp = out.astype(int)     # python_row i <- cpp_row out[i]

    def _py_to_cpp(self, U_py):
        if self._perm_py_from_cpp is None:
            self._derive_perm()
        U_cpp = np.empty_like(U_py)
        U_cpp[self._perm_py_from_cpp, :] = U_py      # invert the cpp->py mapping
        return U_cpp

    def _build_seed(self, t_start, dt_tp, n_knots):
        t_prev, U_prev = self._prev
        n_u = U_prev.shape[0]
        ncol = max(U_prev.shape[1], 1)
        U = np.zeros((n_u, n_knots - 1), order="F")
        last = U_prev[:, min(ncol, U_prev.shape[1]) - 1]
        any_overlap = False
        for k in range(n_knots - 1):
            tk = t_start + (k * dt_tp) * SEC2CENT
            if t_prev[0] <= tk <= t_prev[-1]:
                j = min(int(np.searchsorted(t_prev, tk)), U_prev.shape[1] - 1)
                U[:, k] = U_prev[:, j]
                any_overlap = True
            else:
                U[:, k] = last if self._warm_mode == "hold" else 0.0
        return U if any_overlap else None

    # ---- decomposed trajOpt with seed injection ----
    def calculate_trajectory(self, t_start, duration, x_0, os_0, goals, verbose=False):
        self.planner.setVerbosity(verbose)
        dt_fine = self.planner_settings.dt_tvlqr
        N_fine = int(np.ceil(duration / dt_fine)) + 1
        t_end = t_start + duration * SEC2CENT
        vecsPy = self._propagate_environment(os_0, t_start, t_end, dt_fine, N_fine, goals)
        x0c = np.copy(np.asarray(x_0, float).flatten(), order="C")
        dt_tp = float(self.planner_settings.dt_tp)

        trajPy, vecs_dtPy, costSettings = self.planner.prepareForAlilqr(
            vecsPy, dt_tp, t_start, t_end, x0c, self._init_bdot)
        used_warm = False
        if self._warm_mode != "off" and self._prev is not None:
            n_knots = len(np.asarray(vecs_dtPy[0]))
            U_seed = self._build_seed(t_start, dt_tp, n_knots)
            if U_seed is not None:
                trajPy = self.planner.generateInitialTrajectory(
                    dt_tp, x0c, U_seed, vecs_dtPy)
                used_warm = True
        ali_settings = self.planner.readParameters()[1]
        t0 = time.monotonic()
        aliOut = self.planner.alilqr(dt_tp, trajPy, vecs_dtPy, costSettings,
                                     ali_settings, False)
        after = self.planner.cleanUpAfterAlilqr(vecsPy, dt_tp, t_start, t_end, aliOut)
        print(f"[tune:{self._tag}] plan warm={int(used_warm)} "
              f"solve={time.monotonic()-t0:.1f}s", flush=True)
        (Xset, Uset_cpp, Tset, Kset_cpp, Sset, lqr_times) = after[3]
        Uset = reorder_controls_cpp_to_python(Uset_cpp, self.est_sat.actuators)
        Kset = reorder_gains_cpp_to_python(Kset_cpp, self.est_sat.actuators)
        return Trajectory(np.array(lqr_times), Xset, Uset, Kset, Sset)


def build_settings(sat, angle, angle_N, ang_vel=1e5, ang_vel_N=None):
    ps = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)
    ps.verbosity = False
    ps.cost_main.use_full_cost_hessian = True
    ps.pass1.regularization.use_dynamics_hess = 1
    ps.init_traj.bdot_gain = 500
    ps.pass1.aug_lag.penalty_init = 1e-3
    ps.pass1.aug_lag.penalty_scale = 10
    ps.pass1.convergence.max_outer_iter = 10
    ps.pass1.convergence.max_inner_iter = 25
    ps.pass2.aug_lag.penalty_init = 1e5
    ps.pass2.aug_lag.penalty_scale = 10
    ps.pass2.convergence.max_outer_iter = 6
    ps.pass2.convergence.max_inner_iter = 15
    av_N = ang_vel if ang_vel_N is None else ang_vel_N
    ps.cost_main = CostWeights(
        angle=angle, angle_N=angle_N, ang_vel=ang_vel, ang_vel_N=av_N,
        ang_vel_err_dir=1e2, ang_vel_err_dir_N=0.0, ang_vel_mag=0.0,
        ang_vel_mag_N=0.0, control_mult=1.0, ang_cost_func_type=2)
    ps.cost_second = ps.cost_main
    # TVLQR (tracking) weights stay at the campaign values -- tracking measured tight.
    ps.cost_tvlqr = CostWeights(
        angle=1e5, angle_N=1e6, ang_vel=1e6, ang_vel_N=1e8,
        ang_vel_mag=0.0, ang_vel_mag_N=0.0, control_mult=1.0,
        ang_cost_func_type=2)
    return ps


def make_maker(cfg):
    def maker(sat, config):
        ps = build_settings(sat, cfg["angle"], cfg["angle_N"],
                            cfg.get("ang_vel", 1e5))
        if cfg["warm"] == "off" and cfg.get("stock"):
            ctl = Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)
        else:
            ctl = WarmPlanTrack(est_sat=sat, planner_settings=ps,
                                warm_mode=cfg["warm"], init_bdot=cfg.get("bdot", 0),
                                tag=cfg["name"])
        return ctl
    return maker


CONFIGS = [
    dict(name="base_cold",          angle=1e1, angle_N=1e1, warm="off"),
    dict(name="base_warmhold",      angle=1e1, angle_N=1e1, warm="hold"),
    dict(name="base_warmzero",      angle=1e1, angle_N=1e1, warm="zero"),
    dict(name="base_cold_bdot",     angle=1e1, angle_N=1e1, warm="off", bdot=1),
    dict(name="a1e2_cold",          angle=1e2, angle_N=1e2, warm="off"),
    dict(name="a1e2_warmhold",      angle=1e2, angle_N=1e2, warm="hold"),
    dict(name="a1e3_cold",          angle=1e3, angle_N=1e3, warm="off"),
    dict(name="a1e3_warmhold",      angle=1e3, angle_N=1e3, warm="hold"),
    dict(name="termN1e3_cold",      angle=1e1, angle_N=1e3, warm="off"),
    dict(name="termN1e3_warmhold",  angle=1e1, angle_N=1e3, warm="hold"),
    dict(name="a1e2N1e4_warmhold",  angle=1e2, angle_N=1e4, warm="hold"),
    dict(name="a1e2_av1e4_warmhold", angle=1e2, angle_N=1e2, ang_vel=1e4, warm="hold"),
]


def run_one(cfg, tf=T_ORBIT, seed=SEED):
    config = dict(make_config(seed, n_rw=1, task="reduced", tf=tf, dt=1.0, seed=seed),
                  controller="planner")
    r = simulate(config, make_maker(cfg),
                 disturbances=("gg", "drag", "srp", "dipole", "general"),
                 bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
    r["tune_cfg"] = cfg
    with open(os.path.join(OUT, f"tune_seed{seed}_{cfg['name']}.pkl"), "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    return metrics_row(cfg, r)


def metrics_row(cfg, r):
    e = error_series(r)
    t = np.asarray(r["time"], float)

    def med(t0, t1):
        i0, i1 = np.searchsorted(t, [t0, t1])
        return float(np.median(e[i0:max(i1, i0 + 1)]))
    jolts = []
    for w0 in range(1000, 5001, 500):
        i0, i1 = np.searchsorted(t, [w0, w0 + 75])
        if i1 < len(e):
            jolts.append(float(e[i1] - e[i0]))
    pw = [float(v) for v in r.get("plan_wall_s", [])]
    return dict(
        name=cfg["name"], final=float(e[-1]), standing=med(3500, 5400),
        jolt=float(np.median(jolts)) if jolts else None,
        acq5=float(t[np.argmax(e < 5.0)]) if (e < 5.0).any() else None,
        conv1_5400=med(5395, 5405), h_peak=float(np.max(r["h_frac"])),
        n_plans=r.get("n_plans"), n_fb=r.get("n_fallbacks"),
        kills=r.get("n_budget_kills"), solve_med=float(np.median(pw)) if pw else None,
        solve_max=float(np.max(pw)) if pw else None)


def campaign_running():
    return subprocess.run(["pgrep", "-f", "generate_A_baseline"],
                          capture_output=True).returncode == 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--check":
        # Equivalence smoke: decomposed-cold vs stock trajOpt, short horizon.
        rows = []
        for cfg in (dict(name="chk_stock", angle=1e1, angle_N=1e1, warm="off", stock=True),
                    dict(name="chk_decomp", angle=1e1, angle_N=1e1, warm="off"),
                    dict(name="chk_warm", angle=1e1, angle_N=1e1, warm="hold")):
            rows.append(run_one(cfg, tf=1600.0))
            print(rows[-1], flush=True)
        a, b = rows[0], rows[1]
        ok = (abs(a["final"] - b["final"]) < 0.25 and a["n_plans"] == b["n_plans"]
              and (a["n_fb"] or 0) == 0 and (b["n_fb"] or 0) == 0)
        print(f"EQUIVALENCE {'PASS' if ok else 'FAIL'}: stock final {a['final']:.3f} "
              f"vs decomposed {b['final']:.3f}; warm ran with n_fb={rows[2]['n_fb']}")
        return 0 if ok else 1
    if mode == "--sweep":
        if campaign_running():
            print("REFUSING: campaign A generator still running (machine contention). "
                  "Re-invoke --sweep when it completes.")
            return 2
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(12, os.cpu_count() - 4)) as p:
            rows = p.map(run_one, CONFIGS)
        hdr = ["name", "final", "standing", "jolt", "conv1_5400", "h_peak",
               "n_plans", "n_fb", "kills", "solve_med", "solve_max"]
        lines = ["  ".join(hdr)]
        for r in rows:
            lines.append("  ".join(
                f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in hdr))
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(OUT, f"TUNE_SWEEP_seed{SEED}.txt"), "w") as f:
            f.write(txt + "\n")
        return 0
    print("usage: tune_planner.py --check | --sweep")
    return 2


if __name__ == "__main__":
    sys.exit(main())
