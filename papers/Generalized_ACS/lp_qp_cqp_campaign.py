"""
Paper 1 — LP / QP / cQP / LP2 closed-loop allocation campaign harness  (Task 2b).

The triage (RESULTS_TRIAGE_REPORT.md, P1.1) found that no 100-trial x 1000 s
closed-loop allocation campaign exists; the draft's 120 s numbers are unsourced
and the two open-loop JSONs are mutually inconsistent. This is the campaign
harness for the re-run, templated off
``origin/PKMN_3+1:research/closed_loop_comparison.py`` (checked out to
``research/closed_loop_comparison.py``).

Status: PARAMETERIZED AND READY — NOT YET RUN AT SCALE. Use ``--time-one-trial``
for the batch wall-clock estimate; ``--run`` launches the full campaign.

Allocators benchmarked (the ``ALLOCATOR_FACTORIES`` registry):
  * LP   — single-stage direction-preserving LP (current behaviour)
  * QP   — box-constrained least-squares
  * cQP  — QP + linear power gate
  * LP2  — two-stage (lexicographic) LP refinement from Task 2a: pays the
           gyroscopic-compensation torque first, the discretionary pointing
           torque with the leftover authority. See papers/Generalized_ACS/
           two_stage_lp.py and ADCS/controller/mtq_w_rw_LP.py (two_stage_alloc).

Configs (the ``CONFIGS`` registry):
  * 3+1  — 3 MTQ + 1 RW. Underactuated; LP2's gain here is small because the
           precession torque is largely MTQ-infeasible (genuine underactuation,
           see EXEC_BRIEF2_RESULTS.md / two_stage_lp.py Sweep A).
  * 3+3  — 3 MTQ + 3 RW. The regime LP2 targets: tau_gyro is RW-feasible, so
           LP2 protects it while single-stage LP bleeds it under saturation.

What this harness does beyond the template:
  * 100 trials at 1000 s (template was 10 trials at 300 s).
  * A FIXED, explicit 100-element seed list.
  * Allocator-agnostic + config-agnostic registries.
  * Persists per-allocator PER-STEP solve-time arrays, timed here with
    time.perf_counter, plus per-trial convergence (<5 deg) + final/mean/p95.

Run:
  PYTHONPATH=. venv/bin/python papers/Generalized_ACS/lp_qp_cqp_campaign.py --time-one-trial
  PYTHONPATH=. venv/bin/python papers/Generalized_ACS/lp_qp_cqp_campaign.py --run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, "../..")))
sys.path.append(_HERE)  # for `import two_stage_lp`

import numpy as np

from research.allocation_comparison import (
    AllocationResult, LPAllocator, QPAllocator, QPCAllocator, TorqueAllocator,
)
from research.closed_loop_comparison import (
    create_3mtq_1rw_config, create_3mtq_3rw_config,
    simulate_closed_loop, time_varying_b_field,
)
from ADCS.helpers.math_helpers import normalize, skewsym
from two_stage_lp import allocate_two_stage_lp

OUT_DIR = os.path.join(_HERE, "output_data")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- campaign parameters (the re-run spec from RESULTS_TRIAGE_REPORT.md P1.1) -
N_TRIALS = 100
TF = 1000.0
DT = 2.0
CONV_THRESHOLD_DEG = 5.0
SEEDS = list(range(1000, 1000 + N_TRIALS))   # fixed, explicit, reproducible

# ---- config registry --------------------------------------------------------
CONFIGS: dict[str, callable] = {
    "3+1": lambda: create_3mtq_1rw_config(kp=5e-5, kd=1e-3, tf=TF, dt=DT),
    "3+3": lambda: create_3mtq_3rw_config(kp=5e-5, kd=1e-3, tf=TF, dt=DT),
}

# ---- initial-condition variants (Addendum A section 3 explicit ask) ---------
# Stratified at fractions of the BeaverCube h_max reference, per Patrick's
# setup guidance: a curve, not two dots; 0.9 h_max is where the demo's alpha
# collapsed to ~0.03 and where LP2 should separate cleanest from LP1.
H_MAX_REF: float = 16.2e-3       # N m s -- BeaverCube reference h_max
H0_VARIANTS: dict[str, float] = {
    "h0_0.0xh": 0.0  * H_MAX_REF,    # zero stored momentum
    "h0_0.3xh": 0.3  * H_MAX_REF,
    "h0_0.6xh": 0.6  * H_MAX_REF,
    "h0_0.9xh": 0.9  * H_MAX_REF,    # alpha-collapse regime per the demo
}


# ---- LP2: two-stage allocator as a research.TorqueAllocator ------------------
class LP2Allocator(TorqueAllocator):
    """
    Two-stage (lexicographic) LP allocator — the Task 2a refinement.

    The shared TorqueAllocator.allocate() interface passes only the *fused*
    tau_des (plus omega, h_rw). To split it, LP2 needs the body inertia J,
    supplied at construction (a fixed satellite property). It recovers
        tau_gyro = omega x (J omega + A_rw h_rw),   tau_pd = tau_des - tau_gyro
    and runs the two-stage LP from papers/Generalized_ACS/two_stage_lp.py
    (the same algorithm validated there and wired into
    ADCS.controller.MTQ_w_RW_LP.allocate_two_stage_lp).
    """

    def __init__(self, J: np.ndarray):
        super().__init__("LP2")
        self.J = np.asarray(J, dtype=float)

    def allocate(self, tau_des, b_body, A_rw, A_mtq_axes,
                 u_rw_max, u_mtq_max, omega=None, h_rw=None, J_rw=None):
        t0 = time.perf_counter()
        tau_des = np.asarray(tau_des, float).reshape(3)
        n_rw = A_rw.shape[1] if A_rw.size else 0
        n_mtq = A_mtq_axes.shape[1] if A_mtq_axes.size else 0

        omega = np.zeros(3) if omega is None else np.asarray(omega, float).reshape(3)
        h_rw = np.zeros(n_rw) if h_rw is None else np.asarray(h_rw, float).reshape(-1)
        h_rw_body = A_rw @ h_rw if n_rw else np.zeros(3)
        tau_gyro = np.cross(omega, self.J @ omega + h_rw_body)
        tau_pd = tau_des - tau_gyro

        u_rw, u_mtq, _ = allocate_two_stage_lp(
            tau_pd, tau_gyro, A_rw, A_mtq_axes, b_body,
            np.asarray(u_rw_max, float), np.asarray(u_mtq_max, float))

        A_mtq = -skewsym(b_body) @ A_mtq_axes if n_mtq else np.zeros((3, 0))
        A_total = np.hstack([A_rw, A_mtq])
        u = np.concatenate([u_rw, u_mtq])
        tau_ach = A_total @ u if u.size else np.zeros(3)
        t_mag = float(np.linalg.norm(tau_des))
        if t_mag > 1e-12:
            tau_hat = tau_des / t_mag
            alpha = max(0.0, float(np.dot(tau_ach, tau_hat)) / t_mag)
            a_n = float(np.linalg.norm(tau_ach))
            if a_n > 1e-15:
                cos = np.clip(float(np.dot(tau_ach, tau_des)) / (a_n * t_mag), -1, 1)
                dir_err = float(np.degrees(np.arccos(cos)))
            else:
                dir_err = 0.0
            mag_ratio = a_n / t_mag
        else:
            alpha, dir_err, mag_ratio = 1.0, 0.0, 1.0
        return AllocationResult(
            u_rw=u_rw, u_mtq=u_mtq, tau_achieved=tau_ach, tau_desired=tau_des,
            alpha=alpha, direction_error_deg=dir_err, magnitude_ratio=mag_ratio,
            solve_time_us=(time.perf_counter() - t0) * 1e6,
            solver_success=True, method=self.name)


def make_allocators(config) -> dict[str, TorqueAllocator]:
    """Build a fresh allocator set for a given config (LP2 needs config.J)."""
    return {
        "LP":  LPAllocator(),
        "QP":  QPAllocator(),
        "cQP": QPCAllocator("A"),
        "LP2": LP2Allocator(J=config.J),
    }


class TimingAllocator(TorqueAllocator):
    """Wraps an allocator; records wall-clock of every allocate() call."""

    def __init__(self, inner: TorqueAllocator):
        self.inner = inner
        self.name = inner.name
        self.solve_times_us: list[float] = []

    def allocate(self, *args, **kwargs):
        t0 = time.perf_counter()
        res = self.inner.allocate(*args, **kwargs)
        self.solve_times_us.append((time.perf_counter() - t0) * 1e6)
        return res


def make_scenario(seed: int, h0_mag: float) -> dict:
    """
    One reproducible scenario at a given stored-momentum magnitude.

    Paired across h-levels: the SAME seed yields the same omega/q/goal at every
    h0_mag, with only the stored-momentum magnitude scaling. h_dir_body (the
    direction of stored momentum in the body frame) is drawn from the same rng
    so it is also paired across levels.
    """
    rng = np.random.default_rng(seed)
    axis = normalize(rng.standard_normal(3))
    angle = rng.uniform(0.1, 0.5)
    q0 = normalize(np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)]))
    omega0 = rng.standard_normal(3) * 0.02
    h_dir_body = normalize(rng.standard_normal(3))   # random body-frame direction
    sign_31 = 1.0 if rng.random() < 0.5 else -1.0    # sign for 3+1 wheel scalar
    return {"omega0": omega0, "q0": q0, "h_dir_body": h_dir_body,
            "sign_31": sign_31, "q_goal": np.array([1.0, 0.0, 0.0, 0.0]),
            "seed": seed, "h0_mag": float(h0_mag)}


def _scenario_for(config, seed: int, h0_mag: float) -> dict:
    """
    Slice the shared scenario for the config's RW geometry.

    * 3+1: the wheel can only store momentum along its spin axis, so the
      initial stored momentum is a SCALAR along that axis (sign randomized,
      magnitude = h0_mag) -- "wheel spun up near its limit" per Patrick's note.
    * 3+3: random body-frame direction at fixed magnitude h0_mag, projected
      onto the (orthonormal) wheel axes to recover per-wheel scalars.
    """
    s = make_scenario(seed, h0_mag)
    n_rw = config.A_rw.shape[1]
    if n_rw == 1:
        h_rw = np.array([s["sign_31"] * h0_mag])
    else:
        h_body = h0_mag * s["h_dir_body"]
        h_rw = config.A_rw.T @ h_body
    s["x0"] = np.concatenate([s["omega0"], s["q0"], h_rw])
    return s


def run_one_trial(config, allocator: TorqueAllocator, scenario: dict) -> dict:
    res = simulate_closed_loop(
        config=config, allocator=allocator,
        x0=scenario["x0"], q_goal=scenario["q_goal"],
        b_field_func=time_varying_b_field(orbit_period=5400.0))
    return {
        "seed": scenario["seed"],
        "final_error_deg": float(res.final_error),
        "rms_error_deg": float(res.rms_error),
        "mean_alpha": float(res.mean_alpha),
        "converged": bool(res.final_error < CONV_THRESHOLD_DEG),
    }


def _aggregate(trials: list[dict], solve_times_us: list[float]) -> dict:
    finals = np.array([t["final_error_deg"] for t in trials], dtype=float)
    st = np.array(solve_times_us, dtype=float)
    return {
        "n_trials": len(trials),
        "converged_pct": float(100.0 * np.mean(finals < CONV_THRESHOLD_DEG)),
        "final_error_deg": {
            "mean": float(np.mean(finals)), "median": float(np.median(finals)),
            "p95": float(np.percentile(finals, 95)), "max": float(np.max(finals)),
        },
        "solve_time_us": {
            "n_solves": int(st.size),
            "mean": float(np.mean(st)) if st.size else None,
            "p50": float(np.percentile(st, 50)) if st.size else None,
            "p95": float(np.percentile(st, 95)) if st.size else None,
            "p99": float(np.percentile(st, 99)) if st.size else None,
            "max": float(np.max(st)) if st.size else None,
        },
    }


def time_one_trial() -> None:
    """Run one scenario through every cell; report the full-campaign estimate."""
    total = 0.0
    for cfg_label, cfg_factory in CONFIGS.items():
        config = cfg_factory()
        for h_label, h0 in H0_VARIANTS.items():
            scenario = _scenario_for(config, SEEDS[0], h0)
            print(f"\n[{cfg_label} | {h_label}={h0}] one trial (tf={TF:.0f}s, "
                  f"{int(TF / DT)} steps):")
            for label, alloc in make_allocators(config).items():
                timed = TimingAllocator(alloc)
                t0 = time.perf_counter()
                stats = run_one_trial(config, timed, scenario)
                wall = time.perf_counter() - t0
                total += wall
                st = np.array(timed.solve_times_us)
                print(f"  {label:5s}: {wall:6.2f} s/trial   "
                      f"final_err={stats['final_error_deg']:6.2f} deg   "
                      f"solve {st.mean():.1f} us/step "
                      f"(p95 {np.percentile(st, 95):.1f})")
    batch = total * N_TRIALS
    n_alloc = len(make_allocators(CONFIGS["3+1"]()))
    n_cells = len(CONFIGS) * len(H0_VARIANTS) * n_alloc
    print(f"\n  one trial over {n_cells} cells "
          f"({len(CONFIGS)} configs x {len(H0_VARIANTS)} momenta x {n_alloc} allocators): "
          f"{total:.2f} s")
    print(f"  ESTIMATED full campaign ({N_TRIALS} trials): "
          f"{batch:.0f} s  (~{batch / 60:.1f} min)")
    print("  (single-process; trivially pool-parallelizable over trials)")


def run_campaign() -> None:
    """Full 100-trial x 1000 s campaign over both configs, both momenta, all allocators."""
    out = {
        "campaign": "LP/QP/cQP/LP2 closed-loop allocation benchmark",
        "spec": {"n_trials": N_TRIALS, "tf_s": TF, "dt_s": DT,
                 "conv_threshold_deg": CONV_THRESHOLD_DEG, "seeds": SEEDS,
                 "h0_variants": H0_VARIANTS},
        "created": datetime.now().isoformat(timespec="seconds"),
        "results": {},
    }
    per_step = {}
    for cfg_label, cfg_factory in CONFIGS.items():
        out["results"][cfg_label] = {}
        for h_label, h0 in H0_VARIANTS.items():
            out["results"][cfg_label][h_label] = {}
            for label, alloc in make_allocators(cfg_factory()).items():
                config = cfg_factory()
                timed = TimingAllocator(alloc)
                trials = [run_one_trial(config, timed,
                                        _scenario_for(config, s, h0))
                          for s in SEEDS]
                agg = _aggregate(trials, timed.solve_times_us)
                out["results"][cfg_label][h_label][label] = {
                    "per_trial": trials, "aggregate": agg}
                per_step[f"{cfg_label}_{h_label}_{label}"] = timed.solve_times_us
                print(f"  [{cfg_label} | {h_label}] {label:5s}: "
                      f"conv={agg['converged_pct']:5.1f}%  "
                      f"mean={agg['final_error_deg']['mean']:7.2f} deg  "
                      f"p95={agg['final_error_deg']['p95']:7.2f}  "
                      f"solve p95={agg['solve_time_us']['p95']:.1f} us")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUT_DIR, f"lp_qp_cqp_campaign_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        os.path.join(OUT_DIR, f"lp_qp_cqp_solve_times_{stamp}.npz"),
        **{k: np.asarray(v) for k, v in per_step.items()})
    print(f"  wrote {json_path} (+ solve-time npz)")
    return json_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-one-trial", action="store_true",
                    help="run one trial, report the batch wall-clock estimate")
    ap.add_argument("--run", action="store_true",
                    help="run the full 100-trial campaign over both configs")
    args = ap.parse_args()
    if args.run:
        run_campaign()
    else:
        time_one_trial()
