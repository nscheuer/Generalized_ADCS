import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_SCRIPT = REPO_ROOT / "papers" / "3MTQ+1RW" / "new" / "debug_saltro_3+1_reduced.py"


def _extract_json_marker(stdout: str, stderr: str) -> Dict[str, Any]:
    merged = "\n".join([stdout, stderr])
    matches = re.findall(r"__JSON__(\{.*\})", merged)
    if not matches:
        raise RuntimeError("Could not find __JSON__ payload in subprocess output")
    return json.loads(matches[-1])


def _run_py(code: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Subprocess failed\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return _extract_json_marker(proc.stdout, proc.stderr)


def _saltro_case(tf: float, dt: float, goal_mode: str, boresight_name: str | None) -> Dict[str, Any]:
    code = f'''
import json, runpy
ns = runpy.run_path(r"{str(TARGET_SCRIPT)}")
run = ns["run_saltro_3p1_reduced"]
_, trk, tim = run(
    tf={tf},
    dt={dt},
    plot=False,
    verbose=False,
    goal_mode={goal_mode!r},
    use_saltro_3_1_debug_settings=True,
    vector_goal_boresight_name={boresight_name!r},
)
print("__JSON__" + json.dumps({{"timing": tim, "tracking": trk}}))
'''
    return _run_py(code)


def _legacy_altro_case(tf: float, dt: float) -> Dict[str, Any]:
    code = f'''
import json, time
import numpy as np
import ADCS as ADCS
from ADCS.CONOPS.goals import ECI_Goal

sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0], dtype=float)
os0 = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)
goal = ECI_Goal(np.array([0.0, 0.0, -1.0]))

ps = ADCS.controller.helpers.PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=50, dt_tvlqr=1.0)
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
ps.cost_main = ADCS.controller.helpers.CostWeights(
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
ps.cost_second = ps.cost_main
ps.cost_tvlqr = ADCS.controller.helpers.CostWeights(
    angle=1e5,
    angle_N=1e6,
    ang_vel=1e6,
    ang_vel_N=1e8,
    ang_vel_mag=0.0,
    ang_vel_mag_N=0.0,
    control_mult=1.0,
    ang_cost_func_type=2,
)
ctl = ADCS.controller.Plan_and_Track_LQR(est_sat=sat, planner_settings=ps)

# Use same tracking metric helper as SALTRO debug script for apples-to-apples comparison.
from runpy import run_path
ns = run_path(r"{str(TARGET_SCRIPT)}")
metric = ns["_tracking_error_stats_deg"]

start = time.perf_counter()
res = ADCS.simulate(x=x0, satellite=sat, controller=ctl, goal=goal, os0=os0, dt={dt}, tf={tf})
end = time.perf_counter()
run = res.first()
trk = metric(np.asarray(run.state_hist), np.asarray(run.target_hist), np.asarray(sat.get_boresight()))
print("__JSON__" + json.dumps({{"timing": {{"total_s": float(end - start)}}, "tracking": trk}}))
'''
    return _run_py(code)


def _fmt(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SALTRO vector/quaternion goals against legacy trajectory_planner")
    parser.add_argument("--tf", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--vector-boresight", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    saltro_quat = _saltro_case(args.tf, args.dt, goal_mode="quat_slew90", boresight_name=None)
    saltro_vector = _saltro_case(args.tf, args.dt, goal_mode="eci_vector", boresight_name=args.vector_boresight)
    legacy = _legacy_altro_case(args.tf, args.dt)

    rows = [
        ("SALTRO quat", saltro_quat),
        ("SALTRO vector", saltro_vector),
        ("Legacy planner", legacy),
    ]

    print("\n=== Parity Comparison (tf={:.1f}s, dt={:.1f}s) ===".format(args.tf, args.dt))
    print("{:<16} {:>10} {:>10} {:>10} {:>10}".format("Case", "plan_s", "sim_s", "err_final", "err_mean"))
    for name, payload in rows:
        timing = payload.get("timing", {})
        tracking = payload.get("tracking", {})
        plan_s = timing.get("planning_s", float("nan"))
        sim_s = timing.get("simulation_s", timing.get("total_s", float("nan")))
        final_e = tracking.get("final", float("nan"))
        mean_e = tracking.get("mean", float("nan"))
        print(
            "{:<16} {:>10} {:>10} {:>10} {:>10}".format(
                name,
                _fmt(plan_s),
                _fmt(sim_s),
                _fmt(final_e),
                _fmt(mean_e),
            )
        )

    result = {
        "meta": {
            "tf": args.tf,
            "dt": args.dt,
            "vector_boresight": args.vector_boresight,
            "python": sys.executable,
        },
        "saltro_quat": saltro_quat,
        "saltro_vector": saltro_vector,
        "legacy_planner": legacy,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
