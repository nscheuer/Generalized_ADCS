"""Debug harness for the P2.6 disturbed test (cold-gas leak + RW momentum).

Mirrors the existing OldPlanner/SALTRO debug examples
(papers/Planner/debug_altro_3+1_reduced.py, papers/3MTQ+1RW/new/
debug_saltro_3+1_reduced.py): a closed-loop ADCS.simulate run plotted with the
standard ADCS GUI (AnimationPlot clickthrough + Attitude / AngularVelocity /
Control / TargetHistogram / TargetPlot panels), but with the cold-gas leak
disturbance and the knobs we traced this session.

Plus a bonus first step: a single planner solve from a high-RW-momentum state
with the per-iteration AL-iLQR solver trace (cost / constraint violation /
regularization), captured from the C++ via the TP_DEBUG_ITER / TP_DEBUG_RESET
hooks added to OldPlanner.cpp. Set TRACE_SOLVE=False to skip it.

Why the knobs matter (traced this session)
------------------------------------------
  DT_TP        50 -> warm-start RK4 rollout diverges to NaN at high momentum ->
               reg-proof reset loop (grind, now bailed by a guard). 10 -> ok.
  H0           initial RW momentum for the trace solve (the grind trigger).
  ANG_COST_TYPE 2=acos, 1=0.5(1-c)^2, 0=1-c.
  LEAK / MTQ / ANGLE_SCALE / USE_DDP / AM_WEIGHT.

Run:  ../../venv/bin/python debug_p2.6_disturbed.py
"""
import os
import re
import sys
import time
import tempfile
import importlib.util

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================ KNOBS =======================================
LEAK          = 2.0e-4   # cold-gas leak torque magnitude [N.m]
H0            = 0.08     # initial RW momentum along leak axis for the trace solve
DT_TP         = 1.0     # planner integration step [s]   (50 -> NaN grind, 10 -> ok)
MTQ           = 50.0     # MTQ dipole [A.m^2]
ANGLE_SCALE   = 1.0      # TVLQR attitude-weight multiplier
ANG_COST_TYPE = 2        # 2=acos, 1=0.5(1-c)^2, 0=1-c
USE_DDP       = False    # use_dynamics_hess (full DDP vs Gauss-Newton)
AM_WEIGHT     = 1.0e4    # RW angular-momentum (dump) cost weight
TF            = 300.0    # closed-loop sim duration [s] (dt_tp=10 planning is ~Nknots heavy)
TRACE_SOLVE   = True     # do the bonus single-solve per-iteration trace
# ==========================================================================

ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "papers", "Planner"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load(os.path.join(ROOT, "papers", "Planner", "generate_p2.6_coldgas.py"), "p26gen")
import ADCS  # noqa: E402
import _paper2_sim as P  # noqa: E402
from ADCS.satellite_hardware.disturbances import Prop_Disturbance  # noqa: E402

# generate_p2.6_coldgas forces matplotlib Agg (headless batch). Switch back to an
# interactive backend so the GUI windows display; PNG save is the fallback.
for _be in ((([os.environ["MPLBACKEND"]] if os.environ.get("MPLBACKEND", "").lower() not in ("", "agg") else []))
            + ["macosx", "tkagg", "qtagg"]):
    try:
        plt.switch_backend(_be)
        break
    except Exception:
        continue
print(f"[debug] matplotlib backend: {matplotlib.get_backend()}")

m.MTQ_DIPOLE = MTQ
LEAK_VEC = LEAK * m.LEAK_AXIS


def make_controller(real_sat):
    ps = P.make_planner_settings(real_sat)
    ps.dt_tp = DT_TP
    ps.pass1.regularization.use_dynamics_hess = int(USE_DDP)
    ps.cost_main.use_full_cost_hessian = bool(USE_DDP)
    ps.cost_main.ang_cost_func_type = ANG_COST_TYPE
    ps.cost_tvlqr.ang_cost_func_type = ANG_COST_TYPE
    ps.cost_tvlqr.angle *= ANGLE_SCALE
    ps.cost_tvlqr.angle_N *= ANGLE_SCALE
    ps.rw_AM_weight = AM_WEIGHT
    ps.rw_stic_weight = 0.0
    ps.plan_for_gendist = True       # oracle feedforward (debug): exact leak
    ps.gendist_torq = LEAK_VEC.copy()
    return ADCS.controller.Plan_and_Track_LQR(est_sat=real_sat, planner_settings=ps)


print(f"[debug] leak={LEAK:g} dt_tp={DT_TP:g} MTQ={MTQ:g} angType={ANG_COST_TYPE} "
      f"ddp={USE_DDP} ang_scale={ANGLE_SCALE:g}")

# =====================================================================
# Bonus: single planner solve from a high-momentum state, with the
# per-iteration AL-iLQR solver trace captured from the C++.
# =====================================================================
if TRACE_SOLVE:
    sat_t = m.make_serious_sat(estimated=False)
    n_rw = sum(a.__class__.__name__ == "RW" for a in sat_t.actuators)
    ctrl_t = make_controller(sat_t)
    os0 = P.default_os0()
    orb = m.Orbit(os0=os0, end_time=os0.J2000 + (m.SUNLIT_OFFSET_S + 1600) * m.SEC2CENT,
                  dt=1.0, use_J2=True, fast=False, verbose=False)
    osk = orb.get_os(J2000=os0.J2000 + (m.SUNLIT_OFFSET_S + 1000) * m.SEC2CENT)
    gl = ADCS.GoalList(goal_timeline={0.0: ADCS.goals.AntiVelocity_Goal()},
                       time_units="seconds", start_juliantime=os0.J2000)
    x0 = P.x0(n_rw)
    x0[7:7 + n_rw] = H0 * m.LEAK_AXIS

    os.environ["TP_DEBUG_ITER"] = "1"
    os.environ["TP_DEBUG_RESET"] = "1"
    capfile = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False).name
    saved_fd, cap_fd = os.dup(2), os.open(capfile, os.O_WRONLY | os.O_TRUNC)
    sys.stderr.flush(); os.dup2(cap_fd, 2)
    t0 = time.perf_counter(); ok_t, err_t = True, ""
    try:
        ctrl_t.calculate_trajectory(osk.J2000, 600.0, x0.copy(), osk, gl)
    except Exception as exc:  # noqa: BLE001
        ok_t, err_t = False, str(exc)
    finally:
        sys.stderr.flush(); os.dup2(saved_fd, 2); os.close(cap_fd); os.close(saved_fd)
    cap = open(capfile).read()
    pat = re.compile(r"\[TP_ITER\] outer=(\S+) inner=(\S+) iter=(\S+) cmax=(\S+) "
                     r"cost=(\S+) costClean=(\S+) grad=(\S+) rho=(\S+)")
    rows = [[float(x) for x in mm.groups()] for line in cap.splitlines()
            if (mm := pat.match(line))]
    nres = cap.count("[TP_RESET]")
    itr = np.array(rows) if rows else np.zeros((0, 8))
    print(f"[debug] trace solve {time.perf_counter()-t0:.1f}s ok={ok_t} "
          f"iters={len(itr)} resets={nres}")
    if nres > 10000:
        print(f"  *** {nres} resets -> NaN/indefinite reset loop (grind). Lower DT_TP. ***")
    if len(itr):
        x = np.arange(len(itr))
        figi, ax = plt.subplots(2, 2, figsize=(11, 7), num="per-iteration solver trace")
        ax[0, 0].semilogy(x, np.abs(itr[:, 4]) + 1e-30); ax[0, 0].set_title("AL cost")
        ax[0, 1].semilogy(x, np.abs(itr[:, 5]) + 1e-30); ax[0, 1].set_title("clean cost")
        ax[1, 0].semilogy(x, np.abs(itr[:, 3]) + 1e-30); ax[1, 0].set_title("max constraint violation")
        ax[1, 1].semilogy(x, np.abs(itr[:, 7]) + 1e-30); ax[1, 1].set_title("regularization rho")
        for a in ax.flat:
            a.set_xlabel("iteration"); a.grid(True, which="both", alpha=0.3)
        figi.suptitle(f"per-iteration trace (leak={LEAK:g}, h0={H0:g}, dt_tp={DT_TP:g})")
        figi.tight_layout()
        _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
        os.makedirs(_out, exist_ok=True)
        figi.savefig(os.path.join(_out, f"dbg_iters_dttp{DT_TP:g}_h{H0:g}_leak{LEAK:g}.png"), dpi=130)

# =====================================================================
# Main: closed-loop ADCS.simulate under the leak, plotted with the EXACT
# standard ADCS GUI (AnimationPlot clickthrough + standard panels).
# =====================================================================
real_sat = m.make_serious_sat(estimated=False)
leak = Prop_Disturbance(torque_nominal=LEAK_VEC.copy())
leak.current_torque = LEAK_VEC.copy()
real_sat.disturbances = list(real_sat.disturbances) + [leak]
controller = make_controller(real_sat)

os0 = P.default_os0()
x_0 = P.x0(sum(a.__class__.__name__ == "RW" for a in real_sat.actuators))
goal = ADCS.goals.AntiVelocity_Goal()

print(f"[debug] closed-loop ADCS.simulate  tf={TF:g}s ...")
results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=TF,
)

ADCS.plot(
    results,
    ADCS.plots.AnimationPlot(),
    layout=(1, 1),
    title=f"P2.6 disturbed (leak={LEAK:g} N.m) — attitude animation",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="All Actuator Commands", units="cmd"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 3),
    title=f"P2.6 disturbed — closed-loop (dt_tp={DT_TP:g}, MTQ={MTQ:g})",
)
print(DT_TP)
plt.show()
