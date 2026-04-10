import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

import ADCS as ADCS
import matplotlib.pyplot as plt
import numpy as np
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings
from ADCS.helpers.plot.subplot import Subplot
from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_factory.actuators import create_cubewheel_smallplus_rw, create_isis_magnetorquer_board
from ADCS.satellite_factory.sensors import create_Clydespace_3U_array, create_ICM20948_IMU, create_isis_magnetometer
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.disturbances import (
    Drag_Disturbance,
    GG_Disturbance,
    GeometryConfig,
    GeometryFace,
    SRP_Disturbance,
)
from ADCS.satellite_hardware.satellite.satellite import Satellite

# Global toggles for all hardware bias/noise terms used in this constructor.
ENABLE_BIAS = False
ENABLE_NOISE = False


def create_clean_3_1_satellite() -> Satellite:
    """Create a clean 3MTQ+1RW cubesat matching sat_3_1_hybrid.py - no sensors, no disturbances."""
    mass = 4.0
    COM = np.zeros(3)
    J = np.array(
        [
            [0.03136490806, 5.88304e-05, -0.00671361357],
            [5.88304e-05, 0.03409127827, -0.00012334756],
            [-0.00671361357, -0.00012334756, 0.01004091997],
        ],
        dtype=float,
    )

    # Create 3 MTQs with clean bias/noise (matching sat_3_1_hybrid.py)
    mtq_bias = [Bias(bias=0.0, std_bias=0.0) for _ in range(3)]
    mtq_noise = [Noise(noise=0.0, std_noise=0.0) for _ in range(3)]
    mtqs = create_isis_magnetorquer_board(bias=mtq_bias, noise=mtq_noise, estimate_bias=False)
    
    # Create 1 RW (matching sat_3_1_hybrid.py)
    rw = create_cubewheel_smallplus_rw(
        axis=np.array([0.0, 0.0, 1.0]),
        bias=Bias(bias=0.0, std_bias=0.0),
        noise=Noise(noise=0.0, std_noise=0.0),
        h_meas_noise=Noise(noise=0.0, std_noise=0.0),
        estimate_bias=False,
    )

    boresight = np.array([0.0, 1.0, 0.0])

    return Satellite(
        mass=mass,
        COM=COM,
        J_0=J,
        disturbances=[],
        sensors=[],
        actuators=mtqs + [rw],
        boresight=boresight,
    )


class RWMomentumPlotSingle(Subplot):
    def __init__(
        self,
        rw_state_index: int = 7,
        *,
        title: str = "RW Momentum State",
        units: str = "N·m·s",
        label: str = "$h_{rw}$",
    ):
        self.rw_state_index = rw_state_index
        self.title = title
        self.units = units
        self.label = label

    def plot(self, ax, sim) -> None:
        runs = sim.runs if hasattr(sim, "runs") else [sim]

        for run in runs:
            if run.state_hist is None or len(run.state_hist) == 0:
                continue
            t = np.asarray(run.time_s)
            X = np.vstack(run.state_hist)
            if self.rw_state_index >= X.shape[1]:
                continue
            ax.plot(t, X[:, self.rw_state_index], alpha=0.25)

        ax.set_title(self.title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{self.label} [{self.units}]")
        ax.grid(True)


class QuaternionGoalErrorPlotSingle(Subplot):
    def __init__(
        self,
        q_goal: np.ndarray | None = None,
        *,
        title: str = "Attitude Error vs Goal",
        units: str = "deg",
        label: str = r"$\theta_{err}$",
    ):
        self.q_goal = None
        if q_goal is not None:
            q_goal = np.asarray(q_goal, dtype=float).reshape(4)
            self.q_goal = q_goal / max(np.linalg.norm(q_goal), 1e-15)
        self.title = title
        self.units = units
        self.label = label

    @staticmethod
    def _quat_error_deg(q: np.ndarray, q_ref: np.ndarray) -> float:
        q = np.asarray(q, dtype=float).reshape(4)
        q_ref = np.asarray(q_ref, dtype=float).reshape(4)
        q = q / max(np.linalg.norm(q), 1e-15)
        q_ref = q_ref / max(np.linalg.norm(q_ref), 1e-15)
        d = float(np.clip(np.abs(np.dot(q, q_ref)), -1.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(d)))

    @staticmethod
    def _boresight_eci(q_b2i: np.ndarray, bore_body_unit: np.ndarray) -> np.ndarray:
        q0, q1, q2, q3 = q_b2i
        r = np.array(
            [
                [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                [2 * (q1 * q2 + q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 - q0 * q1)],
                [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
            ],
            dtype=float,
        )
        out = r @ bore_body_unit
        return out / max(np.linalg.norm(out), 1e-15)

    @staticmethod
    def _vec_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
        u = np.asarray(u, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)
        u = u / max(np.linalg.norm(u), 1e-15)
        v = v / max(np.linalg.norm(v), 1e-15)
        d = float(np.clip(np.dot(u, v), -1.0, 1.0))
        return float(np.degrees(np.arccos(d)))

    def plot(self, ax, sim) -> None:
        runs = sim.runs if hasattr(sim, "runs") else [sim]

        for run in runs:
            if run.state_hist is None or len(run.state_hist) == 0:
                continue

            t = np.asarray(run.time_s)
            X = np.vstack(run.state_hist)
            if X.shape[1] < 7:
                continue

            n = min(len(t), len(X))
            target_hist = getattr(run, "target_hist", None)
            boresight_hist = getattr(run, "boresight_hist", None)

            if target_hist is not None and len(target_hist) > 0:
                n = min(n, len(target_hist))
                err_deg = np.full(n, np.nan, dtype=float)

                for i in range(n):
                    row = np.asarray(target_hist[i], dtype=float).reshape(-1)
                    if row.size != 4:
                        continue

                    q = X[i, 3:7]
                    if not np.isnan(row[0]):
                        err_deg[i] = self._quat_error_deg(q, row)
                        continue

                    if boresight_hist is None or i >= len(boresight_hist):
                        continue

                    bore_body = np.asarray(boresight_hist[i], dtype=float).reshape(-1)
                    if bore_body.size != 3 or np.linalg.norm(bore_body) <= 0:
                        continue

                    target_vec = row[1:4]
                    if np.linalg.norm(target_vec) <= 0:
                        continue

                    bore_unit = bore_body / np.linalg.norm(bore_body)
                    target_unit = target_vec / np.linalg.norm(target_vec)
                    bore_eci = self._boresight_eci(q, bore_unit)
                    err_deg[i] = self._vec_angle_deg(bore_eci, target_unit)

                ax.plot(t[:n], err_deg, alpha=0.25)
                continue

            q_hist = np.asarray(X[:, 3:7], dtype=float)
            q_norm = np.linalg.norm(q_hist, axis=1, keepdims=True)
            q_norm = np.where(q_norm > 1e-15, q_norm, 1e-15)
            q_hist = q_hist / q_norm

            if self.q_goal is not None:
                dots = np.clip(np.abs(q_hist[:n] @ self.q_goal), -1.0, 1.0)
                err_deg = np.degrees(2.0 * np.arccos(dots))
                ax.plot(t[:n], err_deg, alpha=0.25)

        ax.set_title(self.title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{self.label} [{self.units}]")
        ax.grid(True)

real_sat = create_clean_3_1_satellite()
x_0 = np.array([0.01, 0.01, -0.01] + [1, 0, 0, 0] + [0.0])  # w, q, h

planner_settings = PlannerSettings(est_sat=real_sat)
planner_settings.passes[0].dt = 5.0

# Tune SALTRO planner settings to match debug_saltro_3+1v.py
planner_settings.passes[0].ilqr.cost_tol = 1e-5
planner_settings.passes[0].ilqr.max_iters = 20

planner_settings.passes[0].aug_lag.max_outer_iters = 30
planner_settings.passes[0].aug_lag.constraint_tol = 1e-3

cost = planner_settings.passes[0].cost
cost.angle = 1e2
cost.ang_vel = 1e5
cost.ang_vel_mag = 0.0
cost.ang_vel_err_dir = 0.0
cost.control_mult = 1.0
cost.mtq_control_weight = 1.0
cost.rw_control_weight = 1.0
cost.magic_control_weight = 0.0
cost.rw_AM_weight = 0.0
cost.rw_stic_weight = 0.0
cost.RWh_max_mult = 0.0
cost.RWh_stiction_mult = 0.0
cost.RWh_ok_mult = 0.0
cost.angle_N = 0.0
cost.ang_vel_N = 0.0
cost.ang_vel_mag_N = 0.0
cost.ang_vel_err_dir_N = 0.0
cost.ang_cost_func_type = 0
cost.use_cost_hess = True

planner_settings.passes[0].reg.reg_init = 1e-6
planner_settings.passes[0].reg.reg_max = 1e10
planner_settings.passes[0].reg.reg_scale = 10.0
planner_settings.passes[0].reg.use_dynamics_hess = False
planner_settings.passes[0].reg.use_constraint_hess = False

planner_settings.passes[0].linesearch.max_iters = 24
planner_settings.passes[0].linesearch.beta1 = 1e-10
planner_settings.passes[0].linesearch.beta2 = 5000.0

controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

os0 = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
    V=np.array([8.0, 0.0, 0.0]),
)

goal = ADCS.goals.ECI_Goal(np.array([0, 0, -1]))

# Precompute and set the active trajectory so we can plot open-loop explicitly
# and avoid the internal default open-loop plot in ADCS.simulate.
traj = controller.calculate_trajectory(
    t_start=os0.J2000,
    duration=1000.0,
    x_0=x_0,
    os_0=os0,
    goals=ADCS.GoalList({os0.J2000: goal}),
    verbose=False,
)
controller.set_active_trajectory(traj)

target_ref, w_ref = goal.to_ref(os0)
open_loop_results = traj.to_simulation_results(
    satellite=real_sat,
    target=target_ref,
    w_target=w_ref,
)

# TargetPlot requires boresight history; open-loop trajectory conversion does not
# populate it, so provide a constant boresight for all samples.
open_loop_run = open_loop_results.runs[0]
open_loop_bore = np.asarray(real_sat.get_boresight(), dtype=float).reshape(3)
open_loop_run.boresight_hist = [open_loop_bore.copy() for _ in range(len(open_loop_run.state_hist))]

ADCS.plot(
    open_loop_results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="All Actuator Commands", units="Command"),
    ADCS.plots.ControlPlotSingle(index=3, title="RW Command", units="N·m", label="$u_{rw}$"),
    RWMomentumPlotSingle(title="RW Momentum State (Open-Loop)", label="$h_{rw}$"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 3),
    title="3+1 SALTRO Open-Loop (Vector Goal)",
)

results = ADCS.simulate(
    x=x_0,
    satellite=real_sat,
    controller=controller,
    goal=goal,
    os0=os0,
    dt=1.0,
    tf=1000.0,
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1, 1),
    title="3+1 SALTRO Reduced (Vector Goal)",
)

ADCS.plot(
    results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="All Actuator Commands", units="Command"),
    ADCS.plots.ControlPlotSingle(index=3, title="RW Command", units="N·m", label="$u_{rw}$"),
    RWMomentumPlotSingle(title="RW Momentum State (Closed-Loop)", label="$h_{rw}$"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 3),
    title="3+1 SALTRO Closed-Loop (Vector Goal)",
)

plt.show()
