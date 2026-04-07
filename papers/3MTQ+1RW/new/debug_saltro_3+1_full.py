import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

import ADCS as ADCS
import matplotlib.pyplot as plt
import numpy as np
from ADCS.controller.neo_planner.NEO_planner_settings import PlannerSettings
from ADCS.helpers.plot.subplot import Subplot


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

real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0])  # w, q, h

planner_settings = PlannerSettings(est_sat=real_sat)
planner_settings.passes[0].dt = 1.0

controller = ADCS.controller.SALTRO(est_sat=real_sat, planner_settings=planner_settings)

os0 = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(),
    J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]),
    V=np.array([0.0, 7.5, 0.0]),
)

q_goal = np.array([0.0, 1.0, 0.0, 0.0])
goal = ADCS.goals.Fixed_Attitude_Goal(q_ref=q_goal)

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

ADCS.plot(
    open_loop_results,
    ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
    ADCS.plots.ControlPlotCombined(title="All Actuator Commands", units="Command"),
    ADCS.plots.ControlPlotSingle(index=3, title="RW Command", units="N·m", label="$u_{rw}$"),
    RWMomentumPlotSingle(title="RW Momentum State (Open-Loop)", label="$h_{rw}$"),
    ADCS.plots.TargetHistogram(bin_width=5.0),
    ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
    layout=(2, 3),
    title="3+1 SALTRO Open-Loop (Quat Goal)",
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
    ADCS.plots.AnimationPlot(),
    layout=(1, 1),
    title="3+1 SALTRO Reduced (Quat Goal)",
)

ADCS.plot(
    results,
    ADCS.plots.AttitudePlot(sources=["real", "reference"]),
    layout=(1, 1),
    title="3+1 SALTRO Reduced (Quat Goal)",
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
    title="3+1 SALTRO Closed-Loop (Quat Goal)",
)

plt.show()
