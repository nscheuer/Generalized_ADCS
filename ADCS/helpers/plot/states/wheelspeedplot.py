"""Reaction wheel speed from the stored momentum states."""

import numpy as np

from ADCS.helpers.plot.subplot import Subplot


class WheelSpeedPlot(Subplot):
    """Plot each reaction wheel's speed relative to the body, in rad/s."""

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", [sim])
        for run in runs:
            wheels = run.satellite.rw_actuators
            if not wheels or not run.state_hist:
                continue
            speed = np.asarray([state.h for state in run.state_hist], dtype=float)
            time = np.asarray(run.time_s) if run.time_s is not None else np.arange(len(speed))
            for i, wheel in enumerate(wheels):
                ax.plot(time[:len(speed)], speed[:, i] / wheel.J, label=f"RW {i + 1}")
        ax.set(title="Reaction Wheel Speed", xlabel="Time [s]", ylabel="Speed [rad/s]")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()
