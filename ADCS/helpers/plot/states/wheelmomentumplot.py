"""Reaction-wheel angular momentum from the stored momentum states."""

import numpy as np

from ADCS.helpers.plot.subplot import Subplot


class WheelMomentumPlot(Subplot):
    """Plot each wheel's signed momentum and the total stored body momentum.

    Wheel momentum is read from ``State.h`` in N m s. The total is the norm
    of the vector sum of wheel momenta along their body-frame axes.
    """

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", [sim])
        for run in runs:
            wheels = run.satellite.rw_actuators
            if not wheels or not run.state_hist:
                continue

            h = np.asarray([state.h for state in run.state_hist], dtype=float)
            time = (np.asarray(run.time_s, dtype=float)
                    if run.time_s is not None else np.arange(len(h)))
            axes = np.asarray([wheel.axis for wheel in wheels], dtype=float)

            for i in range(len(wheels)):
                ax.plot(time[:len(h)], h[:, i], label=f"RW {i + 1}")
            ax.plot(time[:len(h)], np.linalg.norm(h @ axes, axis=1),
                    color="black", linewidth=2, label="Stored |h|")

        ax.set(title="Reaction-Wheel Momentum", xlabel="Time [s]",
               ylabel="Angular momentum [N m s]")
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()
