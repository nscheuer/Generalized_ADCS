"""Plot estimated disturbance-parameter histories."""

from __future__ import annotations

import numpy as np

from ..subplot import Subplot


__all__ = ["DisturbanceParameterPlot"]


def _parameters(disturbances) -> np.ndarray:
    values = []
    for disturbance in disturbances or ():
        length = int(getattr(disturbance, "estimated_vector_length", 0))
        if length <= 0:
            continue
        values.append(np.asarray(disturbance.main_param, dtype=float).reshape(-1))
    return np.concatenate(values) if values else np.empty(0)


class DisturbanceParameterPlot(Subplot):
    """Plot estimated disturbance parameters against truth parameters.

    Truth parameters are read from the true satellite disturbance models in
    the simulation result; estimates are read from each ``EstimatorState``.
    """

    def __init__(self, *, title: str = "Disturbance Parameters", labels=None, units: str = "", plot_torque: bool = False):
        self.title = title
        self.labels = labels
        self.units = units
        self.plot_torque = plot_torque

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None) or [sim]
        run = runs[0]
        estimates = np.vstack([
            np.asarray(state.dist_param, dtype=float).reshape(-1)
            for state in run.est_state_hist
        ])
        if self.plot_torque:
            truth = np.vstack([
                np.asarray(run.satellite.dist_torques(state, os), dtype=float).reshape(3)
                for state, os in zip(run.state_hist, run.os_hist)
            ])
        else:
            truth = _parameters(run.satellite.disturbances)
        time = np.asarray(run.time_s, dtype=float)
        n = estimates.shape[1]
        labels = self.labels or [rf"$d_{{{i}}}$" for i in range(n)]
        if len(labels) != n:
            raise ValueError("labels length must match disturbance-parameter dimension")
        for index in range(n):
            ax.plot(time, estimates[:, index], "--", label=f"{labels[index]} estimate")
            if self.plot_torque and truth.shape[1] == n:
                ax.plot(time, truth[:, index], "-", label=f"{labels[index]} truth")
            elif not self.plot_torque and truth.size == n:
                ax.plot(time, np.full(time.shape, truth[index]), "-", label=f"{labels[index]} truth")
        ax.set_title(self.title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(self.units or "Parameter")
        ax.grid(True)
        ax.legend()
