__all__ = ["OrbitDensityPlot", "OrbitDensityModelPlot"]

import numpy as np

from ..subplot import Subplot
from ADCS.orbits.universal_constants import EarthConstants


class OrbitDensityPlot(Subplot):
    r"""
    Time-history plot of atmospheric density along the spacecraft orbit.

    This class visualizes the atmospheric density experienced by the spacecraft
    as a function of time, using orbital state history provided by the simulation.
    It is intended for quick inspection of drag-related environmental conditions
    during the mission timeline.

    The user can configure axis scaling, units, and plot title without needing
    to understand the underlying density model or orbital propagation.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Physical units used for atmospheric density.
    :type units:
        str

    :param log_y:
        If True, the density axis is displayed on a logarithmic scale.
    :type log_y:
        bool

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Atmospheric Density",
        units: str = "kg/m$^3$",
        log_y: bool = True,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is not None and isinstance(runs, (list, tuple)) and len(runs) > 0:
            print(f"[OrbitDensityPlot] MCSimulationResults detected: showing run 0 of {len(runs)}")
            sim = runs[0]

        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        rho = np.array([float(os.rho) for os in sim.os_hist], dtype=float)

        ax.plot(t, rho, label=r"$\rho$")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"$\\rho$ [{self.units}]")
        ax.set_title(self.title, loc="left", pad=10)
        if self.log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.legend()


class OrbitDensityModelPlot(Subplot):
    r"""
    Altitude-dependent visualization of the atmospheric density model.

    This class plots the atmospheric density model as a function of altitude,
    independent of time history. It evaluates the density model associated with
    the first available orbital state and displays the resulting profile.

    The plot is useful for understanding the structure and range of the selected
    density model, with configurable altitude limits and axis scaling.

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Physical units used for atmospheric density.
    :type units:
        str

    :param h_max_km:
        Maximum altitude in kilometers shown on the plot.
    :type h_max_km:
        float

    :param n_points:
        Number of altitude samples used to evaluate the density model.
    :type n_points:
        int

    :param log_x:
        If True, the density axis is displayed on a logarithmic scale.
    :type log_x:
        bool

    """
    def __init__(
        self,
        *,
        title: str = "Atmospheric Density Model",
        units: str = "kg/m$^3$",
        h_max_km: float = 1000.0,
        n_points: int = 300,
        log_x: bool = True,
    ):
        self.title = title
        self.units = units
        self.h_max_km = h_max_km
        self.n_points = n_points
        self.log_x = log_x

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is not None and isinstance(runs, (list, tuple)) and len(runs) > 0:
            print(f"[OrbitDensityModelPlot] MCSimulationResults detected: showing run 0 of {len(runs)}")
            sim = runs[0]

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        os0 = sim.os_hist[0]
        if getattr(os0, "density_model", None) is None:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No density_model available in os_hist[0]", ha="center", va="center")
            return

        h = np.linspace(0.0, float(self.h_max_km), int(self.n_points))
        rho = np.array([float(os0.density_model.interpolate(hi)) for hi in h], dtype=float)

        ax.plot(rho, h, label=r"$\rho(h)$")
        ax.set_xlabel(f"$\\rho$ [{self.units}]")
        ax.set_ylabel("Altitude [km]")
        ax.set_title(self.title, loc="left", pad=10)
        if self.log_x:
            ax.set_xscale("log")
        ax.grid(True, which="both")
        ax.legend()
