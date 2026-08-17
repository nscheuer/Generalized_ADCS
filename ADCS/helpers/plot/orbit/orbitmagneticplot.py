__all__ = ["OrbitMagneticPlot", "OrbitMagneticPlotSingle"]

import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot
from typing import Sequence


class OrbitMagneticPlot(Subplot):
    r"""
    Multi-panel visualization of the geomagnetic field along the orbit.

    This class displays the three Cartesian components of the magnetic field in
    the inertial frame, together with the field magnitude, arranged in a fixed
    grid layout. It is intended for quick inspection of magnetic environment
    variations during orbital motion.

    Users mainly control appearance options such as colors, axis scaling, and
    labeling.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot group.
    :type title:
        str

    :param units:
        Physical units of the magnetic field values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z magnetic field components.
    :type colors:
        tuple[str, str, str]

    :param mag_color:
        Color used for the magnetic field magnitude plot.
    :type mag_color:
        str

    :param log_y:
        If True, all magnetic field axes use logarithmic scaling.
    :type log_y:
        bool

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Magnetic Field (ECI)",
        units: str = "T",
        colors=("tab:blue", "tab:orange", "tab:green"),
        mag_color: str = "tab:red",
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.mag_color = mag_color
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())
        ax_bx = ax.figure.add_subplot(gs[0, 0])
        ax_bmag = ax.figure.add_subplot(gs[0, 1])
        ax_by = ax.figure.add_subplot(gs[1, 0])
        ax_bz = ax.figure.add_subplot(gs[1, 1])

        first = runs[0]
        t = getattr(first, self.time)

        alpha = max(0.15, 1.0 / len(runs))

        for run in runs:
            if run.os_hist is None or len(run.os_hist) == 0:
                continue

            B = np.vstack([np.asarray(os.B) for os in run.os_hist])
            bmag = np.linalg.norm(B, axis=1)

            ax_bx.plot(t, B[:, 0], color=self.colors[0], alpha=alpha)
            ax_by.plot(t, B[:, 1], color=self.colors[1], alpha=alpha)
            ax_bz.plot(t, B[:, 2], color=self.colors[2], alpha=alpha)
            ax_bmag.plot(t, bmag, color=self.mag_color, alpha=alpha)

        labels = [r"$B_x$", r"$B_y$", r"$B_z$"]
        axes = [ax_bx, ax_by, ax_bz]

        for i, ax_i in enumerate(axes):
            ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.grid(True, which="both")

        ax_bmag.set_ylabel(rf"$\|B\|$ [{self.units}]")
        if self.log_y:
            ax_bmag.set_yscale("log")
        ax_bmag.grid(True, which="both")

        ax_by.set_xlabel("Time [s]")
        ax_bz.set_xlabel("Time [s]")
        ax_bx.set_title(self.title, loc="left", pad=10)


class OrbitMagneticPlotSingle(Subplot):
    r"""
    Single-component visualization of the geomagnetic field.

    This class plots one selected magnetic field component or the field magnitude
    as a function of time. It is useful when focusing on a specific direction or
    when embedding a single magnetic signal into a larger figure.

    The component, labeling, and visual style are fully user-configurable.

    :param component:
        Magnetic field component to plot. Must be one of x, y, z, or m for magnitude.
    :type component:
        str

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title of the plot. If None, a default title is used.
    :type title:
        str or None

    :param units:
        Physical units of the magnetic field values.
    :type units:
        str

    :param color:
        Color used for the plotted magnetic field signal.
    :type color:
        str or None

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param labels:
        Optional mapping from component identifiers to display labels.
    :type labels:
        dict[str, str] or None

    """
    def __init__(
        self,
        *,
        component: str,
        time: str = "time_s",
        title: str | None = None,
        units: str = "T",
        color: str | None = None,
        log_y: bool = False,
        labels: dict[str, str] | None = None,
    ):
        if component not in {"x", "y", "z", "m"}:
            raise ValueError("component must be one of 'x', 'y', 'z', or 'm'")
        self.component = component
        self.time = time
        self.units = units
        self.color = color
        self.log_y = log_y
        self.labels = labels or {
            "x": r"$B_x$",
            "y": r"$B_y$",
            "z": r"$B_z$",
            "m": r"$\|B\|$",
        }
        self.title = title

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        first = runs[0]
        t = getattr(first, self.time)

        alpha = max(0.15, 1.0 / len(runs))

        for run in runs:
            if run.os_hist is None or len(run.os_hist) == 0:
                continue

            B = np.vstack([np.asarray(os.B) for os in run.os_hist])

            if self.component == "x":
                y = B[:, 0]
                label = self.labels["x"]
                color = self.color or "tab:blue"
                title = self.title or "Magnetic Field $B_x$ (ECI)"
            elif self.component == "y":
                y = B[:, 1]
                label = self.labels["y"]
                color = self.color or "tab:orange"
                title = self.title or "Magnetic Field $B_y$ (ECI)"
            elif self.component == "z":
                y = B[:, 2]
                label = self.labels["z"]
                color = self.color or "tab:green"
                title = self.title or "Magnetic Field $B_z$ (ECI)"
            else:
                y = np.linalg.norm(B, axis=1)
                label = self.labels["m"]
                color = self.color or "tab:red"
                title = self.title or r"Magnetic Field $\|B\|$ (ECI)"

            ax.plot(t, y, color=color, alpha=alpha)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [{self.units}]")
        ax.set_title(title, loc="left", pad=10)

        if self.log_y:
            ax.set_yscale("log")

        ax.grid(True, which="both")



class OrbitMagneticPlotCombined(Subplot):
    r"""
    Combined plot of all magnetic field components on a single axis.

    This class overlays the x, y, and z components of the geomagnetic field in
    the inertial frame onto one set of axes. It enables direct comparison of
    component magnitudes and trends over time.

    The plot focuses on user-controlled labeling, color selection, and axis
    scaling.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Physical units of the magnetic field values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z magnetic field components.
    :type colors:
        tuple[str, str, str]

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param labels:
        Labels used for the magnetic field components.
    :type labels:
        list[str] or None

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Magnetic Field (ECI)",
        units: str = "T",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
        labels: Sequence[str] | None = None,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y
        self.labels = labels or [r"$B_x$", r"$B_y$", r"$B_z$"]

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        first = runs[0]
        t = getattr(first, self.time)

        alpha = max(0.15, 1.0 / len(runs))

        for run in runs:
            if run.os_hist is None or len(run.os_hist) == 0:
                continue

            B = np.vstack([np.asarray(os.B) for os in run.os_hist])

            for i in range(3):
                ax.plot(
                    t,
                    B[:, i],
                    color=self.colors[i],
                    alpha=alpha,
                )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Magnetic Field [{self.units}]")
        ax.set_title(self.title, loc="left", pad=10)

        if self.log_y:
            ax.set_yscale("log")

        ax.grid(True, which="both")
