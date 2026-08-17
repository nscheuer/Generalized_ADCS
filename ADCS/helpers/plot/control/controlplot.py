__all__ = ["ControlPlot", "ControlPlotSingle", "ControlPlotCombined"]

import math
import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot
from typing import Sequence


def _extract_u_max(sim) -> list[float] | None:
    sat = getattr(sim, "satellite", None)
    acts = getattr(sat, "actuators", None) if sat is not None else None
    if acts is None:
        return None
    u_max = []
    for a in acts:
        if hasattr(a, "u_max"):
            u_max.append(float(a.u_max))
        else:
            u_max.append(np.nan)
    return u_max


class ControlPlot(Subplot):
    r"""
    Multi-panel visualization of all control input channels.

    This class provides a compact overview of each control channel stored in the
    simulation control history, arranged automatically in a grid of subplots.
    Each channel is plotted separately with optional actuator saturation limits.
    The class is intended for quick inspection of control activity rather than
    detailed tuning.

    The plot reads control inputs from the simulation and displays them as functions
    of time. If actuator limits are available from the simulation satellite model,
    symmetric bounds are overlaid for reference.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the subplot group.
    :type title:
        str

    :param units:
        Physical units of the control inputs, displayed on the y-axis labels.
    :type units:
        str

    :param labels:
        Optional list of labels for each control channel.
    :type labels:
        list[str] or None

    :param log_y:
        If True, the y-axis of each subplot is displayed on a logarithmic scale.
    :type log_y:
        bool

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Control Inputs",
        units: str = "",
        labels: Sequence[str] | None = None,
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        runs = sim.runs if hasattr(sim, "runs") else [sim]

        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        if runs[0].control_hist is None or len(runs[0].control_hist) == 0:
            fig = ax.figure
            ax_text = fig.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No control history available", ha="center", va="center")
            return

        t0 = getattr(runs[0], self.time)
        U0 = np.vstack(runs[0].control_hist)
        n_ctrl = U0.shape[1]

        u_max_list = _extract_u_max(runs[0])
        if u_max_list is not None:
            u_max_list = (u_max_list + [np.nan] * n_ctrl)[:n_ctrl]

        ncols = int(math.ceil(math.sqrt(n_ctrl)))
        nrows = int(math.ceil(n_ctrl / ncols))

        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax.get_subplotspec())
        axes = []
        for i in range(n_ctrl):
            r, c = divmod(i, ncols)
            axes.append(ax.figure.add_subplot(gs[r, c]))

        labels = (
            [rf"$u_{{{i}}}$" for i in range(n_ctrl)]
            if self.labels is None
            else self.labels
        )

        for run in runs:
            t = getattr(run, self.time)
            U = np.vstack(run.control_hist)
            for i, ax_i in enumerate(axes):
                ax_i.plot(t, U[:, i], alpha=0.25)

        for i, ax_i in enumerate(axes):
            if u_max_list is not None and np.isfinite(u_max_list[i]):
                ax_i.axhline(u_max_list[i], linestyle="--", linewidth=1.2)
                ax_i.axhline(-u_max_list[i], linestyle="--", linewidth=1.2)

            ax_i.set_ylabel(
                f"{labels[i]} {f'[{self.units}]' if self.units else ''}".strip()
            )
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.grid(True, which="both")

        for j in range(n_ctrl, nrows * ncols):
            r, c = divmod(j, ncols)
            ax_unused = ax.figure.add_subplot(gs[r, c])
            ax_unused.axis("off")

        for ax_i in axes[-ncols:]:
            ax_i.set_xlabel("Time [s]")

        axes[0].set_title(self.title, loc="left", pad=10)



class ControlPlotSingle(Subplot):
    r"""
    Visualization of a single control input channel.

    This class plots one selected control channel versus time, allowing focused
    inspection of a specific actuator or control dimension. Optional actuator
    saturation limits are shown when available.

    The plot is suitable for detailed analysis or comparison across multiple
    figures when individual control channels are of interest.

    :param index:
        Index of the control channel to plot.
    :type index:
        int

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title of the plot. If None, a default title is used.
    :type title:
        str or None

    :param units:
        Physical units of the control input.
    :type units:
        str

    :param label:
        Label for the plotted control signal.
    :type label:
        str or None

    :param color:
        Matplotlib color specification for the control signal.
    :type color:
        str or None

    :param log_y:
        If True, the y-axis is displayed on a logarithmic scale.
    :type log_y:
        bool

    """
    def __init__(
        self,
        index: int,
        *,
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        label: str | None = None,
        color: str | None = None,
        log_y: bool = False,
    ):
        self.index = index
        self.time = time
        self.title = title
        self.units = units
        self.label = label
        self.color = color
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        run = sim.runs[0] if hasattr(sim, "runs") else sim

        if run.control_hist is None or len(run.control_hist) == 0:
            self._plot_no_data(ax)
            return

        t = getattr(run, self.time)
        U = np.vstack(run.control_hist)
        n_ctrl = U.shape[1]

        if self.index < 0 or self.index >= n_ctrl:
            raise ValueError(
                f"Control index {self.index} out of bounds for {n_ctrl} channels."
            )

        u_max_list = _extract_u_max(run)
        umax = u_max_list[self.index] if u_max_list and self.index < len(u_max_list) else None

        y = U[:, self.index]

        lbl = self.label if self.label else rf"$u_{{{self.index}}}$"
        tit = self.title if self.title else f"Control Channel {self.index}"
        c = self.color if self.color else "tab:blue"

        (ln,) = ax.plot(t, y, color=c, label=lbl)
        color = ln.get_color()

        if umax is not None and np.isfinite(umax):
            ax.axhline(umax, linestyle="--", linewidth=1.2, color=color)
            ax.axhline(-umax, linestyle="--", linewidth=1.2, color=color)

        ax.set_ylabel(f"{lbl} [{self.units}]" if self.units else lbl)
        ax.set_xlabel("Time [s]")
        ax.set_title(tit)
        ax.legend()
        ax.grid(True, which="both")

        if self.log_y:
            ax.set_yscale("log")


    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title or "Control Input", loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "No control history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


class ControlPlotCombined(Subplot):
    r"""
    Combined visualization of all control input channels on a single axis.

    This class plots all control channels together in one figure, enabling direct
    comparison of magnitudes and temporal behavior. Each channel is assigned a
    distinct label and optional color. Actuator saturation limits are shown when
    available.

    This representation is useful for high-level assessment of control effort and
    relative channel usage.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title of the plot.
    :type title:
        str

    :param units:
        Physical units of the control inputs.
    :type units:
        str

    :param labels:
        Optional list of labels for each control channel.
    :type labels:
        list[str] or None

    :param log_y:
        If True, the y-axis is displayed on a logarithmic scale.
    :type log_y:
        bool

    :param colors:
        Optional list of colors used cyclically for control channels.
    :type colors:
        list[str] or None

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Control Inputs",
        units: str = "",
        labels: Sequence[str] | None = None,
        log_y: bool = False,
        colors: Sequence[str] | None = None,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.colors = colors

    def plot(self, ax, sim) -> None:
        runs = sim.runs if hasattr(sim, "runs") else [sim]

        if runs[0].control_hist is None or len(runs[0].control_hist) == 0:
            self._plot_no_data(ax)
            return

        U0 = np.vstack(runs[0].control_hist)
        n_ctrl = U0.shape[1]

        u_max_list = _extract_u_max(runs[0])
        if u_max_list is not None:
            u_max_list = (u_max_list + [np.nan] * n_ctrl)[:n_ctrl]

        labels = (
            [rf"$u_{{{i}}}$" for i in range(n_ctrl)]
            if self.labels is None
            else self.labels
        )

        for run in runs:
            t = getattr(run, self.time)
            U = np.vstack(run.control_hist)

            for i in range(n_ctrl):
                color_arg = {}
                if self.colors:
                    color_arg["color"] = self.colors[i % len(self.colors)]
                ax.plot(t, U[:, i], alpha=0.25, **color_arg)

        for i in range(n_ctrl):
            if u_max_list is not None and np.isfinite(u_max_list[i]):
                ax.axhline(u_max_list[i], linestyle="--", linewidth=1.2)
                ax.axhline(-u_max_list[i], linestyle="--", linewidth=1.2)

        ax.set_ylabel(f"Control Input [{self.units}]" if self.units else "Control Input")
        ax.set_xlabel("Time [s]")
        ax.set_title(self.title)

        if self.log_y:
            ax.set_yscale("log")

        ax.grid(True, which="both")

    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "No control history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
