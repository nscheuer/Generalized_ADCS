__all__ = ["ControlPlot", "ControlPlotSingle", "ControlPlotCombined"]

import math
import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


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
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Control Inputs",
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        if sim.control_hist is None or len(sim.control_hist) == 0:
            fig = ax.figure
            ax_text = fig.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(
                0.5, 0.5, "No control history available", ha="center", va="center"
            )
            return

        t = getattr(sim, self.time)
        U = np.vstack(sim.control_hist)
        n_ctrl = U.shape[1]

        u_max_list = _extract_u_max(sim)
        if u_max_list is not None and len(u_max_list) < n_ctrl:
            u_max_list = u_max_list + [np.nan] * (n_ctrl - len(u_max_list))
        if u_max_list is not None and len(u_max_list) > n_ctrl:
            u_max_list = u_max_list[:n_ctrl]

        ncols = int(math.ceil(math.sqrt(n_ctrl)))
        nrows = int(math.ceil(n_ctrl / ncols))

        gs = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=ax.get_subplotspec()
        )

        axes = []
        for i in range(n_ctrl):
            r, c = divmod(i, ncols)
            axes.append(ax.figure.add_subplot(gs[r, c]))

        if self.labels is None:
            labels = [rf"$u_{{{i}}}$" for i in range(n_ctrl)]
        else:
            if len(self.labels) != n_ctrl:
                raise ValueError(
                    f"labels length ({len(self.labels)}) must match number of controls ({n_ctrl})"
                )
            labels = self.labels

        for i, ax_i in enumerate(axes):
            (ln,) = ax_i.plot(t, U[:, i], label=labels[i])
            color = ln.get_color()

            if u_max_list is not None:
                umax = u_max_list[i]
                if np.isfinite(umax):
                    ax_i.axhline(umax, linestyle="--", linewidth=1.2, color=color)
                    ax_i.axhline(-umax, linestyle="--", linewidth=1.2, color=color)

            ax_i.set_ylabel(
                f"{labels[i]} {f'[{self.units}]' if self.units else ''}".strip()
            )
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.legend()
            ax_i.grid(True, which="both")

        for j in range(n_ctrl, nrows * ncols):
            r, c = divmod(j, ncols)
            ax_unused = ax.figure.add_subplot(gs[r, c])
            ax_unused.axis("off")

        for ax_i in axes[-ncols:]:
            ax_i.set_xlabel("Time [s]")

        axes[0].set_title(self.title, loc="left", pad=10)


class ControlPlotSingle(Subplot):
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
        if sim.control_hist is None or len(sim.control_hist) == 0:
            self._plot_no_data(ax)
            return

        t = getattr(sim, self.time)
        U = np.vstack(sim.control_hist)
        n_ctrl = U.shape[1]

        if self.index < 0 or self.index >= n_ctrl:
            raise ValueError(
                f"Control index {self.index} out of bounds for {n_ctrl} channels."
            )

        u_max_list = _extract_u_max(sim)
        umax = None
        if u_max_list is not None and self.index < len(u_max_list):
            umax = u_max_list[self.index]

        y = U[:, self.index]

        lbl = self.label if self.label else rf"$u_{{{self.index}}}$"
        tit = self.title if self.title else f"Control Channel {self.index}"
        c = self.color if self.color else "tab:blue"

        (ln,) = ax.plot(t, y, color=c, label=lbl)
        color = ln.get_color()

        if umax is not None and np.isfinite(umax):
            ax.axhline(umax, linestyle="--", linewidth=1.2, color=color)
            ax.axhline(-umax, linestyle="--", linewidth=1.2, color=color)

        ylabel = f"{lbl} [{self.units}]" if self.units else lbl
        ax.set_ylabel(ylabel)
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
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Control Inputs",
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
        colors: list[str] | None = None,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.colors = colors

    def plot(self, ax, sim) -> None:
        if sim.control_hist is None or len(sim.control_hist) == 0:
            self._plot_no_data(ax)
            return

        t = getattr(sim, self.time)
        U = np.vstack(sim.control_hist)
        n_ctrl = U.shape[1]

        u_max_list = _extract_u_max(sim)
        if u_max_list is not None and len(u_max_list) < n_ctrl:
            u_max_list = u_max_list + [np.nan] * (n_ctrl - len(u_max_list))
        if u_max_list is not None and len(u_max_list) > n_ctrl:
            u_max_list = u_max_list[:n_ctrl]

        if self.labels is None:
            labels = [rf"$u_{{{i}}}$" for i in range(n_ctrl)]
        else:
            if len(self.labels) != n_ctrl:
                raise ValueError(
                    f"Label count ({len(self.labels)}) does not match control channels ({n_ctrl})"
                )
            labels = self.labels

        for i in range(n_ctrl):
            color_arg = {}
            if self.colors:
                color_arg["color"] = self.colors[i % len(self.colors)]

            (ln,) = ax.plot(t, U[:, i], label=labels[i], **color_arg)
            color = ln.get_color()

            if u_max_list is not None:
                umax = u_max_list[i]
                if np.isfinite(umax):
                    ax.axhline(umax, linestyle="--", linewidth=1.2, color=color)
                    ax.axhline(-umax, linestyle="--", linewidth=1.2, color=color)

        ylabel = f"Control Input [{self.units}]" if self.units else "Control Input"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(self.title)

        if self.log_y:
            ax.set_yscale("log")

        if n_ctrl > 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.legend()

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
