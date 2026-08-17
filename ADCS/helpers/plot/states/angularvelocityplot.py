__all__ = ["AngularVelocityPlot"]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ..subplot import Subplot
from ADCS.state import State
from typing import Sequence


def _normalize_sources(sources: list[str] | None) -> list[str]:
    if not sources:  # None or []
        return ["real"]
    out = []
    for s in sources:
        s2 = str(s).strip().lower()
        if s2 not in {"real", "reference", "estimated"}:
            raise ValueError("sources must be a list containing any of: 'real', 'reference', 'estimated'")
        if s2 not in out:
            out.append(s2)
    return out


def _get_w_series(sim, source: str) -> np.ndarray | None:
    """Return Nx3 angular velocity history for a given source, or None if unavailable."""
    if source == "real":
        if sim.state_hist is None or len(sim.state_hist) == 0:
            return None
        X = State.stack(sim.state_hist)
        return X[:, 0:3]

    if source == "estimated":
        if getattr(sim, "est_state_hist", None) is None or len(sim.est_state_hist) == 0:
            return None
        Xh = State.stack(sim.est_state_hist)
        return Xh[:, 0:3]

    if source == "reference":
        if getattr(sim, "w_target_hist", None) is None or len(sim.w_target_hist) == 0:
            return None
        Wt = np.vstack(sim.w_target_hist)
        return Wt[:, 0:3]

    raise ValueError(f"Unknown source: {source}")


def _source_style(source: str) -> dict:
    # Keep same component colors; vary linestyle by source
    if source == "real":
        return {"linestyle": "-"}
    if source == "estimated":
        return {"linestyle": "--"}
    if source == "reference":
        return {"linestyle": ":"}
    return {"linestyle": "-"}


def _source_suffix(source: str) -> str:
    return {"real": " (real)", "estimated": " (est)", "reference": " (ref)"}[source]


class AngularVelocityPlot(Subplot):
    r"""
    Multi-panel visualization of spacecraft angular velocity components.

    This class displays the body-frame angular velocity components and their
    magnitude as functions of time, arranged in a fixed grid layout. Multiple
    sources such as real, estimated, and reference angular rates can be overlaid
    for comparison.

    The plot is configured through user settings that control which sources are
    shown, visual styling, units, and axis scaling.

    :param sources:
        List of angular velocity sources to display. Supported values are real,
        estimated, and reference. If None, only the real angular velocity is shown.
    :type sources:
        list[str] or None

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot group.
    :type title:
        str

    :param units:
        Physical units of the angular velocity values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z angular velocity components.
    :type colors:
        tuple[str, str, str]

    :param log_y:
        If True, the y-axes use logarithmic scaling.
    :type log_y:
        bool

    """
    def __init__(
        self,
        *,
        sources: Sequence[str] | None = None,  # ["real","reference","estimated"]
        time: str = "time_s",
        title: str = "Angular Rates in Body Frame",
        units: str = "rad/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
    ):
        self.sources = _normalize_sources(sources)
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())
        sub_axes_wx = ax.figure.add_subplot(gs[0, 0])
        sub_axes_mag = ax.figure.add_subplot(gs[0, 1])
        sub_axes_wy = ax.figure.add_subplot(gs[1, 0])
        sub_axes_wz = ax.figure.add_subplot(gs[1, 1])

        labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
        axes = [sub_axes_wx, sub_axes_wy, sub_axes_wz]

        alpha = max(0.15, 1.0 / len(runs))
        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            for source in self.sources:
                w = _get_w_series(run, source)
                if w is None:
                    continue

                N = w.shape[0]
                tt = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

                style = _source_style(source)

                w_mag = np.linalg.norm(w, axis=1)

                for i, ax_i in enumerate(axes):
                    ax_i.plot(
                        tt,
                        w[:, i],
                        color=self.colors[i],
                        alpha=alpha,
                        label=None,
                        **style,
                    )
                    plotted_any = True

                sub_axes_mag.plot(
                    tt,
                    w_mag,
                    color="tab:red",
                    alpha=alpha,
                    label=None,
                    **style,
                )

        if not plotted_any:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No angular rate history available", ha="center", va="center")
            return

        # Formatting
        for i, ax_i in enumerate(axes):
            ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.grid(True, which="both")

            # clean legend: sources only (linestyle proxy)
            if len(self.sources) > 1:
                handles, labs = [], []
                for src in self.sources:
                    handles.append(ax_i.plot([], [], color="k", **_source_style(src))[0])
                    labs.append(src)
                ax_i.legend(handles, labs)
            else:
                ax_i.legend().set_visible(False)

        sub_axes_mag.set_ylabel(rf"$\|\omega\|$ [{self.units}]")
        if self.log_y:
            sub_axes_mag.set_yscale("log")
        sub_axes_mag.grid(True, which="both")
        if len(self.sources) > 1:
            handles, labs = [], []
            for src in self.sources:
                handles.append(sub_axes_mag.plot([], [], color="k", **_source_style(src))[0])
                labs.append(src)
            sub_axes_mag.legend(handles, labs)
        else:
            sub_axes_mag.legend().set_visible(False)

        # X labels (pick from first run that has time)
        any_t = None
        for run in runs:
            if getattr(run, self.time, None) is not None:
                any_t = True
                break

        xlabel = "Time [s]" if any_t else "Sample"
        sub_axes_wy.set_xlabel(xlabel)
        sub_axes_wz.set_xlabel(xlabel)
        sub_axes_mag.set_xlabel(xlabel)

        sub_axes_wx.set_title(self.title, loc="left", pad=10)



class AngularVelocityPlotSingle(Subplot):
    r"""
    Visualization of a single angular velocity component or magnitude.

    This class plots one selected angular velocity component or the angular
    velocity magnitude as a function of time. Multiple sources may be overlaid
    to compare real, estimated, and reference rates.

    User configuration focuses on component selection, displayed sources,
    labeling, units, and axis scaling.

    :param component:
        Angular velocity component to plot. Must be one of x, y, z, or m for
        magnitude.
    :type component:
        str

    :param sources:
        List of angular velocity sources to display. Supported values are real,
        estimated, and reference.
    :type sources:
        list[str] or None

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title of the plot. If None, a default title is used.
    :type title:
        str or None

    :param units:
        Physical units of the angular velocity values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z angular velocity components.
    :type colors:
        tuple[str, str, str]

    :param mag_color:
        Color used for the angular velocity magnitude.
    :type mag_color:
        str

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    """
    def __init__(
        self,
        *,
        component: str,  # 'x', 'y', 'z', or 'm'
        sources: Sequence[str] | None = None,
        time: str = "time_s",
        title: str | None = None,
        units: str = "rad/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        mag_color: str = "tab:red",
        log_y: bool = False,
    ):
        if component not in {"x", "y", "z", "m"}:
            raise ValueError("component must be one of 'x', 'y', 'z', or 'm'")

        self.component = component
        self.sources = _normalize_sources(sources)
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

        comp_idx = {"x": 0, "y": 1, "z": 2}.get(self.component, None)
        base_label = {
            "x": r"$\omega_x$",
            "y": r"$\omega_y$",
            "z": r"$\omega_z$",
            "m": r"$\|\omega\|$",
        }[self.component]

        if self.component in {"x", "y", "z"}:
            base_color = self.colors[comp_idx]
            default_title = f"{base_label} vs Time"
        else:
            base_color = self.mag_color
            default_title = r"$\|\omega\|$ vs Time"

        alpha = max(0.15, 1.0 / len(runs))
        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            for source in self.sources:
                w = _get_w_series(run, source)
                if w is None:
                    continue

                N = w.shape[0]
                tt = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

                if self.component in {"x", "y", "z"}:
                    y = w[:, comp_idx]
                else:
                    y = np.linalg.norm(w, axis=1)

                style = _source_style(source)

                ax.plot(
                    tt,
                    y,
                    color=base_color,
                    alpha=alpha,
                    label=None,
                    **style,
                )
                plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title or default_title, loc="left", pad=10)
            ax.text(
                0.5,
                0.5,
                "No angular rate history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        ax.set_ylabel(f"{base_label} [{self.units}]")
        ax.set_xlabel("Time [s]" if any(getattr(r, self.time, None) is not None for r in runs) else "Sample")
        ax.set_title(self.title or default_title)
        if self.log_y:
            ax.set_yscale("log")

        # clean legend: sources only
        handles, labs = [], []
        for src in self.sources:
            handles.append(ax.plot([], [], color="k", **_source_style(src))[0])
            labs.append(src)
        ax.legend(handles, labs) if len(self.sources) > 1 else ax.legend().set_visible(False)

        ax.grid(True, which="both")



class AngularVelocityPlotCombined(Subplot):
    r"""
    Combined plot of all angular velocity components on a single axis.

    This class overlays the x, y, and z body-frame angular velocity components
    on one set of axes, optionally including multiple data sources. It provides
    a compact, high-level view of rotational motion and relative component
    magnitudes.

    The plot emphasizes user-defined configuration of sources, colors, units,
    and axis scaling.

    :param sources:
        List of angular velocity sources to display. Supported values are real,
        estimated, and reference.
    :type sources:
        list[str] or None

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Physical units of the angular velocity values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z angular velocity components.
    :type colors:
        tuple[str, str, str]

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    """
    def __init__(
        self,
        *,
        sources: Sequence[str] | None = None,
        time: str = "time_s",
        title: str = "Angular Rates (Body Frame)",
        units: str = "rad/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
    ):
        self.sources = _normalize_sources(sources)
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        t_label = "Time [s]" if any(getattr(r, self.time, None) is not None for r in runs) else "Sample"
        labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

        alpha = max(0.15, 1.0 / len(runs))
        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            for source in self.sources:
                w = _get_w_series(run, source)
                if w is None:
                    continue

                N = w.shape[0]
                tt = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

                style = _source_style(source)

                for i in range(3):
                    ax.plot(
                        tt,
                        w[:, i],
                        color=self.colors[i],
                        alpha=alpha,
                        label=None,
                        **style,
                    )

                plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(
                0.5,
                0.5,
                "No angular rate history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        ax.set_ylabel(f"Angular Velocity [{self.units}]")
        ax.set_xlabel(t_label)
        ax.set_title(self.title)
        if self.log_y:
            ax.set_yscale("log")

        # clean legend: show component colors + source linestyles
        handles, labs = [], []

        # component color proxies
        for i in range(3):
            handles.append(ax.plot([], [], color=self.colors[i], linestyle="-")[0])
            labs.append(labels[i])

        # source linestyle proxies (if multiple)
        if len(self.sources) > 1:
            for src in self.sources:
                handles.append(ax.plot([], [], color="k", **_source_style(src))[0])
                labs.append(src)

        ax.legend(handles, labs, loc="upper right")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
