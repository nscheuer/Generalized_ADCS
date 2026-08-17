import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot
from ADCS.state import State


def _normalize_sources_q(sources: list[str] | None) -> list[str]:
    if not sources:  # None or []
        return ["real"]
    out = []
    for s in sources:
        s2 = str(s).strip().lower()
        if s2 not in {"real", "estimated"}:
            raise ValueError("sources must be a list containing any of: 'real', 'estimated'")
        if s2 not in out:
            out.append(s2)
    return out


def _get_q_series(sim, source: str) -> np.ndarray | None:
    """Return Nx4 quaternion history for a given source, or None if unavailable."""
    if source == "real":
        if sim.state_hist is None or len(sim.state_hist) == 0:
            return None
        X = State.stack(sim.state_hist)
        return _canonicalize_quaternion(X[:, 3:7])

    if source == "estimated":
        if getattr(sim, "est_state_hist", None) is None or len(sim.est_state_hist) == 0:
            return None
        Xh = State.stack(sim.est_state_hist)
        return _canonicalize_quaternion(Xh[:, 3:7])

    raise ValueError(f"Unknown source: {source}")


def _source_style_q(source: str) -> dict:
    return {"linestyle": "-" if source == "real" else "--"}


def _source_suffix_q(source: str) -> str:
    return " (real)" if source == "real" else " (est)"


def _canonicalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Enforce a unique quaternion sign convention:
    q0 >= 0 for every timestep.
    """
    q = np.asarray(q, dtype=float).copy()
    mask = q[:, 0] < 0
    q[mask] *= -1.0
    return q



class QuaternionPlot(Subplot):
    r"""
    Multi-panel visualization of quaternion components over time.

    This class displays the four quaternion components in a fixed 2x2 layout,
    allowing comparison between real and estimated attitude representations.
    The plot is intended to give a clear overview of quaternion behavior without
    requiring knowledge of the underlying attitude propagation.

    User configuration focuses on selecting data sources, colors, units, and
    the time reference used for the x-axis.

    :param sources:
        List of quaternion sources to display. Supported values are real and
        estimated. If None, only the real quaternion is shown.
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
        Optional units string appended to quaternion component labels.
    :type units:
        str

    :param colors:
        Colors used for the quaternion components q0, q1, q2, and q3.
    :type colors:
        tuple[str, str, str, str]

    """
    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str = "Quaternion Components",
        units: str = "",
        colors=("tab:blue", "tab:orange", "tab:green", "tab:red"),
    ):
        self.sources = _normalize_sources_q(sources)
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())
        axes = [
            ax.figure.add_subplot(gs[0, 0]),  # q0
            ax.figure.add_subplot(gs[0, 1]),  # q1
            ax.figure.add_subplot(gs[1, 0]),  # q2
            ax.figure.add_subplot(gs[1, 1]),  # q3
        ]

        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]

        alpha = max(0.15, 1.0 / len(runs))
        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            for source in self.sources:
                q = _get_q_series(run, source)
                if q is None:
                    continue

                N = q.shape[0]
                tt = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

                style = _source_style_q(source)

                for i, ax_i in enumerate(axes):
                    ax_i.plot(
                        tt,
                        q[:, i],
                        color=self.colors[i],
                        alpha=alpha,
                        label=None,
                        **style,
                    )
                    ax_i.set_ylabel(f"{labels[i]} {self.units}".strip())
                    ax_i.grid(True, which="both")

                plotted_any = True

        if not plotted_any:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No quaternion history available", ha="center", va="center")
            return

        # Clean legends: sources only (linestyle proxies)
        if len(self.sources) > 1:
            for ax_i in axes:
                handles, labs = [], []
                for src in self.sources:
                    handles.append(ax_i.plot([], [], color="k", **_source_style_q(src))[0])
                    labs.append(src)
                ax_i.legend(handles, labs)
        else:
            for ax_i in axes:
                leg = ax_i.legend()
                if leg is not None:
                    leg.set_visible(False)

        any_t = any(getattr(r, self.time, None) is not None for r in runs)
        xlabel = "Time [s]" if any_t else "Sample"
        axes[2].set_xlabel(xlabel)
        axes[3].set_xlabel(xlabel)

        axes[0].set_title(self.title, loc="left", pad=10)



class QuaternionPlotSingle(Subplot):
    r"""
    Visualization of a single quaternion component over time.

    This class plots one selected quaternion component, optionally overlaying
    real and estimated values. It is intended for focused inspection of a
    specific quaternion element.

    User settings control which component is displayed, visual styling,
    labeling, and the time reference.

    :param component:
        Quaternion component index to plot. Must be one of 0, 1, 2, or 3.
    :type component:
        int

    :param sources:
        List of quaternion sources to display. Supported values are real and
        estimated.
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
        Optional units string appended to the y-axis label.
    :type units:
        str

    :param color:
        Color used for the plotted quaternion component.
    :type color:
        str or None

    :param colors:
        Default colors for quaternion components q0, q1, q2, and q3.
    :type colors:
        tuple[str, str, str, str]

    """
    def __init__(
        self,
        *,
        component: int,  # 0,1,2,3
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        color: str | None = None,
        colors=("tab:blue", "tab:orange", "tab:green", "tab:red"),
    ):
        if component not in {0, 1, 2, 3}:
            raise ValueError("component must be an integer: 0, 1, 2, or 3")

        self.component = component
        self.sources = _normalize_sources_q(sources)
        self.time = time
        self.units = units
        self.color = color
        self.colors = colors
        self.title = title

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
        label = labels[self.component]
        base_color = self.color or self.colors[self.component]
        default_title = self.title or f"{label} vs Time"

        alpha = max(0.15, 1.0 / len(runs))
        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            for source in self.sources:
                q = _get_q_series(run, source)
                if q is None:
                    continue

                N = q.shape[0]
                tt = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

                y = q[:, self.component]
                style = _source_style_q(source)

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
            ax.set_title(default_title, loc="left", pad=10)
            ax.text(
                0.5,
                0.5,
                "No quaternion history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        ylabel = f"{label} [{self.units}]" if self.units else label
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]" if any(getattr(r, self.time, None) is not None for r in runs) else "Sample")
        ax.set_title(default_title)

        # Clean legend: sources only
        if len(self.sources) > 1:
            handles, labs = [], []
            for src in self.sources:
                handles.append(ax.plot([], [], color="k", **_source_style_q(src))[0])
                labs.append(src)
            ax.legend(handles, labs)
        else:
            leg = ax.legend()
            if leg is not None:
                leg.set_visible(False)

        ax.grid(True, which="both")



class QuaternionPlotCombined(Subplot):
    r"""
    Combined plot of all quaternion components on a single axis.

    This class overlays all four quaternion components on one set of axes,
    optionally including both real and estimated data. It provides a compact
    view suitable for quick comparison of relative component behavior.

    The plot emphasizes user-defined configuration of sources, colors, units,
    and the time reference.

    :param sources:
        List of quaternion sources to display. Supported values are real and
        estimated.
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
        Optional units string appended to the y-axis label.
    :type units:
        str

    :param colors:
        Colors used for the quaternion components q0, q1, q2, and q3.
    :type colors:
        tuple[str, str, str, str]

    """

    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str = "Quaternion Components",
        units: str = "",
        colors=("tab:blue", "tab:orange", "tab:green", "tab:red"),
    ):
        self.sources = _normalize_sources_q(sources)
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
        alpha = max(0.15, 1.0 / len(runs))

        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            for source in self.sources:
                q = _get_q_series(run, source)
                if q is None:
                    continue

                N = q.shape[0]
                tt = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

                style = _source_style_q(source)

                for i in range(4):
                    ax.plot(
                        tt,
                        q[:, i],
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
                "No quaternion history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        ax.set_xlabel("Time [s]" if any(getattr(r, self.time, None) is not None for r in runs) else "Sample")
        ax.set_ylabel(f"Quaternion {self.units}".strip())
        ax.set_title(self.title)

        # Clean legend: component color proxies + (optional) source linestyle proxies
        handles, labs = [], []

        # Component proxies (colors)
        for i in range(4):
            handles.append(ax.plot([], [], color=self.colors[i], linestyle="-")[0])
            labs.append(labels[i])

        # Source proxies (linestyle)
        if len(self.sources) > 1:
            for src in self.sources:
                handles.append(ax.plot([], [], color="k", **_source_style_q(src))[0])
                labs.append(src)

        ax.legend(handles, labs)
        ax.grid(True, which="both")
