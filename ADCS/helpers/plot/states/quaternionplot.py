import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


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
        X = np.vstack(sim.state_hist)
        return _canonicalize_quaternion(X[:, 3:7])

    if source == "estimated":
        if getattr(sim, "est_state_hist", None) is None or len(sim.est_state_hist) == 0:
            return None
        Xh = np.vstack(sim.est_state_hist)
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
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())
        axes = [
            ax.figure.add_subplot(gs[0, 0]),  # q0
            ax.figure.add_subplot(gs[0, 1]),  # q1
            ax.figure.add_subplot(gs[1, 0]),  # q2
            ax.figure.add_subplot(gs[1, 1]),  # q3
        ]

        t = getattr(sim, self.time, None)
        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]

        plotted_any = False

        for source in self.sources:
            q = _get_q_series(sim, source)
            if q is None:
                continue

            N = q.shape[0]
            tt = np.asarray(t)[:N] if t is not None else np.arange(N)

            style = _source_style_q(source)
            suf = _source_suffix_q(source)

            for i, ax_i in enumerate(axes):
                ax_i.plot(tt, q[:, i], color=self.colors[i], label=labels[i] + suf, **style)
                ax_i.set_ylabel(f"{labels[i]} {self.units}".strip())
                ax_i.grid(True, which="both")

            plotted_any = True

        if not plotted_any:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No quaternion history available", ha="center", va="center")
            return

        for ax_i in axes:
            ax_i.legend()

        axes[2].set_xlabel("Time [s]" if t is not None else "Sample")
        axes[3].set_xlabel("Time [s]" if t is not None else "Sample")
        axes[0].set_title(self.title, loc="left", pad=10)


class QuaternionPlotSingle(Subplot):
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
        t = getattr(sim, self.time, None)

        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]
        label = labels[self.component]
        base_color = self.color or self.colors[self.component]
        default_title = self.title or f"{label} vs Time"

        plotted_any = False

        for source in self.sources:
            q = _get_q_series(sim, source)
            if q is None:
                continue

            N = q.shape[0]
            tt = np.asarray(t)[:N] if t is not None else np.arange(N)

            y = q[:, self.component]
            style = _source_style_q(source)
            suf = _source_suffix_q(source)

            ax.plot(tt, y, color=base_color, label=label + suf, **style)
            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(default_title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No quaternion history available", ha="center", va="center", transform=ax.transAxes)
            return

        ylabel = f"{label} [{self.units}]" if self.units else label
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]" if t is not None else "Sample")
        ax.set_title(default_title)
        ax.legend()
        ax.grid(True, which="both")


class QuaternionPlotCombined(Subplot):
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
        t = getattr(sim, self.time, None)
        labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$"]

        plotted_any = False

        for source in self.sources:
            q = _get_q_series(sim, source)
            if q is None:
                continue

            N = q.shape[0]
            tt = np.asarray(t)[:N] if t is not None else np.arange(N)

            style = _source_style_q(source)
            suf = _source_suffix_q(source)

            for i in range(4):
                ax.plot(tt, q[:, i], color=self.colors[i], label=labels[i] + suf, **style)

            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No quaternion history available", ha="center", va="center", transform=ax.transAxes)
            return

        ax.set_xlabel("Time [s]" if t is not None else "Sample")
        ax.set_ylabel(f"Quaternion {self.units}".strip())
        ax.set_title(self.title)
        ax.legend()
        ax.grid(True, which="both")
