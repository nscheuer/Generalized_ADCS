__all__ = ["AngularVelocityPlot"]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ..subplot import Subplot


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
        X = np.vstack(sim.state_hist)
        return X[:, 0:3]

    if source == "estimated":
        if getattr(sim, "est_state_hist", None) is None or len(sim.est_state_hist) == 0:
            return None
        Xh = np.vstack(sim.est_state_hist)
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
    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real","reference","estimated"]
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
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())
        sub_axes_wx = ax.figure.add_subplot(gs[0, 0])
        sub_axes_mag = ax.figure.add_subplot(gs[0, 1])
        sub_axes_wy = ax.figure.add_subplot(gs[1, 0])
        sub_axes_wz = ax.figure.add_subplot(gs[1, 1])

        t = getattr(sim, self.time, None)

        labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]
        axes = [sub_axes_wx, sub_axes_wy, sub_axes_wz]

        plotted_any = False

        for source in self.sources:
            w = _get_w_series(sim, source)
            if w is None:
                continue

            N = w.shape[0]
            tt = np.asarray(t)[:N] if t is not None else np.arange(N)

            style = _source_style(source)
            suf = _source_suffix(source)

            w_mag = np.linalg.norm(w, axis=1)

            for i, ax_i in enumerate(axes):
                ax_i.plot(
                    tt,
                    w[:, i],
                    color=self.colors[i],
                    label=labels[i] + suf,
                    **style,
                )
                ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
                if self.log_y:
                    ax_i.set_yscale("log")
                ax_i.grid(True, which="both")
                plotted_any = True

            sub_axes_mag.plot(
                tt,
                w_mag,
                color="tab:red",
                label=r"$\|\omega\|$" + suf,
                **style,
            )
            sub_axes_mag.set_ylabel(rf"$\|\omega\|$ [{self.units}]")
            if self.log_y:
                sub_axes_mag.set_yscale("log")
            sub_axes_mag.grid(True, which="both")

        if not plotted_any:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No angular rate history available", ha="center", va="center")
            return

        for ax_i in axes:
            ax_i.legend()
        sub_axes_mag.legend()

        sub_axes_wy.set_xlabel("Time [s]" if t is not None else "Sample")
        sub_axes_wz.set_xlabel("Time [s]" if t is not None else "Sample")
        sub_axes_mag.set_xlabel("Time [s]" if t is not None else "Sample")

        sub_axes_wx.set_title(self.title, loc="left", pad=10)


class AngularVelocityPlotSingle(Subplot):
    def __init__(
        self,
        *,
        component: str,  # 'x', 'y', 'z', or 'm'
        sources: list[str] | None = None,
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
        t = getattr(sim, self.time, None)

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

        plotted_any = False

        for source in self.sources:
            w = _get_w_series(sim, source)
            if w is None:
                continue

            N = w.shape[0]
            tt = np.asarray(t)[:N] if t is not None else np.arange(N)

            if self.component in {"x", "y", "z"}:
                y = w[:, comp_idx]
            else:
                y = np.linalg.norm(w, axis=1)

            style = _source_style(source)
            suf = _source_suffix(source)

            ax.plot(tt, y, color=base_color, label=base_label + suf, **style)
            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title or default_title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No angular rate history available", ha="center", va="center", transform=ax.transAxes)
            return

        ax.set_ylabel(f"{base_label} [{self.units}]")
        ax.set_xlabel("Time [s]" if t is not None else "Sample")
        ax.set_title(self.title or default_title)
        if self.log_y:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both")


class AngularVelocityPlotCombined(Subplot):
    def __init__(
        self,
        *,
        sources: list[str] | None = None,
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
        t = getattr(sim, self.time, None)
        labels = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"]

        plotted_any = False

        for source in self.sources:
            w = _get_w_series(sim, source)
            if w is None:
                continue

            N = w.shape[0]
            tt = np.asarray(t)[:N] if t is not None else np.arange(N)

            style = _source_style(source)
            suf = _source_suffix(source)

            for i in range(3):
                ax.plot(tt, w[:, i], color=self.colors[i], label=labels[i] + suf, **style)

            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No angular rate history available", ha="center", va="center", transform=ax.transAxes)
            return

        ax.set_ylabel(f"Angular Velocity [{self.units}]")
        ax.set_xlabel("Time [s]" if t is not None else "Sample")
        ax.set_title(self.title)
        if self.log_y:
            ax.set_yscale("log")

        ax.legend(loc="upper right")
        ax.grid(True, which="both", linestyle="--", alpha=0.7)
