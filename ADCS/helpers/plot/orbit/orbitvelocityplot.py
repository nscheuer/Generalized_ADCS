__all__ = ["OrbitVelocityPlot", "OrbitVelocityPlotSingle", "OrbitVelocityPlotCombined"]

import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot

def _normalize_orbit_sources(sources: list[str] | None) -> list[str]:
    if not sources:  # None or []
        return ["real"]
    allowed = {"real", "estimated"}
    out: list[str] = []
    for s in sources:
        s2 = str(s).strip().lower()
        if s2 not in allowed:
            raise ValueError(f"Invalid sources {s2!r}. Allowed: {sorted(allowed)}")
        if s2 not in out:
            out.append(s2)
    return out


def _get_V_series(sim, source: str) -> np.ndarray | None:
    """
    Return Nx3 ECI velocity history for a given source.

    - real      -> sim.os_hist[*].V
    - estimated -> sim.est_os_hist[*].os.V   (unwrap EstimatedOrbital_State)
    """
    if source == "real":
        hist = getattr(sim, "os_hist", None)
        get_V = lambda os: os.V
    elif source == "estimated":
        hist = getattr(sim, "est_os_hist", None)
        get_V = lambda os: os.os.V
    else:
        raise ValueError(f"Unknown source: {source}")

    if hist is None or len(hist) == 0:
        return None

    rows = [
        np.asarray(get_V(os), dtype=float).reshape(3)
        for os in hist
        if os is not None
    ]

    if not rows:
        return None

    return np.vstack(rows)


def _source_style_orbit(source: str) -> dict:
    return {"linestyle": "-" if source == "real" else "--"}


def _source_suffix_orbit(source: str) -> str:
    return " (real)" if source == "real" else " (est)"


# -----------------------------------------------------------------------------
# plots
# -----------------------------------------------------------------------------

class OrbitVelocityPlot(Subplot):
    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str = "Orbit Velocity (ECI)",
        units: str = "km/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        mag_color: str = "tab:red",
        log_y: bool = False,
    ):
        self.sources = _normalize_orbit_sources(sources)
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.mag_color = mag_color
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())
        ax_vx = ax.figure.add_subplot(gs[0, 0])
        ax_vmag = ax.figure.add_subplot(gs[0, 1])
        ax_vy = ax.figure.add_subplot(gs[1, 0])
        ax_vz = ax.figure.add_subplot(gs[1, 1])

        t0 = getattr(sim, self.time, None)

        labels = [r"$v_x$", r"$v_y$", r"$v_z$"]
        axes = [ax_vx, ax_vy, ax_vz]

        plotted_any = False

        for source in self.sources:
            V = _get_V_series(sim, source)
            if V is None:
                continue

            N = V.shape[0]
            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            style = _source_style_orbit(source)
            suf = _source_suffix_orbit(source)

            vmag = np.linalg.norm(V, axis=1)

            for i, ax_i in enumerate(axes):
                ax_i.plot(
                    t,
                    V[:, i],
                    color=self.colors[i],
                    label=labels[i] + suf,
                    **style,
                )
                ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
                if self.log_y:
                    ax_i.set_yscale("log")
                ax_i.grid(True, which="both")

            ax_vmag.plot(
                t,
                vmag,
                color=self.mag_color,
                label=r"$\|v\|$" + suf,
                **style,
            )
            ax_vmag.set_ylabel(f"$\\|v\\|$ [{self.units}]")
            if self.log_y:
                ax_vmag.set_yscale("log")
            ax_vmag.grid(True, which="both")

            plotted_any = True

        if not plotted_any:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No orbit velocity history available", ha="center", va="center")
            return

        for ax_i in axes:
            ax_i.legend()
        ax_vmag.legend()

        ax_vy.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax_vz.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax_vx.set_title(self.title, loc="left", pad=10)


class OrbitVelocityPlotSingle(Subplot):
    def __init__(
        self,
        *,
        component: str,
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str | None = None,
        units: str = "km/s",
        color: str | None = None,
        log_y: bool = False,
        labels: dict[str, str] | None = None,
    ):
        if component not in {"x", "y", "z", "m"}:
            raise ValueError("component must be one of 'x', 'y', 'z', or 'm'")
        self.component = component
        self.sources = _normalize_orbit_sources(sources)
        self.time = time
        self.units = units
        self.color = color
        self.log_y = log_y
        self.labels = labels or {
            "x": r"$v_x$",
            "y": r"$v_y$",
            "z": r"$v_z$",
            "m": r"$\|v\|$",
        }
        self.title = title

    def plot(self, ax, sim) -> None:
        t0 = getattr(sim, self.time, None)

        default_title = self.title or "Orbit Velocity (ECI)"

        plotted_any = False

        for source in self.sources:
            V = _get_V_series(sim, source)
            if V is None:
                continue

            N = V.shape[0]
            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            style = _source_style_orbit(source)
            suf = _source_suffix_orbit(source)

            if self.component == "x":
                y = V[:, 0]
                label = self.labels["x"]
                base_color = self.color or "tab:blue"
            elif self.component == "y":
                y = V[:, 1]
                label = self.labels["y"]
                base_color = self.color or "tab:orange"
            elif self.component == "z":
                y = V[:, 2]
                label = self.labels["z"]
                base_color = self.color or "tab:green"
            else:
                y = np.linalg.norm(V, axis=1)
                label = self.labels["m"]
                base_color = self.color or "tab:red"

            ax.plot(t, y, color=base_color, label=label + suf, **style)
            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(default_title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No orbit velocity history available", ha="center", va="center")
            return

        ax.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax.set_ylabel(f"{self.labels[self.component]} [{self.units}]")
        ax.set_title(self.title or default_title)
        if self.log_y:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both")


class OrbitVelocityPlotCombined(Subplot):
    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str = "Orbit Velocity (ECI)",
        units: str = "km/s",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
        labels: list[str] | None = None,
    ):
        self.sources = _normalize_orbit_sources(sources)
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y
        self.labels = labels or [r"$v_x$", r"$v_y$", r"$v_z$"]

    def plot(self, ax, sim) -> None:
        t0 = getattr(sim, self.time, None)

        plotted_any = False

        for source in self.sources:
            V = _get_V_series(sim, source)
            if V is None:
                continue

            N = V.shape[0]
            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            style = _source_style_orbit(source)
            suf = _source_suffix_orbit(source)

            for i in range(3):
                ax.plot(
                    t,
                    V[:, i],
                    color=self.colors[i],
                    label=self.labels[i] + suf,
                    **style,
                )

            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No orbit velocity history available", ha="center", va="center")
            return

        ax.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax.set_ylabel(f"Velocity [{self.units}]")
        ax.set_title(self.title, loc="left", pad=10)

        if self.log_y:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True, which="both")
