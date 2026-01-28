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


class OrbitVelocityPlot(Subplot):
    r"""
    Multi-panel visualization of spacecraft velocity components in ECI.

    This class displays the Cartesian velocity components and the velocity
    magnitude as functions of time, arranged in a fixed grid layout. Real and
    estimated velocity histories can be shown together for comparison.

    User configuration focuses on selecting data sources, visual styling,
    units, and axis scaling.

    :param sources:
        List of orbit velocity sources to display. Supported values are real and
        estimated. If None, only the real velocity is shown.
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
        Physical units of the velocity components.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z velocity components.
    :type colors:
        tuple[str, str, str]

    :param mag_color:
        Color used for the velocity magnitude plot.
    :type mag_color:
        str

    :param log_y:
        If True, the y-axes use logarithmic scaling.
    :type log_y:
        bool

    """
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
    r"""
    Single-component visualization of spacecraft velocity in ECI.

    This class plots one selected velocity component or the velocity magnitude
    as a function of time. It supports overlaying real and estimated data and is
    suited for focused inspection of a single velocity dimension.

    The user controls the component selection, labeling, color, and axis scaling.

    :param component:
        Velocity component to plot. Must be one of x, y, z, or m for magnitude.
    :type component:
        str

    :param sources:
        List of orbit velocity sources to display. Supported values are real and
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
        Physical units of the velocity values.
    :type units:
        str

    :param color:
        Color used for the plotted velocity signal.
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
    r"""
    Combined plot of all spacecraft velocity components in ECI.

    This class overlays the x, y, and z velocity components on a single set of
    axes, enabling direct comparison of component magnitudes and trends over
    time. Real and estimated sources may be displayed simultaneously.

    The plot emphasizes user-defined appearance options such as colors, labels,
    and axis scaling.

    :param sources:
        List of orbit velocity sources to display. Supported values are real and
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
        Physical units of the velocity values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z velocity components.
    :type colors:
        tuple[str, str, str]

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param labels:
        Labels used for the velocity components.
    :type labels:
        list[str] or None

    """
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
