__all__ = ["OrbitPositionPlot"]

import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


def _normalize_orbit_sources(sources: list[str] | None) -> list[str]:
    if not sources:  # None or []a
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


def _get_R_series(sim, source: str) -> np.ndarray | None:
    if source == "real":
        hist = getattr(sim, "os_hist", None)
        get_R = lambda os: os.R
    elif source == "estimated":
        hist = getattr(sim, "est_os_hist", None)
        get_R = lambda os: os.os.R  # unwrap EstimatedOrbital_State
    else:
        raise ValueError(f"Unknown source: {source}")

    if hist is None or len(hist) == 0:
        return None

    rows = [
        np.asarray(get_R(os), dtype=float).reshape(3)
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


class OrbitPositionPlot(Subplot):
    r"""
    Multi-panel visualization of spacecraft position components in ECI.

    This class plots the Cartesian position components and the position magnitude
    of the spacecraft as functions of time, using orbital state histories from the
    simulation. Real and estimated positions can be shown together for comparison,
    arranged automatically in a fixed grid layout.

    The plot emphasizes user-controlled settings such as data sources, colors,
    units, and axis scaling.

    :param sources:
        List of orbit position sources to display. Supported values are real and
        estimated. If None, only the real position is shown.
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
        Physical units of the position components.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z position components.
    :type colors:
        tuple[str, str, str]

    :param mag_color:
        Color used for the position magnitude plot.
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
        title: str = "Orbit Position (ECI)",
        units: str = "km",
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
        ax_rx = ax.figure.add_subplot(gs[0, 0])
        ax_rmag = ax.figure.add_subplot(gs[0, 1])
        ax_ry = ax.figure.add_subplot(gs[1, 0])
        ax_rz = ax.figure.add_subplot(gs[1, 1])

        t0 = getattr(sim, self.time, None)

        labels = [r"$r_x$", r"$r_y$", r"$r_z$"]
        axes = [ax_rx, ax_ry, ax_rz]

        plotted_any = False

        for source in self.sources:
            R = _get_R_series(sim, source)
            if R is None:
                continue

            N = R.shape[0]
            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            style = _source_style_orbit(source)
            suf = _source_suffix_orbit(source)

            rmag = np.linalg.norm(R, axis=1)

            for i, ax_i in enumerate(axes):
                ax_i.plot(t, R[:, i], color=self.colors[i], label=labels[i] + suf, **style)
                ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
                if self.log_y:
                    ax_i.set_yscale("log")
                ax_i.grid(True, which="both")

            # magnitude on its own axis
            ax_rmag.plot(t, rmag, color=self.mag_color, label=r"$\|r\|$" + suf, **style)
            ax_rmag.set_ylabel(f"$\\|r\\|$ [{self.units}]")
            if self.log_y:
                ax_rmag.set_yscale("log")
            ax_rmag.grid(True, which="both")

            plotted_any = True

        if not plotted_any:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No orbit history available", ha="center", va="center")
            return

        for ax_i in axes:
            ax_i.legend()
        ax_rmag.legend()

        ax_ry.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax_rz.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax_rx.set_title(self.title, loc="left", pad=10)


class OrbitPositionPlotSingle(Subplot):
    r"""
    Single-component visualization of spacecraft position in ECI.

    This class plots one selected position component or the position magnitude
    as a function of time. It supports displaying real and estimated data together,
    making it suitable for focused analysis of a single spatial dimension.

    The user controls which component is shown, along with labeling, color, and
    axis scaling options.

    :param component:
        Position component to plot. Must be one of x, y, z, or m for magnitude.
    :type component:
        str

    :param sources:
        List of orbit position sources to display. Supported values are real and
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
        Physical units of the position values.
    :type units:
        str

    :param color:
        Color used for the plotted position signal.
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
        units: str = "km",
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
            "x": r"$r_x$",
            "y": r"$r_y$",
            "z": r"$r_z$",
            "m": r"$\|r\|$",
        }
        self.title = title

    def plot(self, ax, sim) -> None:
        t0 = getattr(sim, self.time, None)

        default_title = self.title or "Orbit Position (ECI)"

        plotted_any = False

        for source in self.sources:
            R = _get_R_series(sim, source)
            if R is None:
                continue

            N = R.shape[0]
            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            style = _source_style_orbit(source)
            suf = _source_suffix_orbit(source)

            if self.component == "x":
                y = R[:, 0]
                label = self.labels["x"]
                base_color = self.color or "tab:blue"
                title = self.title or "Orbit Position $r_x$ (ECI)"
            elif self.component == "y":
                y = R[:, 1]
                label = self.labels["y"]
                base_color = self.color or "tab:orange"
                title = self.title or "Orbit Position $r_y$ (ECI)"
            elif self.component == "z":
                y = R[:, 2]
                label = self.labels["z"]
                base_color = self.color or "tab:green"
                title = self.title or "Orbit Position $r_z$ (ECI)"
            else:
                y = np.linalg.norm(R, axis=1)
                label = self.labels["m"]
                base_color = self.color or "tab:red"
                title = self.title or "Orbit Position $\\|r\\|$ (ECI)"

            ax.plot(t, y, color=base_color, label=label + suf, **style)
            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(default_title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No orbit history available", ha="center", va="center")
            return

        ax.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax.set_ylabel(f"{self.labels[self.component]} [{self.units}]")
        ax.set_title(self.title or default_title)
        if self.log_y:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both")


class OrbitPositionPlotCombined(Subplot):
    r"""
    Combined plot of all spacecraft position components in ECI.

    This class overlays the x, y, and z position components on a single set of
    axes, allowing direct comparison of component magnitudes and trends over
    time. Real and estimated sources can be displayed simultaneously.

    The plot focuses on user-defined appearance options such as colors, labels,
    and axis scaling.

    :param sources:
        List of orbit position sources to display. Supported values are real and
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
        Physical units of the position values.
    :type units:
        str

    :param colors:
        Colors used for the x, y, and z position components.
    :type colors:
        tuple[str, str, str]

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param labels:
        Labels used for the position components.
    :type labels:
        list[str] or None

    """
    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real","estimated"]
        time: str = "time_s",
        title: str = "Orbit Position (ECI)",
        units: str = "km",
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
        self.labels = labels or [r"$r_x$", r"$r_y$", r"$r_z$"]

    def plot(self, ax, sim) -> None:
        t0 = getattr(sim, self.time, None)

        plotted_any = False

        for source in self.sources:
            R = _get_R_series(sim, source)
            if R is None:
                continue

            N = R.shape[0]
            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            style = _source_style_orbit(source)
            suf = _source_suffix_orbit(source)

            for i in range(3):
                ax.plot(
                    t,
                    R[:, i],
                    color=self.colors[i],
                    label=self.labels[i] + suf,
                    **style,
                )

            plotted_any = True

        if not plotted_any:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No orbit history available", ha="center", va="center")
            return

        ax.set_xlabel("Time [s]" if t0 is not None else "Sample")
        ax.set_ylabel(f"Position [{self.units}]")
        ax.set_title(self.title, loc="left", pad=10)

        if self.log_y:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True, which="both")
