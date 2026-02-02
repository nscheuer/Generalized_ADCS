__all__ = ["SensorsPlot", "SensorsPlotSingle", "SensorsPlotCombined"]

import math
import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


def _normalize_sources(sources):
    if sources is None or len(sources) == 0:
        return ["real"]
    allowed = {"real", "clean"}
    bad = [s for s in sources if s not in allowed]
    if bad:
        raise ValueError(f"Invalid sources {bad}. Allowed: {sorted(allowed)}")
    # de-dup but keep order
    out = []
    for s in sources:
        if s not in out:
            out.append(s)
    return out


def _get_sensor_matrix(sim, which: str):
    if which == "real":
        hist = sim.sensor_hist
    elif which == "clean":
        hist = sim.clean_sensor_hist
    else:
        raise ValueError("which must be 'real' or 'clean'")

    if hist is None or len(hist) == 0:
        return None
    return np.vstack([np.asarray(v) for v in hist])


class SensorsPlot(Subplot):
    r"""
    Multi-panel visualization of sensor measurement histories.

    This class displays each sensor channel in its own subplot, arranged
    automatically in a grid. Multiple data sources, such as real and clean
    sensor readings, may be overlaid for comparison.

    The plot is configured primarily through user settings for sources,
    labeling, units, and axis scaling.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot group.
    :type title:
        str

    :param units:
        Physical units of the sensor readings.
    :type units:
        str

    :param labels:
        Optional list of labels for each sensor channel.
    :type labels:
        list[str] or None

    :param log_y:
        If True, the y-axes use logarithmic scaling.
    :type log_y:
        bool

    :param sources:
        List of sensor data sources to display. Supported values are real and clean.
    :type sources:
        list[str] or None

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Sensors",
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
        sources: list[str] | None = None,  # ["real", "clean"]
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.sources = _normalize_sources(sources)

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Find first run with usable data for layout
        first_mat = None
        first_run = None
        for run in runs:
            mats = {src: _get_sensor_matrix(run, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is not None:
                first_mat = first
                first_run = run
                break

        if first_mat is None or first_run is None:
            self._plot_no_data(ax)
            return

        n_sens = int(first_mat.shape[1])

        # layout grid
        ncols = int(math.ceil(math.sqrt(n_sens)))
        nrows = int(math.ceil(n_sens / ncols))
        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax.get_subplotspec())

        axes = []
        for i in range(n_sens):
            r, c = divmod(i, ncols)
            axes.append(ax.figure.add_subplot(gs[r, c]))

        # labels
        if self.labels is None:
            labels = [rf"$y_{{{i}}}$" for i in range(n_sens)]
        else:
            if len(self.labels) != n_sens:
                raise ValueError(f"labels length ({len(self.labels)}) must match sensor dimension ({n_sens})")
            labels = self.labels

        style = {"real": "-", "clean": "--"}
        alpha = max(0.15, 1.0 / len(runs))

        plotted_any = False

        # Plot all runs
        for run in runs:
            t0 = getattr(run, self.time, None)

            mats = {src: _get_sensor_matrix(run, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is None:
                continue

            N = first.shape[0]
            for m in mats.values():
                if m is not None:
                    N = min(N, m.shape[0])

            t = None
            if t0 is not None:
                t = np.asarray(t0)[:N]
            else:
                t = np.arange(N)

            for i, ax_i in enumerate(axes):
                for src in self.sources:
                    Y = mats.get(src, None)
                    if Y is None:
                        continue
                    ax_i.plot(
                        t,
                        Y[:N, i],
                        linestyle=style[src],
                        alpha=alpha,
                        label=None,
                    )
                    plotted_any = True

        if not plotted_any:
            self._plot_no_data(ax)
            return

        # Format each subplot + clean legend (sources only)
        for i, ax_i in enumerate(axes):
            ylabel = f"{labels[i]} [{self.units}]" if self.units else labels[i]
            ax_i.set_ylabel(ylabel)

            if self.log_y:
                ax_i.set_yscale("log")

            ax_i.grid(True, which="both")

            # legend only shows sources
            if len(self.sources) > 1:
                handles, labs = [], []
                for src in self.sources:
                    handles.append(ax_i.plot([], [], linestyle=style[src], color="k")[0])
                    labs.append(src)
                ax_i.legend(handles, labs)

        # hide unused cells
        for j in range(n_sens, nrows * ncols):
            r, c = divmod(j, ncols)
            ax_unused = ax.figure.add_subplot(gs[r, c])
            ax_unused.axis("off")

        for ax_i in axes[-ncols:]:
            ax_i.set_xlabel("Time [s]")

        axes[0].set_title(self.title, loc="left", pad=10)


    def _plot_no_data(self, ax):
        ax_text = ax.figure.add_subplot(ax.get_subplotspec())
        ax_text.axis("off")
        ax_text.set_title(self.title, loc="left", pad=10)
        ax_text.text(0.5, 0.5, "No sensor history available", ha="center", va="center")


class SensorsPlotSingle(Subplot):
    r"""
    Visualization of a single sensor channel over time.

    This class plots one selected sensor channel, optionally overlaying multiple
    data sources such as real and clean measurements. It is intended for focused
    inspection of a specific sensor output.

    User settings control the selected channel, labeling, color, and axis scaling.

    :param index:
        Index of the sensor channel to plot.
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
        Physical units of the sensor readings.
    :type units:
        str

    :param label:
        Label used for the sensor channel.
    :type label:
        str or None

    :param color:
        Color used for the plotted sensor signal.
    :type color:
        str or None

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param sources:
        List of sensor data sources to display. Supported values are real and clean.
    :type sources:
        list[str] or None

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
        sources: list[str] | None = None,  # ["real", "clean"]
    ):
        self.index = index
        self.time = time
        self.title = title
        self.units = units
        self.label = label
        self.color = color
        self.log_y = log_y
        self.sources = _normalize_sources(sources)

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        # Find first run with data to validate index
        first_mat = None
        for run in runs:
            mats = {src: _get_sensor_matrix(run, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is not None:
                first_mat = first
                break

        if first_mat is None:
            self._plot_no_data(ax)
            return

        n_sens = int(first_mat.shape[1])
        if self.index < 0 or self.index >= n_sens:
            raise ValueError(f"Sensor index {self.index} out of bounds for {n_sens} channels.")

        lbl = self.label or rf"$y_{{{self.index}}}$"
        title = self.title or f"Sensor Channel {self.index}"

        style = {"real": "-", "clean": "--"}
        alpha = max(0.15, 1.0 / len(runs))

        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            mats = {src: _get_sensor_matrix(run, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is None:
                continue

            N = first.shape[0]
            for m in mats.values():
                if m is not None:
                    N = min(N, m.shape[0])

            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            for src_i, src in enumerate(self.sources):
                Y = mats.get(src, None)
                if Y is None:
                    continue
                kw = {}
                if self.color and src_i == 0:
                    kw["color"] = self.color
                ax.plot(
                    t,
                    Y[:N, self.index],
                    linestyle=style[src],
                    alpha=alpha,
                    label=None,
                    **kw,
                )
                plotted_any = True

        if not plotted_any:
            self._plot_no_data(ax)
            return

        ylabel = f"{lbl} [{self.units}]" if self.units else lbl
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(title)

        if self.log_y:
            ax.set_yscale("log")

        # Clean legend: sources only
        handles, labs = [], []
        for src in self.sources:
            handles.append(ax.plot([], [], linestyle=style[src], color="k")[0])
            labs.append(src if len(self.sources) > 1 else lbl)
        ax.legend(handles, labs)

        ax.grid(True, which="both")


    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title or "Sensor", loc="left", pad=10)
        ax.text(0.5, 0.5, "No sensor history available", ha="center", va="center", transform=ax.transAxes)


class SensorsPlotCombined(Subplot):
    r"""
    Combined plot of all sensor channels on a single axis.

    This class overlays all sensor channels on one set of axes, optionally
    including multiple data sources. It provides a compact, high-level view
    of sensor activity and relative magnitudes.

    The plot emphasizes user-defined configuration of labels, colors, units,
    and axis scaling.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param units:
        Physical units of the sensor readings.
    :type units:
        str

    :param labels:
        Optional list of labels for each sensor channel.
    :type labels:
        list[str] or None

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param colors:
        Optional list of colors used cyclically for sensor channels.
    :type colors:
        list[str] or None

    :param sources:
        List of sensor data sources to display. Supported values are real and clean.
    :type sources:
        list[str] or None

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Sensors",
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
        colors: list[str] | None = None,
        sources: list[str] | None = None,  # ["real", "clean"]
    ):
        self.time = time
        self.title = title
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.colors = colors
        self.sources = _normalize_sources(sources)

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        # Find first run with data for dimensioning
        first_mat = None
        for run in runs:
            mats = {src: _get_sensor_matrix(run, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is not None:
                first_mat = first
                break

        if first_mat is None:
            self._plot_no_data(ax)
            return

        n_sens = int(first_mat.shape[1])

        if self.labels is None:
            labels = [rf"$y_{{{i}}}$" for i in range(n_sens)]
        else:
            if len(self.labels) != n_sens:
                raise ValueError(f"labels length ({len(self.labels)}) must match sensor dimension ({n_sens})")
            labels = self.labels

        style = {"real": "-", "clean": "--"}
        alpha = max(0.15, 1.0 / len(runs))

        plotted_any = False

        for run in runs:
            t0 = getattr(run, self.time, None)

            mats = {src: _get_sensor_matrix(run, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is None:
                continue

            N = first.shape[0]
            for m in mats.values():
                if m is not None:
                    N = min(N, m.shape[0])

            t = np.asarray(t0)[:N] if t0 is not None else np.arange(N)

            for i in range(n_sens):
                color_arg = {}
                if self.colors:
                    color_arg["color"] = self.colors[i % len(self.colors)]

                for src in self.sources:
                    Y = mats.get(src, None)
                    if Y is None:
                        continue
                    ax.plot(
                        t,
                        Y[:N, i],
                        linestyle=style[src],
                        alpha=alpha,
                        label=None,
                        **color_arg,
                    )
                    plotted_any = True

        if not plotted_any:
            self._plot_no_data(ax)
            return

        ylabel = f"Sensor Reading [{self.units}]" if self.units else "Sensor Reading"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(self.title)

        if self.log_y:
            ax.set_yscale("log")

        # Legend (compact): colors proxy for channels (optional), linestyle proxy for sources
        handles, labs = [], []

        if self.colors:
            kmax = min(n_sens, len(self.colors))
            for i in range(kmax):
                handles.append(ax.plot([], [], color=self.colors[i], linestyle="-")[0])
                labs.append(labels[i])
            if n_sens > kmax:
                handles.append(ax.plot([], [], color="k", linestyle="-")[0])
                labs.append("...")

        if len(self.sources) > 1:
            for src in self.sources:
                handles.append(ax.plot([], [], color="k", linestyle=style[src])[0])
                labs.append(src)

        if handles:
            ax.legend(handles, labs, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.legend()

        ax.grid(True, which="both", linestyle="--", alpha=0.7)


    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(0.5, 0.5, "No sensor history available", ha="center", va="center", transform=ax.transAxes)
