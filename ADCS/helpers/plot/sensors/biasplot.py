__all__ = ["BiasPlot", "BiasPlotSingle", "BiasPlotCombined"]

import math
import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


def _normalize_bias_sources(sources):
    if sources is None or len(sources) == 0:
        return ["real"]
    allowed = {"real", "estimated"}
    bad = [s for s in sources if s not in allowed]
    if bad:
        raise ValueError(f"Invalid sources {bad}. Allowed: {sorted(allowed)}")
    out = []
    for s in sources:
        if s not in out:
            out.append(s)
    return out


def _flatten_object_bias_snapshot(snapshot) -> np.ndarray:
    """
    snapshot: one time-step of bias history.
      - numeric array-like (any shape), OR
      - object array/list of per-device arrays (possibly different lengths)
    Returns:
      1D float array of concatenated biases for that timestep.
    """
    if snapshot is None:
        return None

    s = np.asarray(snapshot)

    # Numeric case: flatten
    if s.dtype != object:
        return np.asarray(s, dtype=float).reshape(-1)

    # Object case: concatenate per-device arrays
    parts = []
    for item in s.ravel():
        if item is None:
            continue
        arr = np.asarray(item, dtype=float).reshape(-1)
        parts.append(arr)

    if len(parts) == 0:
        return np.array([], dtype=float)

    return np.concatenate(parts, axis=0)


def _get_bias_matrix(sim, kind: str, which: str):
    """
    kind: 'sensor' or 'actuator'
    which: 'real' or 'estimated'
    Returns: (N, D) float matrix or None
    """
    if kind == "sensor":
        hist = sim.sensor_bias if which == "real" else sim.est_sensor_bias
    elif kind == "actuator":
        hist = sim.actuator_bias if which == "real" else sim.est_actuator_bias
    else:
        raise ValueError("kind must be 'sensor' or 'actuator'")

    if hist is None or len(hist) == 0:
        return None

    rows = []
    D0 = None
    for k, snap in enumerate(hist):
        v = _flatten_object_bias_snapshot(snap)
        if v is None:
            continue

        if D0 is None:
            D0 = v.size
        elif v.size != D0:
            raise ValueError(
                f"Inconsistent bias vector length at k={k}: got {v.size}, expected {D0}. "
                "This usually means the number of sensors/actuators or bias dimensions changed."
            )

        rows.append(v)

    if len(rows) == 0:
        return None

    return np.vstack(rows).astype(float)


def _get_time_axis(sim, time_attr: str, N: int) -> np.ndarray:
    """
    Returns a safe length-N x-axis.
    - If sim.<time_attr> exists and can be cast to float, use it (trim to N).
    - Otherwise, fall back to np.arange(N).
    """
    t = getattr(sim, time_attr, None)
    if t is None:
        return np.arange(N)

    t = np.asarray(t)
    if t.size == 0:
        return np.arange(N)

    # If dtype=object or otherwise not numeric, try casting. If it fails, fall back.
    try:
        t = t.astype(float)
    except Exception:
        return np.arange(N)

    if t.size < N:
        # if someone recorded fewer time points than biases
        return np.arange(N)

    return t[:N]


class BiasPlot(Subplot):
    r"""
    Multi-panel visualization of sensor or actuator bias histories.

    This class displays each bias component as a separate subplot, arranged
    automatically in a grid. Real and estimated bias histories can be shown
    simultaneously for comparison, depending on the selected sources.

    The primary user controls are the bias type, displayed sources, labeling,
    units, and axis scaling.

    :param kind:
        Type of bias to plot. Must be either sensor or actuator.
    :type kind:
        str

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot group.
    :type title:
        str or None

    :param units:
        Physical units of the bias values.
    :type units:
        str

    :param labels:
        Optional list of labels for each bias component.
    :type labels:
        list[str] or None

    :param log_y:
        If True, the y-axes use logarithmic scaling.
    :type log_y:
        bool

    :param sources:
        List of bias sources to display. Supported values are real and estimated.
    :type sources:
        list[str] or None

    """
    def __init__(
        self,
        *,
        kind: str = "sensor",  # 'sensor' or 'actuator'
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
        sources: list[str] | None = None,  # ["real", "estimated"]
    ):
        self.kind = kind
        self.time = time
        self.title = title or f"{kind.capitalize()} Bias"
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.sources = _normalize_bias_sources(sources)

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # Find first run with usable data (for layout)
        mats0 = None
        first_run = None
        for run in runs:
            mats = {src: _get_bias_matrix(run, self.kind, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is not None:
                mats0 = mats
                first_run = run
                break

        if mats0 is None or first_run is None:
            self._plot_no_data(ax)
            return

        n_bias = mats0[next(k for k, v in mats0.items() if v is not None)].shape[1]

        ncols = int(math.ceil(math.sqrt(n_bias)))
        nrows = int(math.ceil(n_bias / ncols))
        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax.get_subplotspec())

        axes = []
        for i in range(n_bias):
            r, c = divmod(i, ncols)
            axes.append(ax.figure.add_subplot(gs[r, c]))

        labels = self.labels or [rf"$b_{{{i}}}$" for i in range(n_bias)]
        if len(labels) != n_bias:
            raise ValueError("labels length must match bias dimension")

        style = {"real": "-", "estimated": "--"}
        alpha = max(0.15, 1.0 / len(runs))

        # Plot all runs
        for run in runs:
            mats = {src: _get_bias_matrix(run, self.kind, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is None:
                continue

            N = min(m.shape[0] for m in mats.values() if m is not None)
            t = _get_time_axis(run, self.time, N)

            for i, ax_i in enumerate(axes):
                for src in self.sources:
                    B = mats.get(src, None)
                    if B is None:
                        continue
                    ax_i.plot(
                        t,
                        B[:N, i],
                        linestyle=style[src],
                        alpha=alpha,
                        label=None,
                    )

        # Ax formatting + clean legends
        for i, ax_i in enumerate(axes):
            ylabel = f"{labels[i]} [{self.units}]" if self.units else labels[i]
            ax_i.set_ylabel(ylabel)
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.grid(True, which="both")

            # One legend per subplot (sources only)
            handles, labs = [], []
            for src in self.sources:
                handles.append(ax_i.plot([], [], linestyle=style[src], color="k")[0])
                labs.append(src)
            if len(self.sources) > 1:
                ax_i.legend(handles, labs)

        # Turn off unused slots
        for j in range(n_bias, nrows * ncols):
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
        ax_text.text(
            0.5, 0.5, "No bias history available", ha="center", va="center"
        )


class BiasPlotSingle(Subplot):
    r"""
    Visualization of a single bias component over time.

    This class plots one selected bias component for sensor or actuator biases.
    Real and estimated sources may be overlaid for direct comparison. It is
    intended for focused inspection of a specific bias term.

    User settings determine which bias component is shown and how it is styled.

    :param index:
        Index of the bias component to plot.
    :type index:
        int

    :param kind:
        Type of bias to plot. Must be either sensor or actuator.
    :type kind:
        str

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title of the plot. If None, a default title is used.
    :type title:
        str or None

    :param units:
        Physical units of the bias values.
    :type units:
        str

    :param label:
        Label used for the bias component.
    :type label:
        str or None

    :param color:
        Color used for the plotted bias signal.
    :type color:
        str or None

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param sources:
        List of bias sources to display. Supported values are real and estimated.
    :type sources:
        list[str] or None

    """
    def __init__(
        self,
        index: int,
        *,
        kind: str = "sensor",
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        label: str | None = None,
        color: str | None = None,
        log_y: bool = False,
        sources: list[str] | None = None,
    ):
        self.index = index
        self.kind = kind
        self.time = time
        self.title = title
        self.units = units
        self.label = label
        self.color = color
        self.log_y = log_y
        self.sources = _normalize_bias_sources(sources)

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        # Find first run with data to validate index
        first_mat = None
        first_run = None
        for run in runs:
            mats = {src: _get_bias_matrix(run, self.kind, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is not None:
                first_mat = first
                first_run = run
                break

        if first_mat is None or first_run is None:
            self._plot_no_data(ax)
            return

        n_bias = first_mat.shape[1]
        if not (0 <= self.index < n_bias):
            raise ValueError(f"Bias index {self.index} out of bounds for {n_bias}")

        lbl = self.label or rf"$b_{{{self.index}}}$"
        title = self.title or f"{self.kind.capitalize()} Bias {self.index}"
        style = {"real": "-", "estimated": "--"}
        alpha = max(0.15, 1.0 / len(runs))

        t_label = "Time [s]"
        plotted_any = False

        for run in runs:
            mats = {src: _get_bias_matrix(run, self.kind, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is None:
                continue

            N = min(m.shape[0] for m in mats.values() if m is not None)
            t = _get_time_axis(run, self.time, N)

            for src_i, src in enumerate(self.sources):
                B = mats.get(src, None)
                if B is None:
                    continue

                kw = {}
                if self.color is not None and src_i == 0:
                    kw["color"] = self.color

                ax.plot(
                    t,
                    B[:N, self.index],
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
        ax.set_xlabel(t_label)
        ax.set_title(title)

        if self.log_y:
            ax.set_yscale("log")

        # Clean legend: sources only
        handles, labs = [], []
        for src in self.sources:
            handles.append(ax.plot([], [], linestyle=style[src], color="k")[0])
            labs.append(src)
        if len(self.sources) > 1:
            ax.legend(handles, labs)
        else:
            ax.legend(handles, labs)

        ax.grid(True, which="both")


    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title or "Bias", loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "No bias history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


class BiasPlotCombined(Subplot):
    r"""
    Combined plot of all bias components on a single axis.

    This class overlays all bias components on one set of axes, optionally
    including both real and estimated sources. It is useful for high-level
    comparison of bias magnitudes and trends across components.

    The plot emphasizes user-defined labeling, colors, and axis scaling.

    :param kind:
        Type of bias to plot. Must be either sensor or actuator.
    :type kind:
        str

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str or None

    :param units:
        Physical units of the bias values.
    :type units:
        str

    :param labels:
        Optional list of labels for each bias component.
    :type labels:
        list[str] or None

    :param log_y:
        If True, the y-axis uses logarithmic scaling.
    :type log_y:
        bool

    :param colors:
        Optional list of colors used cyclically for bias components.
    :type colors:
        list[str] or None

    :param sources:
        List of bias sources to display. Supported values are real and estimated.
    :type sources:
        list[str] or None

    """
    def __init__(
        self,
        *,
        kind: str = "sensor",
        time: str = "time_s",
        title: str | None = None,
        units: str = "",
        labels: list[str] | None = None,
        log_y: bool = False,
        colors: list[str] | None = None,
        sources: list[str] | None = None,
    ):
        self.kind = kind
        self.time = time
        self.title = title or f"{kind.capitalize()} Bias"
        self.units = units
        self.labels = labels
        self.log_y = log_y
        self.colors = colors
        self.sources = _normalize_bias_sources(sources)

    def plot(self, ax, sim) -> None:
        runs = getattr(sim, "runs", None)
        if runs is None:
            runs = [sim]

        # Find first run with data for dimensions
        first_mat = None
        for run in runs:
            mats = {src: _get_bias_matrix(run, self.kind, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is not None:
                first_mat = first
                first_run = run
                break

        if first_mat is None:
            self._plot_no_data(ax)
            return

        n_bias = first_mat.shape[1]
        labels = self.labels or [rf"$b_{{{i}}}$" for i in range(n_bias)]
        if len(labels) != n_bias:
            raise ValueError("labels length must match bias dimension")

        style = {"real": "-", "estimated": "--"}
        alpha = max(0.15, 1.0 / len(runs))
        t_label = "Time [s]"
        plotted_any = False

        for run in runs:
            mats = {src: _get_bias_matrix(run, self.kind, src) for src in self.sources}
            first = next((m for m in mats.values() if m is not None), None)
            if first is None:
                continue

            N = min(m.shape[0] for m in mats.values() if m is not None)
            t = _get_time_axis(run, self.time, N)

            for src in self.sources:
                B = mats.get(src, None)
                if B is None:
                    continue

                for i in range(n_bias):
                    color_arg = {}
                    if self.colors:
                        color_arg["color"] = self.colors[i % len(self.colors)]

                    ax.plot(
                        t,
                        B[:N, i],
                        linestyle=style[src],
                        alpha=alpha,
                        label=None,
                        **color_arg,
                    )

                plotted_any = True

        if not plotted_any:
            self._plot_no_data(ax)
            return

        ylabel = f"Bias [{self.units}]" if self.units else "Bias"
        ax.set_ylabel(ylabel)
        ax.set_xlabel(t_label)
        ax.set_title(self.title)

        if self.log_y:
            ax.set_yscale("log")

        # Clean legend:
        # - components: via color proxies (optional)
        # - sources: via linestyle proxies
        handles, labs = [], []

        if self.colors:
            for i in range(min(n_bias, len(self.colors))):
                handles.append(ax.plot([], [], color=self.colors[i % len(self.colors)], linestyle="-")[0])
                labs.append(labels[i])
            if n_bias > len(self.colors):
                handles.append(ax.plot([], [], color="k", linestyle="-")[0])
                labs.append("...")

        if len(self.sources) > 1:
            for src in self.sources:
                handles.append(ax.plot([], [], color="k", linestyle=style[src])[0])
                labs.append(src)

        if handles:
            ax.legend(handles, labs, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        else:
            ax.legend()

        ax.grid(True, which="both", linestyle="--", alpha=0.7)


    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "No bias history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
