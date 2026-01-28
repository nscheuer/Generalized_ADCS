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
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        mats = {src: _get_bias_matrix(sim, self.kind, src) for src in self.sources}
        first = next((m for m in mats.values() if m is not None), None)
        if first is None:
            self._plot_no_data(ax)
            return

        n_bias = first.shape[1]
        N = min(m.shape[0] for m in mats.values() if m is not None)
        t = _get_time_axis(sim, self.time, N)

        ncols = int(math.ceil(math.sqrt(n_bias)))
        nrows = int(math.ceil(n_bias / ncols))
        gs = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=ax.get_subplotspec()
        )

        axes = []
        for i in range(n_bias):
            r, c = divmod(i, ncols)
            axes.append(ax.figure.add_subplot(gs[r, c]))

        labels = self.labels or [rf"$b_{{{i}}}$" for i in range(n_bias)]
        if len(labels) != n_bias:
            raise ValueError("labels length must match bias dimension")

        style = {"real": "-", "estimated": "--"}

        for i, ax_i in enumerate(axes):
            for src in self.sources:
                B = mats[src]
                if B is None:
                    continue
                ax_i.plot(
                    t,
                    B[:N, i],
                    linestyle=style[src],
                    label=f"{labels[i]} ({src})" if len(self.sources) > 1 else labels[i],
                )

            ylabel = f"{labels[i]} [{self.units}]" if self.units else labels[i]
            ax_i.set_ylabel(ylabel)

            if self.log_y:
                ax_i.set_yscale("log")

            ax_i.legend()
            ax_i.grid(True, which="both")

        # Turn off unused slots in the grid
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
        mats = {src: _get_bias_matrix(sim, self.kind, src) for src in self.sources}
        first = next((m for m in mats.values() if m is not None), None)
        if first is None:
            self._plot_no_data(ax)
            return

        n_bias = first.shape[1]
        if not (0 <= self.index < n_bias):
            raise ValueError(f"Bias index {self.index} out of bounds for {n_bias}")

        N = min(m.shape[0] for m in mats.values() if m is not None)
        t = _get_time_axis(sim, self.time, N)

        lbl = self.label or rf"$b_{{{self.index}}}$"
        title = self.title or f"{self.kind.capitalize()} Bias {self.index}"

        style = {"real": "-", "estimated": "--"}

        for src_i, src in enumerate(self.sources):
            B = mats[src]
            if B is None:
                continue
            kw = {}
            if self.color is not None and src_i == 0:
                kw["color"] = self.color
            ax.plot(t, B[:N, self.index], linestyle=style[src], label=src, **kw)

        ylabel = f"{lbl} [{self.units}]" if self.units else lbl
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(title)

        if self.log_y:
            ax.set_yscale("log")

        ax.legend()
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
        mats = {src: _get_bias_matrix(sim, self.kind, src) for src in self.sources}
        first = next((m for m in mats.values() if m is not None), None)
        if first is None:
            self._plot_no_data(ax)
            return

        n_bias = first.shape[1]
        N = min(m.shape[0] for m in mats.values() if m is not None)
        t = _get_time_axis(sim, self.time, N)

        labels = self.labels or [rf"$b_{{{i}}}$" for i in range(n_bias)]
        if len(labels) != n_bias:
            raise ValueError("labels length must match bias dimension")

        style = {"real": "-", "estimated": "--"}

        for i in range(n_bias):
            color_arg = {}
            if self.colors:
                color_arg["color"] = self.colors[i % len(self.colors)]

            for src in self.sources:
                B = mats[src]
                if B is None:
                    continue
                ax.plot(
                    t,
                    B[:N, i],
                    linestyle=style[src],
                    label=f"{labels[i]} ({src})" if len(self.sources) > 1 else labels[i],
                    **color_arg,
                )

        ylabel = f"Bias [{self.units}]" if self.units else "Bias"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.set_title(self.title)

        if self.log_y:
            ax.set_yscale("log")

        if n_bias > 5 or len(self.sources) > 1:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
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
