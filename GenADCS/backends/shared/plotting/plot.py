"""Composable Matplotlib plots for SimulationResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from GenADCS.backends.shared.simulationresult import SimulationResult

MODES = ("single", "spaghetti", "1sigma", "10-90")


def _runs(sim):
    runs = [sim] if isinstance(sim, SimulationResult) else list(sim)
    if not runs or any(not isinstance(run, SimulationResult) for run in runs):
        raise TypeError("sim must be a SimulationResult or a nonempty sequence of them")
    return runs


@dataclass
class Subplot:
    """A sampled field to draw in one panel of a figure."""

    field: str
    satellite: str = "sat_000"
    title: str | None = None
    ylabel: str | None = None
    labels: Sequence[str] | None = None
    colors: Sequence[str] | None = None
    plot_kwargs: dict = field(default_factory=dict)
    axes_kwargs: dict = field(default_factory=dict)

    def _series(self, run):
        satellite = run.satellites[self.satellite]
        values = getattr(satellite, self.field)
        if values is None:
            raise ValueError(f"{self.field} was not recorded for {self.satellite}")
        return satellite.times_s, np.asarray(values, dtype=float)

    def draw(self, ax, runs, mode):
        series = [self._series(run) for run in runs]
        width = series[0][1].shape[1]
        if any(values.shape[1] != width for _, values in series):
            raise ValueError("all runs must have the same number of components")
        labels = list(self.labels) if self.labels is not None else [f"{self.field} {i}" for i in range(width)]
        if len(labels) != width:
            raise ValueError("labels must match the number of components")
        if self.colors is not None and len(self.colors) != width:
            raise ValueError("colors must match the number of components")
        colors = self.colors or [None] * width
        if mode == "single":
            series = series[:1]
        if mode in ("single", "spaghetti"):
            for index, (times, values) in enumerate(series):
                for component in range(width):
                    style = dict(self.plot_kwargs)
                    if mode == "spaghetti":
                        style.setdefault("alpha", max(0.15, 1 / len(series) ** 0.5))
                    if colors[component] is not None:
                        style.setdefault("color", colors[component])
                    ax.plot(times, values[:, component],
                            label=labels[component] if index == 0 else None, **style)
        else:
            times = series[0][0]
            if any(not np.array_equal(t, times) for t, _ in series[1:]):
                raise ValueError("statistical modes require identical sampled times across runs")
            samples = np.stack([values for _, values in series])
            if mode == "1sigma":
                center = samples.mean(axis=0)
                spread = samples.std(axis=0)
                low, high = center - spread, center + spread
            else:
                center = np.median(samples, axis=0)
                low, high = np.percentile(samples, [10, 90], axis=0)
            for component in range(width):
                style = dict(self.plot_kwargs)
                if colors[component] is not None:
                    style.setdefault("color", colors[component])
                line, = ax.plot(times, center[:, component], label=labels[component], **style)
                ax.fill_between(times, low[:, component], high[:, component],
                                color=line.get_color(), alpha=0.2)
        ax.set(title=self.title or self.field.replace("_", " ").title(),
               xlabel="Time [s]", ylabel=self.ylabel)
        ax.set(**self.axes_kwargs)
        ax.grid(True, alpha=0.3)
        if width > 1:
            ax.legend()


class AttitudePlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("labels", ("q0", "q1", "q2", "q3"))
        super().__init__("satellite_attitude", **kwargs)


class SatelliteAttitudeQuaternionPlot(Subplot):
    """Draw all four scalar-first attitude quaternion components."""

    def __init__(self, *, satellite="sat_000", title="Attitude Quaternion",
                 xlabel="Time [s]", ylabel="Quaternion", labels=None,
                 colors=None, axes_kwargs=None, plot_kwargs=None,
                 title_fontsize=None, label_fontsize=None, tick_fontsize=None,
                 **line_kwargs):
        axes_options = dict(axes_kwargs or {})
        axes_options.setdefault("xlabel", xlabel)
        for name in ("xlim", "ylim", "xscale", "yscale", "aspect", "facecolor"):
            if name in line_kwargs:
                axes_options[name] = line_kwargs.pop(name)
        style = {**(plot_kwargs or {}), **line_kwargs}
        super().__init__("satellite_attitude", satellite=satellite, title=title,
                         ylabel=ylabel, labels=labels or ("qW", "qX", "qY", "qZ"),
                         colors=colors, plot_kwargs=style, axes_kwargs=axes_options)
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize

    def draw(self, ax, runs, mode):
        super().draw(ax, runs, mode)
        if self.title_fontsize is not None:
            ax.title.set_fontsize(self.title_fontsize)
        if self.label_fontsize is not None:
            ax.xaxis.label.set_fontsize(self.label_fontsize)
            ax.yaxis.label.set_fontsize(self.label_fontsize)
        if self.tick_fontsize is not None:
            ax.tick_params(labelsize=self.tick_fontsize)


class _QuaternionComponentPlot(SatelliteAttitudeQuaternionPlot):
    component_index = 0
    component_name = "qW"

    def __init__(self, **kwargs):
        kwargs.setdefault("title", self.component_name)
        kwargs.setdefault("ylabel", self.component_name)
        kwargs.setdefault("labels", (self.component_name,))
        color = kwargs.pop("color", None)
        if color is not None:
            kwargs.setdefault("colors", (color,))
        super().__init__(**kwargs)

    def _series(self, run):
        times, quaternion = super()._series(run)
        return times, quaternion[:, self.component_index:self.component_index + 1]


class SatelliteAttitudeQWPlot(_QuaternionComponentPlot):
    component_index = 0
    component_name = "qW"


class SatelliteAttitudeQXPlot(_QuaternionComponentPlot):
    component_index = 1
    component_name = "qX"


class SatelliteAttitudeQYPlot(_QuaternionComponentPlot):
    component_index = 2
    component_name = "qY"


class SatelliteAttitudeQZPlot(_QuaternionComponentPlot):
    component_index = 3
    component_name = "qZ"


class RatePlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("ylabel", "Rate [rad/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("satellite_rate", **kwargs)


class PositionPlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("ylabel", "Position [m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("satellite_position", **kwargs)


class VelocityPlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("ylabel", "Velocity [m/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("satellite_velocity", **kwargs)


class WheelMomentumPlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("ylabel", "Momentum [N m s]")
        super().__init__("satellite_wheel_momenta", **kwargs)


class WheelSpeedPlot(WheelMomentumPlot):
    """Convert stored wheel momenta using the satellite configuration."""

    def _series(self, run):
        times, momentum = super()._series(run)
        satellite = run.satellites[self.satellite].to_satellite()
        inertia = [getattr(wheel, "rotational_inertia", getattr(wheel, "J", None))
                   for wheel in satellite.adcs.reaction_wheels]
        if len(inertia) != momentum.shape[1] or any(value is None or value <= 0 for value in inertia):
            raise ValueError("wheel configuration does not match recorded momenta")
        return times, momentum / np.asarray(inertia)

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Reaction Wheel Speed")
        kwargs.setdefault("ylabel", "Speed [rad/s]")
        super().__init__(**kwargs)


def plot(sim, *subplots: Subplot, mode="single", layout=None, title=None,
         sharex=True, figure_kwargs=None, subplot_kwargs=None, show=False):
    """Arrange panels in one figure and return ``(figure, axes)``.

    ``sim`` accepts one SimulationResult or several runs for ensemble modes.
    Matplotlib figure options go in ``figure_kwargs``; subplot options go in
    ``subplot_kwargs``. Each panel also accepts Matplotlib line and axes options.
    """
    import matplotlib.pyplot as plt

    runs = _runs(sim)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not subplots:
        raise ValueError("provide at least one subplot")
    if any(not isinstance(panel, Subplot) for panel in subplots):
        raise TypeError("all panels must be Subplot instances")
    rows, columns = layout or (len(subplots), 1)
    if rows * columns < len(subplots):
        raise ValueError("layout is too small for the panels")
    fig, axes = plt.subplots(rows, columns, sharex=sharex,
                             **(figure_kwargs or {}), **(subplot_kwargs or {}))
    flat = np.atleast_1d(axes).ravel()
    for ax, panel in zip(flat, subplots):
        panel.draw(ax, runs, mode)
    for ax in flat[len(subplots):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show(block=False)
    return fig, axes
