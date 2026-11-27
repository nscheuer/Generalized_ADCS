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


class EulerAnglePlot(SatelliteAttitudeQuaternionPlot):
    """Plot 3-2-1 Euler angles in radians, converted from scalar-first quaternions."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Euler Angles (3-2-1)")
        kwargs.setdefault("ylabel", "Angle [rad]")
        kwargs.setdefault("labels", ("Roll", "Pitch", "Yaw"))
        super().__init__(**kwargs)

    def _series(self, run):
        times, q = super()._series(run)
        w, x, y, z = q.T
        angles = np.column_stack((
            np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
            np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0)),
            np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
        ))
        return times, angles


class MRPPlot(SatelliteAttitudeQuaternionPlot):
    """Plot modified Rodrigues parameters with the shadow set for bounded values."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Modified Rodrigues Parameters")
        kwargs.setdefault("ylabel", "MRP")
        kwargs.setdefault("labels", ("σx", "σy", "σz"))
        super().__init__(**kwargs)

    def _series(self, run):
        times, q = super()._series(run)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / norm
        q = np.where(q[:, :1] < 0, -q, q)
        denom = 1.0 + q[:, :1]
        mrp = q[:, 1:] / denom
        shadow = np.linalg.norm(mrp, axis=1) > 1.0
        mrp[shadow] = -mrp[shadow] / np.sum(mrp[shadow] ** 2, axis=1, keepdims=True)
        return times, mrp


class RodriguezPlot(SatelliteAttitudeQuaternionPlot):
    """Plot Rodrigues parameters in the principal quaternion hemisphere."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Rodrigues Parameters")
        kwargs.setdefault("ylabel", "Rodrigues parameter")
        kwargs.setdefault("labels", ("ρx", "ρy", "ρz"))
        super().__init__(**kwargs)

    def _series(self, run):
        times, q = super()._series(run)
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / norm
        q = np.where(q[:, :1] < 0, -q, q)
        if np.any(np.abs(q[:, 0]) < 1e-12):
            raise ValueError("Rodrigues parameters are singular at a 180-degree rotation")
        return times, q[:, 1:] / q[:, :1]


class BodyAngularRatePlot(Subplot):
    """Plot body-frame angular rate components in rad/s."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Body Angular Rate")
        kwargs.setdefault("ylabel", "Rate [rad/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        kwargs["plot_kwargs"] = {**kwargs.pop("plot_kwargs", {}), **{
            name: kwargs.pop(name) for name in tuple(kwargs)
            if name in {"linewidth", "linestyle", "marker", "alpha", "drawstyle"}
        }}
        super().__init__("satellite_rate", **kwargs)


class InertialAngularRatePlot(BodyAngularRatePlot):
    """Plot inertial-frame angular rate, rotated from body-frame samples."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Inertial Angular Rate")
        super().__init__(**kwargs)

    def _series(self, run):
        from GenADCS.backends.python.helpers.math.frames.frames import body_to_inertial

        times, rate = super()._series(run)
        satellite = run.satellites[self.satellite]
        quaternion = satellite.satellite_attitude
        if quaternion is None:
            raise ValueError("satellite_attitude must be recorded for inertial rate")
        inertial = np.empty_like(rate)
        for index in range(len(rate)):
            inertial[index] = body_to_inertial(rate[index], quaternion[index])
        return times, inertial


class AngularMagnitudePlot(BodyAngularRatePlot):
    """Plot the norm of the angular rate vector in rad/s."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Angular Rate Magnitude")
        kwargs.setdefault("ylabel", "Magnitude [rad/s]")
        kwargs.setdefault("labels", ("|ω|",))
        super().__init__(**kwargs)

    def _series(self, run):
        times, rate = super()._series(run)
        return times, np.linalg.norm(rate, axis=1, keepdims=True)


class RatePlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("ylabel", "Rate [rad/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("satellite_rate", **kwargs)


class EstimatedAttitudePlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Estimated Attitude Quaternion")
        kwargs.setdefault("ylabel", "Quaternion")
        kwargs.setdefault("labels", ("qW", "qX", "qY", "qZ"))
        super().__init__("estimated_attitude", **kwargs)


class EstimatedRatePlot(Subplot):
    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Estimated Body Angular Rate")
        kwargs.setdefault("ylabel", "Rate [rad/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("estimated_rate", **kwargs)


class _EstimatedCovariancePlot(Subplot):
    def __init__(self, field, *, option="individual", **kwargs):
        if option not in ("individual", "trace", "eigenvalues"):
            raise ValueError("option must be 'individual', 'trace', or 'eigenvalues'")
        self.option = option
        if "labels" not in kwargs:
            kwargs["labels"] = (("trace",) if option == "trace" else
                                ("largest", "middle", "smallest") if option == "eigenvalues" else
                                ("x", "y", "z"))
        super().__init__(field, **kwargs)

    def _series(self, run):
        times, values = super()._series(run)
        matrices = values.reshape((-1, 3, 3))
        if self.option == "trace":
            return times, np.sqrt(np.maximum(np.trace(matrices, axis1=1, axis2=2), 0.0))[:, None]
            eigenvalues = np.linalg.eigvalsh(matrices)[:, ::-1]
            return times, np.sqrt(np.maximum(eigenvalues, 0.0))
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)
        return times, np.sqrt(np.maximum(diagonal, 0.0))


class EstimatedAttitudeCovariancePlot(_EstimatedCovariancePlot):
    def __init__(self, *, option="individual", **kwargs):
        kwargs.setdefault("title", "Estimated Attitude Standard Deviation")
        kwargs.setdefault("ylabel", "Standard deviation [rad]")
        super().__init__("estimated_attitude_covariance", option=option, **kwargs)


class EstimatedRateCovariancePlot(_EstimatedCovariancePlot):
    def __init__(self, *, option="individual", **kwargs):
        kwargs.setdefault("title", "Estimated Rate Standard Deviation")
        kwargs.setdefault("ylabel", "Standard deviation [rad/s]")
        super().__init__("estimated_rate_covariance", option=option, **kwargs)


class MeasurementInnovationPlot(Subplot):
    def __init__(self, *, option="individual", **kwargs):
        if option not in ("individual", "trace"):
            raise ValueError("option must be 'individual' or 'trace'")
        self.option = option
        kwargs.setdefault("title", "Measurement Innovation")
        kwargs.setdefault("ylabel", "Innovation [rad]")
        kwargs.setdefault("labels", ("magnitude",) if option == "trace" else ("x", "y", "z"))
        super().__init__("measurement_innovation", **kwargs)

    def _series(self, run):
        times, values = super()._series(run)
        if self.option == "trace":
            return times, np.linalg.norm(values, axis=1)[:, None]
        return times, values


class MeasurementInnovationCovariancePlot(_EstimatedCovariancePlot):
    def __init__(self, *, option="individual", **kwargs):
        kwargs.setdefault("title", "Measurement Innovation Covariance")
        kwargs.setdefault("ylabel", "Innovation standard deviation [rad]")
        super().__init__("measurement_innovation_covariance", option=option, **kwargs)


class _NormalizedSquaredMetricPlot(Subplot):
    def __init__(self, field, *, option="individual", **kwargs):
        if option != "individual":
            raise ValueError("normalized squared metrics are scalar; option must be 'individual'")
        self.option = option
        kwargs.setdefault("labels", ("value",))
        super().__init__(field, **kwargs)


class NormalizedInnovationSquaredPlot(_NormalizedSquaredMetricPlot):
    def __init__(self, *, option="individual", **kwargs):
        kwargs.setdefault("title", "Normalized Innovation Squared")
        kwargs.setdefault("ylabel", "NIS")
        super().__init__("normalized_innovation_squared", option=option, **kwargs)


class NormalizedEstimationErrorSquaredPlot(_NormalizedSquaredMetricPlot):
    def __init__(self, *, option="individual", **kwargs):
        kwargs.setdefault("title", "Normalized Estimation Error Squared")
        kwargs.setdefault("ylabel", "NEES")
        super().__init__("normalized_estimation_error_squared", option=option, **kwargs)


class MeasurementResidualPlot(Subplot):
    def __init__(self, *, option="individual", **kwargs):
        if option not in ("individual", "trace"):
            raise ValueError("option must be 'individual' or 'trace'")
        self.option = option
        kwargs.setdefault("title", "Measurement Residual")
        kwargs.setdefault("ylabel", "Residual [rad]")
        kwargs.setdefault("labels", ("magnitude",) if option == "trace" else ("x", "y", "z"))
        super().__init__("measurement_residual", **kwargs)

    def _series(self, run):
        times, values = super()._series(run)
        return (times, np.linalg.norm(values, axis=1, keepdims=True)) if self.option == "trace" else (times, values)


class KalmanGainPlot(Subplot):
    def __init__(self, *, option="individual", **kwargs):
        if option not in ("individual", "trace", "eigenvalues"):
            raise ValueError("option must be 'individual', 'trace', or 'eigenvalues'")
        self.option = option
        kwargs.setdefault("title", "Kalman Gain")
        kwargs.setdefault("ylabel", "Gain")
        kwargs.setdefault("labels", tuple(f"K{i}" for i in range(18)) if option == "individual" else
                           (("Frobenius norm",) if option == "trace" else ("largest singular value", "middle singular value", "smallest singular value")))
        super().__init__("kalman_gain", **kwargs)

    def _series(self, run):
        times, values = super()._series(run)
        matrices = values.reshape((-1, 6, 3))
        if self.option == "trace":
            return times, np.linalg.norm(matrices, axis=(1, 2))[:, None]
            singular = np.linalg.svd(matrices, compute_uv=False)
            return times, singular
        return times, values


def _quaternion_rotation_error_deg(
    first: np.ndarray, second: np.ndarray, option: str
) -> np.ndarray:
    """Compute rotation-vector error in degrees between two quaternion trajectories."""
    valid = np.all(np.isfinite(first), axis=1) & np.all(np.isfinite(second), axis=1)
    width = 1 if option == "magnitude" else 3
    out = np.full((first.shape[0], width), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    q1 = first[valid]
    q2 = second[valid]
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    relative = np.empty_like(q1)
    relative[:, 0] = q1[:, 0] * q2[:, 0] + np.sum(q1[:, 1:] * q2[:, 1:], axis=1)
    relative[:, 1:] = (
        q1[:, 0, None] * q2[:, 1:]
        - q2[:, 0, None] * q1[:, 1:]
        - np.cross(q1[:, 1:], q2[:, 1:])
    )
    relative = np.where(relative[:, :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, :1], 0.0, None))
    rotation = np.divide(
        relative[:, 1:] * angle,
        vector_norm,
        out=np.zeros_like(relative[:, 1:]),
        where=vector_norm > 1e-15,
    )
    if option == "magnitude":
        rotation = np.linalg.norm(rotation, axis=1, keepdims=True)
    out[valid] = np.rad2deg(rotation)
    return out


class AbsoluteKnowledgeErrorQuaternionPlot(Subplot):
    """Attitude estimation error as a rotation vector or its magnitude."""

    def __init__(self, option="individual", **kwargs):
        if option not in ("individual", "magnitude"):
            raise ValueError("option must be 'individual' or 'magnitude'")
        self.option = option
        kwargs.setdefault("title", "Absolute Attitude Knowledge Error")
        kwargs.setdefault("ylabel", "Error [deg]" if option == "individual" else "Magnitude [deg]")
        kwargs.setdefault("labels", ("x", "y", "z") if option == "individual" else ("magnitude",))
        super().__init__("estimated_attitude", **kwargs)

    def _series(self, run):
        satellite = run.satellites[self.satellite]
        estimated = satellite.estimated_attitude
        truth = satellite.satellite_attitude
        if estimated is None or truth is None:
            raise ValueError("satellite_attitude and estimated_attitude must be recorded")
        return satellite.times_s, _quaternion_rotation_error_deg(truth, estimated, self.option)


class AbsolutePointingErrorPlot(Subplot):
    """Absolute Pointing Error (APE): truth attitude vs. goal attitude."""

    def __init__(self, option="individual", **kwargs):
        if option not in ("individual", "magnitude"):
            raise ValueError("option must be 'individual' or 'magnitude'")
        self.option = option
        kwargs.setdefault("title", "Absolute Pointing Error (APE)")
        kwargs.setdefault("ylabel", "Error [deg]" if option == "individual" else "Magnitude [deg]")
        kwargs.setdefault("labels", ("x", "y", "z") if option == "individual" else ("magnitude",))
        super().__init__("goal_attitude", **kwargs)

    def _series(self, run):
        satellite = run.satellites[self.satellite]
        truth = satellite.satellite_attitude
        goal = satellite.goal_attitude
        if truth is None or goal is None:
            raise ValueError("satellite_attitude and goal_attitude must be recorded for APE")
        return satellite.times_s, _quaternion_rotation_error_deg(truth, goal, self.option)


class AttitudeControlErrorPlot(Subplot):
    """Attitude Control Error (ACE): estimated attitude vs. goal attitude (requires estimator)."""

    def __init__(self, option="individual", **kwargs):
        if option not in ("individual", "magnitude"):
            raise ValueError("option must be 'individual' or 'magnitude'")
        self.option = option
        kwargs.setdefault("title", "Attitude Control Error (ACE)")
        kwargs.setdefault("ylabel", "Error [deg]" if option == "individual" else "Magnitude [deg]")
        kwargs.setdefault("labels", ("x", "y", "z") if option == "individual" else ("magnitude",))
        super().__init__("goal_attitude", **kwargs)

    def _series(self, run):
        satellite = run.satellites[self.satellite]
        estimated = satellite.estimated_attitude
        goal = satellite.goal_attitude
        if estimated is None or goal is None:
            raise ValueError(
                "estimated_attitude and goal_attitude must be recorded (ACE requires an estimator)"
            )
        return satellite.times_s, _quaternion_rotation_error_deg(estimated, goal, self.option)


class AbsoluteKnowledgeErrorRatePlot(Subplot):
    """Rate estimation error as components or its magnitude."""

    def __init__(self, option="individual", **kwargs):
        if option not in ("individual", "magnitude"):
            raise ValueError("option must be 'individual' or 'magnitude'")
        self.option = option
        kwargs.setdefault("title", "Absolute Rate Knowledge Error")
        kwargs.setdefault("ylabel", "Error [rad/s]" if option == "individual" else "Magnitude [rad/s]")
        kwargs.setdefault("labels", ("x", "y", "z") if option == "individual" else ("magnitude",))
        super().__init__("estimated_rate", **kwargs)

    def _series(self, run):
        satellite = run.satellites[self.satellite]
        estimated = satellite.estimated_rate
        truth = satellite.satellite_rate
        if estimated is None or truth is None:
            raise ValueError("satellite_rate and estimated_rate must be recorded")
        error = truth - estimated
        if self.option == "magnitude":
            error = np.linalg.norm(error, axis=1, keepdims=True)
        return satellite.times_s, error


class GyroscopePlot(Subplot):
    """Plot recorded clean and noisy, biased gyroscope components when present."""

    def __init__(self, *, satellite="sat_000", title="Gyroscope Measurement",
                 ylabel="Angular rate [rad/s]", clean=True, reading=True,
                 plot_kwargs=None, **line_kwargs):
        self.fields = tuple(name for name, enabled in
                            (("gyroscope_clean", clean), ("gyroscope_reading", reading))
                            if enabled)
        if not self.fields:
            raise ValueError("enable clean and/or reading gyroscope data")
        self.satellite = satellite
        self.title = title
        self.ylabel = ylabel
        self.plot_kwargs = {**(plot_kwargs or {}), **line_kwargs}

    def draw(self, ax, runs, mode):
        styles = (("gyroscope_clean", "clean", "C0", "-"),
                  ("gyroscope_reading", "noisy + bias", "C1", "--"))
        any_component = False
        for run_index, run in enumerate(runs[:1] if mode == "single" else runs):
            satellite = run.satellites[self.satellite]
            for field_name, label, color, linestyle in styles:
                if field_name not in self.fields:
                    continue
                values = getattr(satellite, field_name)
                if values is None or values.shape[1] == 0:
                    continue
                any_component = True
                for axis, component in enumerate(("x", "y", "z")):
                    style = {**self.plot_kwargs, "color": color, "linestyle": linestyle}
                    ax.plot(satellite.times_s, values[:, axis],
                            label=f"{label} {component}" if run_index == 0 else None, **style)
        if not any_component:
            raise ValueError("no gyroscope measurements were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel=self.ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()


class MagnetometerPlot(GyroscopePlot):
    """Plot recorded clean and noisy, biased magnetometer components."""

    def __init__(self, *, satellite="sat_000", title="Magnetometer Measurement",
                 ylabel="Magnetic field [T]", clean=True, reading=True,
                 plot_kwargs=None, **line_kwargs):
        self.fields = tuple(name for name, enabled in
                            (("magnetometer_clean", clean), ("magnetometer_reading", reading))
                            if enabled)
        if not self.fields:
            raise ValueError("enable clean and/or reading magnetometer data")
        self.satellite = satellite
        self.title = title
        self.ylabel = ylabel
        self.plot_kwargs = {**(plot_kwargs or {}), **line_kwargs}

    def draw(self, ax, runs, mode):
        styles = (("magnetometer_clean", "clean", "C0", "-"),
                  ("magnetometer_reading", "noisy + bias", "C1", "--"))
        any_component = False
        for run_index, run in enumerate(runs[:1] if mode == "single" else runs):
            satellite = run.satellites[self.satellite]
            for field_name, label, color, linestyle in styles:
                if field_name not in self.fields:
                    continue
                values = getattr(satellite, field_name)
                if values is None or values.shape[1] == 0:
                    continue
                any_component = True
                for axis, component in enumerate(("x", "y", "z")):
                    ax.plot(satellite.times_s, values[:, axis],
                            label=f"{label} {component}" if run_index == 0 else None,
                            **{**self.plot_kwargs, "color": color, "linestyle": linestyle})
        if not any_component:
            raise ValueError("no magnetometer measurements were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel=self.ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()


class StarTrackerPlot(Subplot):
    """Plot clean and measured star tracker attitude quaternions."""

    def __init__(self, *, satellite="sat_000", title="Star Tracker Quaternion",
                 clean=True, reading=True, **kwargs):
        self.fields = tuple(name for name, enabled in
                            (("autonomous_star_tracker_clean", clean),
                             ("autonomous_star_tracker_reading", reading)) if enabled)
        if not self.fields:
            raise ValueError("enable clean and/or reading star tracker data")
        self.satellite, self.title = satellite, title

    def draw(self, ax, runs, mode):
        styles = (("autonomous_star_tracker_clean", "clean", "-"),
                  ("autonomous_star_tracker_reading", "noisy + bias", "--"))
        drawn = False
        for ri, run in enumerate(runs[:1] if mode == "single" else runs):
            sat = run.satellites[self.satellite]
            for name, label, ls in styles:
                values = getattr(sat, name)
                if name not in self.fields or values is None or values.shape[1] == 0:
                    continue
                drawn = True
                for i, component in enumerate(("q0", "q1", "q2", "q3")):
                    ax.plot(sat.times_s, values[:, i], color=f"C{i}", linestyle=ls,
                            label=f"{label} {component}" if ri == 0 else None)
        if not drawn:
            raise ValueError("no star tracker measurements were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel="Quaternion")
        ax.set_xlim(left=0.0, right=max(run.satellites[self.satellite].times_s[-1] for run in runs))
        ax.grid(True, alpha=.3)
        ax.legend()


class CoarseSunSensorPlot(Subplot):
    """Plot clean and measured coarse Sun sensor intensity."""

    def __init__(self, *, satellite="sat_000", title="Coarse Sun Sensor", clean=True, reading=True):
        self.satellite, self.title = satellite, title
        self.fields = tuple(n for n, use in (("coarse_sun_sensor_clean", clean),
                                             ("coarse_sun_sensor_reading", reading)) if use)
        if not self.fields:
            raise ValueError("enable clean and/or reading data")

    def draw(self, ax, runs, mode):
        drawn = False
        for ri, run in enumerate(runs[:1] if mode == "single" else runs):
            sat = run.satellites[self.satellite]
            for name, label, ls in (("coarse_sun_sensor_clean", "clean", "-"),
                                    ("coarse_sun_sensor_reading", "noisy + bias", "--")):
                values = getattr(sat, name)
                if name not in self.fields or values is None or values.shape[1] == 0:
                    continue
                ax.plot(sat.times_s, values[:, 0], linestyle=ls, color="C0" if name.endswith("clean") else "C1",
                        label=label if ri == 0 else None)
                drawn = True
        if not drawn:
            raise ValueError("no coarse Sun sensor measurements were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel="Sensor reading")
        ax.set_xlim(0.0, max(run.satellites[self.satellite].times_s[-1] for run in runs))
        ax.grid(True, alpha=.3)
        ax.legend()


class FineSunSensorPlot(Subplot):
    """Plot clean and measured fine Sun sensor alpha and beta angles."""

    def __init__(self, *, satellite="sat_000", clean=True, reading=True,
                 title="Fine Sun Sensor Angles", sensor="fine_sun_sensor"):
        self.satellite, self.title = satellite, title
        self.fields = tuple(n for n, use in ((sensor + "_clean", clean),
                                             (sensor + "_reading", reading)) if use)
        if not self.fields:
            raise ValueError("enable clean and/or reading fine Sun sensor data")

    def draw(self, ax, runs, mode):
        styles = ((self.fields[0].rsplit("_", 1)[0] + "_clean", "clean", "-"),
                  (self.fields[0].rsplit("_", 1)[0] + "_reading", "noisy + bias", "--"))
        drawn = False
        selected = runs[:1] if mode == "single" else runs
        for run_index, run in enumerate(selected):
            sat = run.satellites[self.satellite]
            for name, label, linestyle in styles:
                values = getattr(sat, name)
                if name not in self.fields or values is None or values.shape[1] == 0:
                    continue
                drawn = True
                for component, angle_name in enumerate(("alpha", "beta")):
                    ax.plot(sat.times_s, np.rad2deg(values[:, component]),
                            linestyle=linestyle, color=f"C{component}",
                            label=f"{label} {angle_name}" if run_index == 0 else None)
        if not drawn:
            raise ValueError("no fine direction sensor measurements were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel="Angle [deg]")
        ax.set_xlim(0., max(run.satellites[self.satellite].times_s[-1] for run in runs))
        ax.grid(True, alpha=.3)
        ax.legend()


class FineEarthSensorPlot(FineSunSensorPlot):
    """Plot clean and measured fine Earth sensor alpha and beta angles."""

    def __init__(self, *, satellite="sat_000", clean=True, reading=True,
                 title="Fine Earth Sensor Angles"):
        self.satellite, self.title = satellite, title
        super().__init__(satellite=satellite, clean=clean, reading=reading,
                         title=title, sensor="fine_earth_sensor")


class GNSSPositionPlot(Subplot):
    """Plot clean and measured GNSS inertial position components."""

    def __init__(self, *, satellite="sat_000", title="GNSS Position", clean=True, reading=True):
        self.satellite, self.title = satellite, title
        self.fields = tuple(n for n, use in (("gnss_clean", clean), ("gnss_reading", reading)) if use)
        if not self.fields:
            raise ValueError("enable clean and/or reading GNSS data")

    def draw(self, ax, runs, mode):
        drawn = False
        selected = runs[:1] if mode == "single" else runs
        for ri, run in enumerate(selected):
            sat = run.satellites[self.satellite]
            for name, label, linestyle in (("gnss_clean", "clean", "-"),
                                           ("gnss_reading", "noisy + bias", "--")):
                values = getattr(sat, name)
                if name not in self.fields or values is None or values.shape[1] == 0:
                    continue
                drawn = True
                for i, axis_name in enumerate(("x", "y", "z")):
                    ax.plot(sat.times_s, values[:, i] / 1000., linestyle=linestyle,
                            color=f"C{i}", label=f"{label} {axis_name}" if ri == 0 else None)
        if not drawn:
            raise ValueError("no GNSS position data were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel="Position [km]")
        ax.grid(True, alpha=.3)
        ax.legend()


class GNSSVelocityPlot(GNSSPositionPlot):
    """Plot clean and measured GNSS inertial velocity components."""

    def __init__(self, *, satellite="sat_000", title="GNSS Velocity", clean=True, reading=True):
        self.satellite, self.title = satellite, title
        self.fields = tuple(n for n, use in (("gnss_clean", clean), ("gnss_reading", reading)) if use)
        if not self.fields:
            raise ValueError("enable clean and/or reading GNSS data")

    def draw(self, ax, runs, mode):
        drawn = False
        selected = runs[:1] if mode == "single" else runs
        for ri, run in enumerate(selected):
            sat = run.satellites[self.satellite]
            for name, label, linestyle in (("gnss_clean", "clean", "-"),
                                           ("gnss_reading", "noisy + bias", "--")):
                values = getattr(sat, name)
                if name not in self.fields or values is None or values.shape[1] ==  0:
                    continue
                drawn = True
                for i, axis_name in enumerate(("x", "y", "z")):
                    ax.plot(sat.times_s, values[:, i + 3] / 1000., linestyle=linestyle,
                            color=f"C{i}", label=f"{label} {axis_name}" if ri == 0 else None)
        if not drawn:
            raise ValueError("no GNSS velocity data were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel="Velocity [km/s]")
        ax.grid(True, alpha=.3)
        ax.legend()
        if not self.fields:
            raise ValueError("enable clean and/or reading fine Earth sensor data")


class PositionPlot(Subplot):
    def __init__(self, *, satellite="sat_000", **kwargs):
        kwargs.setdefault("ylabel", "Position [m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        style = {**kwargs.pop("plot_kwargs", {}), **{name: kwargs.pop(name) for name in tuple(kwargs)
                 if name in {"linewidth", "linestyle", "marker", "alpha", "drawstyle"}}}
        super().__init__("satellite_position", satellite=satellite, plot_kwargs=style, **kwargs)


class VelocityPlot(Subplot):
    def __init__(self, *, satellite="sat_000", **kwargs):
        kwargs.setdefault("ylabel", "Velocity [m/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        style = {**kwargs.pop("plot_kwargs", {}), **{name: kwargs.pop(name) for name in tuple(kwargs)
                 if name in {"linewidth", "linestyle", "marker", "alpha", "drawstyle"}}}
        super().__init__("satellite_velocity", satellite=satellite, plot_kwargs=style, **kwargs)


class AccelerometerPlot(GyroscopePlot):
    """Plot clean and measured accelerometer specific-force components."""

    def __init__(self, *, satellite="sat_000", title="Accelerometer Measurement",
                 ylabel="Specific force [m/s²]", clean=True, reading=True,
                 plot_kwargs=None, **line_kwargs):
        self.fields = tuple(n for n, use in (("accelerometer_clean", clean),
                                             ("accelerometer_reading", reading)) if use)
        if not self.fields:
            raise ValueError("enable clean and/or reading accelerometer data")
        self.satellite, self.title, self.ylabel = satellite, title, ylabel
        self.plot_kwargs = {**(plot_kwargs or {}), **line_kwargs}

    def draw(self, ax, runs, mode):
        drawn = False
        for ri, run in enumerate(runs[:1] if mode == "single" else runs):
            sat = run.satellites[self.satellite]
            for name, label, linestyle in (("accelerometer_clean", "clean", "-"),
                                           ("accelerometer_reading", "noisy + bias", "--")):
                values = getattr(sat, name)
                if name not in self.fields or values is None or values.shape[1] == 0:
                    continue
                drawn = True
                for i, axis in enumerate(("x", "y", "z")):
                    ax.plot(sat.times_s, values[:, i], color=f"C{i}", linestyle=linestyle,
                            label=f"{label} {axis}" if ri == 0 else None, **self.plot_kwargs)
        if not drawn:
            raise ValueError("no accelerometer measurements were recorded")
        ax.set(title=self.title, xlabel="Time [s]", ylabel=self.ylabel)
        ax.set_xlim(0., max(run.satellites[self.satellite].times_s[-1] for run in runs))
        ax.grid(True, alpha=.3)
        ax.legend()


class AltitudePlot(Subplot):
    """Plot WGS84 geodetic altitude from recorded inertial position."""

    def __init__(self, *, satellite="sat_000", title="Altitude", ylabel="Altitude [km]", **kwargs):
        style = {**kwargs.pop("plot_kwargs", {}), **{name: kwargs.pop(name) for name in tuple(kwargs)
                 if name in {"linewidth", "linestyle", "marker", "alpha", "drawstyle"}}}
        super().__init__("satellite_position", satellite=satellite, title=title,
                         ylabel=ylabel, labels=("Altitude",), plot_kwargs=style, **kwargs)

    def _series(self, run):
        from GenADCS.backends.python.helpers.math.frames import gcrf_position_to_geodetic

        times, positions = super()._series(run)
        epoch = run.epoch_jd
        geodetic = np.asarray([
            gcrf_position_to_geodetic(position, epoch + time / 86400.0,
                                      epoch + time / 86400.0)
            for time, position in zip(times, positions)
        ])
        return times, geodetic[:, 2:3]


class LatLonPlot(Subplot):
    """Plot WGS84 geodetic latitude and longitude over simulation time."""

    def __init__(self, *, satellite="sat_000", title="Latitude and Longitude", **kwargs):
        kwargs.setdefault("ylabel", "Angle [deg]")
        kwargs.setdefault("labels", ("Latitude", "Longitude"))
        style = {**kwargs.pop("plot_kwargs", {}), **{name: kwargs.pop(name) for name in tuple(kwargs)
                 if name in {"linewidth", "linestyle", "marker", "alpha", "drawstyle"}}}
        super().__init__("satellite_position", satellite=satellite, title=title,
                         plot_kwargs=style, **kwargs)

    def _series(self, run):
        from GenADCS.backends.python.helpers.math.frames import gcrf_position_to_geodetic

        times, positions = super()._series(run)
        epoch = run.epoch_jd
        geodetic = np.asarray([
            gcrf_position_to_geodetic(position, epoch + time / 86400.0,
                                      epoch + time / 86400.0)
            for time, position in zip(times, positions)
        ])
        return times, geodetic[:, :2]


class SunVectorPlot(Subplot):
    """Plot the recorded spacecraft-to-Sun vector in inertial metres."""

    def __init__(self, *, satellite="sat_000", title="Relative Sun Vector", **kwargs):
        kwargs.setdefault("ylabel", "Vector [m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("relative_sun_vector", satellite=satellite, title=title, **kwargs)


class EarthVectorPlot(Subplot):
    """Plot the recorded spacecraft-to-Earth vector in inertial metres."""

    def __init__(self, *, satellite="sat_000", title="Relative Earth Vector", **kwargs):
        kwargs.setdefault("ylabel", "Vector [m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("relative_earth_vector", satellite=satellite, title=title, **kwargs)


class EclipseStatePlot(Subplot):
    """Plot the recorded solar illumination fraction from zero to one."""

    def __init__(self, *, satellite="sat_000", title="Illumination Fraction", **kwargs):
        kwargs.setdefault("ylabel", "Illumination [0–1]")
        kwargs.setdefault("labels", ("Illumination",))
        super().__init__("eclipse_state", satellite=satellite, title=title, **kwargs)


class AtmosphericDensityPlot(Subplot):
    """Plot atmospheric mass density in kg/m³."""

    def __init__(self, *, satellite="sat_000", title="Atmospheric Density", **kwargs):
        kwargs.setdefault("ylabel", "Density [kg/m³]")
        kwargs.setdefault("labels", ("Density",))
        kwargs.setdefault("axes_kwargs", {"yscale": "log"})
        super().__init__("atmospheric_density", satellite=satellite, title=title, **kwargs)


class RelativeAtmosphericVelocityPlot(Subplot):
    """Plot velocity relative to the rotating atmosphere in inertial axes."""

    def __init__(self, *, satellite="sat_000", title="Relative Atmospheric Velocity", **kwargs):
        kwargs.setdefault("ylabel", "Velocity [m/s]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("relative_atmospheric_velocity", satellite=satellite,
                         title=title, **kwargs)


class MagneticFieldPlot(Subplot):
    """Plot inertial magnetic field components in tesla."""

    def __init__(self, *, satellite="sat_000", title="Magnetic Field", **kwargs):
        kwargs.setdefault("ylabel", "Magnetic field [T]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("magnetic_field", satellite=satellite, title=title, **kwargs)


class MagneticFieldMagnitudePlot(MagneticFieldPlot):
    """Plot the magnitude of the inertial magnetic field in tesla."""

    def __init__(self, **kwargs):
        kwargs.setdefault("title", "Magnetic Field Magnitude")
        kwargs.setdefault("ylabel", "Magnitude [T]")
        kwargs.setdefault("labels", ("|B|",))
        super().__init__(**kwargs)

    def _series(self, run):
        times, field = super()._series(run)
        return times, np.linalg.norm(field, axis=1, keepdims=True)


class AerodynamicForcePlot(Subplot):
    """Plot aerodynamic force components in inertial newtons."""

    def __init__(self, *, satellite="sat_000", title="Aerodynamic Force", **kwargs):
        kwargs.setdefault("ylabel", "Force [N]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("aerodynamic_force", satellite=satellite, title=title, **kwargs)


class AerodynamicTorquePlot(Subplot):
    """Plot aerodynamic torque components in body-frame newton metres."""

    def __init__(self, *, satellite="sat_000", title="Aerodynamic Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("aerodynamic_torque", satellite=satellite, title=title, **kwargs)


class SRPForcePlot(Subplot):
    """Plot solar radiation pressure force in inertial newtons."""

    def __init__(self, *, satellite="sat_000", title="Solar Radiation Pressure Force", **kwargs):
        kwargs.setdefault("ylabel", "Force [N]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("solar_radiation_force", satellite=satellite, title=title, **kwargs)


class SRPTorquePlot(Subplot):
    """Plot solar radiation pressure torque in body-frame newton metres."""

    def __init__(self, *, satellite="sat_000", title="Solar Radiation Pressure Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("solar_radiation_torque", satellite=satellite, title=title, **kwargs)


class LunisolarForcePlot(Subplot):
    """Plot differential Sun and Moon gravity force in inertial newtons."""

    def __init__(self, *, satellite="sat_000", title="Lunisolar Force", **kwargs):
        kwargs.setdefault("ylabel", "Force [N]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("lunisolar_force", satellite=satellite, title=title, **kwargs)


class GravityGradientTorquePlot(Subplot):
    """Plot body-frame gravity-gradient torque in newton metres."""

    def __init__(self, *, satellite="sat_000", title="Gravity Gradient Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("gravity_gradient_torque", satellite=satellite, title=title, **kwargs)


class JForcePlot(Subplot):
    """Plot the combined J2–J6 perturbation force, excluding central gravity."""

    def __init__(self, *, satellite="sat_000", title="Zonal Harmonic Force", **kwargs):
        kwargs.setdefault("ylabel", "Force [N]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("JForcePlot", satellite=satellite, title=title, **kwargs)

    def _series(self, run):
        satellite = run.satellites[self.satellite]
        names = ("J2_force", "J3_force", "J4_force", "J5_force", "J6_force")
        available = [getattr(satellite, name) for name in names]
        if all(value is None for value in available):
            raise ValueError("record at least one J2_force through J6_force to plot JForcePlot")
        total = sum((value for value in available if value is not None), np.zeros_like(next(v for v in available if v is not None)))
        return satellite.times_s, total


class MagneticTorquePlot(Subplot):
    """Plot body-frame residual magnetic torque in newton metres."""

    def __init__(self, *, satellite="sat_000", title="Magnetic Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("magnetic_torque", satellite=satellite, title=title, **kwargs)


class ForcePlot(Subplot):
    """Plot the combined force acting on a satellite in inertial axes."""

    def __init__(self, *, satellite="sat_000", title="Total Force", **kwargs):
        kwargs.setdefault("ylabel", "Force [N]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("total_force", satellite=satellite, title=title, **kwargs)


class TorquePlot(Subplot):
    """Plot the combined torque acting on a satellite in body axes."""

    def __init__(self, *, satellite="sat_000", title="Total Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("total_torque", satellite=satellite, title=title, **kwargs)


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


class GoalAttitudePlot(Subplot):
    """Plot goal attitude quaternion components (scalar-first)."""

    def __init__(self, *, satellite="sat_000", title="Goal Attitude", **kwargs):
        kwargs.setdefault("ylabel", "Quaternion")
        kwargs.setdefault("labels", ("qW", "qX", "qY", "qZ"))
        super().__init__("goal_attitude", satellite=satellite, title=title, **kwargs)


class GoalRatePlot(Subplot):
    """Plot goal body angular rate components."""

    def __init__(self, *, satellite="sat_000", title="Goal Angular Rate", **kwargs):
        kwargs.setdefault("ylabel", "Angular Rate [rad/s]")
        kwargs.setdefault("labels", ("wx", "wy", "wz"))
        super().__init__("goal_rate", satellite=satellite, title=title, **kwargs)


class FeedforwardTorquePlot(Subplot):
    """Plot feedforward torque in body coordinates."""

    def __init__(self, *, satellite="sat_000", title="Feedforward Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("feedforward_torque", satellite=satellite, title=title, **kwargs)


class FeedforwardAttitudePlot(Subplot):
    """Plot feedforward attitude quaternion components (scalar-first)."""

    def __init__(self, *, satellite="sat_000", title="Feedforward Attitude", **kwargs):
        kwargs.setdefault("ylabel", "Quaternion")
        kwargs.setdefault("labels", ("qW", "qX", "qY", "qZ"))
        super().__init__("feedforward_attitude", satellite=satellite, title=title, **kwargs)


class FeedforwardRatePlot(Subplot):
    """Plot feedforward body angular rate components."""

    def __init__(self, *, satellite="sat_000", title="Feedforward Angular Rate", **kwargs):
        kwargs.setdefault("ylabel", "Angular Rate [rad/s]")
        kwargs.setdefault("labels", ("wx", "wy", "wz"))
        super().__init__("feedforward_rate", satellite=satellite, title=title, **kwargs)


class ReferenceTorquePlot(Subplot):
    """Plot controller reference torque in body coordinates."""

    def __init__(self, *, satellite="sat_000", title="Reference Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        kwargs.setdefault("labels", ("x", "y", "z"))
        super().__init__("reference_torque", satellite=satellite, title=title, **kwargs)


class CommandedWheelTorquePlot(Subplot):
    """Plot commanded torque per reaction wheel."""

    def __init__(self, *, satellite="sat_000", title="Commanded Reaction Wheel Torque", **kwargs):
        kwargs.setdefault("ylabel", "Torque [N m]")
        super().__init__("commanded_wheel_torque", satellite=satellite, title=title, **kwargs)


class CommandedMagneticDipolePlot(Subplot):
    """Plot commanded magnetic dipole per magnetorquer."""

    def __init__(self, *, satellite="sat_000", title="Commanded Magnetic Dipole", **kwargs):
        kwargs.setdefault("ylabel", "Dipole [A m^2]")
        super().__init__("commanded_magnetic_dipole", satellite=satellite, title=title, **kwargs)


def _extract_sm_metadata(sat_res):
    """Recover OBC/ADCS mode names, goal names, and OBC config from ``SatelliteResult``."""
    cfg = sat_res.satellite_config or {}
    obc = None
    adcs_sm = None
    try:
        from GenADCS.backends.shared.simulationresult.satellite_configuration import _unpacked
        if cfg.get("adcs_state_machine") is not None:
            adcs_sm = _unpacked(cfg["adcs_state_machine"])
        if cfg.get("obc") is not None:
            obc = _unpacked(cfg["obc"])
    except Exception:
        pass

    obc_names = ()
    adcs_names = ()
    goal_names = ()
    try:
        from GenADCS.backends.python.hardware.ADCS_subsystem.state_machine import (
            GoalTable,
            prepare_adcs_sm,
        )
        from GenADCS.backends.python.hardware.OBC_subsystem.runtime import prepare_obc

        gt = GoalTable()
        adcs_rt = prepare_adcs_sm(adcs_sm, gt)
        if adcs_rt.enabled:
            adcs_names = adcs_rt.mode_names
        if obc is not None and obc.state_machine is not None:
            obc_rt = prepare_obc(obc, adcs_rt if adcs_rt.enabled else None, gt)
            obc_names = obc_rt.mode_names
        _, _, goal_names = gt.to_arrays()
    except Exception:
        pass
    return obc, adcs_sm, obc_names, adcs_names, goal_names


def _matches_source(pattern: str, mode_name: str) -> bool:
    if pattern == "*" or pattern == mode_name:
        return True
    if pattern.endswith(".*"):
        return mode_name.startswith(pattern[:-1])
    return mode_name.startswith(f"{pattern}.")


def _matches_target(target: str, mode_name: str, obc) -> bool:
    if target == mode_name:
        return True
    if obc is not None and obc.state_machine is not None:
        for m in obc.state_machine.modes:
            if m.name == target and m.submodes and m.initial_submode is not None:
                return mode_name == f"{m.name}.{m.initial_submode}"
    return False


def _obc_transition_reason(obc, prev_name: str, next_name: str, t_s: float) -> str:
    """Determine a concise justification label for an OBC mode switch."""
    if obc is None or obc.state_machine is None:
        return "Switch"
    sm = obc.state_machine
    from GenADCS.backends.python.hardware.OBC_subsystem.state_machine import (
        After,
        Completion,
        FDIRTriggered,
        InFOV,
        Telecommand,
    )

    candidates = [
        tr for tr in sm.transitions
        if _matches_source(tr.source, prev_name) and _matches_target(tr.target, next_name, obc)
    ]
    candidates.sort(key=lambda tr: -tr.priority)
    for tr in candidates:
        trig = tr.trigger
        if isinstance(trig, Telecommand):
            # Prefer Telecommand if a matching timeline event is near t_s, or if it's the only candidate.
            timeline = list(sm.timeline) + list(obc.timeline)
            if any(entry[1] == trig.command and abs(float(entry[0]) - t_s) <= max(1.0, 0.05 * max(1.0, t_s)) for entry in timeline) or len(candidates) == 1:
                return f"TC: {trig.command}"
        elif isinstance(trig, After):
            return f"Auto: After({trig.duration:g}s)"
        elif isinstance(trig, Completion):
            return "Auto: Completion"
        elif isinstance(trig, InFOV):
            return "Swath: ~InFOV" if trig.negate else "Swath: InFOV"
        elif isinstance(trig, FDIRTriggered):
            return f"FDIR: {trig.monitor}" if trig.monitor else "FDIR"

    if candidates and isinstance(candidates[0].trigger, Telecommand):
        return f"TC: {candidates[0].trigger.command}"

    # Check Mode.auto_target.
    for m in sm.modes:
        if m.name == prev_name and m.auto_target and _matches_target(m.auto_target, next_name, obc):
            return "Auto: Completion"
        for sub in m.submodes:
            if f"{m.name}.{sub.name}" == prev_name and sub.auto_target:
                resolved = f"{m.name}.{sub.auto_target}" if "." not in sub.auto_target else sub.auto_target
                if _matches_target(resolved, next_name, obc):
                    return "Auto: Completion"

    # Check FDIR default/monitor target.
    if obc.fdir is not None and obc.fdir.monitors:
        if _matches_target(obc.fdir.safe_mode, next_name, obc):
            return "FDIR"
        for mon in obc.fdir.monitors:
            if mon.target_mode and _matches_target(mon.target_mode, next_name, obc):
                return f"FDIR: {mon.name}"
    return "Switch"


class OBCModePlot(Subplot):
    """Plot OBC modes and submodes as colored horizontal segments with transition justifications."""

    def __init__(self, *, satellite="sat_000", title="OBC Mode & Submode Timeline",
                 max_annotations: int = 20, **kwargs):
        kwargs.setdefault("ylabel", "OBC Mode")
        super().__init__("obc_mode", satellite=satellite, title=title, **kwargs)
        self.max_annotations = max_annotations

    def draw(self, ax, runs, mode):
        import matplotlib.pyplot as plt

        sat_res = runs[0].satellites[self.satellite]
        raw = getattr(sat_res, "obc_mode", None)
        if raw is None:
            raise ValueError(f"obc_mode was not recorded for {self.satellite}")
        times = np.asarray(sat_res.times_s, dtype=float)
        ids = np.asarray(raw[:, 0], dtype=int)

        obc, _, obc_names, _, _ = _extract_sm_metadata(sat_res)
        if not obc_names:
            max_id = int(np.max(ids)) if ids.size else 0
            obc_names = tuple(f"Mode {i}" for i in range(max_id + 1))

        n_modes = len(obc_names)
        main_modes: list[str] = []
        for name in obc_names:
            main = name.split(".", 1)[0]
            if main not in main_modes:
                main_modes.append(main)

        cmap = plt.get_cmap("tab10")
        main_colors = {main: cmap(i % 10) for i, main in enumerate(main_modes)}
        sub_styles = ("-", "--", "-.", ":")

        # Shade main-mode bands and track submode line styles.
        main_y_indices: dict[str, list[int]] = {m: [] for m in main_modes}
        sub_style_for: dict[int, str] = {}
        for idx, name in enumerate(obc_names):
            main = name.split(".", 1)[0]
            sub_idx = len(main_y_indices[main])
            main_y_indices[main].append(idx)
            sub_style_for[idx] = sub_styles[sub_idx % len(sub_styles)]

        for main, y_list in main_y_indices.items():
            ax.axhspan(min(y_list) - 0.4, max(y_list) + 0.4,
                       color=main_colors[main], alpha=0.08, zorder=0)

        # Build contiguous intervals and draw horizontal segments + vertical transitions.
        change_indices = np.where(np.diff(ids) != 0)[0] + 1
        starts = np.concatenate(([0], change_indices))
        ends = np.concatenate((change_indices, [len(ids) - 1]))

        seen_labels: set[str] = set()
        transitions: list[tuple[float, int, int, str]] = []
        for seg_idx, (s, e) in enumerate(zip(starts, ends)):
            m_id = int(ids[s])
            m_name = obc_names[m_id] if 0 <= m_id < n_modes else f"Mode {m_id}"
            main = m_name.split(".", 1)[0]
            color = main_colors.get(main, "C0")
            t0 = float(times[s])
            t1 = float(times[e])
            lbl = m_name if m_name not in seen_labels else None
            seen_labels.add(m_name)
            ax.hlines(m_id, t0, t1, colors=[color], linestyles=sub_style_for.get(m_id, "-"),
                      linewidth=3.0, label=lbl, zorder=3)
            if seg_idx > 0:
                prev_id = int(ids[starts[seg_idx - 1]])
                prev_name = obc_names[prev_id] if 0 <= prev_id < n_modes else f"Mode {prev_id}"
                reason = _obc_transition_reason(obc, prev_name, m_name, t0)
                ax.plot([t0, t0], [prev_id, m_id], color="0.35", linestyle=":",
                        linewidth=1.1, zorder=2)
                transitions.append((t0, prev_id, m_id, reason))

        # Annotate transitions without excessive clutter.
        if len(transitions) <= self.max_annotations:
            to_annotate = transitions
        else:
            seen_reasons: set[str] = set()
            to_annotate = []
            for tr_item in transitions:
                key = f"{tr_item[1]}->{tr_item[2]}:{tr_item[3]}"
                if key not in seen_reasons:
                    seen_reasons.add(key)
                    to_annotate.append(tr_item)
                if len(to_annotate) >= self.max_annotations:
                    break

        for idx, (t_tr, y_prev, y_next, reason) in enumerate(to_annotate):
            y_mid = 0.5 * (y_prev + y_next)
            y_off = 10 if (idx % 2 == 0) else -14
            ax.annotate(
                reason,
                xy=(t_tr, y_mid),
                xytext=(4, y_off),
                textcoords="offset points",
                fontsize=7.5,
                color="0.15",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75", alpha=0.88),
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7),
                zorder=5,
            )

        ax.set_yticks(range(n_modes))
        ax.set_yticklabels(obc_names)
        ax.set_ylim(-0.6, max(0.6, n_modes - 0.4))
        ax.set(title=self.title or "OBC Mode & Submode Timeline",
               xlabel="Time [s]", ylabel=self.ylabel)
        ax.set(**self.axes_kwargs)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, min(3, n_modes // 3)))


class ADCSModePlot(Subplot):
    """Plot ADCS modes as horizontal lines with vertical markers for active Goal changes."""

    def __init__(self, *, satellite="sat_000", title="ADCS Mode & Goal Transitions",
                 max_goal_annotations: int = 20, **kwargs):
        kwargs.setdefault("ylabel", "ADCS Mode")
        super().__init__("adcs_mode", satellite=satellite, title=title, **kwargs)
        self.max_goal_annotations = max_goal_annotations

    def draw(self, ax, runs, mode):
        import matplotlib.pyplot as plt

        sat_res = runs[0].satellites[self.satellite]
        raw_mode = getattr(sat_res, "adcs_mode", None)
        if raw_mode is None:
            raise ValueError(f"adcs_mode was not recorded for {self.satellite}")
        times = np.asarray(sat_res.times_s, dtype=float)
        mode_ids = np.asarray(raw_mode[:, 0], dtype=int)
        raw_goal = getattr(sat_res, "adcs_goal", None)
        goal_ids = np.asarray(raw_goal[:, 0], dtype=int) if raw_goal is not None else None

        _, _, _, adcs_names, goal_names = _extract_sm_metadata(sat_res)
        if not adcs_names:
            max_id = int(np.max(mode_ids)) if mode_ids.size else 0
            adcs_names = tuple(f"ADCS {i}" for i in range(max_id + 1))

        n_modes = len(adcs_names)
        cmap = plt.get_cmap("tab10")
        mode_colors = {i: cmap(i % 10) for i in range(n_modes)}

        change_idx = np.where(np.diff(mode_ids) != 0)[0] + 1
        starts = np.concatenate(([0], change_idx))
        ends = np.concatenate((change_idx, [len(mode_ids) - 1]))

        seen_modes: set[int] = set()
        for seg_idx, (s, e) in enumerate(zip(starts, ends)):
            m_id = int(mode_ids[s])
            m_name = adcs_names[m_id] if 0 <= m_id < n_modes else f"ADCS {m_id}"
            lbl = f"Mode: {m_name}" if m_id not in seen_modes else None
            seen_modes.add(m_id)
            ax.hlines(m_id, float(times[s]), float(times[e]),
                      colors=[mode_colors.get(m_id, "C0")], linewidth=3.2,
                      label=lbl, zorder=3)
            if seg_idx > 0:
                prev_id = int(mode_ids[starts[seg_idx - 1]])
                t_tr = float(times[s])
                ax.plot([t_tr, t_tr], [prev_id, m_id], color="0.35",
                        linestyle=":", linewidth=1.1, zorder=2)

        # Vertical lines for Goal changes.
        if goal_ids is not None and goal_ids.size:
            goal_changes = np.concatenate(([0], np.where(np.diff(goal_ids) != 0)[0] + 1))
            goal_cmap = plt.get_cmap("Set2")
            seen_goals: set[int] = set()
            for idx, gc in enumerate(goal_changes):
                g_id = int(goal_ids[gc])
                g_name = goal_names[g_id] if 0 <= g_id < len(goal_names) else f"Goal {g_id}"
                g_color = goal_cmap(g_id % 8)
                t_g = float(times[gc])
                g_lbl = f"Goal: {g_name}" if g_id not in seen_goals else None
                seen_goals.add(g_id)
                ax.axvline(t_g, color=g_color, linestyle="--", linewidth=1.4,
                           alpha=0.85, label=g_lbl, zorder=1)
                if len(goal_changes) <= self.max_goal_annotations:
                    y_pos = n_modes - 0.35 if (idx % 2 == 0) else -0.35
                    ax.annotate(
                        g_name,
                        xy=(t_g, y_pos),
                        xytext=(3, 0),
                        textcoords="offset points",
                        fontsize=7.0,
                        color="0.2",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=g_color, alpha=0.85),
                        zorder=4,
                    )

        ax.set_yticks(range(n_modes))
        ax.set_yticklabels(adcs_names)
        ax.set_ylim(-0.6, max(0.6, n_modes - 0.4))
        ax.set(title=self.title or "ADCS Mode & Goal Transitions",
               xlabel="Time [s]", ylabel=self.ylabel)
        ax.set(**self.axes_kwargs)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)


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
    from .close_all import ensure_close_all_button
    ensure_close_all_button()
    flat = np.atleast_1d(axes).ravel()
    for ax, panel in zip(flat, subplots):
        try:
            panel.draw(ax, runs, mode)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            ax.clear()
            ax.set_title(panel.title or panel.field.replace("_", " ").title())
            ax.text(0.5, 0.5, f"Unable to plot data:\n{exc}",
                    ha="center", va="center", wrap=True,
                    transform=ax.transAxes, color="crimson")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    for ax in flat[len(subplots):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if show:
        plt.show(block=False)
    return fig, axes
