__all__ = ["DensityNoise"]

import numpy as np
from typing import Sequence

from ADCS.satellite_hardware.errors.noise import Noise


class DensityNoise(Noise):
    r"""
    Noise specified as a **noise density**, converted for the model's averaging interval.

    Data sheets quote sensor noise as a density: a gyro's angle random walk
    in deg/√h or rad/s/√Hz, a magnetometer's noise in nT/√Hz. A reading
    averaged over an interval :math:`\Delta t` then has standard
    deviation :math:`\eta/\sqrt{\Delta t}`: shorter averages are noisier and
    longer averages are cleaner. This interval describes the measurement
    represented by the noise model; it need not match how often software polls
    a sensor. A plain
    :class:`~ADCS.satellite_hardware.errors.noise.Noise` has one standard
    deviation whatever the rate, which is right for quantisation and
    electronics noise but wrong for the density part.

    This subclass takes the density and the sample time and produces the
    per-reading standard deviation

    .. math::

        \sigma = \sqrt{\sigma_n^2 + \eta^2 / \Delta t},

    where :math:`\sigma_n` (``std_noise``) is the rate-independent floor. The
    result is stored in ``std_noise`` of the base class, so every consumer
    (the reading draw, the measurement covariance the estimator builds, the
    warm-start weights) sees the scaled value with no change.
    :meth:`at_sample_time` re-derives the model for another interval.

    Parameters
    ----------
    density : float or array
        Noise density :math:`\eta` in reading units per root hertz
        (equivalently reading units times :math:`\sqrt{\mathrm{s}}`).
    sample_time : float
        Averaging interval :math:`\Delta t` in seconds. ``density`` must be
        in reading units per :math:`\sqrt{\mathrm{Hz}}` (reading units times
        :math:`\sqrt{\mathrm{s}}`). Convert densities quoted per another time
        unit before passing them.
    std_noise : float or array, optional
        Rate-independent part :math:`\sigma_n` of the standard deviation (default 0).
    noise : float or array, optional
        Mean or nominal offset :math:`n_0` (default 0).
    bounds : (float, float), optional
        Lower and upper limits applied after sampling.
    """

    def __init__(
        self,
        density: np.ndarray | float,
        sample_time: float,
        *,
        std_noise: np.ndarray | float = 0.0,
        noise: np.ndarray | float = 0.0,
        bounds: Sequence[np.ndarray | float] = (-np.array([np.inf]), np.array([np.inf])),
    ) -> None:
        density = np.atleast_1d(np.asarray(density, dtype=float))
        floor = np.atleast_1d(np.asarray(std_noise, dtype=float))
        try:
            density, floor = np.broadcast_arrays(density, floor)
        except ValueError:
            raise ValueError("density and std_noise must be broadcastable to the same shape.")
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("density must be finite and non-negative.")
        if np.any(~np.isfinite(floor)) or np.any(floor < 0.0):
            raise ValueError("std_noise must be finite and non-negative.")
        sample_time = float(sample_time)
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be positive and finite: a noise density only gives a standard deviation at a rate.")
        self.density = density.copy()
        self.std_floor = floor.copy()
        self.sample_time = sample_time
        super().__init__(noise=noise, std_noise=np.sqrt(floor**2 + density**2 / sample_time), bounds=bounds)

    def at_sample_time(self, sample_time: float) -> "DensityNoise":
        """The same density model evaluated for another sample interval."""
        return DensityNoise(self.density, sample_time, std_noise=self.std_floor, noise=self.noise, bounds=self.bounds)

    def broadcast_to(self, length: int, *, owner: str = "sensor") -> "DensityNoise":
        """Size this model for a ``length``-output sensor (see :meth:`Noise.broadcast_to`)."""
        size = int(np.asarray(self.std_noise).size)
        if size == length:
            return self.copy()
        if size != 1:
            raise ValueError(f"{owner} noise model has {size} entries for a {length}-output sensor")

        def repeat(value):
            array = np.asarray(value, dtype=float).reshape(-1)
            return np.full(length, array[0]) if array.size == 1 else array.copy()

        return DensityNoise(
            repeat(self.density),
            self.sample_time,
            std_noise=repeat(self.std_floor),
            noise=repeat(self.noise),
            bounds=(repeat(self.bounds[0]), repeat(self.bounds[1])),
        )

    def copy(self) -> "DensityNoise":
        return DensityNoise(self.density, self.sample_time, std_noise=self.std_floor, noise=self.noise, bounds=self.bounds)
