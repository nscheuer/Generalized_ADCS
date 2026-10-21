__all__ = ["DensityModel"]

import numpy as np
from typing import List
import numpy as np
from numpy.typing import NDArray

# --- Default SMAD model data (Simple Model of Atmospheric Density) ---
SMAD_altrange: NDArray[np.float64] = np.array([
    0, 100, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 550,
    600, 650, 700, 750, 800, 850, 900, 950, 1000
], dtype=float)  # km

SMAD_rhovsalt: NDArray[np.float64] = np.array([
    1.2, 5.69e-7, 2.02e-9, 7.66e-10, 2.90e-10, 1.46e-10, 7.30e-11, 4.10e-11,
    2.30e-11, 1.38e-11, 8.33e-12, 5.24e-12, 3.29e-12, 1.39e-12, 6.15e-13,
    2.84e-13, 1.37e-13, 6.87e-14, 3.63e-14, 2.02e-14, 1.21e-14, 7.69e-15,
    5.24e-15, 3.78e-15, 2.86e-15
], dtype=float)  # kg/m³

class DensityModel:
    r"""
    Simple atmospheric density interpolation model.

    This class implements a lightweight empirical atmospheric density model
    based on tabulated altitude–density reference data. The density
    :math:`\rho(h)` is obtained by interpolating the logarithm of density
    between known data points, making the model suitable for preliminary
    orbital dynamics and drag analyses.

    The model is conceptually equivalent to a simplified version of standard
    atmospheric models (e.g. SMAD – *Space Mission Analysis and Design*),
    where density is treated as a deterministic function of altitude.

    Mathematical Model
    ------------------
    Given a discrete set of altitude–density pairs

    .. math::

        \{(h_i, \rho_i)\}_{i=0}^{N-1},

    the density at an arbitrary altitude :math:`h` is computed via linear
    interpolation:

    .. math::

        \rho(h) = \exp\left(
        \ln\rho_i +
        \frac{\ln\rho_{i+1} - \ln\rho_i}{h_{i+1} - h_i}
        \left(h - h_i\right)\right),
        \quad h_i \le h \le h_{i+1}.

    This density model is commonly used in aerodynamic drag formulations:

    .. math::

        F_D = \frac{1}{2} C_D A \rho(h) v^2,

    where :math:`C_D` is the drag coefficient, :math:`A` is the reference area,
    and :math:`v` is the relative velocity magnitude.

    :param altitude_range:
        Reference altitude samples :math:`h_i` in kilometers (km).
        Values must be non-negative and strictly ordered.
    :type altitude_range: numpy.typing.NDArray[numpy.float64]

    :param rho_range:
        Atmospheric density samples :math:`\rho_i` in kg/m³ corresponding
        to ``altitude_range``.
        All values must be strictly positive.
    :type rho_range: numpy.typing.NDArray[numpy.float64]

    :raises ValueError:
        If the altitude and density arrays have mismatched shapes,
        contain negative altitudes, or contain non-positive density values.

    .. note::

        This model does **not** account for temporal, solar, geomagnetic,
        or latitudinal variations in atmospheric density. It is intended
        for simplified analyses and educational use.

    """

    def __init__(
        self,
        altitude_range: NDArray[np.float64] = SMAD_altrange,
        rho_range: NDArray[np.float64] = SMAD_rhovsalt,
    ) -> None:
        r"""
        Initialize the atmospheric density model.

        This constructor stores and validates the altitude–density reference
        data used by the interpolation model. The provided arrays define the
        discrete function :math:`\rho(h)` that is interpolated by
        :meth:`~ADCS.etc.DensityModel.interpolate`.

        :param altitude_range:
            Array of reference altitudes :math:`h_i` in kilometers (km).
            Must be non-negative and have the same shape as ``rho_range``.
        :type altitude_range: numpy.typing.NDArray[numpy.float64]

        :param rho_range:
            Array of atmospheric densities :math:`\rho_i` in kg/m³.
            Must be strictly positive and match ``altitude_range`` in size.
        :type rho_range: numpy.typing.NDArray[numpy.float64]

        :raises ValueError:
            If array shapes do not match, altitudes are negative,
            or densities are non-positive.

        :return: None
        :rtype: None

        """

        # Store references (or copies if you prefer)
        self.altitude_range = np.array(altitude_range, dtype=float)
        self.rho_range = np.array(rho_range, dtype=float)

        # --- Validation ---
        if self.altitude_range.shape != self.rho_range.shape:
            raise ValueError("altitude_range and rho_range must have the same shape.")
        if np.any(self.altitude_range < 0):
            raise ValueError("Altitude values must be non-negative.")
        if np.any(self.rho_range <= 0):
            raise ValueError("Density values must be positive.")
        self._log_rho_range = np.log(self.rho_range)

    def interpolate(self, altitude_km: float) -> float:
        r"""
        Interpolate atmospheric density at a given altitude.

        This method evaluates the atmospheric density :math:`\rho(h)` at a
        specified altitude using log linear interpolation over the stored
        reference data.

        Mathematically, for an altitude :math:`h` between two reference points
        :math:`h_i` and :math:`h_{i+1}`, the density is given by

        .. math::

            \rho(h) = \exp\left(
            \ln\rho_i +
            \frac{\ln\rho_{i+1} - \ln\rho_i}{h_{i+1} - h_i}
            (h - h_i)\right).

        The atmosphere is approximately *exponential* in altitude, so the
        interpolation is performed **log-linearly** (linear in :math:`\ln\rho`
        vs. altitude); plain linear-in-altitude interpolation over a sparse
        table is wrong by orders of magnitude between samples (e.g. ~600x at
        50 km). Above the top reference altitude the density continues to
        **decay exponentially** using the scale height of the last table
        interval (so e.g. GEO does not retain a spurious LEO-tail density);
        below the lowest reference altitude the density is clamped to the
        lowest sample (sub-surface / decayed-orbit guard).

        :param altitude_km:
            Altitude above Earth’s mean radius in kilometers (km).
        :type altitude_km: float

        :return:
            Interpolated atmospheric density :math:`\rho(h)` in kg/m³.
        :rtype: float

        """
        h = float(altitude_km)
        alt = self.altitude_range
        rho = self.rho_range
        log_rho = self._log_rho_range

        if h <= alt[0]:
            # Below the table (incl. sub-surface / negative): clamp, no crash.
            return float(rho[0])
        if h >= alt[-1]:
            # Continue the exponential decay of the last interval rather than
            # holding a constant (the old constant extrapolation gave high
            # orbits a spurious non-zero LEO-tail density).
            denom = log_rho[-2] - log_rho[-1]
            if denom <= 0.0:
                return float(rho[-1])
            scale_height = (alt[-1] - alt[-2]) / denom  # km, > 0
            return float(rho[-1] * np.exp(-(h - alt[-1]) / scale_height))

        # Interior: linear in ln(rho) vs altitude == exponential interpolation.
        return float(np.exp(np.interp(h, alt, log_rho)))

    def interpolate_many(self, altitude_km):
        r"""Interpolate atmospheric density for an array of altitudes.

        Values within the table use log linear interpolation. Values above
        the table continue with the last interval's exponential scale height.

        :param altitude_km: Altitudes above Earth's mean radius [km].
        :type altitude_km: numpy.ndarray
        :return: Atmospheric mass densities [kg/m³].
        :rtype: numpy.ndarray
        """
        heights = np.asarray(altitude_km, dtype=float)
        log_density = np.interp(heights, self.altitude_range, self._log_rho_range)
        density = np.exp(log_density)
        above = heights >= self.altitude_range[-1]
        if np.any(above):
            log_rho = self._log_rho_range
            denominator = log_rho[-2] - log_rho[-1]
            if denominator > 0.0:
                scale_height = (self.altitude_range[-1] - self.altitude_range[-2]) / denominator
                density = np.where(
                    above,
                    self.rho_range[-1] * np.exp(-(heights - self.altitude_range[-1]) / scale_height),
                    density,
                )
            else:
                density = np.where(above, self.rho_range[-1], density)
        return density

    def __repr__(self) -> str:
        r"""
        Return a concise string representation of the density model.

        The representation reports only the number of reference
        altitude–density samples stored in the model, making it suitable
        for debugging and logging without exposing large numerical arrays.

        :return:
            Human-readable summary string of the form
            ``DensityModel(n=<number_of_samples>)``.
        :rtype: str

        """

        return f"DensityModel(n={len(self.altitude_range)})"
