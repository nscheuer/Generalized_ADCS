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
    Represents a simple atmospheric density model for interpolating between
    tabulated altitude–density pairs.

    This class provides a lightweight empirical model for approximating
    atmospheric density :math:`\rho(h)` as a function of altitude above
    Earth’s surface. It performs 1D linear interpolation over a predefined
    dataset of altitude and density values, such as those derived from
    standard atmosphere models (e.g., *SMAD*).

    Attributes
    ----------
    altitude_range : ndarray of float
        Array of reference altitude points :math:`h_i` [km].
    rho_range : ndarray of float
        Corresponding air density values :math:`\rho_i` [kg/m³].

    Notes
    -----
    The interpolation assumes a monotonic relationship between altitude and
    density, i.e. :math:`\frac{d\rho}{dh} < 0`.  
    It is suitable for use in orbital dynamics models and drag force
    calculations of low-Earth orbit satellites:

    .. math::

        F_D = \tfrac{1}{2} C_D A \rho(h) v^2

    where :math:`C_D` is the drag coefficient, :math:`A` is the reference area,
    and :math:`v` is the relative velocity magnitude.
    """


    def __init__(
        self,
        altitude_range: NDArray[np.float64] = SMAD_altrange,
        rho_range: NDArray[np.float64] = SMAD_rhovsalt,
    ) -> None:
        r"""
        Initialize the atmospheric density model with altitude–density data.

        Parameters
        ----------
        altitude_range : ndarray of float, optional
            Array of altitude values :math:`h_i` in **kilometers (km)**.
            Must be strictly non-negative and of the same length as `rho_range`.
        rho_range : ndarray of float, optional
            Corresponding atmospheric density values :math:`\rho_i` in **kg/m³**.
            Must be strictly positive and match the shape of `altitude_range`.

        Raises
        ------
        ValueError
            If `altitude_range` and `rho_range` have mismatched shapes,
            contain negative altitudes, or non-positive density values.

        Notes
        -----
        The model represents a simple 1D interpolation curve :math:`\rho(h)`
        using tabulated reference data, such as from *SMAD* (Space Mission Analysis and Design).
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

    def interpolate(self, altitude_km: float) -> float:
        r"""
        Interpolate and return the atmospheric density at a given altitude.

        Parameters
        ----------
        altitude_km : float
            Altitude above Earth's mean radius [km].

        Returns
        -------
        float
            Interpolated atmospheric density :math:`\rho(h)` [kg/m³].

        Notes
        -----
        The interpolation is performed linearly in altitude–density space:

        .. math::

            \rho(h) = \rho_i + \frac{(\rho_{i+1} - \rho_i)}{(h_{i+1} - h_i)} (h - h_i)

        where :math:`(h_i, \rho_i)` are tabulated reference points.
        """

        return float(np.interp(altitude_km, self.altitude_range, self.rho_range))

    def __repr__(self) -> str:
        r"""
        Return a concise string representation of the model.

        Returns
        -------
        str
            String summary showing the number of altitude–density samples, e.g.:

            ``'DensityModel(n=50)'``

        Notes
        -----
        Intended for debugging and logging. Does not display array contents.
        """

        return f"DensityModel(n={len(self.altitude_range)})"