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
    :math:`\rho(h)` is obtained by one–dimensional linear interpolation
    between known data points, making the model suitable for preliminary
    orbital dynamics and drag analyses.

    The model is conceptually equivalent to a simplified version of standard
    atmospheric models (e.g. SMAD – *Space Mission Analysis and Design*),
    where density is assumed to be a deterministic and monotonically
    decreasing function of altitude.

    Mathematical Model
    ------------------
    Given a discrete set of altitude–density pairs

    .. math::

        \{(h_i, \rho_i)\}_{i=0}^{N-1},

    the density at an arbitrary altitude :math:`h` is computed via linear
    interpolation:

    .. math::

        \rho(h) =
        \rho_i +
        \frac{\rho_{i+1} - \rho_i}{h_{i+1} - h_i}
        \left(h - h_i\right),
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
        Interpolate atmospheric density at a given altitude.

        This method evaluates the atmospheric density :math:`\rho(h)` at a
        specified altitude using linear interpolation over the stored
        reference data.

        Mathematically, for an altitude :math:`h` between two reference points
        :math:`h_i` and :math:`h_{i+1}`, the density is given by

        .. math::

            \rho(h) =
            \rho_i +
            \frac{\rho_{i+1} - \rho_i}{h_{i+1} - h_i}
            (h - h_i).

        If :math:`h` lies outside the reference range, the boundary density
        values are returned (constant extrapolation).

        :param altitude_km:
            Altitude above Earth’s mean radius in kilometers (km).
        :type altitude_km: float

        :return:
            Interpolated atmospheric density :math:`\rho(h)` in kg/m³.
        :rtype: float

        """

        return float(np.interp(altitude_km, self.altitude_range, self.rho_range))

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