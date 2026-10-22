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

# --- Solar-activity amplification ---
# Thermospheric density swings strongly over the ~11-year solar cycle, and the
# swing grows with altitude (negligible low down, ~2 orders of magnitude near
# 1000 km). The baseline table above is taken as quiet-Sun (solar-minimum); a
# multiplicative amplification carries it to active-Sun (solar-maximum):
#     rho(h, level) = rho_quiet(h) * SolarMaxAmplification(h) ** level,   level in [0, 1]
# (level = 0 -> quiet/min, 1 -> active/max). The factors below are representative
# solar-max/solar-min density ratios vs altitude from standard thermosphere
# behaviour (Jacchia / CIRA / NRLMSISE-class models) -- an engineering
# approximation, not a substitute for a full F10.7/Ap-driven model.
SOLAR_AMP_altrange: NDArray[np.float64] = np.array(
    [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)  # km
SOLAR_MAX_AMPLIFICATION: NDArray[np.float64] = np.array(
    [1.0, 1.0, 1.6, 3.0, 6.0, 12.0, 25.0, 45.0, 75.0, 110.0, 150.0], dtype=float)


def solar_amplification(altitude_km, solar_level: float,
                        amp_alt: NDArray[np.float64] = SOLAR_AMP_altrange,
                        amp_max: NDArray[np.float64] = SOLAR_MAX_AMPLIFICATION):
    r"""
    Multiplicative density factor for a solar-activity ``solar_level`` in [0, 1]
    (0 = quiet/solar-minimum, 1 = active/solar-maximum), log-interpolated in
    altitude. Returns 1.0 at ``solar_level == 0`` (quiet baseline).

    :param altitude_km: Altitude(s) above mean Earth radius [km] (scalar or array).
    :param solar_level: Solar activity level in [0, 1].
    :return: Density amplification factor (same shape as ``altitude_km``).
    """
    amp = np.exp(np.interp(np.asarray(altitude_km, dtype=float), amp_alt, np.log(amp_max)))
    return amp ** float(solar_level)


def f107_to_solar_level(f107: float, f107_quiet: float = 70.0, f107_active: float = 250.0) -> float:
    r"""
    Map a 10.7 cm solar radio flux (F10.7 [sfu]) to a solar-activity level in
    [0, 1] (clamped): ~70 sfu (quiet) -> 0, ~250 sfu (very active) -> 1.
    """
    return float(np.clip((f107 - f107_quiet) / (f107_active - f107_quiet), 0.0, 1.0))


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
        solar_level: float = 0.0,
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
            Array of atmospheric densities :math:`\rho_i` in kg/m³, interpreted
            as the **quiet-Sun (solar-minimum)** profile.
            Must be strictly positive and match ``altitude_range`` in size.
        :type rho_range: numpy.typing.NDArray[numpy.float64]

        :param solar_level:
            Solar-activity level in [0, 1] (0 = quiet/solar-minimum, the default
            and the legacy behavior; 1 = active/solar-maximum). Density is scaled
            by the altitude-dependent :func:`solar_amplification` raised to this
            level. Use :meth:`for_solar_level` / :meth:`from_f107` for clarity.
        :type solar_level: float

        :raises ValueError:
            If array shapes do not match, altitudes are negative,
            densities are non-positive, or ``solar_level`` is negative.

        """

        # Store references (or copies if you prefer)
        self.altitude_range = np.array(altitude_range, dtype=float)
        self.rho_range = np.array(rho_range, dtype=float)
        self.solar_level = float(solar_level)

        # --- Validation ---
        if self.altitude_range.shape != self.rho_range.shape:
            raise ValueError("altitude_range and rho_range must have the same shape.")
        if np.any(self.altitude_range < 0):
            raise ValueError("Altitude values must be non-negative.")
        if np.any(self.rho_range <= 0):
            raise ValueError("Density values must be positive.")
        if self.solar_level < 0.0:
            raise ValueError("solar_level must be >= 0 (0 = quiet, 1 = solar max).")

    @classmethod
    def for_solar_level(cls, solar_level: float, **kwargs) -> "DensityModel":
        r"""Build a model for a solar-activity level in [0, 1] (0 = min, 1 = max)."""
        return cls(solar_level=solar_level, **kwargs)

    @classmethod
    def from_f107(cls, f107: float, **kwargs) -> "DensityModel":
        r"""Build a model from a 10.7 cm solar flux F10.7 [sfu] (see :func:`f107_to_solar_level`)."""
        return cls(solar_level=f107_to_solar_level(f107), **kwargs)

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

        The atmosphere is approximately *exponential* in altitude, so the
        interpolation is performed **log-linearly** (linear in :math:`\ln\rho`
        vs. altitude); plain linear-in-altitude interpolation over a sparse
        table is wrong by orders of magnitude between samples (e.g. ~600x at
        50 km). Above the top reference altitude the density continues to
        **decay exponentially** using the scale height of the last table
        interval (so e.g. GEO does not retain a spurious LEO-tail density);
        below the lowest reference altitude the density is clamped to the
        lowest sample (sub-surface / decayed-orbit guard).

        When ``solar_level > 0`` the quiet-Sun density is multiplied by the
        altitude-dependent :func:`solar_amplification` factor (so e.g. ~650 km
        density rises ~30x from solar minimum to solar maximum).

        Accepts a scalar or an array of altitudes; the return shape matches the
        input (a scalar in gives a Python ``float`` out, for back-compatibility).

        :param altitude_km:
            Altitude(s) above Earth’s mean radius in kilometers (km).
        :type altitude_km: float or numpy.ndarray

        :return:
            Interpolated atmospheric density :math:`\rho(h)` in kg/m³.
        :rtype: float or numpy.ndarray

        """
        h = np.asarray(altitude_km, dtype=float)
        scalar = (h.ndim == 0)
        hf = np.atleast_1d(h)
        alt = self.altitude_range
        rho = self.rho_range
        log_rho = np.log(rho)

        # Interior + below-table clamp: np.interp clamps h<=alt[0] to log_rho[0]
        # (== rho[0]) and is log-linear in between (exponential interpolation).
        out = np.exp(np.interp(hf, alt, log_rho))

        # Above the top sample: continue the exponential decay of the last
        # interval (matches the legacy scalar behavior) rather than holding flat.
        above = hf >= alt[-1]
        if np.any(above):
            denom = log_rho[-2] - log_rho[-1]
            if denom > 0.0:
                scale_height = (alt[-1] - alt[-2]) / denom  # km, > 0
                out = np.where(above, rho[-1] * np.exp(-(hf - alt[-1]) / scale_height), out)
            else:
                out = np.where(above, rho[-1], out)

        # Solar-cycle amplification (1.0 at solar_level == 0 -> legacy behavior).
        if self.solar_level:
            out = out * solar_amplification(hf, self.solar_level)

        out = out.reshape(h.shape)
        return float(out) if scalar else out

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