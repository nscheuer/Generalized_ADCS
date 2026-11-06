__all__ = ["SunSensorWithEarthshine"]

from typing import Optional

import numpy as np

from ADCS.helpers.math_helpers import normalize, normed_vec_jac
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.state import State

from .sunsensor import SunSensor


class SunSensorWithEarthshine(SunSensor):
    r"""
    Coarse Sun sensor that also sees sunlight reflected off the Earth.

    A cosine-law Sun sensor cannot tell direct sunlight from earthshine. When
    the sensor faces the sunlit side of the Earth it reads an extra current,
    which is a real and attitude-dependent error source for coarse sensors in
    low Earth orbit. This subclass adds that term to the base
    :class:`~ADCS.satellite_hardware.sensors.sunsensor.SunSensor`, which is
    otherwise unchanged (field of view, efficiency, bias, noise, eclipse
    handling and estimator interface are all inherited).

    The Earth is treated as a Lambertian reflector seen through its
    sub-satellite point, with a uniform albedo :math:`a`. With the nadir
    direction in the body frame :math:`\hat{\mathbf{n}} = -\mathbf{r}/|\mathbf{r}|`,
    the Sun direction from the Earth's centre :math:`\hat{\mathbf{s}}_\oplus`,
    the Earth radius :math:`R_\oplus` and the sensor axis :math:`\mathbf{a}`,
    the earthshine reading is

    .. math::

        y_\oplus = \eta \, a \left(\frac{R_\oplus}{|\mathbf{r}|}\right)^2
        \max(-\hat{\mathbf{n}}\cdot\hat{\mathbf{s}}_\oplus,\,0)\,
        \max(\mathbf{a}\cdot\hat{\mathbf{n}},\,0),

    and the full clean reading is the base Sun term plus :math:`y_\oplus`.
    The first factor is the fraction of solar irradiance reflected, the
    second the solid-angle fall-off with altitude, the third the solar
    elevation at the sub-satellite point (zero over the night side), and
    the last the cosine law of the sensor itself. For a 500 km orbit and
    :math:`a = 0.3` the term peaks near a quarter of the direct-Sun reading,
    which is the size of the effect seen in flight.

    What this model leaves out: the integration over the visible disc (the
    illuminated part is not a point at nadir), the latitude and cloud
    dependence of the albedo, and the sunlit crescent that is still visible
    from inside the umbra. In eclipse the reading keeps the base class's
    ``NaN`` sentinel, so the sensor is masked exactly as before.

    Only the nadir direction depends on attitude; the solar elevation and the
    altitude factor are scalars built from body-frame vectors that rotate
    together. The Jacobian therefore adds one term, through
    :math:`\partial\hat{\mathbf{n}}/\partial\mathbf{q}`, to the base Jacobian.

    To study earthshine as an unmodelled error, give the truth satellite this
    sensor and the estimator's satellite a plain
    :class:`~ADCS.satellite_hardware.sensors.sunsensor.SunSensor` with the
    same axis and efficiency. With ``albedo=0`` the two are identical.

    Parameters
    ----------
    axis, efficiency, sample_time, bias, noise, estimate_bias
        As for :class:`~ADCS.satellite_hardware.sensors.sunsensor.SunSensor`.
    albedo : float
        Bond albedo :math:`a` of the Earth, dimensionless. 0.3 is the usual
        global mean.
    earth_radius : float
        Earth radius :math:`R_\oplus` in km, defaulting to
        :attr:`~ADCS.orbits.universal_constants.EarthConstants.R_e`.
    """

    def __init__(
        self,
        axis: np.ndarray,
        efficiency: float,
        sample_time: float = 0.1,
        bias: Optional[Bias] = None,
        noise: Optional[Noise] = None,
        estimate_bias: bool = False,
        *,
        albedo: float = 0.3,
        earth_radius: float = EarthConstants.R_e,
    ):
        albedo = float(albedo)
        earth_radius = float(earth_radius)
        if not np.isfinite(albedo) or albedo < 0.0:
            raise ValueError("albedo must be finite and non-negative")
        if not np.isfinite(earth_radius) or earth_radius <= 0.0:
            raise ValueError("earth_radius must be finite and positive")
        self.albedo = albedo
        self.earth_radius = earth_radius
        super().__init__(
            axis=axis,
            efficiency=efficiency,
            sample_time=sample_time,
            bias=bias,
            noise=noise,
            estimate_bias=estimate_bias,
        )

    def _earthshine_scale(self, os: Orbital_State) -> float:
        r"""Attitude-independent part :math:`\eta a (R_\oplus/|r|)^2 \max(-\hat n\cdot\hat s_\oplus, 0)`, from inertial vectors."""
        r_eci = np.asarray(os.R, dtype=float)
        s_eci = np.asarray(os.S, dtype=float)
        r_norm = np.linalg.norm(r_eci)
        solar_elevation = np.maximum(np.dot(r_eci, s_eci) / (r_norm * np.linalg.norm(s_eci)), 0.0)
        return self.efficiency * self.albedo * (self.earth_radius / r_norm) ** 2 * solar_elevation

    def earthshine_reading(self, x: State, os: Orbital_State) -> float:
        r"""The earthshine term :math:`y_\oplus` alone, ignoring eclipse."""
        vecs = os.get_state_vector(x=x)
        nadir = normalize(-vecs["r"])
        return self._earthshine_scale(os) * np.maximum(np.dot(self.axis, nadir), 0.0)

    def clean_reading(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""Base Sun reading plus the earthshine term; ``NaN`` in eclipse as for the base class."""
        reading = super().clean_reading(x, os)
        if not os.is_sunlit():
            return reading
        return reading + self.earthshine_reading(x, os)

    def basestate_jac(self, x: State, os: Orbital_State) -> np.ndarray:
        r"""Base Jacobian plus :math:`\partial y_\oplus/\partial\mathbf{q}` through the nadir direction."""
        jac = super().basestate_jac(x, os)
        if not os.is_sunlit():
            return jac
        vecs = os.get_state_vector(x=x)
        nadir = normalize(-vecs["r"])
        dnadir__dq = normed_vec_jac(-vecs["r"], -vecs["dr"])
        facing = np.dot(self.axis, nadir) > 0.0
        dearth__dq = self._earthshine_scale(os) * facing * (dnadir__dq @ self.axis)
        return jac + np.vstack([np.zeros((3, 1)), np.expand_dims(dearth__dq, 1)])
