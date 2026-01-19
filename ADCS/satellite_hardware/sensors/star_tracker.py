__all__ = ["StarTracker"]

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ADCS.satellite_hardware.sensors.sensor import Sensor
from ADCS.environment import StarCatalog, NavigationStar
from ADCS.satellite_hardware.actuators import Bias, AnisotropicNoise
from ADCS.satellite_hardware.disturbances.disturbance_mode import DisturbanceMode
from ADCS.helpers.math_helpers import drotmatTvecdq, rot_mat
from ADCS.orbits.orbital_state import Orbital_State

class StarTracker(Sensor):
    output_length: int = 3

    def __init__(self, 
        sample_time: float = 0.1, 
        bias: Bias = None, 
        anisotropic_noise: AnisotropicNoise = None, 
        estimate_bias: bool = False,
        boresight: np.ndarray = np.array([0.0, 0.0, 1.0]),
        fov: float = np.deg2rad(4.0),
        sun_exclusion: float = np.deg2rad(25.0),
        star_catalog: Optional[StarCatalog] = None
    ) -> None:
        self.boresight = np.asarray(boresight, dtype=np.float64)
        self.boresight = self.boresight / np.linalg.norm(self.boresight)

        self.fov = float(fov)
        self.sun_exclusion = float(sun_exclusion)
        
        self.catalog = star_catalog if star_catalog is not None else StarCatalog()
        self.current_star: Optional[NavigationStar] = None

        self._R_noise = self._build_noise_rotation()

        super().__init__(
            sample_time=sample_time,
            output_length=3,
            bias=bias,
            noise=anisotropic_noise,
            estimate_bias=estimate_bias
        )

    def _build_noise_rotation(self) -> NDArray[np.float64]:
        z = np.array([0.0, 0.0, 1.0])

        if np.allclose(self.boresight, z):
            return np.eye(3)
        if np.allclose(self.boresight, -z):
            return np.diag([1.0, -1.0, -1.0])

        v = np.cross(z, self.boresight)
        s = np.linalg.norm(v)
        c = np.dot(z, self.boresight)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        return R

    def _get_sun_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        if hasattr(os, 'S') and os.S is not None:
            s = np.asarray(os.S, dtype=np.float64)
            if not np.allclose(s, 0):
                return s
        return None

    def _get_moon_eci(self, os: Orbital_State) -> Optional[NDArray[np.float64]]:
        try:
            if hasattr(os, 'ephem') and os.ephem is not None:
                moon = os.ephem.planets['moon']
                moon_icrf = os.ephem.earth.at(os.sf_pos.t).observe(moon).apparent()
                return np.asarray(moon_icrf.position.km, dtype=np.float64)
        except (KeyError, AttributeError):
            pass
        return None

    def _select_star(self, q: NDArray[np.float64], os: Orbital_State) -> Optional[NavigationStar]:
        A = rot_mat(q)
        boresight_eci = A @ self.boresight
        r_sat_eci = os.R
        sun_eci = self._get_sun_eci(os)
        moon_eci = self._get_moon_eci(os)

        visible = self.catalog.get_visible_stars(
            boresight_eci=boresight_eci,
            fov_rad=self.fov,
            r_sat_eci=r_sat_eci,
            sun_eci=sun_eci,
            moon_eci=moon_eci,
            sun_exclusion_rad=self.sun_exclusion
        )

        if not visible:
            return None

        return min(visible, key=lambda s: s.vmag)

    def clean_reading(self, x: NDArray[np.float64], os: Orbital_State) -> NDArray[np.float64]:
        q = x[3:7]
        star = self._select_star(q, os)
        
        if star is None:
            self.current_star = None
            return np.full(3, np.nan)

        self.current_star = star
        A = rot_mat(q)
        return A.T @ star.s_eci

    def reading(self, x: NDArray[np.float64], os: Orbital_State, dmode: Optional[DisturbanceMode] = None) -> NDArray[np.float64]:
        measurement = super().reading(x, os, dmode)
        
        if not np.any(np.isnan(measurement)):
            norm = np.linalg.norm(measurement)
            if norm > 1e-9:
                measurement = measurement / norm
                
        return measurement

    def basestate_jac(self, x: NDArray[np.float64], os: Orbital_State) -> NDArray[np.float64]:
        if self.current_star is None:
            return np.zeros((7, self.output_length))

        q = x[3:7]
        s_eci = self.current_star.s_eci
        db_dq = drotmatTvecdq(q, s_eci)

        J = np.zeros((7, self.output_length))
        J[3:7, :] = db_dq
        return J

    def bias_jac(self, x: NDArray[np.float64], os: Orbital_State) -> NDArray[np.float64]:
        return np.zeros((0, self.output_length))

    @property
    def noise_covariance(self) -> NDArray[np.float64]:
        if self.noise:
            return self.noise.cov()
        return np.zeros((3, 3))