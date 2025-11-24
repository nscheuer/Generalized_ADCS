__all__ = ["Orbital_State"]

import numpy as np
import warnings
import ppigrf
from ADCS.orbits.density_model import DensityModel
from ADCS.orbits.universal_constants import EarthConstants, TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, rot_mat, drotmatTvecdq, ddrotmatTvecdqdq
from skyfield import api, units, positionlib, toposlib, framelib, vectorlib
from skyfield.functions import T as sffT
from datetime import timezone
from typing import Union, Dict, Optional

class Orbital_State:
    r"""
    Represents the complete orbital state of a spacecraft in the Earth-centered inertial (ECI)
    frame, including position, velocity, environmental vectors, and derived coordinate
    transformations.

    This class provides methods for orbit propagation, Jacobian computation, and
    coordinate transformations between ECI, ECEF, ENU, and geocentric frames.
    It also integrates environmental modeling such as atmospheric density,
    magnetic field, and solar vector direction.

    Attributes
    ----------
    ephem : Ephemeris
        Skyfield ephemeris object containing planetary and time-scale data.
    J2000 : float
        Epoch time measured in Julian centuries since J2000.
    R : ndarray, shape (3,)
        Spacecraft position vector :math:`\mathbf{r}` in the ECI frame [km].
    V : ndarray, shape (3,)
        Spacecraft velocity vector :math:`\mathbf{v}` in the ECI frame [km/s].
    S : ndarray, shape (3,)
        Sun vector :math:`\mathbf{S}` in ECI frame [unitless or km].
    B : ndarray, shape (3,)
        Magnetic field vector :math:`\mathbf{B}` in ECI frame [T].
    rho : float
        Atmospheric density :math:`\rho` [kg/m³].
    density_model : DensityModel
        Atmospheric density model used for interpolation.
    ECI2ENUmat : ndarray, shape (3,3)
        Transformation matrix from ECI to ENU coordinates.
    geocentric : ndarray, shape (3,)
        Geocentric spherical coordinates :math:`[r, \theta, \phi]` [km, rad, rad].
    LLA : ndarray, shape (3,)
        Latitude, longitude, and altitude [rad, rad, km] when computed.
    datetime : datetime
        UTC datetime corresponding to the current epoch.

    Notes
    -----
    - Time conversions use the Skyfield timescale interface (:math:`t_{TAI}`, :math:`t_{UTC}`).
    - Magnetic field computations use the IGRF model via `ppigrf`.
    - The class supports both Euler and RK4 propagation of orbital state vectors:

    .. math::

        \frac{d\mathbf{r}}{dt} = \mathbf{v}, \quad
        \frac{d\mathbf{v}}{dt} = -\frac{\mu_E \mathbf{r}}{r^3} + \mathbf{a}_{J2}

    where :math:`\mu_E` is Earth's gravitational constant and :math:`\mathbf{a}_{J2}` is
    the optional J₂ perturbation acceleration term.
    """
    def __init__(self, ephem: Ephemeris, J2000: float, R: np.ndarray, V: np.ndarray, S: np.ndarray = None, B: np.ndarray = None, rho: float = None, density_model: DensityModel = None, fast: bool = False) -> None:
        self.ephem = ephem
        self.ts = self.ephem.ts
        
        self.J2000 = J2000
        self.R = R
        self.V = V
        self.TAI = self.j2000_to_tai()

        pos_time = self.ts.tai_jd(self.TAI)

        pos: units.Distance = units.Distance(km=self.R.tolist())
        vel_sf: units.Velocity = units.Velocity(km_per_s=self.V.tolist())

        # Create ICRF aka ECI position from J2000
        self.sf_pos: positionlib.ICRF = positionlib.ICRF(
            pos.au.tolist(),
            velocity_au_per_d=vel_sf.au_per_d.tolist(),
            t=pos_time,  # ✅ attach Skyfield Time object
            center=399,
            target=0
        )
        self.datetime = self.sf_pos.t.astimezone(timezone.utc).replace(tzinfo = None)

        # Geographic position
        if not fast:
            self.sf_geo_pos: toposlib.GeographicPosition = api.wgs84.geographic_position_of(self.sf_pos)
            self.LLA = np.array([self.sf_geo_pos.latitude.radians, self.sf_geo_pos.longitude.radians, self.sf_geo_pos.elevation.km])
            R_eci_to_ecef = self.sf_geo_pos.rotation_at(self.sf_pos.t)
            R_ecef_to_enu = np.array([
                [0, 1, 0],  # East  ← +Y_ECEF
                [1, 0, 0],  # North ← +X_ECEF
                [0, 0, 1],  # Up    ← +Z_ECEF
            ])
            self.ECI2ENUmat: np.ndarray = R_ecef_to_enu @ R_eci_to_ecef    
        self.ECEF: units.Distance = self.sf_pos.frame_xyz(framelib.itrs).km

        r = np.linalg.norm(self.ECEF)
        th = np.arccos(self.ECEF[2]/r)
        ph = np.arctan2(self.ECEF[1], self.ECEF[0])
        self.geocentric = np.array([r, th, ph])

        if density_model:
            self.density_model = density_model
        else:
            self.density_model = DensityModel()

        if S is not None:
            self.S = S
        elif fast:
            self.S = np.zeros(3)
        else:
            self.S = self.get_sun_eci()

        if B is not None:
            self.B = B
        elif fast:
            self.B = np.zeros(3)
        else:
            self.B = self.get_b_eci()

        if rho is not None:
            self.rho = rho
        elif fast or self.density_model is None:
            self.rho = 0.0
        else:
            altitude_from_core = np.linalg.norm(self.R)
            self.rho = self.density_model.interpolate(altitude_from_core - EarthConstants.R_e)

        self.vecs: Dict[str, np.ndarray] | None = None
        self._last_x: np.ndarray | None = None

    def copy(self):
        return self.average(self, 0)
    
    def average(self, orbital_state_2, ratio: float = 0.5, fast: bool = False):
        r"""
        Compute the weighted average between this orbital state and another
        :class:`Orbital_State` instance.

        Parameters
        ----------
        orbital_state_2 : Orbital_State
            Second orbital state to interpolate toward.
        ratio : float, optional
            Interpolation factor between 0 and 1.  
            - 0 → returns this state  
            - 1 → returns `orbital_state_2`  
            - 0.5 → midpoint (default)

        Returns
        -------
        Orbital_State
            New :class:`Orbital_State` object representing the interpolated state.

        Notes
        -----
        The averaging is performed linearly on each vector quantity:

        .. math::

            \mathbf{R}_{avg} &= (1 - \alpha)\mathbf{R}_1 + \alpha\mathbf{R}_2 \\
            \mathbf{V}_{avg} &= (1 - \alpha)\mathbf{V}_1 + \alpha\mathbf{V}_2 \\
            \mathbf{S}_{avg} &= (1 - \alpha)\mathbf{S}_1 + \alpha\mathbf{S}_2 \\
            \mathbf{B}_{avg} &= (1 - \alpha)\mathbf{B}_1 + \alpha\mathbf{B}_2 \\
            \rho_{avg} &= (1 - \alpha)\rho_1 + \alpha\rho_2

        where :math:`\alpha` is the weighting ratio.  
        Atmospheric models are preserved from the current object, but a warning
        is issued if their altitude–density grids do not match.
        """

        os2 = orbital_state_2

        j2000 = (1-ratio)*self.J2000 + ratio*os2.J2000
        R = (1-ratio)*self.R + ratio*os2.R
        V = (1-ratio)*self.V + ratio*os2.V
        S = (1-ratio)*self.S + ratio*os2.S
        B = (1-ratio)*self.B + ratio*os2.B
        rho = (1-ratio)*self.rho + ratio*os2.rho

        # Prefer own atmospheric model
        if not np.all(self.density_model.altitude_range == os2.density_model.altitude_range):
            warnings.warn('non-matching altitude range in atmospheric model between 2 orbital states')
        if not np.all(self.density_model.rho_range == os2.density_model.rho_range):
            warnings.warn('non-matching air density vs altitude in atmospheric model between 2 orbital states')
        density_model = self.density_model

        return Orbital_State(self.ephem, j2000, R, V, S, B, rho, density_model, fast=fast)

    def orbit_dynamics(self, J2_perturbation_on: bool = True) -> Union[np.ndarray, np.ndarray]:
        r"""
        Compute the orbital dynamics (two-body or J2-perturbed).

        Parameters
        ----------
        J2_perturbation_on : bool, optional
            If True, includes the Earth's oblateness correction (J₂ term).

        Returns
        -------
        r_dot : ndarray, shape (3,)
            Time derivative of position vector :math:`\dot{\mathbf{r}}` in ECI frame.
        v_dot : ndarray, shape (3,)
            Time derivative of velocity vector :math:`\dot{\mathbf{v}}` in ECI frame (acceleration).

        Notes
        -----
        The central gravitational acceleration is:

        .. math::

            \dot{\mathbf{v}} = -\frac{\mu_E \mathbf{r}}{\|\mathbf{r}\|^3}

        and the optional J₂ perturbation term is:

        .. math::

            \mathbf{a}_{J2} = \frac{3}{2} J_2 \frac{\mu_E R_E^2}{r^5}
            \begin{bmatrix}
            x \left(5 \frac{z^2}{r^2} - 1\right) \\
            y \left(5 \frac{z^2}{r^2} - 1\right) \\
            z \left(5 \frac{z^2}{r^2} - 3\right)
            \end{bmatrix}
        """
        # Unpack position and velocity from the state
        r_ECIk = self.R       # position vector (ECI) [km]
        v_ECIk = self.V       # velocity vector (ECI) [km/s]

        # Basic orbital parameters
        rn = np.linalg.norm(r_ECIk)   # magnitude of position vector
        xk, yk, zk = r_ECIk

        # Earth's constants (from a constants module or class)
        mu_e = EarthConstants.mu_e    # gravitational parameter [km^3/s^2]
        R_e = EarthConstants.R_e      # Earth radius [km]
        J2coeff = EarthConstants.J2coeff        # J2 coefficient (≈ 1.08263e−3)

        # Central (two-body) gravitational acceleration
        v_dot = -mu_e * r_ECIk / rn**3

        # Optional: add J2 perturbation acceleration
        if J2_perturbation_on:
            z2 = zk**2
            r2 = rn**2
            factor = (3/2) * J2coeff * (mu_e * R_e**2) / rn**5
            a_J2 = factor * np.array([
                xk * (5*z2/r2 - 1),
                yk * (5*z2/r2 - 1),
                zk * (5*z2/r2 - 3)
            ])
            v_dot += a_J2

        # Time derivative of position is velocity
        r_dot = v_ECIk

        # Return full state derivative vector [ṙ, v̇]
        return r_dot, v_dot
    
    def orbit_dynamics_jacobians(self, J2_perturbation_on: bool = True) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute the Jacobian matrices of the orbital dynamics equations.

        Parameters
        ----------
        J2_perturbation_on : bool, optional
            Enables J₂ perturbation effects in the Jacobian.

        Returns
        -------
        drd_dr : ndarray, shape (3,3)
            Partial derivative :math:`\frac{\partial \dot{\mathbf{r}}}{\partial \mathbf{r}}`.
        drd_dv : ndarray, shape (3,3)
            Partial derivative :math:`\frac{\partial \dot{\mathbf{r}}}{\partial \mathbf{v}}`.
        dvd_dr : ndarray, shape (3,3)
            Partial derivative :math:`\frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{r}}`.
        dvd_dv : ndarray, shape (3,3)
            Partial derivative :math:`\frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{v}}`.

        Notes
        -----
        For the two-body problem:

        .. math::

            \frac{\partial \dot{\mathbf{v}}}{\partial \mathbf{r}} =
            -\frac{\mu_E}{r^3} \left[ I - 3 \frac{\mathbf{r}\mathbf{r}^T}{r^2} \right]

        When J₂ is included, additional perturbation derivatives are added according
        to the Earth's gravitational potential expansion.
        """
        # Extract state
        r_ECI = self.R  # position vector [km]
        v_ECI = self.V  # velocity vector [km/s]
        rn = np.linalg.norm(r_ECI)
        nr = r_ECI / rn  # normalized position vector
        zk = r_ECI[2]    # z-component

        # Gravitational constant (Earth)
        mu_e = EarthConstants.mu_e
        R_e = EarthConstants.R_e
        J2coeff = EarthConstants.J2coeff

        # --- Core Two-body dynamics ---
        v_dot = -mu_e * r_ECI / rn**3
        r_dot = v_ECI

        # --- Initialize Jacobian blocks ---
        drd_dr = np.zeros((3, 3))
        drd_dv = np.eye(3)
        dvd_dv = np.zeros((3, 3))

        # Gravitational gradient matrix (∂a/∂r) for central gravity
        dvd_dr = -mu_e * (np.eye(3) - 3.0 * np.outer(nr, nr)) / rn**3

        # --- Optional: J2 Perturbation Jacobian ---
        if J2_perturbation_on:
            # Matrix term used for J2 effect (from JGM-3 model, via Wikipedia)
            j2_mult = np.diagflat(np.array([1.0, 1.0, 3.0]) * rn**2.0 - np.ones(3) * 5.0 * zk * zk)

            coeff = mu_e * (1.0 / rn**7.0) * (J2coeff * R_e**2) * (3.0 / 2.0)

            # Add J2 acceleration term
            v_dot += -coeff * r_ECI @ j2_mult

            # Unit vector in z-direction
            unit_z = np.array([0.0, 0.0, 1.0])

            # Jacobian correction term for J2 (as per original symbolic approach)
            dvd_dr += -coeff * (
                -7.0 * (np.outer(r_ECI, r_ECI @ j2_mult)) / rn**2.0
                + j2_mult
                + 2.0 * (
                    np.outer(-5.0 * zk * unit_z + r_ECI, r_ECI)
                    + 2.0 * zk * np.outer(r_ECI, unit_z)
                )
            )

        # Return the 4 sub-Jacobian blocks
        return drd_dr, drd_dv, dvd_dr, dvd_dv
        
    def propagate_orbit(self, dt: float, J2_perturbation_on: bool = True, fast: bool = True):
        r"""
        Propagate the orbital state forward in time using first-order integration.

        Parameters
        ----------
        dt : float
            Time step [s].
        J2_perturbation_on : bool, optional
            If True, include J₂ perturbation effects.

        Returns
        -------
        Orbital_State
            New orbital state at :math:`t + \Delta t`.

        Notes
        -----
        This uses a simple Euler integration step:

        .. math::

            \mathbf{r}_{k+1} = \mathbf{r}_k + \dot{\mathbf{r}}_k \Delta t \\
            \mathbf{v}_{k+1} = \mathbf{v}_k + \dot{\mathbf{v}}_k \Delta t
        """
        r_ECI = self.R
        v_ECI = self.V

        k1a, k1b = self.orbit_dynamics(J2_perturbation_on)
        r_out = r_ECI+k1a*dt
        v_out = v_ECI+k1b*dt

        return Orbital_State(self.ephem, self.J2000 + (dt/TimeConstants.cent2sec), r_out, v_out, S=None, B=None, rho=None, density_model=None, fast=fast)

    def propagate_orbit_rk4(self, dt: float, J2_perturbation_on: bool = True, fast: bool = True):
        r"""
        Propagate the orbit using 4th-order Runge–Kutta (RK4) integration.

        Parameters
        ----------
        dt : float
            Time step [s].
        J2_perturbation_on : bool, optional
            If True, include J₂ perturbation effects.

        Returns
        -------
        Orbital_State
            Updated orbital state after RK4 integration.

        Notes
        -----
        Runge–Kutta integration steps:

        .. math::

            k_1 = f(t, y) \\
            k_2 = f(t + \tfrac{1}{2}\Delta t, y + \tfrac{1}{2}k_1 \Delta t) \\
            k_3 = f(t + \tfrac{1}{2}\Delta t, y + \tfrac{1}{2}k_2 \Delta t) \\
            k_4 = f(t + \Delta t, y + k_3 \Delta t) \\
            y_{k+1} = y_k + \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4)
        """

        r_ECI = self.R
        v_ECI = self.V

        k1a, k1b = self.orbit_dynamics(J2_perturbation_on)
        k2a_in = r_ECI+k1a*0.5*dt
        k2b_in = v_ECI+k1b*0.5*dt
        os2_in = Orbital_State(self.ephem, self.J2000 + (0.5*dt/TimeConstants.cent2sec), k2a_in, k2b_in, S=None, B=None, rho=None, density_model=None, fast=fast)

        k2a, k2b = os2_in.orbit_dynamics(J2_perturbation_on)
        k3a_in = r_ECI+k2a*0.5*dt
        k3b_in = v_ECI+k2b*0.5*dt
        os3_in = Orbital_State(self.ephem, self.J2000 + (0.5*dt/TimeConstants.cent2sec), k3a_in, k3b_in, S=None, B=None, rho=None, density_model=None, fast=fast)

        k3a, k3b = os3_in.orbit_dynamics(J2_perturbation_on)
        k4a_in = r_ECI+k3a*dt
        k4b_in = v_ECI+k3b*dt
        os4_in = Orbital_State(self.ephem, self.J2000 + (1.0*dt/TimeConstants.cent2sec), k4a_in, k4b_in, S=None, B=None, rho=None, density_model=None, fast=fast)

        k4a, k4b = os4_in.orbit_dynamics(J2_perturbation_on)
        r_out = r_ECI + (dt/6.0)*(k1a+k2a*2.0+k3a*2.0+k4a)
        v_out = v_ECI + (dt/6.0)*(k1b+k2b*2.0+k3b*2.0+k4b)

        return Orbital_State(self.ephem, self.J2000 + (dt/TimeConstants.cent2sec), r_out, v_out, S=None, B=None, rho=None, density_model=None, fast=fast)
    
    def propagate_jacobians(self, dt: float, J2_perturbation_on: bool = True) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Propagate the state-transition Jacobians using Euler integration.

        Parameters
        ----------
        dt : float
            Time step [s].
        J2_perturbation_on : bool, optional
            If True, include J₂ effects.

        Returns
        -------
        dr1__dr0 : ndarray, shape (3,3)
            ∂r₁/∂r₀.
        dr1__dv0 : ndarray, shape (3,3)
            ∂r₁/∂v₀.
        dv1__dr0 : ndarray, shape (3,3)
            ∂v₁/∂r₀.
        dv1__dv0 : ndarray, shape (3,3)
            ∂v₁/∂v₀.

        Notes
        -----
        Linearized update:

        .. math::

            \frac{d}{dt}
            \begin{bmatrix}
            \mathbf{r} \\ \mathbf{v}
            \end{bmatrix}
            =
            \begin{bmatrix}
            0 & I \\
            A & 0
            \end{bmatrix}
            \begin{bmatrix}
            \mathbf{r} \\ \mathbf{v}
            \end{bmatrix}
        """

        
        r0 = self.R
        v0 = self.V

        rd0, vd0 = self.orbit_dynamics(J2_perturbation_on)
        r1 = r0 + rd0*dt
        v1 = v0 + vd0*dt

        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = self.orbit_dynamics_jacobians(J2_perturbation_on)

        dr1__dr0 = np.eye(3)+dt*drd0__dr0
        dr1__dv0 = dt*drd0__dv0
        dv1__dr0 = dt*dvd0__dr0
        dv1__dv0 = np.eye(3)+dt*dvd0__dv0

        return dr1__dr0, dr1__dv0, dv1__dr0, dv1__dv0
    
    def propagate_jacobians_rk4(self, dt: float, J2_perturbation_on: bool = True) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Propagate the state-transition Jacobian matrix using RK4 integration.

        Parameters
        ----------
        dt : float
            Time step [s].
        J2_perturbation_on : bool, optional
            If True, include J₂ perturbation effects.

        Returns
        -------
        dr_out_dr0, dv_out_dr0, dr_out_dv0, dv_out_dv0 : ndarray
            Sub-blocks of the final 6x6 state transition matrix :math:`\Phi(t+\Delta t, t)`.

        Notes
        -----
        The full state transition matrix is numerically propagated using the same
        RK4 scheme applied to the dynamics Jacobian differential equation.
        """

        
        r0 = self.R
        v0 = self.V

        rd0, vd0 = self.orbit_dynamics(J2_perturbation_on)
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = self.orbit_dynamics_jacobians(J2_perturbation_on)
        dsd0__ds0 = np.block([[drd0__dr0, dvd0__dr0], [drd0__dv0, dvd0__dv0]])
        r1 = r0 + rd0*0.5*dt
        v1 = v0 + vd0*0.5*dt
        ds1__ds0 = np.eye(6)+0.5*dt*dsd0__ds0
        os1 = Orbital_State(self.ephem, self.J2000+(0.5*dt/TimeConstants.cent2sec), r1, v1, S=None, B=None, rho=None, density_model=None, fast=True)

        rd1, vd1 = os1.orbit_dynamics(J2_perturbation_on)
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = os1.orbit_dynamics_jacobians(J2_perturbation_on)
        dsd1__ds1 = np.block([[drd0__dr0, dvd0__dr0], [drd0__dv0, dvd0__dv0]])
        dsd1__ds0 = ds1__ds0@dsd1__ds1
        r2 = r0+rd1*0.5*dt
        v2 = v0+vd1*0.5*dt
        ds2__ds0 = np.eye(6)+0.5*dt*dsd1__ds0
        os2 = Orbital_State(self.ephem, self.J2000+(0.5*dt/TimeConstants.cent2sec), r2, v2, S=None, B=None, rho=None, density_model=None, fast=True)

        rd2, vd2 = os2.orbit_dynamics(J2_perturbation_on)
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = os2.orbit_dynamics_jacobians(J2_perturbation_on)
        dsd2__ds2 = np.block([[drd0__dr0, dvd0__dr0], [drd0__dv0, dvd0__dv0]])
        dsd2__ds0 = ds2__ds0@dsd2__ds2
        r3 = r0+rd2*dt
        v3 = v0+vd2*dt
        ds3__ds0 = np.eye(6)+dt*dsd2__ds0
        os3 = Orbital_State(self.ephem, self.J2000+(1.0*dt/TimeConstants.cent2sec), r3, v3, S=None, B=None, rho=None, density_model=None, fast=True)

        rd3, vd3 = os3.orbit_dynamics(J2_perturbation_on)
        drd0__dr0, drd0__dv0, dvd0__dr0, dvd0__dv0 = os3.orbit_dynamics_jacobians(J2_perturbation_on)
        dsd3__ds3 = np.block([[drd0__dr0, dvd0__dr0], [drd0__dv0, dvd0__dv0]])
        dsd3__ds0 = ds3__ds0@dsd3__ds3
        r4 = r0+(dt/6.0)*(rd0+2.0*rd1+2.0*rd2+rd3)
        v4 = v0+(dt/6.0)*(vd0+2.0*vd1+2.0*vd2+vd3)
        ds4__ds0 = np.eye(6)+(dt/6.0)*(dsd0__ds0 + 2.0*dsd1__ds0 + 2.0*dsd2__ds0 + dsd3__ds0)
        os4 = Orbital_State(self.ephem, self.J2000+(1.0*dt/TimeConstants.cent2sec), r4, v4, S=None, B=None, rho=None, density_model=None, fast=True)

        return ds4__ds0[0:3,0:3],ds4__ds0[3:6,0:3],ds4__ds0[0:3,3:6],ds4__ds0[3:6,3:6]
           
    def eci_to_ecef(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector from ECI (J2000) frame to ECEF frame.

        Parameters
        ----------
        vec : ndarray, shape (3,)
            Vector in ECI coordinates.

        Returns
        -------
        ndarray
            Corresponding vector in ECEF frame.

        Notes
        -----
        Uses Skyfield's Earth rotation model :math:`R_{ECI \rightarrow ECEF}(t)`.
        """

        return framelib.itrs.rotation_at(self.sf_pos.t)@vec
    
    def ecef_to_eci(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector from ECEF to ECI coordinates.

        Parameters
        ----------
        vec : ndarray, shape (3,)
            Vector in ECEF frame.

        Returns
        -------
        ndarray
            Vector expressed in ECI frame.
        """

        return sffT(framelib.itrs.rotation_at(self.sf_pos.t))@vec
    
    def ecef_to_geocentric(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Convert between ECEF and geocentric spherical coordinate representations.

        Parameters
        ----------
        vec : ndarray, shape (3,)
            Vector in either ECEF or geocentric frame.

        Returns
        -------
        ndarray
            Transformed vector in target frame.

        Notes
        -----
        The transformation uses orthonormal bases aligned with
        local "up", "north", and "east" directions on a spherical Earth.
        """

        n_ecef = normalize(self.ECEF)
        svec = normalize(np.cross(np.array([0.0, 0.0, 1.0]),n_ecef))
        R = np.vstack([n_ecef,normalize(np.cross(svec,n_ecef)),svec])
        return R.T@vec
    
    def geocentric_to_ecef(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Convert between ECEF and geocentric spherical coordinate representations.

        Parameters
        ----------
        vec : ndarray, shape (3,)
            Vector in either ECEF or geocentric frame.

        Returns
        -------
        ndarray
            Transformed vector in target frame.

        Notes
        -----
        The transformation uses orthonormal bases aligned with
        local "up", "north", and "east" directions on a spherical Earth.
        """
        n_ecef = normalize(self.ECEF) #"up"
        svec = normalize(np.cross(np.array([0.0, 0.0, 1.0]),n_ecef))  #"east" on earch-centered sphere
        return vec[0]*n_ecef + svec*vec[2] + normalize(np.cross(svec,n_ecef))*vec[1]
    
    def eci_to_enu(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector between ECI and ENU frames.

        Parameters
        ----------
        vec : ndarray, shape (3,)
            Vector in source frame.

        Returns
        -------
        ndarray
            Vector in target frame.
        """

        return vec@self.ECI2ENUmat.T

    def enu_to_eci(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Transform a vector between ECI and ENU frames.

        Parameters
        ----------
        vec : ndarray, shape (3,)
            Vector in source frame.

        Returns
        -------
        ndarray
            Vector in target frame.
        """

        return vec@self.ECI2ENUmat
    
    def get_b_eci(self) -> np.ndarray:
        r"""
        Compute the Earth's magnetic field vector in the ECI frame.

        Returns
        -------
        ndarray, shape (3,)
            Magnetic field vector :math:`\mathbf{B}` [Tesla] in ECI frame.

        Notes
        -----
        Uses IGRF (via `ppigrf`) in geocentric spherical coordinates:

        .. math::

            (B_r, B_\theta, B_\phi) \rightarrow ECEF \rightarrow ECI

        The output is converted from nT to Tesla.
        """

        r = self.geocentric[0]
        theta_deg = np.rad2deg(self.geocentric[1])
        phi_deg = np.rad2deg(self.geocentric[2])

        b_r, b_th, b_ph = ppigrf.igrf_gc(r, theta_deg, phi_deg, self.datetime)
        b_array = np.array([b_r, b_th, b_ph])

        b_ecef = self.geocentric_to_ecef(np.squeeze(b_array))
        b_eci = self.ecef_to_eci(b_ecef)
        return b_eci*1e-9 # Conversion from nT to T
    
    def j2000_to_tai(self):
        r"""
        Convert J2000 epoch time to TAI Julian Date.

        Returns
        -------
        float
            Julian Date (TAI).

        Notes
        -----
        Conversion formula:

        .. math::

            JD_{TAI} = 2451545.0 + 36525 \times (J2000)
        """

        return self.J2000*36525.0+2451545.0

    def get_sun_eci(self) -> np.ndarray:
        r"""
        Compute the Sun's position vector in ECI coordinates.

        Returns
        -------
        ndarray, shape (3,)
            Sun position vector :math:`\mathbf{r}_{sun}` [km] in ECI frame.

        Notes
        -----
        Uses Skyfield ephemerides for Earth–Sun vector calculation:

        .. math::

            \mathbf{r}_{sun,ECI} = \mathbf{r}_{sun} - \mathbf{r}_{earth}
        """

        timescale_object: api.Timescale = self.ephem.ts
        pos_time = timescale_object.tai_jd(self.TAI)

        earth: vectorlib.VectorSum = self.ephem.earth
        sun: vectorlib.VectorSum = self.ephem.sun

        sun_icrf: positionlib.ICRF = self.ephem.earth.at(pos_time).observe(self.ephem.sun).apparent()

        # Extract Sun position vector (km) in ECI coordinates
        sun_eci: np.ndarray = sun_icrf.position.km
        return sun_eci


    def update_vecs(self, x: np.ndarray) -> None:
        q0 = x[3:7]

        R = self.R
        V = self.V
        B = self.B
        S = self.S
        rho = self.rho

        rmat_ECI2B = rot_mat(q0).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V

        dR_B__dq = drotmatTvecdq(q0,R)
        dB_B__dq = drotmatTvecdq(q0,B)
        dV_B__dq = drotmatTvecdq(q0,V)
        dS_B__dq = drotmatTvecdq(q0,S)
        ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
        ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
        ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
        ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)

        self.vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq}
        self._last_x = x
    
    def get_state_vector(self, x: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        if not np.array_equal(x, self._last_x):
            self.update_vecs(x=x)
        return self.vecs


    def is_sunlit(self) -> bool:
        return self.sf_pos.is_sunlit(self.ephem.planets)