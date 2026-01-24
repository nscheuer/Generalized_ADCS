__all__ = ["Orbit"]

import numpy as np
import ppigrf as ppigrf
import warnings
from typing import List, Union
from tqdm import tqdm
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import matrix_row_normalize

class Orbit:
    r"""
    Time-ordered container and propagator for orbital states.

    This class represents an orbit as a discrete, time-indexed collection
    of :class:`~ADCS.orbits.orbital_state.Orbital_State` objects. It supports
    orbit propagation, interpolation, sub-sampling, coordinate-frame
    transformations, and environment-related vector extraction
    (e.g. geomagnetic field).

    The class can be initialized in three distinct modes:

    1. **Singleton orbit**  
       A single orbital state with no propagation.

    2. **Propagated orbit**  
       A single initial state propagated forward in time using a
       fourth-order Runge–Kutta (RK4) integrator.

    3. **Predefined orbit**  
       A list of already-computed orbital states.

    Mathematical Background
    -----------------------
    Orbit propagation is performed by numerically integrating the
    translational equations of motion:

    .. math::

        \ddot{\mathbf{r}} =
        -\frac{\mu}{r^3}\mathbf{r}
        + \mathbf{a}_{J_2},

    where :math:`\mu` is Earth’s gravitational parameter and
    :math:`\mathbf{a}_{J_2}` is the optional oblateness perturbation:

    .. math::

        \mathbf{a}_{J_2}
        = \frac{3 J_2 \mu R_E^2}{2 r^5}
        \begin{bmatrix}
            x \left(5 \frac{z^2}{r^2} - 1\right) \\
            y \left(5 \frac{z^2}{r^2} - 1\right) \\
            z \left(5 \frac{z^2}{r^2} - 3\right)
        \end{bmatrix}.

    The resulting trajectory is sampled at discrete epochs
    :math:`t_k` expressed in J2000 centuries.

    :param os0:
        Initial orbital state or list of orbital states.
    :type os0:
        Orbital_State or list[Orbital_State]

    :param end_time:
        Final propagation time in J2000 centuries.
    :type end_time: float or None

    :param dt:
        Propagation time step in seconds.
    :type dt: float or None

    :param use_J2:
        Enable or disable the J2 gravitational perturbation.
    :type use_J2: bool

    :param fast:
        Enable fast (approximate) propagation mode if supported
        by :class:`~ADCS.orbits.orbital_state.Orbital_State`.
    :type fast: bool

    :param verbose:
        Enable progress bar output during propagation.
    :type verbose: bool

    :raises ValueError:
        If input arguments are inconsistent or unsupported.

    """
    def __init__(
        self,
        os0: Union[Orbital_State, List[Orbital_State]],
        end_time: float = None,
        dt: float = None,
        use_J2: bool = True,
        fast: bool = True,
        verbose: bool = True
    ) -> None:
        r"""
        Initialize an orbit from an initial condition or a list of states.

        Depending on the inputs, this constructor either stores a single
        orbital state, propagates it forward in time using RK4 integration,
        or constructs an orbit from a provided list of orbital states.

        :param os0:
            Initial orbital state or list of orbital states.
        :type os0:
            Orbital_State or list[Orbital_State]

        :param end_time:
            Final propagation time in J2000 centuries.
        :type end_time: float or None

        :param dt:
            Propagation time step in seconds.
        :type dt: float or None

        :param use_J2:
            Enable J2 perturbation during propagation.
        :type use_J2: bool

        :param fast:
            Enable fast propagation mode.
        :type fast: bool

        :param verbose:
            Enable progress bar during propagation.
        :type verbose: bool

        :return:
            ``None``
        :rtype: None

        """

        if isinstance(os0, Orbital_State):

            start_time = os0.J2000
            if end_time is None or dt is None or end_time == start_time:
                self.states = {os0.J2000: os0.copy()}
                self.times = np.array([os0.J2000])
                return

            duration = end_time - start_time
            l = np.floor(duration / (dt / TimeConstants.cent2sec))

            times = (
                [start_time]
                + [start_time + j * (dt / TimeConstants.cent2sec) for j in range(1 + int(l))]
                + [end_time]
            )

            # Remove duplicates and sort
            times = sorted(list(set(times)))

            # Storage for states
            states0: List[Orbital_State] = [np.nan for _ in times]
            states0[0] = os0.copy()
            times[0] = os0.J2000

            for j in tqdm(range(1, len(times)), desc="Propagating Orbit", unit="step", disable=not verbose):

                dt_step = (times[j] - times[j - 1]) * TimeConstants.cent2sec

                states0[j] = states0[j - 1].propagate_orbit_rk4(
                    dt=dt_step,
                    J2_perturbation_on=use_J2,
                    fast=fast
                )

            # Build time-indexed dictionary
            self.states = {state.J2000: state for state in states0}
            self.times = np.array(sorted(self.states.keys()))

        elif isinstance(os0, list) and all(isinstance(j, Orbital_State) for j in os0):
            unique_times = {j.J2000 for j in os0}
            self.states = {j.J2000: j.copy() for j in os0 if j.J2000 in unique_times}
            self.times = np.array(sorted(self.states.keys()))

        else:
            raise ValueError("Orbit must be initialized with Orbital_State or List[Orbital_State]")
        
    def get_os(self, J2000: float) -> Orbital_State:
        r"""
        Retrieve or interpolate the orbital state at a given epoch.

        If an orbital state exists at the requested time, it is returned.
        Otherwise, linear interpolation between the nearest neighboring
        states is performed using the averaging method defined in
        :class:`~ADCS.orbits.orbital_state.Orbital_State`.

        :param J2000:
            Requested epoch in J2000 centuries.
        :type J2000: float

        :return:
            Orbital state at the specified time.
        :rtype: Orbital_State

        :raises ValueError:
            If the requested time lies outside the orbit span.

        """
        t = J2000
        if t>self.max_time():
            raise ValueError("get_os() called with t > max_time")
        if t<self.min_time():
            raise ValueError("get_os() called with t < min_time")
        close = np.isclose(self.times, t, rtol=0.0, atol=1e-2/TimeConstants.cent2sec)
        if np.any(close):
            inds = np.flatnonzero(close)
            if len(inds) == 1:
                ind = inds[0]
                return self.states[self.times[ind]].copy()
            elif len(inds) > 1:
                warnings.warn("get_os() has more than one match!")
                close_times = self.times[inds]
                inds2 = np.argmin(np.abs(np.array(close_times) - t))
                if np.isscalar(inds2):
                    ind = int(inds2)
                else:
                    ind = int(inds2[0])
                return self.states[close_times[ind]].copy()

        i0 = np.flatnonzero(self.times<t)[-1]
        i1 = np.flatnonzero(self.times>t)[0]
        t0 = self.times[i0]
        t1 = self.times[i1]
        return self.states[t0].average(self.states[t1],ratio = (t-t0)/(t1-t0))

    def get_range(self, t_0: float, t_1: float, dt: float = None):
        r"""
        Extract a sub-orbit over a specified time interval.

        This method returns either an existing subset of orbital states
        or a newly interpolated orbit sampled at a specified time step.

        :param t_0:
            Start time in J2000 centuries.
        :type t_0: float

        :param t_1:
            End time in J2000 centuries.
        :type t_1: float

        :param dt:
            Optional sampling interval in seconds.
        :type dt: float or None

        :return:
            New orbit or orbital state(s) within the given time range.
        :rtype:
            Orbit or Orbital_State

        :raises ValueError:
            If the time bounds are invalid or outside the orbit span.

        """
        if t_1<t_0:
            raise ValueError('times are in wrong order')

        if t_1==t_0:
            if dt is not None:
                return self.get_os(t_0)
            elif t_0 in self.times:
                return self.get_os(t_0)
            raise ValueError('times are equal, no matching time exactly. Try again with a specified dt or a wider time bracket. (or use the get_os() method)')
        if t_0>self.max_time():
            raise ValueError('first orbital state is not within this orbit (too far in future)')
        if t_0<self.min_time():
            raise ValueError('first orbital state is not within this orbit (too far in past)')
        if t_1>self.max_time():
            raise ValueError('last orbital state is not within this orbit (too far in future)')
        if t_1<self.min_time():
            raise ValueError('last orbital state is not within this orbit (too far in past)')

        if dt is None:
            newstates = [self.states[j] for j in self.times if (j<=t_1 and j>=t_0)]
            if len(newstates)==0:
                raise ValueError('there are no pre-created states in this time span')
            orbit_out = Orbit(newstates)
            return orbit_out
        else:
            ts = np.concatenate([np.arange(t_0,t_1,dt/TimeConstants.cent2sec),[t_1]])
            # ts = np.unique(ts)
            return self.new_orbit_from_times(ts)
        
    def new_orbit_from_times(self, time_list: List[float]):
        r"""
        Construct a new orbit sampled at specific epochs.

        For each requested time, the orbital state is obtained using
        :meth:`~ADCS.orbits.orbit.Orbit.get_os`.

        :param time_list:
            List of epochs in J2000 centuries.
        :type time_list: list[float]

        :return:
            New orbit containing interpolated states.
        :rtype: Orbit

        :raises ValueError:
            If any requested time lies outside the orbit span.

        """
        if not np.all([self.time_in_span(j) for j in time_list]):
            print(self.max_time(), self.min_time(),min(time_list),max(time_list))
            raise ValueError('at least one time is not within this orbit span')
        newstates = [self.get_os(j) for j in time_list]
        return Orbit(newstates)
    
    def next_state(self, input: Orbital_State | float) -> Orbital_State:
        r"""
        Return the next available orbital state after a given time.

        :param input:
            Reference time or orbital state.
        :type input:
            Orbital_State or float

        :return:
            Next orbital state in chronological order.
        :rtype: Orbital_State

        :raises ValueError:
            If the input time lies outside the orbit span.

        """
        if isinstance(input,Orbital_State):
            t = input.J2000
        elif isinstance(input,float):
            t = input
        else:
            raise ValueError("Must be j2000 time or orbital state")

        if t>self.max_time():
            raise ValueError('this orbital state is not within this orbit (too far in future)')
        if t<self.min_time():
            raise ValueError('this orbital state is not within this orbit (too far in past)')

        ind = np.flatnonzero(self.times>=t)[0]

        return self.states[self.times[ind]]
    
    def min_time(self) -> float:
        r"""
        Return the earliest epoch in the orbit.

        :return:
            Minimum J2000 time.
        :rtype: float

        """
        return np.amin(self.times)

    def max_time(self) -> float:
        r"""
        Return the latest epoch in the orbit.

        :return:
            Maximum J2000 time.
        :rtype: float

        """
        return np.amax(self.times)

    def time_in_span(self,t) -> bool:
        r"""
        Check whether a time lies within the orbit span.

        :param t:
            Time in J2000 centuries.
        :type t: float

        :return:
            ``True`` if the time is within the orbit span.
        :rtype: bool

        """
        return t<=self.max_time() and t>=self.min_time()
    
    def geocentric_to_ecef_orbit(self, b_vec: np.ndarray) -> np.ndarray:
        r"""
        Convert geocentric spherical vectors to ECEF coordinates along the orbit.

        This method applies a local orthonormal frame transformation defined
        by the orbit geometry.

        :param b_vec:
            Geocentric vector components :math:`(b_r, b_\theta, b_\phi)`.
        :type b_vec: numpy.ndarray

        :return:
            Vectors expressed in the ECEF frame.
        :rtype: numpy.ndarray

        """
        ecef_mat = np.vstack([self.states[j].ECEF for j in self.times])
        n_ecef = matrix_row_normalize(ecef_mat)
        svec = matrix_row_normalize(np.cross(MathConstants.unitvecs[2], n_ecef))
        return b_vec[:,0:1]*n_ecef + svec*b_vec[:,2:] + matrix_row_normalize(np.cross(svec,n_ecef))*b_vec[:,1:2]

    def ecef_to_eci_orbit(self, b_ecef_vec: np.ndarray) -> np.ndarray:
        r"""
        Convert ECEF vectors to ECI coordinates along the orbit.

        :param b_ecef_vec:
            Vectors expressed in the ECEF frame.
        :type b_ecef_vec: numpy.ndarray

        :return:
            Vectors expressed in the ECI frame.
        :rtype: numpy.ndarray

        """
        return np.stack([self.states[self.times[j]].ecef_to_eci(b_ecef_vec[j,:]) for j in range(len(self.times))])

    def get_b_eci_orbit(self) -> np.ndarray:
        r"""
        Compute the geomagnetic field in the ECI frame along the orbit.

        The magnetic field is evaluated using the IGRF model in geocentric
        coordinates and transformed into the inertial frame.

        :return:
            Geomagnetic field vectors in ECI coordinates [Tesla].
        :rtype: numpy.ndarray

        """

        geos = np.vstack([self.states[j].geocentric for j in self.times])
        dts = [self.states[j].datetime for j in self.times]
        b_r, b_th, b_ph = ppigrf.igrf_gc(geos[:,0],geos[:,1]*180.0/np.pi,geos[:,2]*180.0/np.pi,dts)
        b_r = np.diagonal(b_r)
        b_th = np.diagonal(b_th)
        b_ph = np.diagonal(b_ph)

        b_ecef = self.geocentric_to_ecef_orbit(np.atleast_2d(np.squeeze(np.stack([b_r, b_th, b_ph])).T))
        b_eci = self.ecef_to_eci_orbit(b_ecef)
        return b_eci*1e-9
    
    def get_vecs(self) -> List[List[np.ndarray]]:
        r"""
        Return commonly used orbit-related vectors.

        The returned lists contain position, velocity, magnetic field,
        Sun vector, and atmospheric density values for each epoch.

        :return:
            Lists of vectors ``[R, V, B, S, rho]`` over the orbit.
        :rtype:
            list[list[numpy.ndarray]]

        """
        R = [self.states[j].R for j in self.times]
        V = [self.states[j].V for j in self.times]
        B = [self.states[j].B for j in self.times]
        S = [self.states[j].S for j in self.times]
        rho = [self.states[j].rho for j in self.times]
        return [R,V,B,S,rho]
