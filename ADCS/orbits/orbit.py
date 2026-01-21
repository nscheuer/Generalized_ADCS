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
    def __init__(
        self,
        os0: Union[Orbital_State, List[Orbital_State]],
        end_time: float = None,
        dt: float = None,
        use_J2: bool = True,
        fast: bool = True,
        verbose: bool = True
    ) -> None:

        if isinstance(os0, Orbital_State):

            start_time = os0.J2000

            # ---------------------------------------------------------
            # Case 1: Singleton orbit (no propagation)
            # ---------------------------------------------------------
            if end_time is None or dt is None or end_time == start_time:
                self.states = {os0.J2000: os0.copy()}
                self.times = np.array([os0.J2000])
                return

            # ---------------------------------------------------------
            # Case 2: Propagate orbit with RK4
            # ---------------------------------------------------------
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

            # ---------------------------------------------------------
            # tqdm progress bar added HERE
            # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Case 3: List of Orbital_State objects
        # ---------------------------------------------------------
        elif isinstance(os0, list) and all(isinstance(j, Orbital_State) for j in os0):
            unique_times = {j.J2000 for j in os0}
            self.states = {j.J2000: j.copy() for j in os0 if j.J2000 in unique_times}
            self.times = np.array(sorted(self.states.keys()))

        else:
            raise ValueError("Orbit must be initialized with Orbital_State or List[Orbital_State]")
        
    def get_os(self, J2000: float) -> Orbital_State:
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
        if not np.all([self.time_in_span(j) for j in time_list]):
            print(self.max_time(), self.min_time(),min(time_list),max(time_list))
            raise ValueError('at least one time is not within this orbit span')
        newstates = [self.get_os(j) for j in time_list]
        return Orbit(newstates)
    
    def next_state(self, input: Orbital_State | float) -> Orbital_State:
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
        return np.amin(self.times)

    def max_time(self) -> float:
        return np.amax(self.times)

    def time_in_span(self,t) -> bool:
        return t<=self.max_time() and t>=self.min_time()
    
    def geocentric_to_ecef_orbit(self, b_vec: np.ndarray) -> np.ndarray:
        ecef_mat = np.vstack([self.states[j].ECEF for j in self.times])
        n_ecef = matrix_row_normalize(ecef_mat)
        svec = matrix_row_normalize(np.cross(MathConstants.unitvecs[2], n_ecef))
        return b_vec[:,0:1]*n_ecef + svec*b_vec[:,2:] + matrix_row_normalize(np.cross(svec,n_ecef))*b_vec[:,1:2]

    def ecef_to_eci_orbit(self, b_ecef_vec: np.ndarray) -> np.ndarray:
        return np.stack([self.states[self.times[j]].ecef_to_eci(b_ecef_vec[j,:]) for j in range(len(self.times))])

    def get_b_eci_orbit(self) -> np.ndarray:
        geos = np.vstack([self.states[j].geocentric for j in self.times])
        dts = [self.states[j].datetime for j in self.times]
        # IGRF expects radius in km, theta and phi in degrees
        # geos[:,0] is radius in meters, so divide by 1000
        b_r, b_th, b_ph = ppigrf.igrf_gc(geos[:,0]/1000.0, geos[:,1]*180.0/np.pi, geos[:,2]*180.0/np.pi, dts)
        b_r = np.diagonal(b_r)
        b_th = np.diagonal(b_th)
        b_ph = np.diagonal(b_ph)

        b_ecef = self.geocentric_to_ecef_orbit(np.atleast_2d(np.squeeze(np.stack([b_r, b_th, b_ph])).T))
        b_eci = self.ecef_to_eci_orbit(b_ecef)
        return b_eci*1e-9

    def get_sun_eci_orbit(self) -> np.ndarray:
        """Compute sun vectors for all timesteps using vectorized skyfield calls.
        
        Returns:
            np.ndarray: (N, 3) array of sun vectors in ECI frame [km]
        """
        from skyfield import api, positionlib
        
        # Get reference state for ephemeris access
        ref_state = self.states[self.times[0]]
        ephem = ref_state.ephem
        
        # Build array of TAI times
        tai_times = np.array([self.states[t].TAI for t in self.times])
        
        # Create skyfield time objects (vectorized)
        ts = ephem.ts
        t_sf = ts.tai_jd(tai_times)
        
        # Compute sun positions for all times at once
        sun_icrf = ephem.earth.at(t_sf).observe(ephem.sun).apparent()
        sun_eci = sun_icrf.position.km.T  # (N, 3)
        
        return sun_eci
    
    def get_vecs(self) -> List[List[np.ndarray]]:
        R = [self.states[j].R for j in self.times]
        V = [self.states[j].V for j in self.times]
        B = [self.states[j].B for j in self.times]
        S = [self.states[j].S for j in self.times]
        rho = [self.states[j].rho for j in self.times]
        return [R,V,B,S,rho]
