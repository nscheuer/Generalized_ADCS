# orbit.py
__all__ = ["Orbit"]

import numpy as np
import ppigrf
import warnings
from typing import List, Union
from tqdm import tqdm
from skyfield import api, units, positionlib, framelib
from datetime import timezone

from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants, EarthConstants
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import matrix_row_normalize
from ADCS.orbits.density_model import DensityModel


class Orbit:
    r"""
    Time-ordered container and propagator for orbital states.

    Notes
    -----
    The ``fast`` argument is accepted for backward compatibility but is ignored:
    this implementation always computes real environment vectors (Sun and magnetic
    field) and full frame transforms for each stored
    :class:`~ADCS.orbits.orbital_state.Orbital_State`.

    Performance
    -----------
    For propagated orbits, this constructor propagates the translational dynamics
    using pure NumPy and then batch-computes the expensive environment quantities
    (Skyfield frames / Sun vector and ppigrf geomagnetic field) for the full time
    array once, before constructing individual :class:`~ADCS.orbits.orbital_state.Orbital_State`
    objects.

    :param os0:
        Initial orbital state or list of orbital states.
    :type os0: Orbital_State or list[Orbital_State]

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
        Backward-compatible parameter (ignored).
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
        verbose: bool = True,
    ) -> None:
        r"""
        Initialize an orbit from an initial condition or a list of states.

        :param os0:
            Initial orbital state or list of orbital states.
        :type os0: Orbital_State or list[Orbital_State]

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
            Backward-compatible parameter (ignored).
        :type fast: bool

        :param verbose:
            Enable progress bar during propagation.
        :type verbose: bool

        :return:
            ``None``
        :rtype: None

        """
        _ = fast  # ignored
        use_J2 = bool(use_J2)

        if isinstance(os0, Orbital_State):
            start_time = float(os0.J2000)

            # Singleton orbit
            if end_time is None or dt is None or float(end_time) == start_time:
                st = os0.copy()
                self.states = {st.J2000: st}
                self.times = np.array([st.J2000], dtype=float)
                return

            end_time = float(end_time)
            dt = float(dt)

            duration = end_time - start_time
            l = np.floor(duration / (dt / TimeConstants.cent2sec))

            times = (
                [start_time]
                + [start_time + j * (dt / TimeConstants.cent2sec) for j in range(1 + int(l))]
                + [end_time]
            )
            times = sorted(list(set(times)))
            times[0] = start_time

            N = len(times)
            times_arr = np.asarray(times, dtype=float)

            # --- Propagate translational dynamics (RK4) without constructing Orbital_State each step ---
            R_hist = np.empty((N, 3), dtype=float)
            V_hist = np.empty((N, 3), dtype=float)

            R_hist[0, :] = np.asarray(os0.R, dtype=float).reshape(3)
            V_hist[0, :] = np.asarray(os0.V, dtype=float).reshape(3)

            mu_e = float(getattr(os0, "mu_e", EarthConstants.mu_e))
            R_e = float(getattr(os0, "R_e", EarthConstants.R_e))
            J2coeff = float(getattr(os0, "J2coeff", EarthConstants.J2coeff))

            for j in tqdm(range(1, N), desc="Propagating Orbit", unit="step", disable=not verbose):
                dt_step = (times_arr[j] - times_arr[j - 1]) * TimeConstants.cent2sec

                r0 = R_hist[j - 1, :]
                v0 = V_hist[j - 1, :]

                k1r, k1v = Orbital_State._orbit_dynamics_raw(r0, v0, mu_e, R_e, J2coeff, use_J2)
                k2r, k2v = Orbital_State._orbit_dynamics_raw(r0 + 0.5 * dt_step * k1r, v0 + 0.5 * dt_step * k1v, mu_e, R_e, J2coeff, use_J2)
                k3r, k3v = Orbital_State._orbit_dynamics_raw(r0 + 0.5 * dt_step * k2r, v0 + 0.5 * dt_step * k2v, mu_e, R_e, J2coeff, use_J2)
                k4r, k4v = Orbital_State._orbit_dynamics_raw(r0 + dt_step * k3r, v0 + dt_step * k3v, mu_e, R_e, J2coeff, use_J2)

                R_hist[j, :] = r0 + (dt_step / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
                V_hist[j, :] = v0 + (dt_step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

            # --- Batch-compute environment and frame transforms once ---
            ephem = os0.ephem
            ts = ephem.ts

            TAI = times_arr * 36525.0 + 2451545.0
            t_sf = ts.tai_jd(TAI)

            # Rotation matrices ECI->ECEF (vectorized)
            R_eci2ecef_raw = framelib.itrs.rotation_at(t_sf)
            R_eci2ecef_raw = np.asarray(R_eci2ecef_raw, dtype=float)

            if R_eci2ecef_raw.ndim == 3 and R_eci2ecef_raw.shape[0] == 3 and R_eci2ecef_raw.shape[1] == 3:
                R_eci2ecef = np.transpose(R_eci2ecef_raw, (2, 0, 1))  # (N,3,3)
            elif R_eci2ecef_raw.ndim == 3 and R_eci2ecef_raw.shape[-2:] == (3, 3):
                R_eci2ecef = R_eci2ecef_raw  # (N,3,3)
            else:
                # Fallback: treat as a single matrix (should not happen for multi-time)
                R_eci2ecef = np.repeat(R_eci2ecef_raw.reshape(1, 3, 3), N, axis=0)

            R_ecef2eci = np.transpose(R_eci2ecef, (0, 2, 1))

            # ECEF positions
            ECEF_hist = np.einsum("nij,nj->ni", R_eci2ecef, R_hist)

            # Geocentric (r, theta, phi) from ECEF
            r_ecef = np.linalg.norm(ECEF_hist, axis=1)
            theta = np.arccos(ECEF_hist[:, 2] / r_ecef)
            phi = np.arctan2(ECEF_hist[:, 1], ECEF_hist[:, 0])
            geo_hist = np.stack([r_ecef, theta, phi], axis=1)

            # Datetimes (naive UTC)
            dts = []
            for dt_i in t_sf.utc_datetime():
                if getattr(dt_i, "tzinfo", None) is not None:
                    dts.append(dt_i.astimezone(timezone.utc).replace(tzinfo=None))
                else:
                    dts.append(dt_i)

            # Sun vector (ECI) in km (vectorized)
            sun_icrf = ephem.earth.at(t_sf).observe(ephem.sun).apparent()
            sun_km = np.asarray(sun_icrf.position.km, dtype=float)
            if sun_km.ndim == 2 and sun_km.shape[0] == 3 and sun_km.shape[1] == N:
                S_hist = sun_km.T
            elif sun_km.ndim == 2 and sun_km.shape[0] == N and sun_km.shape[1] == 3:
                S_hist = sun_km
            else:
                S_hist = np.reshape(sun_km, (N, 3))

            # Magnetic field (IGRF) in geocentric components (vectorized call)
            b_r, b_th, b_ph = ppigrf.igrf_gc(
                geo_hist[:, 0],
                geo_hist[:, 1] * 180.0 / np.pi,
                geo_hist[:, 2] * 180.0 / np.pi,
                dts,
            )
            b_r = np.asarray(b_r)
            b_th = np.asarray(b_th)
            b_ph = np.asarray(b_ph)

            # ppigrf sometimes returns diagonal matrices when passed time lists; handle both
            if b_r.ndim == 2 and b_r.shape[0] == b_r.shape[1] == N:
                b_r = np.diagonal(b_r)
            if b_th.ndim == 2 and b_th.shape[0] == b_th.shape[1] == N:
                b_th = np.diagonal(b_th)
            if b_ph.ndim == 2 and b_ph.shape[0] == b_ph.shape[1] == N:
                b_ph = np.diagonal(b_ph)

            b_r = np.asarray(b_r, dtype=float).reshape(N)
            b_th = np.asarray(b_th, dtype=float).reshape(N)
            b_ph = np.asarray(b_ph, dtype=float).reshape(N)

            # Local basis for geocentric->ECEF conversion (vectorized, matches existing Orbit.geocentric_to_ecef_orbit)
            n_ecef = ECEF_hist / r_ecef[:, None]
            zhat = MathConstants.unitvecs[2].reshape(1, 3)
            svec = np.cross(zhat, n_ecef)
            svec = svec / np.linalg.norm(svec, axis=1)[:, None]
            tvec = np.cross(svec, n_ecef)
            tvec = tvec / np.linalg.norm(tvec, axis=1)[:, None]

            # Convert to ECEF, then to ECI
            b_ecef = b_r[:, None] * n_ecef + b_ph[:, None] * svec + b_th[:, None] * tvec
            B_hist = np.einsum("nij,nj->ni", R_ecef2eci, b_ecef) * 1e-9

            # Density model (reuse os0 model if present)
            density_model = getattr(os0, "density_model", None)
            if density_model is None:
                density_model = DensityModel()

            alt_core = np.linalg.norm(R_hist, axis=1) - EarthConstants.R_e
            try:
                rho_hist = np.asarray(density_model.interpolate(alt_core), dtype=float).reshape(N)
            except Exception:
                rho_hist = np.array([float(density_model.interpolate(float(h))) for h in alt_core], dtype=float)

            # ENU transform (vectorized)
            pos = units.Distance(km=[R_hist[:, 0], R_hist[:, 1], R_hist[:, 2]])
            vel = units.Velocity(km_per_s=[V_hist[:, 0], V_hist[:, 1], V_hist[:, 2]])
            sf_pos_vec = positionlib.ICRF(pos.au, velocity_au_per_d=vel.au_per_d, t=t_sf, center=399, target=0)
            sf_geo_pos_vec = api.wgs84.geographic_position_of(sf_pos_vec)

            lat = np.asarray(sf_geo_pos_vec.latitude.radians, dtype=float).reshape(N)
            lon = np.asarray(sf_geo_pos_vec.longitude.radians, dtype=float).reshape(N)
            elev = np.asarray(sf_geo_pos_vec.elevation.km, dtype=float).reshape(N)
            LLA_hist = np.stack([lat, lon, elev], axis=1)

            R_eci_to_ecef_geo_raw = sf_geo_pos_vec.rotation_at(t_sf)
            R_eci_to_ecef_geo_raw = np.asarray(R_eci_to_ecef_geo_raw, dtype=float)
            if R_eci_to_ecef_geo_raw.ndim == 3 and R_eci_to_ecef_geo_raw.shape[0] == 3 and R_eci_to_ecef_geo_raw.shape[1] == 3:
                R_eci_to_ecef_geo = np.transpose(R_eci_to_ecef_geo_raw, (2, 0, 1))
            elif R_eci_to_ecef_geo_raw.ndim == 3 and R_eci_to_ecef_geo_raw.shape[-2:] == (3, 3):
                R_eci_to_ecef_geo = R_eci_to_ecef_geo_raw
            else:
                R_eci_to_ecef_geo = np.repeat(R_eci_to_ecef_geo_raw.reshape(1, 3, 3), N, axis=0)

            R_ecef_to_enu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            ECI2ENU_hist = np.einsum("ij,njk->nik", R_ecef_to_enu, R_eci_to_ecef_geo)

            # Sunlit flag (vectorized)
            try:
                sunlit_hist = np.asarray(sf_pos_vec.is_sunlit(ephem.planets), dtype=bool).reshape(N)
            except Exception:
                sunlit_hist = np.array([False] * N, dtype=bool)

            # --- Construct Orbital_State objects cheaply without per-state skyfield/ppigrf calls ---
            states0: List[Orbital_State] = [None] * N
            for i in range(N):
                st = Orbital_State.__new__(Orbital_State)

                st.ephem = ephem
                st.ts = ts

                st.J2000 = float(times_arr[i])
                st.R = R_hist[i, :].copy()
                st.V = V_hist[i, :].copy()

                st.mu_e = mu_e
                st.R_e = R_e
                st.J2coeff = J2coeff

                st.TAI = float(TAI[i])
                st.datetime = dts[i]

                st._R_eci2ecef = R_eci2ecef[i, :, :].copy()
                st._R_ecef2eci = R_ecef2eci[i, :, :].copy()

                st.ECEF = ECEF_hist[i, :].copy()
                st.geocentric = geo_hist[i, :].copy()

                # Local basis
                st._n_ecef = n_ecef[i, :].copy()
                st._svec = svec[i, :].copy()
                st._tvec = tvec[i, :].copy()
                st._ecef_to_geo = np.vstack([st._n_ecef, st._tvec, st._svec])

                # Environment
                st.density_model = density_model
                st.S = S_hist[i, :].copy()
                st.B = B_hist[i, :].copy()
                st.rho = float(rho_hist[i])

                # Geographic + ENU
                st.LLA = LLA_hist[i, :].copy()
                st.ECI2ENUmat = ECI2ENU_hist[i, :, :].copy()
                st.sf_geo_pos = None  # kept for compatibility; LLA/ECI2ENUmat are provided

                # Provide a scalar sf_pos mainly for compatibility; heavy calls are avoided elsewhere
                try:
                    pos_i = units.Distance(km=st.R.tolist())
                    vel_i = units.Velocity(km_per_s=st.V.tolist())
                    st.sf_pos = positionlib.ICRF(pos_i.au, velocity_au_per_d=vel_i.au_per_d, t=t_sf[i], center=399, target=0)
                except Exception:
                    st.sf_pos = None

                st._sunlit = bool(sunlit_hist[i])

                st.vecs = None
                st._last_x = None

                states0[i] = st

            self.states = {st.J2000: st for st in states0}
            self.times = np.array(sorted(self.states.keys()), dtype=float)

        elif isinstance(os0, list) and all(isinstance(j, Orbital_State) for j in os0):
            # Ensure uniqueness by time and copy states (copy is fast and does not recompute environment)
            unique_times = {j.J2000 for j in os0}
            self.states = {j.J2000: j.copy() for j in os0 if j.J2000 in unique_times}
            self.times = np.array(sorted(self.states.keys()), dtype=float)

        else:
            raise ValueError("Orbit must be initialized with Orbital_State or List[Orbital_State]")

    def get_os(self, J2000: float) -> Orbital_State:
        r"""
        Retrieve or interpolate the orbital state at a given epoch.

        :param J2000:
            Requested epoch in J2000 centuries.
        :type J2000: float

        :return:
            Orbital state at the specified time.
        :rtype: Orbital_State

        :raises ValueError:
            If the requested time lies outside the orbit span.

        """
        t = float(J2000)
        if t > self.max_time():
            raise ValueError("get_os() called with t > max_time")
        if t < self.min_time():
            raise ValueError("get_os() called with t < min_time")

        close = np.isclose(self.times, t, rtol=0.0, atol=1e-2 / TimeConstants.cent2sec)
        if np.any(close):
            inds = np.flatnonzero(close)
            if len(inds) == 1:
                ind = int(inds[0])
                return self.states[self.times[ind]].copy()
            warnings.warn("get_os() has more than one match!")
            close_times = self.times[inds]
            ind2 = int(np.argmin(np.abs(close_times - t)))
            return self.states[close_times[ind2]].copy()

        i0 = np.flatnonzero(self.times < t)[-1]
        i1 = np.flatnonzero(self.times > t)[0]
        t0 = self.times[i0]
        t1 = self.times[i1]
        return self.states[t0].average(self.states[t1], ratio=(t - t0) / (t1 - t0))

    def get_range(self, t_0: float, t_1: float, dt: float = None):
        r"""
        Extract a sub-orbit over a specified time interval.

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
        :rtype: Orbit or Orbital_State

        :raises ValueError:
            If the time bounds are invalid or outside the orbit span.

        """
        t_0 = float(t_0)
        t_1 = float(t_1)

        if t_1 < t_0:
            raise ValueError("times are in wrong order")

        if t_1 == t_0:
            if dt is not None:
                return self.get_os(t_0)
            if t_0 in self.times:
                return self.get_os(t_0)
            raise ValueError(
                "times are equal, no matching time exactly. Try again with a specified dt or a wider time bracket. "
                "(or use the get_os() method)"
            )

        if t_0 > self.max_time():
            raise ValueError("first orbital state is not within this orbit (too far in future)")
        if t_0 < self.min_time():
            raise ValueError("first orbital state is not within this orbit (too far in past)")
        if t_1 > self.max_time():
            raise ValueError("last orbital state is not within this orbit (too far in future)")
        if t_1 < self.min_time():
            raise ValueError("last orbital state is not within this orbit (too far in past)")

        if dt is None:
            newstates = [self.states[j] for j in self.times if (j <= t_1 and j >= t_0)]
            if len(newstates) == 0:
                raise ValueError("there are no pre-created states in this time span")
            return Orbit(newstates)
        ts = np.concatenate([np.arange(t_0, t_1, float(dt) / TimeConstants.cent2sec), [t_1]])
        return self.new_orbit_from_times(ts.tolist())

    def new_orbit_from_times(self, time_list: List[float]):
        r"""
        Construct a new orbit sampled at specific epochs.

        :param time_list:
            List of epochs in J2000 centuries.
        :type time_list: list[float]

        :return:
            New orbit containing interpolated states.
        :rtype: Orbit

        :raises ValueError:
            If any requested time lies outside the orbit span.

        """
        if not np.all([self.time_in_span(float(j)) for j in time_list]):
            raise ValueError("at least one time is not within this orbit span")
        newstates = [self.get_os(float(j)) for j in time_list]
        return Orbit(newstates)

    def next_state(self, input: Orbital_State | float) -> Orbital_State:
        r"""
        Return the next available orbital state after a given time.

        :param input:
            Reference time or orbital state.
        :type input: Orbital_State or float

        :return:
            Next orbital state in chronological order.
        :rtype: Orbital_State

        :raises ValueError:
            If the input time lies outside the orbit span.

        """
        if isinstance(input, Orbital_State):
            t = float(input.J2000)
        elif isinstance(input, float):
            t = float(input)
        else:
            raise ValueError("Must be j2000 time or orbital state")

        if t > self.max_time():
            raise ValueError("this orbital state is not within this orbit (too far in future)")
        if t < self.min_time():
            raise ValueError("this orbital state is not within this orbit (too far in past)")

        ind = int(np.flatnonzero(self.times >= t)[0])
        return self.states[self.times[ind]]

    def min_time(self) -> float:
        r"""
        Return the earliest epoch in the orbit.

        :return:
            Minimum J2000 time.
        :rtype: float

        """
        return float(np.amin(self.times))

    def max_time(self) -> float:
        r"""
        Return the latest epoch in the orbit.

        :return:
            Maximum J2000 time.
        :rtype: float

        """
        return float(np.amax(self.times))

    def time_in_span(self, t) -> bool:
        r"""
        Check whether a time lies within the orbit span.

        :param t:
            Time in J2000 centuries.
        :type t: float

        :return:
            ``True`` if the time is within the orbit span.
        :rtype: bool

        """
        t = float(t)
        return t <= self.max_time() and t >= self.min_time()

    def geocentric_to_ecef_orbit(self, b_vec: np.ndarray) -> np.ndarray:
        r"""
        Convert geocentric spherical vectors to ECEF coordinates along the orbit.

        :param b_vec:
            Geocentric vector components (b_r, b_theta, b_phi).
        :type b_vec: numpy.ndarray

        :return:
            Vectors expressed in the ECEF frame.
        :rtype: numpy.ndarray

        """
        ecef_mat = np.vstack([self.states[j].ECEF for j in self.times])
        n_ecef = matrix_row_normalize(ecef_mat)
        svec = matrix_row_normalize(np.cross(MathConstants.unitvecs[2], n_ecef))
        return (
            b_vec[:, 0:1] * n_ecef
            + svec * b_vec[:, 2:3]
            + matrix_row_normalize(np.cross(svec, n_ecef)) * b_vec[:, 1:2]
        )

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
        return np.stack([self.states[self.times[j]].ecef_to_eci(b_ecef_vec[j, :]) for j in range(len(self.times))])

    def get_b_eci_orbit(self) -> np.ndarray:
        r"""
        Compute the geomagnetic field in the ECI frame along the orbit.

        If per-state magnetic field vectors are already available, they are returned
        directly.

        :return:
            Geomagnetic field vectors in ECI coordinates [Tesla].
        :rtype: numpy.ndarray

        """
        try:
            return np.vstack([self.states[j].B for j in self.times])
        except Exception:
            # Fallback to legacy computation if any state is missing B
            geos = np.vstack([self.states[j].geocentric for j in self.times])
            dts = [self.states[j].datetime for j in self.times]
            b_r, b_th, b_ph = ppigrf.igrf_gc(geos[:, 0], geos[:, 1] * 180.0 / np.pi, geos[:, 2] * 180.0 / np.pi, dts)
            b_r = np.asarray(b_r)
            b_th = np.asarray(b_th)
            b_ph = np.asarray(b_ph)
            if b_r.ndim == 2:
                b_r = np.diagonal(b_r)
            if b_th.ndim == 2:
                b_th = np.diagonal(b_th)
            if b_ph.ndim == 2:
                b_ph = np.diagonal(b_ph)
            b_ecef = self.geocentric_to_ecef_orbit(np.vstack([b_r, b_th, b_ph]).T)
            b_eci = self.ecef_to_eci_orbit(b_ecef)
            return b_eci * 1e-9

    def get_vecs(self) -> List[List[np.ndarray]]:
        r"""
        Return commonly used orbit-related vectors.

        :return:
            Lists of vectors [R, V, B, S, rho] over the orbit.
        :rtype: list[list[numpy.ndarray]]

        """
        R = [self.states[j].R for j in self.times]
        V = [self.states[j].V for j in self.times]
        B = [self.states[j].B for j in self.times]
        S = [self.states[j].S for j in self.times]
        rho = [self.states[j].rho for j in self.times]
        return [R, V, B, S, rho]
