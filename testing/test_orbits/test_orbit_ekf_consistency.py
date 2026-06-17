"""
Consistency / correctness guards for Orbit_EKF (test-hardening backlog #7).

The shipped test file testing/test_orbit_EKF.py is a plot/demo script: it has
NO test function and NO assertions, so pytest collects nothing and the EKF was
entirely unverified. Under that blind spot three real defects survived:

1. (covariance) Orbit_EKF used the CONTINUOUS dynamics Jacobian
   A = df/dx (= [[0, I],[G(r), 0]], zero diagonal blocks, no dt) DIRECTLY as
   the discrete state-transition matrix in P_pred = Fk P0 Fk^T + Q. That is
   dimensionally wrong; the predicted covariance is meaningless and the
   filter is catastrophically over-confident (time-averaged NEES ~1e9 vs the
   ~6 expected for a consistent 6-state filter). The correct STM is the
   Jacobian of the actual propagation map (propagate_orbit_rk4); the
   codebase already provides propagate_jacobians_rk4.
2. (guard) __init__ did `return ValueError(...)` instead of `raise`, so a
   GPS-less satellite produced a confusing `TypeError: __init__() should
   return None` and skipped reset(), instead of a clear ValueError.
3. (guard) reset()'s shape-check messages referenced non-existent
   attributes self.P / self.Q, raising AttributeError instead of the
   intended ValueError on a mis-shaped P/Q.

The FD test is the rigorous core: it compares the EKF's predicted covariance
against an INDEPENDENT finite-difference Jacobian of the very propagation the
EKF performs -- not the EKF compared to itself.
"""

import numpy as np
import pytest

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.errors import Noise, ErrorMode
from ADCS.satellite_hardware.sensors import GPS
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.estimators.orbit_estimators import Orbit_EKF

_R0 = 7000.0 * np.array([0.0, -np.sqrt(0.5), -np.sqrt(0.5)])
_V0 = np.array([7.546, 0.0, 0.0])
_DT = 10.0


def _gps_sat():
    n = Noise(noise=np.zeros(6), std_noise=np.full(6, 0.02))
    return Satellite(sensors=[GPS(noise=n.copy())])


def _fresh_ekf(P0, Q0, ephem, start):
    os_hat0 = Orbital_State(ephem=ephem, J2000=start, R=_R0.copy(), V=_V0.copy())
    return Orbit_EKF(est_sat=_gps_sat(), J2000=start, os_hat=os_hat0,
                     P_hat=P0, Q_hat=Q0, dt=_DT)


def test_ekf_time_update_covariance_matches_fd_stm():
    """Pure time update (no measurements): the EKF's predicted covariance
    must equal Phi_fd P0 Phi_fd^T + Q, where Phi_fd is the central-difference
    Jacobian of the SAME propagate_orbit_rk4 the EKF runs for the state.

    RED on origin/main: Fk = continuous A (||A - Phi_fd|| ~ 16) -> P_pred is
    off by orders of magnitude. GREEN after: ||Fk - Phi_fd|| ~ 1e-8.
    """
    ephem = Ephemeris()
    start = 0.22
    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1e-3**2, 1e-3**2, 1e-3**2])
    Q0 = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])

    ekf = _fresh_ekf(P0, Q0, ephem, start)
    ekf.update(GPS_measurements=[], J2000=start)
    P_pred = np.asarray(ekf.os_hat.P, float)

    # Independent FD STM of the exact propagation the EKF performs.
    def prop(x):
        o = Orbital_State(ephem=ephem, J2000=start, R=x[:3], V=x[3:],
                          S=None, B=None, rho=None, fast=False)
        op = o.propagate_orbit_rk4(dt=_DT, J2_perturbation_on=True, fast=True)
        return np.hstack([op.R, op.V])

    x0 = np.hstack([_R0, _V0])
    Phi_fd = np.zeros((6, 6))
    for i in range(6):
        h = 1e-4 * max(1.0, abs(x0[i]))
        dx = np.zeros(6); dx[i] = h
        Phi_fd[:, i] = (prop(x0 + dx) - prop(x0 - dx)) / (2.0 * h)

    P_ref = Phi_fd @ P0 @ Phi_fd.T + Q0
    np.testing.assert_allclose(P_pred, P_ref, rtol=1e-3, atol=1e-9)


def test_ekf_filter_consistency_nees():
    """End-to-end: time-averaged NEES of the 6-state estimate. The continuous
    -A-as-STM bug yields NEES ~1e9 (RED). With the discrete STM the filter is
    sane; bound is generous (residual ~2e2 is process-noise tuning, the same
    open Q-convention theme as the SRUKF backlog -- deliberately not chased
    here) but >1e6 lower than the broken filter."""
    np.random.seed(0)
    ephem = Ephemeris()
    start = 0.22 - 1 * TimeConstants.sec2cent
    tf = 2000.0
    end = 0.22 + tf * TimeConstants.sec2cent

    os0 = Orbital_State(ephem=ephem, J2000=start, R=_R0.copy(), V=_V0.copy())
    orb = Orbit(os0=os0, end_time=end, dt=_DT, use_J2=True, fast=False)

    gps = GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, 0.02)))
    est_sat = Satellite(sensors=[gps])

    P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 1e-3**2, 1e-3**2, 1e-3**2])
    Q0 = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
    e0 = np.random.multivariate_normal(np.zeros(6), P0)
    os_hat0 = Orbital_State(ephem=ephem, J2000=start,
                            R=_R0 + e0[:3], V=_V0 + e0[3:])
    ekf = Orbit_EKF(est_sat=est_sat, J2000=start, os_hat=os_hat0,
                    P_hat=P0, Q_hat=Q0, dt=_DT)

    N = int(tf / _DT)
    nees = np.zeros(N)
    t = 0.0
    for k in range(N):
        J2000 = 0.22 + t * TimeConstants.sec2cent
        os_true = orb.get_os(J2000=J2000)
        meas = gps.reading(x=None, os=os_true,
                           dmode=ErrorMode(add_bias=True, add_noise=True,
                                           update_bias=True, update_noise=True))
        est = ekf.update(GPS_measurements=[meas], J2000=J2000)
        e = np.hstack([os_true.R, os_true.V]) - np.hstack([est.os.R, est.os.V])
        nees[k] = e @ np.linalg.solve(np.asarray(est.P, float), e)
        t += _DT

    mean_nees = float(np.mean(nees[5:]))
    assert np.isfinite(mean_nees)
    assert mean_nees < 1.0e3, f"filter inconsistent: mean NEES {mean_nees:.3e}"


def test_ekf_requires_gps_sensor_raises_valueerror():
    """GPS-less satellite must raise ValueError. RED on origin/main:
    `return ValueError(...)` in __init__ -> TypeError (__init__ should
    return None), not ValueError."""
    ephem = Ephemeris()
    start = 0.22
    no_gps = Satellite(sensors=[])
    os_hat0 = Orbital_State(ephem=ephem, J2000=start, R=_R0.copy(), V=_V0.copy())
    with pytest.raises(ValueError):
        Orbit_EKF(est_sat=no_gps, J2000=start, os_hat=os_hat0,
                  P_hat=np.eye(6), Q_hat=np.eye(6), dt=_DT)


def test_ekf_reset_rejects_misshaped_covariance_with_valueerror():
    """A non-6x6 P must raise ValueError. RED on origin/main: the error
    message references non-existent self.P -> AttributeError, not
    ValueError."""
    ephem = Ephemeris()
    start = 0.22
    os_hat0 = Orbital_State(ephem=ephem, J2000=start, R=_R0.copy(), V=_V0.copy())
    with pytest.raises(ValueError):
        Orbit_EKF(est_sat=_gps_sat(), J2000=start, os_hat=os_hat0,
                  P_hat=np.eye(5), Q_hat=np.eye(6), dt=_DT)
