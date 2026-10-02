"""Orbit_EKF regressions from the September 2026 estimator audit.

The filter used the continuous Jacobian A = d(xdot)/dx as its state-transition
matrix, so the position block received the velocity variance and the filter
was overconfident by orders of magnitude (Monte Carlo NEES 3.5e9, 5.8 km
position error against 3.4 m with the real transition). It also propagated a
full step before applying a measurement taken at its own epoch, injecting
|v| dt of error into the first innovation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.estimators.orbit_estimators import Orbit_EKF
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.errors import Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import GPS
from ADCS.state import State


J2000_0 = 0.22
DT = 20.0


def _truth(j2000: float = J2000_0) -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(), J2000=j2000,
        R=np.array([6900.0, 1200.0, -300.0]), V=np.array([-1.3, 7.2, 1.1]), fast=True,
    )


def _satellite(std: float) -> EstimatedSatellite:
    return EstimatedSatellite(sensors=[GPS(noise=Noise(noise=np.zeros(6), std_noise=np.full(6, std)))])


def _filter(std: float, P0: np.ndarray, Q0: np.ndarray) -> Orbit_EKF:
    return Orbit_EKF(est_sat=_satellite(std), J2000=J2000_0, os_hat=_truth(), P_hat=P0, Q_hat=Q0, dt=DT)


def test_orbit_ekf_requires_a_gps_sensor():
    with pytest.raises(ValueError, match="GPS"):
        Orbit_EKF(est_sat=EstimatedSatellite(sensors=[]), J2000=J2000_0, os_hat=_truth(),
                  P_hat=np.eye(6), Q_hat=np.zeros((6, 6)), dt=DT)


def test_orbit_ekf_predicts_with_the_discrete_state_transition():
    """P^- must be Phi P Phi^T + Q with Phi the transition over the step, not
    A P A^T + Q with the continuous Jacobian."""
    P0 = np.diag([0.5**2] * 3 + [0.005**2] * 3)
    Q0 = np.diag([1.0e-8] * 3 + [1.0e-12] * 3)
    estimator = _filter(std=1.0, P0=P0, Q0=Q0)
    later = J2000_0 + DT * TimeConstants.sec2cent

    predicted = estimator.update(GPS_measurements=[], J2000=later)

    dr_dr, dr_dv, dv_dr, dv_dv = _truth().propagate_jacobians_rk4(dt=DT, zonal_J=2)
    phi = np.block([[dr_dr, dr_dv], [dv_dr, dv_dv]])
    # the filter's step comes from the J2000 difference, which carries ~7e-8 s
    # of round-off on a 20 s interval (2e-9 relative in P)
    np.testing.assert_allclose(predicted.P, phi @ P0 @ phi.T + Q0, rtol=1.0e-7, atol=1.0e-14)
    # the transition really moves position uncertainty with velocity uncertainty
    assert predicted.P[0, 0] > P0[0, 0] + (DT * 0.005) ** 2 * 0.5


def test_orbit_ekf_transition_matches_finite_differences_of_the_propagator():
    """Independent check of the transition the filter now relies on."""
    base = _truth()
    dr_dr, dr_dv, dv_dr, dv_dv = base.propagate_jacobians_rk4(dt=DT, zonal_J=2)
    phi = np.block([[dr_dr, dr_dv], [dv_dr, dv_dv]])
    x0 = np.concatenate((base.R, base.V))
    fd = np.zeros((6, 6))
    for j in range(6):
        h = 1.0e-3 if j < 3 else 1.0e-6
        plus, minus = x0.copy(), x0.copy()
        plus[j] += h
        minus[j] -= h
        op = Orbital_State(ephem=base.ephem, J2000=J2000_0, R=plus[:3], V=plus[3:], fast=True).propagate_orbit_rk4(dt=DT, zonal_J=2, fast=True)
        om = Orbital_State(ephem=base.ephem, J2000=J2000_0, R=minus[:3], V=minus[3:], fast=True).propagate_orbit_rk4(dt=DT, zonal_J=2, fast=True)
        fd[:, j] = (np.concatenate((op.R, op.V)) - np.concatenate((om.R, om.V))) / (2.0 * h)
    np.testing.assert_allclose(phi, fd, rtol=1.0e-4, atol=1.0e-6)


def test_orbit_ekf_first_update_at_its_own_epoch_does_not_propagate():
    """With an uninformative measurement the posterior is the prior; it used to
    be the prior propagated a full step, |v| dt = 150 km away."""
    P0 = np.diag([0.1**2] * 3 + [0.001**2] * 3)
    estimator = _filter(std=1.0e3, P0=P0, Q0=np.zeros((6, 6)))  # 1000 km GPS noise: K ~ 0
    measurement = estimator.est_sat.GPS_readings(x=State(w=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])), os=_truth())
    posterior = estimator.update(GPS_measurements=measurement, J2000=J2000_0)
    assert np.linalg.norm(posterior.os.R - _truth().R) < 1.0e-3  # km
    assert np.linalg.norm(posterior.os.V - _truth().V) < 1.0e-6

    # a later sample propagates by the actual elapsed time
    later = J2000_0 + DT * TimeConstants.sec2cent
    second = estimator.update(GPS_measurements=[], J2000=later)
    # the near-zero gain still moved the posterior by ~1e-5 km, which propagates
    np.testing.assert_allclose(second.os.R, _truth().propagate_orbit_rk4(dt=DT, zonal_J=2, fast=True).R, rtol=0.0, atol=1.0e-3)


def test_orbit_ekf_reset_does_not_propagate_first_update_at_reset_epoch():
    P0 = np.diag([0.1**2] * 3 + [0.001**2] * 3)
    estimator = _filter(std=1.0e3, P0=P0, Q0=np.zeros((6, 6)))
    estimator.update(GPS_measurements=[], J2000=J2000_0 + DT * TimeConstants.sec2cent)

    reset_epoch = J2000_0 + 2.0 * DT * TimeConstants.sec2cent
    reset_state = _truth(j2000=reset_epoch)
    estimator.reset(
        est_sat=estimator.est_sat,
        J2000=reset_epoch,
        os_hat=reset_state,
        P_hat=P0,
        Q_hat=np.zeros((6, 6)),
        dt=DT,
    )

    updated = estimator.update(GPS_measurements=[], J2000=reset_epoch)
    np.testing.assert_allclose(updated.os.R, reset_state.R)
    np.testing.assert_allclose(updated.os.V, reset_state.V)
