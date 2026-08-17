"""Attitude_Estimator must accept an injected Ephemeris.

Constructing an Ephemeris can download a 16 MB kernel. Attitude_Estimator
previously hard-coded ``Ephemeris()`` in its constructor, so a caller had no
way to share one instance, and no way to build an estimator on a machine that
cannot reach the network -- the documented ``filepath=`` escape hatch could not
reach that call site.
"""

import inspect

import numpy as np
import pytest

from ADCS.estimators.attitude_estimators.attitude_estimator import Attitude_Estimator
from ADCS.orbits.ephemeris import Ephemeris


def test_constructor_exposes_an_ephem_parameter():
    params = inspect.signature(Attitude_Estimator.__init__).parameters
    assert "ephem" in params, "no way to inject an Ephemeris"
    assert params["ephem"].default is None, "ephem must be optional"


def test_injected_ephemeris_is_used_instead_of_building_one(monkeypatch):
    """With one supplied, the constructor must not build its own."""
    from ADCS.estimators.attitude_estimators import SRUAKF
    from testing.test_estimators.ukf.helpers import (
        make_estimate_guess, reduced_state_cov, reduced_process_cov,
        make_satellites, make_baseline_sensors,
    )

    shared = Ephemeris()        # built once here, from cache or packaged copy

    built = []

    def _tripwire(*a, **k):     # must never be called when ephem is supplied
        built.append(1)
        raise AssertionError("constructor built its own Ephemeris despite injection")

    monkeypatch.setattr(
        "ADCS.estimators.attitude_estimators.attitude_estimator.Ephemeris",
        _tripwire,
    )

    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    dt = 5.0
    est = SRUAKF(
        est_sat=est_sat, J2000=0.22,
        x_hat=make_estimate_guess(est_sat),
        P_hat=reduced_state_cov(est_sat),
        Q_hat=reduced_process_cov(est_sat, dt=dt),
        dt=dt, ephem=shared,
    )
    assert not built, "Ephemeris was constructed even though one was injected"
    assert est.prev_os.ephem is shared, "the injected ephemeris was not used"


def test_without_injection_it_still_builds_one(monkeypatch):
    """The default path must keep working -- injection is opt-in."""
    from ADCS.estimators.attitude_estimators import SRUAKF
    from testing.test_estimators.ukf.helpers import (
        make_estimate_guess, reduced_state_cov, reduced_process_cov,
        make_satellites, make_baseline_sensors,
    )

    _, est_sat = make_satellites(sensors=make_baseline_sensors())
    dt = 5.0
    est = SRUAKF(
        est_sat=est_sat, J2000=0.22,
        x_hat=make_estimate_guess(est_sat),
        P_hat=reduced_state_cov(est_sat),
        Q_hat=reduced_process_cov(est_sat, dt=dt),
        dt=dt,
    )
    assert isinstance(est.prev_os.ephem, Ephemeris)


def test_orbital_state_convention_is_matched():
    """Orbital_State already accepted ephem=None; the estimator now does too.

    Note Orbital_State takes ephem as a *required* parameter that tolerates
    None, rather than defaulting it -- so this asserts the tolerated-None
    behaviour, not a default.
    """
    from ADCS.orbits.orbital_state import Orbital_State

    assert "ephem" in inspect.signature(Orbital_State.__init__).parameters
    os_ = Orbital_State(ephem=None, J2000=0.22, R=np.array([7000.0, 0, 0]),
                        V=np.array([0, 7.5, 0]))
    assert isinstance(os_.ephem, Ephemeris), "ephem=None must build one"


def test_shared_ephemeris_is_the_same_object():
    """Two consumers given one ephemeris must not each build their own."""
    from ADCS.orbits.orbital_state import Orbital_State

    shared = Ephemeris()
    # J2000 is in centuries; de421 covers ~1900-2053, so keep both in range.
    a = Orbital_State(ephem=shared, J2000=0.22, R=np.array([7000.0, 0, 0]),
                      V=np.array([0, 7.5, 0]))
    b = Orbital_State(ephem=shared, J2000=0.23, R=np.array([0, 7000.0, 0]),
                      V=np.array([-7.5, 0, 0]))
    assert a.ephem is shared and b.ephem is shared
