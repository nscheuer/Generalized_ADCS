"""The magnetic field model is selectable, and the choice sticks.

``magnetic_model`` picks the implementation behind every ``B`` in the package
(``"igrf_numba"`` by default, ``"igrf_ppigrf"`` for the 0.1.8 behaviour). The
two must give the same field, and a choice made on one state or orbit must
reach every state derived from it -- a state that quietly fell back to the
default would mix models within one simulation.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

from ADCS.environment import DEFAULT_MAGNETIC_MODEL, MAGNETIC_MODELS, field_gc
from ADCS.environment.magnetic_field import _PPIGRF_CHUNK
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants

RTOL = 1e-9
J0 = 0.22


def _worst(ref, got):
    return max(
        float(np.max(np.abs((np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(a), 1.0))))
        for a, b in zip(ref, got)
    )


@pytest.fixture(scope="module")
def ephem():
    return Ephemeris()


def _os0(ephem, model=DEFAULT_MAGNETIC_MODEL):
    return Orbital_State(ephem, J0, np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]),
                         magnetic_model=model)


def _orbit(os0, **kw):
    return Orbit(os0, end_time=J0 + 120 * TimeConstants.sec2cent, dt=10, verbose=False, **kw)


# --------------------------------------------------------------------------
# field_gc
# --------------------------------------------------------------------------


def test_default_is_the_numba_kernel():
    assert DEFAULT_MAGNETIC_MODEL == "igrf_numba"
    assert set(MAGNETIC_MODELS) == {"igrf_numba", "igrf_ppigrf"}


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="magnetic_model must be one of"):
        field_gc("wmm", 6771.2, 45.0, 10.0, datetime(2024, 1, 1))


def test_model_name_is_case_insensitive():
    d = datetime(2024, 1, 1)
    assert field_gc("IGRF_Numba", 6771.2, 45.0, 10.0, d) == field_gc("igrf_numba", 6771.2, 45.0, 10.0, d)


def test_missing_ppigrf_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "ppigrf", None)
    with pytest.raises(ImportError, match=r"Generalized_ADCS\[ppigrf\]"):
        field_gc("igrf_ppigrf", 6771.2, 45.0, 10.0, datetime(2024, 1, 1))


def test_backends_agree_for_a_scalar():
    pytest.importorskip("ppigrf")
    d = datetime(2024, 3, 1, 12)
    ref = field_gc("igrf_ppigrf", 6771.2, 45.0, 10.0, d)
    got = field_gc("igrf_numba", 6771.2, 45.0, 10.0, d)
    assert all(np.ndim(x) == 0 for x in ref)
    assert _worst(ref, got) < RTOL


def test_backends_agree_for_one_date():
    pytest.importorskip("ppigrf")
    rng = np.random.default_rng(1)
    n = 2000
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    th = rng.uniform(0.5, 179.5, n)
    ph = rng.uniform(-180.0, 180.0, n)
    d = datetime(2021, 7, 4)
    ref = field_gc("igrf_ppigrf", r, th, ph, d)
    assert all(np.shape(x) == (n,) for x in ref)
    assert _worst(ref, field_gc("igrf_numba", r, th, ph, d)) < RTOL


def test_backends_agree_with_one_date_per_sample_across_chunks():
    """More samples than one ppigrf chunk, and not a multiple of it."""
    pytest.importorskip("ppigrf")
    rng = np.random.default_rng(2)
    n = 2 * _PPIGRF_CHUNK + 37
    r = 6371.2 + rng.uniform(300.0, 800.0, n)
    th = rng.uniform(0.5, 179.5, n)
    ph = rng.uniform(-180.0, 180.0, n)
    dates = [datetime(2016, 1, 1) + timedelta(days=5.3 * i) for i in range(n)]
    ref = field_gc("igrf_ppigrf", r, th, ph, dates)
    assert all(np.shape(x) == (n,) for x in ref)
    assert _worst(ref, field_gc("igrf_numba", r, th, ph, dates)) < RTOL


def test_ppigrf_rejects_mismatched_date_count():
    pytest.importorskip("ppigrf")
    with pytest.raises(ValueError, match="expected 1 or 5"):
        field_gc("igrf_ppigrf", np.full(5, 6771.2), np.full(5, 45.0), np.full(5, 10.0),
                 [datetime(2024, 1, 1), datetime(2024, 1, 2)])


# --------------------------------------------------------------------------
# Orbital_State / Orbit
# --------------------------------------------------------------------------


def test_state_rejects_unknown_model(ephem):
    with pytest.raises(ValueError):
        _os0(ephem, "nope")


def test_orbits_with_either_model_have_the_same_field(ephem):
    pytest.importorskip("ppigrf")
    a = _orbit(_os0(ephem, "igrf_numba"))
    b = _orbit(_os0(ephem, "igrf_ppigrf"))
    Ba = np.array([a.states[t].B for t in a.times])
    Bb = np.array([b.states[t].B for t in b.times])
    assert np.max(np.abs(Ba - Bb)) < RTOL * np.max(np.abs(Bb))


def test_orbit_inherits_the_initial_states_model(ephem):
    pytest.importorskip("ppigrf")
    orb = _orbit(_os0(ephem, "igrf_ppigrf"))
    assert orb._magnetic_model == "igrf_ppigrf"
    assert all(st.magnetic_model == "igrf_ppigrf" for st in orb.states.values())


def test_explicit_orbit_model_overrides_and_recomputes_the_states(ephem):
    pytest.importorskip("ppigrf")
    os0 = _os0(ephem, "igrf_ppigrf")
    os1 = os0.propagate_orbit_rk4(10.0)
    orb = Orbit([os0, os1], magnetic_model="igrf_numba")
    for st in orb.states.values():
        assert st.magnetic_model == "igrf_numba"
        np.testing.assert_allclose(st.B, st.get_b_eci(), rtol=0, atol=0)
    # The inputs are copied, never modified.
    assert os0.magnetic_model == "igrf_ppigrf"


def test_model_survives_every_way_a_state_is_derived(ephem):
    pytest.importorskip("ppigrf")
    model = "igrf_ppigrf"
    os0 = _os0(ephem, model)
    orb = _orbit(os0)
    mid = J0 + 15 * TimeConstants.sec2cent  # between nodes

    derived = {
        "copy": os0.copy(),
        "propagate_orbit": os0.propagate_orbit(5.0),
        "propagate_orbit_rk4": os0.propagate_orbit_rk4(5.0),
        "average": os0.average(os0.propagate_orbit_rk4(10.0)),
        "get_os": orb.get_os(mid),
        "from_dict": Orbital_State.from_dict(os0.to_dict(), ephem),
    }
    for how, st in derived.items():
        assert st.magnetic_model == model, how

    for how, sub in {
        "get_range": orb.get_range(orb.times[0], orb.times[4]),
        "get_range(dt)": orb.get_range(orb.times[0], orb.times[4], dt=5),
        "new_orbit_from_times": orb.new_orbit_from_times([J0, mid]),
    }.items():
        assert sub._magnetic_model == model, how
        assert all(st.magnetic_model == model for st in sub.states.values()), how


def test_from_dict_without_the_key_uses_the_default(ephem):
    """Dicts serialized before the field existed still load."""
    d = _os0(ephem).to_dict()
    del d["magnetic_model"]
    assert Orbital_State.from_dict(d, ephem).magnetic_model == DEFAULT_MAGNETIC_MODEL
