"""Fast API tests for querying any body present in an ephemeris."""

import numpy as np

from ADCS.environment.ephemeris.de421 import DE421


class _FakeTimeScale:
    def tt_jd(self, jd):
        return (np.asarray(jd, dtype=float) - 2451545.0) / 36525.0


class _FakePosition:
    def __init__(self, km):
        self.position = type("Position", (), {"km": km})()

    def apparent(self):
        return self


class _FakeBody:
    def __init__(self, offset):
        self.offset = offset

    def at(self, times):
        self.times = np.asarray(times)
        return self

    def observe(self, target):
        # Skyfield returns coordinates with axes first: (3,) or (3, N).
        values = self.offset + target.offset + np.atleast_1d(self.times)
        km = np.vstack((values, values + 1.0, values + 2.0))
        if np.asarray(self.times).ndim == 0:
            km = km[:, 0]
        return _FakePosition(km)


class _FakeKernel(dict):
    def names(self):
        return {10: ["SUN"], 399: ["EARTH"], 499: ["MARS"]}


def _ephemeris():
    ephemeris = DE421.__new__(DE421)
    ephemeris.ts = _FakeTimeScale()
    ephemeris.planets = _FakeKernel({
        "earth": _FakeBody(0.0),
        "sun": _FakeBody(1.0),
        "mars": _FakeBody(2.0),
    })
    ephemeris.earth = ephemeris.planets["earth"]
    ephemeris.sun = ephemeris.planets["sun"]
    return ephemeris


def test_position_eci_queries_any_named_kernel_body():
    ephemeris = _ephemeris()

    np.testing.assert_array_equal(ephemeris.position_eci("mars", 0.0), [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(ephemeris.sun_position_eci(0.0), [1.0, 2.0, 3.0])


def test_position_eci_vectorizes_epochs_and_can_return_geometric_position():
    result = _ephemeris().position_eci("mars", [0.0, 1.0], apparent=False)

    np.testing.assert_array_equal(result, [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])


def test_available_bodies_reports_kernel_names_and_aliases():
    assert _ephemeris().available_bodies == ("earth", "mars", "sun")
