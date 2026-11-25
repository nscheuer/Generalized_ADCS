import numpy as np
import pytest


def make_state(backend):
    return backend.State(
        backend.AttitudeState([2, 0, 0, 0], [0.1, 0.2, 0.3]),
        backend.OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
        backend.InternalState([1.25, -0.5]),
    )


def test_record_layout_and_values(state_backend):
    record = make_state(state_backend).to_record()
    assert record.shape == (1,)
    assert record.dtype.names == (
        "quaternion",
        "rates",
        "position",
        "velocity",
        "momentum",
    )
    for field, expected in (
        ("quaternion", [1, 0, 0, 0]),
        ("rates", [0.1, 0.2, 0.3]),
        ("position", [7e6, 0, 0]),
        ("velocity", [0, 7500, 0]),
        ("momentum", [1.25, -0.5]),
    ):
        assert record.dtype[field].subdtype[0] == np.dtype(np.float64)
        np.testing.assert_allclose(record[field][0], expected)


def test_record_roundtrip_and_wheel_count(state_backend):
    state = make_state(state_backend)
    record = state.to_record(wheel_count=2)
    rebuilt = state_backend.State.from_record(record, 2451545.0)
    np.testing.assert_allclose(rebuilt.to_record()["momentum"], record["momentum"])
    with pytest.raises(ValueError):
        state.to_record(wheel_count=1)
    with pytest.raises(ValueError):
        state.to_record(wheel_count=-1)


def test_default_internal_and_keplerian_record_failure(state_backend):
    attitude = state_backend.AttitudeState([1, 0, 0, 0], [0, 0, 0])
    cartesian = state_backend.OrbitalState.from_cartesian([0, 0, 0], [0, 0, 0], 1.0)
    assert (
        np.asarray(
            state_backend.State(attitude, cartesian).internal.wheel_momentum
        ).size
        == 0
    )
    keplerian = state_backend.OrbitalState.from_keplerian(7000, 0, 0, 0, 0, 0, 1.0)
    with pytest.raises(
        Exception, match="Keplerian to Cartesian conversion is not implemented"
    ):
        state_backend.State(attitude, keplerian).to_record()
