import numpy as np
import pytest

from GenADCS.backends.python.environment.ephemeris.de421.de421 import DE421, _load_spk


def test_de421_packaged_kernel_loads_and_is_cached():
    _load_spk.cache_clear()
    model = DE421()
    second = DE421()

    assert DE421._packaged_ephemeris_path().is_file()
    assert "GenADCS/kernels/environment/ephemeris/de421" in str(
        DE421._packaged_ephemeris_path()
    )
    assert model._segments is second._segments
    assert _load_spk.cache_info().hits >= 1
    assert "earth" in model.available_bodies
    assert "sun" in model.available_bodies


def test_de421_sun_state_is_plausible_and_consistent():
    model = DE421()
    epoch = 2_458_849.5
    position = model.position_eci("sun", epoch)
    velocity = model.velocity_eci("sun", epoch)
    state_position, state_velocity = model.state_eci("sun", epoch)

    assert position.shape == velocity.shape == (3,)
    assert 1.4e8 < np.linalg.norm(position) < 1.6e8
    assert 25.0 < np.linalg.norm(velocity) < 35.0
    np.testing.assert_allclose(position, state_position, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(velocity, state_velocity, rtol=1e-12, atol=1e-12)


def test_de421_vectorizes_epochs():
    model = DE421()
    positions = model.position_eci("moon", np.array((2_458_849.5, 2_458_850.5)))

    assert positions.shape == (2, 3)
    assert np.all(np.isfinite(positions))
    assert np.linalg.norm(positions[0]) != np.linalg.norm(positions[1])


@pytest.mark.parametrize("target", ("not-a-body", 123456))
def test_de421_rejects_unknown_body(target):
    with pytest.raises(ValueError, match="unsupported DE421"):
        DE421().position_eci(target, 2_458_849.5)


def test_de421_rejects_epoch_outside_kernel_coverage():
    with pytest.raises(ValueError, match="no SPK segment covers"):
        DE421().position_eci("sun", 1_000_000.0)


def test_de421_rejects_malformed_kernel(tmp_path):
    bad_kernel = tmp_path / "not-an-spk.bsp"
    bad_kernel.write_bytes(b"not an SPK")
    _load_spk.cache_clear()

    with pytest.raises(ValueError, match="not a DAF/SPK"):
        DE421(bad_kernel)
