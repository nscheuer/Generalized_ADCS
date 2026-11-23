"""Magnetic coefficient files are resolved and checksummed once per process, not per evaluation."""

from hashlib import md5

import pytest

from ADCS.environment.magnetic_field import _coefficients


class _CountingManager:
    calls = 0

    def get(self, name, *, destination=None, **kwargs):
        type(self).calls += 1
        return destination


@pytest.fixture
def counting_manager(monkeypatch):
    _coefficients._verified_kernel.cache_clear()
    _CountingManager.calls = 0
    monkeypatch.setattr(_coefficients, "AssetManager", _CountingManager)
    yield _CountingManager
    _coefficients._verified_kernel.cache_clear()


def test_ensure_kernel_checks_each_file_once(tmp_path, counting_manager):
    path = tmp_path / "coeffs.shc"
    path.write_bytes(b"coefficients")
    checksum = md5(b"coefficients").hexdigest()

    for _ in range(5):
        assert _coefficients.ensure_kernel(path, "chaos8_core", checksum) == path

    assert counting_manager.calls == 1


def test_ensure_kernel_does_not_cache_a_failed_check(tmp_path, counting_manager):
    path = tmp_path / "coeffs.shc"
    path.write_bytes(b"corrupt")
    checksum = md5(b"coefficients").hexdigest()

    with pytest.raises(ValueError, match="checksum"):
        _coefficients.ensure_kernel(path, "chaos8_core", checksum)
    path.write_bytes(b"coefficients")
    assert _coefficients.ensure_kernel(path, "chaos8_core", checksum) == path
