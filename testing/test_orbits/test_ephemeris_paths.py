"""DE421 resolves its default kernel through AssetManager without downloading."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import ADCS.helpers.asset_manager as asset_manager
from ADCS.environment.ephemeris import DE421


@pytest.fixture(autouse=True)
def _clear_override(monkeypatch):
    monkeypatch.delenv("ADCS_EPHEMERIS_PATH", raising=False)


@pytest.fixture
def packaged_path(monkeypatch, tmp_path):
    path = tmp_path / "pkg" / "de421.bsp"
    monkeypatch.setattr(DE421, "_packaged_ephemeris_path", staticmethod(lambda: path))
    return path


def test_default_resolution_uses_packaged_kernel_even_when_cache_exists(monkeypatch, tmp_path, packaged_path):
    packaged_path.parent.mkdir()
    packaged_path.write_bytes(b"packaged stand-in")
    cache = tmp_path / "cache" / "generalized_adcs" / "de421.bsp"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cached stand-in")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert DE421._find_existing_ephemeris() == packaged_path


def test_explicit_path_overrides_packaged_kernel(monkeypatch, tmp_path, packaged_path):
    packaged_path.parent.mkdir()
    packaged_path.write_bytes(b"packaged stand-in")
    custom = tmp_path / "custom" / "my_de421.bsp"
    custom.parent.mkdir()
    custom.write_bytes(b"custom stand-in")
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(custom))

    assert DE421._find_existing_ephemeris() == custom


def test_missing_kernel_raises_without_prompt_or_creating_directories(monkeypatch, packaged_path):
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(stdin=io.StringIO()))

    with pytest.raises(FileNotFoundError, match="Missing de421") as error:
        DE421._find_existing_ephemeris()

    assert str(packaged_path) in str(error.value)
    assert not packaged_path.parent.exists()


def test_default_download_path_is_packaged_kernel_directory(packaged_path):
    target = DE421._get_default_ephemeris_path(DE421)

    assert target == packaged_path
    assert target.parent.is_dir()


def test_packaged_path_points_inside_adcs_package():
    path = Path(__file__).resolve().parents[2] / "ADCS" / "environment" / "kernels" / "de421.bsp"
    assert DE421._packaged_ephemeris_path() == path
