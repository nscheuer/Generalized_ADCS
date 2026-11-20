"""Bundled ephemerides are read only; missing kernels use the user cache."""

import io
from types import SimpleNamespace

import pytest

import ADCS.helpers.asset_manager as asset_manager
from ADCS.environment.ephemeris import DE421, DE442, DE442s, DE441


@pytest.fixture(autouse=True)
def _clear_overrides(monkeypatch):
    for name in ("ADCS_EPHEMERIS_PATH", "ADCS_DE442_EPHEMERIS_PATH",
                 "ADCS_DE442S_EPHEMERIS_PATH", "ADCS_DE441_EPHEMERIS_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("model", [DE421, DE442, DE442s])
def test_bundled_kernel_is_used_without_download(monkeypatch, tmp_path, model):
    packaged = tmp_path / "package" / model.KERNEL_FILENAME
    packaged.parent.mkdir()
    packaged.write_bytes(b"bundled")
    monkeypatch.setattr(model, "_packaged_ephemeris_path", classmethod(lambda cls: packaged))
    monkeypatch.setattr(asset_manager.AssetManager, "get", lambda *a, **k: pytest.fail("download attempted"))
    assert model._find_existing_ephemeris() == packaged


@pytest.mark.parametrize("model", [DE421, DE442, DE442s])
def test_missing_kernel_download_target_is_user_cache(monkeypatch, tmp_path, model):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(model, "_packaged_ephemeris_path",
                        classmethod(lambda cls: tmp_path / "read_only_package" / cls.KERNEL_FILENAME))
    requested = []

    def fake_get(self, name, **kwargs):
        requested.append((name, kwargs["destination"]))
        return kwargs["destination"]

    monkeypatch.setattr(asset_manager.AssetManager, "get", fake_get)
    expected = tmp_path / "cache" / "generalized_adcs" / model.KERNEL_FILENAME
    assert model._find_existing_ephemeris() == expected
    assert requested == [(model.KERNEL_FILENAME.removesuffix(".bsp"), expected)]
    assert not (tmp_path / "read_only_package").exists()


def test_explicit_kernel_path_is_respected(monkeypatch, tmp_path):
    custom = tmp_path / "custom" / "de421.bsp"
    custom.parent.mkdir()
    custom.write_bytes(b"custom")
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(custom))
    assert DE421._find_existing_ephemeris() == custom


def test_missing_kernel_does_not_create_package_directory(monkeypatch, tmp_path):
    packaged = tmp_path / "package" / "de421.bsp"
    monkeypatch.setattr(DE421, "_packaged_ephemeris_path", staticmethod(lambda: packaged))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(platform="linux", stdin=io.StringIO()))
    with pytest.raises(FileNotFoundError, match="Missing de421") as error:
        DE421._find_existing_ephemeris()
    assert str(tmp_path / "cache") in str(error.value)
    assert not packaged.parent.exists()


def test_de441_uses_complete_bundled_pair(monkeypatch, tmp_path):
    packaged = tuple(tmp_path / "package" / name for name in DE441.KERNEL_FILENAMES)
    packaged[0].parent.mkdir()
    for path in packaged:
        path.write_bytes(b"bundled")
    monkeypatch.setattr(DE441, "_packaged_ephemeris_paths", classmethod(lambda cls: packaged))
    monkeypatch.setattr(asset_manager.AssetManager, "get", lambda *a, **k: pytest.fail("download attempted"))
    assert DE441._find_existing_ephemeris() == packaged


def test_de441_missing_pair_uses_user_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(DE441, "_packaged_ephemeris_paths",
                        classmethod(lambda cls: tuple(tmp_path / "package" / name for name in cls.KERNEL_FILENAMES)))
    destinations = []

    def fake_get(self, name, **kwargs):
        destinations.append(kwargs["destination"])
        return kwargs["destination"]

    monkeypatch.setattr(asset_manager.AssetManager, "get", fake_get)
    expected = tuple(tmp_path / "cache" / "generalized_adcs" / name for name in DE441.KERNEL_FILENAMES)
    assert DE441._find_existing_ephemeris() == expected
    assert destinations == list(expected)
    assert not (tmp_path / "package").exists()
