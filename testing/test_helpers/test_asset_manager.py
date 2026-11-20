"""Local lookup and download-consent behavior for AssetManager."""

import io
from types import SimpleNamespace

import pytest

import ADCS.helpers.asset_manager as asset_manager
from ADCS.helpers.asset_manager import Asset, AssetManager


@pytest.fixture
def manager(tmp_path):
    asset = Asset("example", "https://example.invalid/example.dat", "example.dat")
    return AssetManager(
        registry={"example": asset},
        locations=[tmp_path / "packaged"],
        cache_dir=tmp_path / "cache",
    )


def test_default_lookup_prefers_bundled_file_then_cache(manager):
    cache = manager.cache_dir / "example.dat"
    packaged = manager.locations[0] / "example.dat"
    packaged.parent.mkdir()
    packaged.write_bytes(b"packaged")
    assert manager.get("example", interactive=False) == packaged

    cache.parent.mkdir()
    cache.write_bytes(b"cached")
    assert manager.get("example", interactive=False) == packaged
    packaged.unlink()
    assert manager.get("example", interactive=False) == cache


@pytest.mark.parametrize("platform, expected", [
    ("linux", "xdg/generalized_adcs"),
    ("darwin", "Library/Caches/generalized_adcs"),
    ("win32", "AppData/Local/generalized_adcs"),
])
def test_user_cache_root_by_platform(monkeypatch, tmp_path, platform, expected):
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(asset_manager.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    if platform == "linux":
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert asset_manager.user_cache_dir() == tmp_path / expected


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_xdg_cache_home_wins_on_every_platform(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert asset_manager.user_cache_dir() == tmp_path / "xdg" / "generalized_adcs"


def test_windows_localappdata_override(monkeypatch, tmp_path):
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert asset_manager.user_cache_dir() == tmp_path / "local" / "generalized_adcs"


def test_destination_uses_only_the_requested_path(manager, tmp_path):
    cache = manager.cache_dir / "example.dat"
    cache.parent.mkdir()
    cache.write_bytes(b"cached")
    destination = tmp_path / "custom" / "example.dat"

    with pytest.raises(FileNotFoundError, match="Missing example"):
        manager.get("example", destination=destination, interactive=False)

    destination.parent.mkdir()
    destination.write_bytes(b"custom")
    assert manager.get("example", destination=destination, interactive=False) == destination


def test_missing_asset_without_terminal_needs_approval(manager, monkeypatch):
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(stdin=io.StringIO()))

    with pytest.raises(FileNotFoundError, match="approve the download"):
        manager.get("example")

    assert not manager.cache_dir.exists()
