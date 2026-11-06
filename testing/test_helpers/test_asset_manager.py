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


def test_default_lookup_checks_cache_then_packaged_location(manager):
    cache = manager.cache_dir / "example.dat"
    packaged = manager.locations[0] / "example.dat"
    packaged.parent.mkdir()
    packaged.write_bytes(b"packaged")
    assert manager.get("example", interactive=False) == packaged

    cache.parent.mkdir()
    cache.write_bytes(b"cached")
    assert manager.get("example", interactive=False) == cache


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
