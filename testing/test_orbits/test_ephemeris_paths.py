"""Ephemeris file resolution: never write into site-packages.

de421.bsp is 16 MB and is deliberately not shipped in the wheel, so a
pip-installed user downloads it on first use. It previously landed *inside*
site-packages, which:

* hard-failed with a misleading "Failed to download" RuntimeError on read-only
  or system-managed installs, when the real cause was a PermissionError; and
* survived ``pip uninstall`` as a 16 MB orphan, being absent from RECORD.

These tests pin the search order and the write location. They never download
anything.
"""

import os
import sys
from pathlib import Path

import pytest

from ADCS.orbits.ephemeris import Ephemeris


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("ADCS_EPHEMERIS_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_download_target_is_not_inside_site_packages():
    """The regression that motivated all of this."""
    target = Ephemeris._cache_ephemeris_path()
    parts = [p.lower() for p in target.parts]
    assert "site-packages" not in parts, f"would write into site-packages: {target}"
    assert "dist-packages" not in parts, f"would write into dist-packages: {target}"


def test_cache_path_is_absolute_and_named():
    target = Ephemeris._cache_ephemeris_path()
    assert target.is_absolute()
    assert target.name == "de421.bsp"
    assert "generalized_adcs" in target.parts


def test_env_override_wins(monkeypatch, tmp_path):
    custom = tmp_path / "somewhere" / "my_de421.bsp"
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(custom))
    assert Ephemeris._cache_ephemeris_path() == custom


def test_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", "~/eph/de421.bsp")
    resolved = Ephemeris._cache_ephemeris_path()
    assert "~" not in str(resolved)
    assert resolved.is_absolute()


def test_xdg_cache_home_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = Ephemeris._cache_ephemeris_path()
    assert target == tmp_path / "generalized_adcs" / "de421.bsp"


def test_platform_default_without_xdg(monkeypatch):
    target = Ephemeris._cache_ephemeris_path()
    home = Path.home()
    if sys.platform == "darwin":
        assert target.is_relative_to(home / "Library" / "Caches")
    elif sys.platform == "win32":
        assert target.is_absolute()
    else:
        assert target.is_relative_to(home / ".cache")


def test_packaged_path_points_at_the_package(monkeypatch):
    packaged = Ephemeris._packaged_ephemeris_path()
    assert packaged.name == "de421.bsp"
    assert packaged.parent.name == "environment"
    # must sit under the ADCS package, not the repo root
    assert packaged.parent.parent.name == "ADCS"


def test_search_order_prefers_the_cache_over_the_packaged_copy(monkeypatch, tmp_path):
    cache = tmp_path / "cache" / "de421.bsp"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"not a real ephemeris")
    packaged = tmp_path / "pkg" / "de421.bsp"
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"also not real")

    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path", staticmethod(lambda: cache))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path", staticmethod(lambda: packaged))
    assert Ephemeris._find_existing_ephemeris() == cache


def test_falls_back_to_the_packaged_copy(monkeypatch, tmp_path):
    missing = tmp_path / "cache" / "de421.bsp"
    packaged = tmp_path / "pkg" / "de421.bsp"
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"stand-in")

    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path", staticmethod(lambda: missing))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path", staticmethod(lambda: packaged))
    assert Ephemeris._find_existing_ephemeris() == packaged


def test_returns_none_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path",
                        staticmethod(lambda: tmp_path / "a" / "de421.bsp"))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path",
                        staticmethod(lambda: tmp_path / "b" / "de421.bsp"))
    assert Ephemeris._find_existing_ephemeris() is None


def test_default_download_path_creates_only_its_own_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = Ephemeris()._get_default_ephemeris_path.__func__(Ephemeris)
    assert target.parent.is_dir(), "the cache directory should be created"
    assert target.parent == tmp_path / "generalized_adcs"


def test_resolution_does_not_touch_the_package_directory(monkeypatch, tmp_path):
    """Resolving a path must not mkdir inside the installed package."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pkg_env_dir = Ephemeris._packaged_ephemeris_path().parent
    existed_before = pkg_env_dir.exists()
    Ephemeris._cache_ephemeris_path()
    Ephemeris._find_existing_ephemeris()
    assert pkg_env_dir.exists() == existed_before, (
        "path resolution must not create directories inside the package")
