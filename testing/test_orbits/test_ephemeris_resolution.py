from __future__ import annotations

from pathlib import Path
import sys

import pytest

from ADCS.orbits.ephemeris import Ephemeris


@pytest.fixture(autouse=True)
def clear_ephemeris_environment(monkeypatch):
    monkeypatch.delenv("ADCS_EPHEMERIS_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_packaged_path_is_inside_the_package():
    path = Ephemeris._packaged_ephemeris_path()
    assert path.name == "de421.bsp"
    assert path.parent.name == "environment"
    assert path.parent.parent.name == "ADCS"


def test_cache_path_is_absolute_and_outside_site_packages():
    path = Ephemeris._cache_ephemeris_path()
    assert path.is_absolute()
    assert path.name == "de421.bsp"
    assert not {"site-packages", "dist-packages"} & {part.lower() for part in path.parts}


def test_environment_override_wins_and_expands_home(monkeypatch, tmp_path):
    explicit = tmp_path / "custom.bsp"
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(explicit))
    assert Ephemeris._cache_ephemeris_path() == explicit
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", "~/custom.bsp")
    assert Ephemeris._cache_ephemeris_path().is_absolute()


def test_xdg_cache_home_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert Ephemeris._cache_ephemeris_path() == tmp_path / "generalized_adcs" / "de421.bsp"


def test_platform_defaults_are_sensible():
    path = Ephemeris._cache_ephemeris_path()
    if sys.platform == "darwin":
        assert path.is_relative_to(Path.home() / "Library" / "Caches")
    else:
        assert path.is_absolute()


def test_existing_ephemeris_search_order(monkeypatch, tmp_path):
    cache = tmp_path / "cache" / "de421.bsp"
    package = tmp_path / "package" / "de421.bsp"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"package")
    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path", staticmethod(lambda: cache))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path", staticmethod(lambda: package))
    assert Ephemeris._find_existing_ephemeris() == package
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    assert Ephemeris._find_existing_ephemeris() == cache


def test_missing_ephemeris_returns_none_and_default_path_creates_its_parent(monkeypatch, tmp_path):
    cache = tmp_path / "cache" / "de421.bsp"
    monkeypatch.setattr(Ephemeris, "_cache_ephemeris_path", staticmethod(lambda: cache))
    monkeypatch.setattr(Ephemeris, "_packaged_ephemeris_path", staticmethod(lambda: tmp_path / "package" / "de421.bsp"))
    assert Ephemeris._find_existing_ephemeris() is None
    assert Ephemeris._get_default_ephemeris_path(Ephemeris) == cache
    assert cache.parent.is_dir()
