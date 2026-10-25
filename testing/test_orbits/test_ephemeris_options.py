"""Ephemeris kernel selection, EOP option and provenance. Never downloads."""

from pathlib import Path

import pytest

from ADCS.orbits.ephemeris import Ephemeris


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("ADCS_EPHEMERIS_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)


def test_unknown_kernel_is_rejected():
    with pytest.raises(ValueError, match="name must be one of"):
        Ephemeris(name="de430")


def test_unknown_eop_source_is_rejected():
    with pytest.raises(ValueError, match="eop must be one of"):
        Ephemeris(eop="bulletinA")


def test_cache_path_follows_the_kernel_name(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert Ephemeris._cache_ephemeris_path("de440s.bsp") == tmp_path / "generalized_adcs" / "de440s.bsp"
    assert Ephemeris._packaged_ephemeris_path("de440s.bsp").name == "de440s.bsp"


def test_env_override_contributes_only_its_directory_for_other_kernels(monkeypatch, tmp_path):
    custom = tmp_path / "eph" / "my_de421.bsp"
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(custom))
    assert Ephemeris._cache_ephemeris_path() == custom
    assert Ephemeris._cache_ephemeris_path("de440s.bsp") == custom.parent / "de440s.bsp"


def test_existing_kernel_is_found_by_name(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    target = tmp_path / "generalized_adcs" / "de442s.bsp"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stand-in")
    assert Ephemeris._find_existing_ephemeris("de442s.bsp") == target


def test_default_provenance_identifies_the_kernel():
    ephem = Ephemeris()
    prov = ephem.provenance()
    assert prov["name"] == Path(ephem.planets.path).stem
    assert len(prov["sha256"]) == 64
    assert prov["eop"] is None
    assert prov["eop_path"] is None and prov["eop_sha256"] is None
    assert ephem.ts.polar_motion_table is None
