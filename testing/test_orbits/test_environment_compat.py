"""Behaviour main supported that the environment refactor must keep.

Each test pins something that worked before ``ADCS.orbits.ephemeris`` and
``ADCS.orbits.density_model`` moved under ``ADCS.environment``. Only the
WMM2025 case may fetch its 4 KB coefficient file on first use.
"""

import importlib

import numpy as np
import pytest
from skyfield.api import load

import ADCS.helpers.asset_manager as asset_manager
from ADCS.environment.environment import Environment
from ADCS.environment.ephemeris import DE421
from ADCS.environment.magnetic_field import DEFAULT_MAGNETIC_MODEL, make_magnetic_field
from ADCS.orbits.orbital_state import Orbital_State

REAL_KERNEL = DE421._packaged_ephemeris_path()
needs_kernel = pytest.mark.skipif(not REAL_KERNEL.is_file(), reason="needs a local de421.bsp")
# Open review items on 2_2/godot_graphics; strict so a fix is noticed.
pending_download_policy = pytest.mark.xfail(
    strict=True, raises=FileNotFoundError, reason="DE421 asks before downloading; the download policy is still undecided")
pending_compat_shims = pytest.mark.xfail(
    strict=True, raises=ModuleNotFoundError,
    reason="old ADCS.orbits import paths were removed without shims; keep or deprecate is undecided")


@pytest.fixture
def no_local_kernels(monkeypatch, tmp_path):
    """Hide every DE421 copy: empty user cache, empty packaged kernel directory."""
    monkeypatch.delenv("ADCS_EPHEMERIS_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(asset_manager, "KERNEL_DIR", tmp_path / "kernels")
    monkeypatch.setattr(DE421, "_packaged_ephemeris_path", staticmethod(lambda: tmp_path / "kernels" / "de421.bsp"))
    return tmp_path


@needs_kernel
def test_de421_honours_ephemeris_path_env(no_local_kernels, monkeypatch):
    custom = no_local_kernels / "elsewhere" / "my_de421.bsp"
    custom.parent.mkdir()
    custom.symlink_to(REAL_KERNEL)
    monkeypatch.setenv("ADCS_EPHEMERIS_PATH", str(custom))

    ephem = DE421()

    assert ephem.planets.path == str(custom)


@needs_kernel
@pending_download_policy
def test_de421_downloads_when_missing_without_a_terminal(no_local_kernels, monkeypatch):
    """Notebooks, CI and pytest have no TTY; main downloaded instead of raising."""
    requested = []

    def fake_download(self, save_path):
        requested.append(save_path)
        return load(str(REAL_KERNEL))

    monkeypatch.setattr(DE421, "_download_ephemeris", fake_download)

    DE421()

    assert requested == [no_local_kernels / "cache" / "generalized_adcs" / "de421.bsp"]


@pytest.mark.parametrize("module, name", [
    ("ADCS.orbits.ephemeris", "Ephemeris"),
    ("ADCS.orbits.density_model", "DensityModel"),
])
@pending_compat_shims
def test_old_import_paths_still_resolve(module, name):
    assert hasattr(importlib.import_module(module), name)


@pending_compat_shims
def test_orbital_state_accepts_density_model_keyword():
    density_model = importlib.import_module("ADCS.orbits.density_model").DensityModel()

    state = Orbital_State(
        J2000=0.26, R=np.array([6878.0, 0.0, 0.0]), V=np.array([0.0, 7.6, 0.0]),
        density_model=density_model,
    )

    assert state.rho == pytest.approx(float(density_model.interpolate(500.0)))


def test_from_dict_with_environment_keeps_non_default_magnetic_model():
    model = "wmm2025"
    assert model != DEFAULT_MAGNETIC_MODEL
    environment = Environment(magnetic_field=make_magnetic_field(model))
    state = Orbital_State(
        J2000=0.26, R=np.array([6878.0, 0.0, 0.0]), V=np.array([0.0, 7.6, 0.0]),
        environment=environment,
    )

    loaded = Orbital_State.from_dict(state.to_dict(), environment=environment)

    assert loaded.environment is environment
    assert loaded.magnetic_model == model
    np.testing.assert_allclose(loaded.B, state.B)
