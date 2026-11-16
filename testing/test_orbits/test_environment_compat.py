"""Behaviour main supported that the environment refactor must keep.

Each test pins something that worked before ``ADCS.orbits.ephemeris`` and
``ADCS.orbits.density_model`` moved under ``ADCS.environment``. Only the
WMM2025 case may fetch its 4 KB coefficient file on first use.
"""

import importlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

import ADCS.helpers.asset_manager as asset_manager
from ADCS.environment.environment import Environment
from ADCS.environment.ephemeris import DE421
from ADCS.environment.magnetic_field import DEFAULT_MAGNETIC_MODEL, make_magnetic_field
from ADCS.orbits.orbital_state import Orbital_State

REAL_KERNEL = DE421._packaged_ephemeris_path()
needs_kernel = pytest.mark.skipif(not REAL_KERNEL.is_file(), reason="needs a local de421.bsp")
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


def test_de421_requires_download_approval_when_missing_without_a_terminal(no_local_kernels, monkeypatch):
    monkeypatch.setattr(asset_manager, "sys", SimpleNamespace(stdin=io.StringIO()))

    with pytest.raises(FileNotFoundError, match="Missing de421") as error:
        DE421()

    assert str(no_local_kernels / "kernels" / "de421.bsp") in str(error.value)


@pytest.mark.parametrize("module, name", [
    ("ADCS.orbits.ephemeris", "Ephemeris"),
    ("ADCS.orbits.density_model", "DensityModel"),
])
def test_old_import_paths_still_resolve(module, name):
    assert hasattr(importlib.import_module(module), name)


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


def test_user_magnetic_field_without_a_name_drives_states_and_orbits():
    from ADCS.environment.magnetic_field import MagneticField
    from ADCS.orbits.orbit import Orbit
    from ADCS.orbits.universal_constants import TimeConstants

    class UniformRadialField(MagneticField):
        def field_gc(self, r, theta, phi, date):
            r = np.asarray(r, dtype=float)
            return 30000.0 * np.ones_like(r), np.zeros_like(r), np.zeros_like(r)

    environment = Environment(magnetic_field=UniformRadialField())
    assert environment.magnetic_model == "UniformRadialField"
    R = np.array([6878.0, 0.0, 0.0])
    state = Orbital_State(J2000=0.26, R=R, V=np.array([0.0, 7.6, 0.0]), environment=environment)
    np.testing.assert_allclose(state.B, 30000.0e-9 * R / np.linalg.norm(R), rtol=1e-6, atol=1e-15)

    dt = 60.0
    orbit = Orbit(state, end_time=state.J2000 + 3 * dt * TimeConstants.sec2cent, dt=dt, environment=environment)
    # get_range / new_orbit_from_times rebuild an Orbit; they must keep the user's model.
    sub = orbit.get_range(orbit.min_time(), orbit.max_time())
    assert sub.environment is environment
    resampled = orbit.new_orbit_from_times([orbit.min_time(), 0.5 * (orbit.min_time() + orbit.max_time())])
    assert resampled.environment is environment


def test_orbit_from_states_rebuilds_states_whose_atmosphere_differs():
    from ADCS.environment.atmosphere import Atmosphere
    from ADCS.orbits.orbit import Orbit

    class ConstantAtmosphere(Atmosphere):
        def density(self, position_eci_km, datetime_utc=None, geodetic_lla=None):
            return 1.0e-9

        def interpolate(self, altitude_km):
            return 1.0e-9

    R, V = np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.6, 0.0])
    smad_states = [Orbital_State(J2000=0.26 + k * 1e-6, R=R, V=V) for k in range(2)]  # default SMAD atmosphere
    assert smad_states[0].rho != pytest.approx(1.0e-9)  # the two atmospheres really differ here
    constant = Environment(atmosphere=ConstantAtmosphere(), ephemeris=smad_states[0].environment.ephemeris)

    orbit = Orbit(smad_states, environment=constant)

    for t in orbit.times:
        state = orbit.states[t]
        assert state.environment is constant
        assert state.rho == pytest.approx(1.0e-9)


def test_orbit_from_states_with_equivalent_models_adopts_without_rebuilding(monkeypatch):
    from ADCS.orbits import orbit as orbit_module

    R, V = np.array([6878.0, 0.0, 0.0]), np.array([0.0, 7.6, 0.0])
    state = Orbital_State(J2000=0.26, R=R, V=V)
    shared = state.environment
    equivalent = Environment(magnetic_field=make_magnetic_field(DEFAULT_MAGNETIC_MODEL), ephemeris=shared.ephemeris)
    assert equivalent is not shared and equivalent.same_models(shared)

    def no_rebuild(*args, **kwargs):
        raise AssertionError("an equivalent state should not be rebuilt")

    monkeypatch.setattr(orbit_module, "Orbital_State", no_rebuild)
    adopted = orbit_module.Orbit.__new__(orbit_module.Orbit)
    adopted.environment = equivalent
    out = adopted._adopt_magnetic_model(state.copy())
    assert out.environment is equivalent
    np.testing.assert_allclose(out.B, state.B)


def test_from_tle_without_environment_uses_the_shared_default():
    from ADCS.environment.environment import default_environment

    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391428221"
    state = Orbital_State.from_TLE(line1, line2)
    assert state.environment is default_environment()


def test_de441_honours_its_ephemeris_path_env(monkeypatch, tmp_path):
    from ADCS.environment.ephemeris import DE441

    directory = tmp_path / "de441"
    directory.mkdir()
    for name in DE441.KERNEL_FILENAMES:
        (directory / name).write_bytes(b"stand-in")
    monkeypatch.setenv("ADCS_DE441_EPHEMERIS_PATH", str(directory / DE441.KERNEL_FILENAMES[0]))

    assert DE441._find_existing_ephemeris() == tuple(directory / name for name in DE441.KERNEL_FILENAMES)
