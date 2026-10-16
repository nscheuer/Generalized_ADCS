"""WMM2025 and CHAOS-8.7 model values, shapes, and kernel acquisition."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import ADCS
from ADCS.environment.magnetic_field import CHAOS8, WMM2025, field_gc
from ADCS.environment.magnetic_field import chaos8, wmm
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.universal_constants import TimeConstants


KERNELS = Path(ADCS.__file__).parent / "environment" / "kernels"


@pytest.mark.parametrize(
    "radius,date,reference",
    [
        # NOAA WMM2025_TEST_VALUES.txt, geodetic latitude 0, longitude 120.
        # At the equator geodetic and geocentric latitude agree. The original
        # north/east/down X/Y/Z are -Btheta/Bphi/-Br respectively.
        (6378.137, datetime(2025, 1, 1), (10580.2, -39677.8, -109.6)),
        (6478.137, datetime(2025, 1, 1), (10152.1, -37688.6, -96.2)),
        (6378.137, datetime(2027, 7, 2, 12), (10381.8, -39701.6, -167.4)),
        (6478.137, datetime(2027, 7, 2, 12), (9969.8, -37711.5, -148.7)),
    ],
)
def test_wmm_matches_noaa_published_values(radius, date, reference):
    np.testing.assert_allclose(WMM2025().field_gc(radius, 90, 120, date), reference, atol=0.06)


def test_wmm_matches_noaa_high_latitude_geodetic_value():
    # NOAA's 2025-01-01, 80 N, 0 E, zero ellipsoidal height test case.
    # WMM's public test coordinates are geodetic, while field_gc is geocentric.
    latitude = np.radians(80.0)
    flattening = 1 / 298.257223563
    eccentricity_sq = flattening * (2 - flattening)
    prime_vertical = 6378.137 / np.sqrt(1 - eccentricity_sq * np.sin(latitude) ** 2)
    x = prime_vertical * np.cos(latitude)
    z = prime_vertical * (1 - eccentricity_sq) * np.sin(latitude)
    geocentric_latitude = np.arctan2(z, x)
    br, bt, bp = WMM2025().field_gc(
        np.hypot(x, z), 90 - np.degrees(geocentric_latitude), 0, datetime(2025, 1, 1)
    )
    tilt = latitude - geocentric_latitude
    north = -br * np.sin(tilt) - bt * np.cos(tilt)
    down = -br * np.cos(tilt) + bt * np.sin(tilt)
    np.testing.assert_allclose((north, bp, down), (6521.6, 145.9, 54791.5), atol=0.06)


@pytest.mark.parametrize(
    "date,radius,theta,phi,reference",
    [
        # CHAOS-8.7 core + static SHC, generated independently with
        # ChaosMagPy 0.16 load_CHAOS_shcfile/synth_values_tdep/static.
        (datetime(2018, 1, 1), 6371.2, 90, 0, (16057.492990, -27640.396552, -2392.204157)),
        (datetime(2026, 1, 1), 6771.2, 45, 10, (-34667.484843, -19101.114795, 983.373701)),
        (datetime(2026, 10, 16), 6371.2, 35, -75, (-54741.519149, -12816.236571, -3738.483419)),
    ],
)
def test_chaos_matches_chaosmagpy_reference(date, radius, theta, phi, reference):
    np.testing.assert_allclose(CHAOS8().field_gc(radius, theta, phi, date), reference, atol=1e-3)


@pytest.mark.parametrize("model,date", [
    (WMM2025(), datetime(2026, 5, 1)),
    (CHAOS8(), datetime(2026, 5, 1)),
])
def test_vector_dates_ecef_and_environment(model, date):
    positions = np.array([[6771.2, 0, 0], [0, 6771.2, 0], [0, 0, 6771.2]])
    dates = [date, datetime(2026, 6, 1), datetime(2026, 7, 1)]
    environment = ADCS.Environment(magnetic_field=model)
    assert environment.magnetic_field is model
    assert getattr(ADCS.Environment.MagneticField, type(model).__name__) is type(model)
    batch = model.field(positions, dates)
    expected = np.stack([model.field(pos, dt) for pos, dt in zip(positions, dates)])
    assert batch.shape == (3, 3)
    np.testing.assert_allclose(batch, expected, atol=1e-13)
    assert np.all(np.isfinite(batch))
    assert np.linalg.norm(batch[0]) > 1e-5
    assert field_gc(model.name, 6771.2, 90, 0, date) == model.field_gc(6771.2, 90, 0, date)


def test_wmm_bounds_and_date_shape():
    model = WMM2025()
    with pytest.raises(ValueError, match="valid from"):
        model.field_gc(6771.2, 45, 0, datetime(2024, 12, 31))
    with pytest.raises(ValueError, match="valid from"):
        model.field_gc(6771.2, 45, 0, datetime(2030, 1, 1))
    with pytest.raises(ValueError, match="expected 1 or 3"):
        model.field_gc(np.full(3, 6771.2), 45, 0, [datetime(2026, 1, 1)] * 2)


def test_model_name_aliases():
    date = datetime(2026, 1, 1)
    assert field_gc("wmm", 6771.2, 45, 0, date) == WMM2025().field_gc(6771.2, 45, 0, date)
    assert field_gc("CHAOS-8", 6771.2, 45, 0, date) == CHAOS8().field_gc(6771.2, 45, 0, date)


@pytest.mark.parametrize("name,model_type", [("wmm", WMM2025), ("chaos-8", CHAOS8)])
def test_legacy_selection_creates_concrete_model(name, model_type):
    state = ADCS.Orbital_State(
        J2000=0.26,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        magnetic_model=name,
    )
    assert isinstance(state.environment.magnetic_field, model_type)
    assert np.all(np.isfinite(state.B))


def test_chaos_bounds_and_static_component():
    model = CHAOS8()
    with pytest.raises(ValueError, match="core coefficients cover"):
        model.field_gc(6771.2, 45, 0, datetime(2030, 1, 1))
    date = datetime(2026, 1, 1)
    full = np.array(model.field_gc(6371.2, 45, 10, date))
    core = np.array(CHAOS8(include_static=False).field_gc(6371.2, 45, 10, date))
    assert np.linalg.norm(full - core) > 0.1


def test_missing_wmm_kernel_downloads(monkeypatch, tmp_path):
    source = KERNELS / "WMM2025.COF"
    monkeypatch.setattr(wmm, "WMM2025_URL", source.as_uri())
    destination = tmp_path / "kernels" / "WMM2025.COF"
    model = WMM2025(destination)
    model.field_gc(6771.2, 45, 10, datetime(2026, 1, 1))
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.with_name(destination.name + ".part").exists()


def test_missing_chaos_kernels_download(monkeypatch, tmp_path):
    core = KERNELS / "CHAOS-8.7_core.shc"
    static = KERNELS / "CHAOS-8.7_static.shc"
    monkeypatch.setattr(chaos8, "CHAOS8_CORE_URL", core.as_uri())
    monkeypatch.setattr(chaos8, "CHAOS8_STATIC_URL", static.as_uri())
    model = CHAOS8(tmp_path / core.name, tmp_path / static.name)
    model.field_gc(6771.2, 45, 10, datetime(2026, 1, 1))
    assert model.core_path.read_bytes() == core.read_bytes()
    assert model.static_path.read_bytes() == static.read_bytes()


def test_corrupt_chaos_download_is_rejected(monkeypatch, tmp_path):
    corrupt = tmp_path / "corrupt.shc"
    corrupt.write_text("not a CHAOS model\n")
    monkeypatch.setattr(chaos8, "CHAOS8_CORE_URL", corrupt.as_uri())
    destination = tmp_path / "kernels" / "CHAOS-8.7_core.shc"
    with pytest.raises(ValueError, match="failed checksum"):
        CHAOS8(core_path=destination, include_static=False).field_gc(
            6771.2, 45, 10, datetime(2026, 1, 1)
        )
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


@pytest.mark.parametrize("model", [WMM2025(), CHAOS8()])
def test_orbit_uses_model_for_propagated_states(model):
    environment = ADCS.Environment(magnetic_field=model)
    initial = ADCS.Orbital_State(
        J2000=0.26,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
        environment=environment,
    )
    orbit = Orbit(
        initial,
        end_time=initial.J2000 + 20 * TimeConstants.sec2cent,
        dt=10,
        verbose=False,
    )
    assert len(orbit.times) >= 3
    for state in orbit.states.values():
        assert state.environment is environment
        assert state.magnetic_model == model.name
        np.testing.assert_allclose(
            state.B,
            state.ecef_to_eci(model.field(state.ECEF, state.datetime)),
            atol=1e-12,
        )
