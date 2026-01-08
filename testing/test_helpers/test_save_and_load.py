import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.save_and_load.save_and_load import save_data, load_data, load_orbital_states
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State


@pytest.mark.fast
def test_save_and_load_roundtrip_tmpdir(tmp_path):
    # --- Create small, deterministic dummy data (fast) ---
    rng = np.random.default_rng(123)

    time_hist = np.arange(5, dtype=float)
    state_hist = rng.standard_normal((5, 10))
    sensor_hist = rng.standard_normal((5, 3))
    u_hist = rng.standard_normal((5, 6))
    boresight_hist = rng.standard_normal((5, 3))

    # --- Create a tiny os_hist (2 states) ---
    ephem = Ephemeris()
    R = np.array([7000.0, 0.0, 0.0])
    V = np.array([0.0, 7.5, 0.0])
    os0 = Orbital_State(ephem=ephem, J2000=0.22, R=R, V=V, fast=True)
    os1 = Orbital_State(ephem=ephem, J2000=0.22 + 1e-6, R=R + 1.0, V=V + 1e-3, fast=True)
    os_hist = [os0, os1]

    # --- Save into tmp directory with nontrivial name ---
    run_dir = save_data(
        "pytest_save_load__nontrivial",
        time_hist,
        state_hist,
        os_hist,
        sensor_hist,
        u_hist,
        boresight_hist,
        labels=["time_hist", "state_hist", "os_hist", "sensor_hist", "u_hist", "boresight_hist"],
        out_dir=tmp_path,
        add_timestamp=True,
    )

    # Basic files exist
    run_path = tmp_path / run_dir.split("/")[-1]  # run_dir is absolute; get folder name
    assert run_path.exists()
    assert (run_path / "manifest.json").exists()
    assert (run_path / "arrays.npz").exists()
    assert (run_path / "orbital_states.pkl").exists()

    # --- Load generic (os_hist comes back as list[dict]) ---
    time2, state2, os_dicts, sensor2, u2, boresight2 = load_data(run_dir)

    assert np.allclose(time2, time_hist)
    assert np.allclose(state2, state_hist)
    assert np.allclose(sensor2, sensor_hist)
    assert np.allclose(u2, u_hist)
    assert np.allclose(boresight2, boresight_hist)

    assert isinstance(os_dicts, list)
    assert len(os_dicts) == 2
    assert isinstance(os_dicts[0], dict)
    assert "J2000" in os_dicts[0] and "R" in os_dicts[0] and "V" in os_dicts[0]

    # --- Load orbital states without ephem (raw dicts) ---
    os_raw = load_orbital_states(run_dir, label="os_hist", ephem=None)
    assert isinstance(os_raw, list)
    assert isinstance(os_raw[0], dict)

    # --- Load orbital states with ephem (reconstruct objects) ---
    os_loaded = load_orbital_states(run_dir, label="os_hist", ephem=Ephemeris(), fast=True)
    assert isinstance(os_loaded, list)
    assert isinstance(os_loaded[0], Orbital_State)

    # Check key fields match (allow float tolerances)
    assert np.isclose(os_loaded[0].J2000, os0.J2000)
    assert np.allclose(os_loaded[0].R, os0.R)
    assert np.allclose(os_loaded[0].V, os0.V)


def test_save_data_rejects_label_mismatch(tmp_path):
    a = np.zeros(3)
    b = np.zeros((2, 2))
    with pytest.raises(ValueError):
        save_data("bad_labels", a, b, labels=["only_one_label"], out_dir=tmp_path, add_timestamp=False)


def test_load_data_missing_manifest(tmp_path):
    # Create an empty folder to simulate a bad run directory
    run_dir = tmp_path / "missing_manifest__nontrivial"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_data(run_dir)
