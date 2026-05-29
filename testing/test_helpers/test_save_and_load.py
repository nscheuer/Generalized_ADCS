import json
import pickle

import numpy as np
import pytest

from ADCS.helpers.save_and_load.save_and_load import load_data, load_orbital_states, save_data
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State


def make_orbital_states():
    ephem = Ephemeris()
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7.5, 0.0])
    os0 = Orbital_State(ephem=ephem, J2000=0.22, R=r0, V=v0, fast=True)
    os1 = Orbital_State(ephem=ephem, J2000=0.220001, R=r0 + 1.0, V=v0 + 1e-3, fast=True)
    return [os0, os1]


def test_save_and_load_roundtrip_for_arrays_and_orbital_states(tmp_path):
    rng = np.random.default_rng(123)
    time_hist = np.arange(5, dtype=float)
    state_hist = rng.standard_normal((5, 10))
    sensor_hist = rng.standard_normal((5, 3))
    u_hist = rng.standard_normal((5, 6))
    boresight_hist = rng.standard_normal((5, 3))
    os_hist = make_orbital_states()

    run_dir = save_data(
        "pytest_save_load",
        time_hist,
        state_hist,
        os_hist,
        sensor_hist,
        u_hist,
        boresight_hist,
        labels=["time_hist", "state_hist", "os_hist", "sensor_hist", "u_hist", "boresight_hist"],
        out_dir=tmp_path,
        add_timestamp=False,
    )

    time2, state2, os_dicts, sensor2, u2, boresight2 = load_data(run_dir)

    assert np.allclose(time2, time_hist)
    assert np.allclose(state2, state_hist)
    assert np.allclose(sensor2, sensor_hist)
    assert np.allclose(u2, u_hist)
    assert np.allclose(boresight2, boresight_hist)
    assert isinstance(os_dicts, list)
    assert isinstance(os_dicts[0], dict)
    assert {"J2000", "R", "V"} <= set(os_dicts[0])


def test_save_data_creates_expected_files_and_manifest_entries(tmp_path):
    array = np.arange(4.0)
    payload = {"mode": "test", "count": 2}
    os_hist = make_orbital_states()

    run_dir = save_data(
        "manifest_case",
        array,
        payload,
        os_hist,
        labels=["array", "payload", "os_hist"],
        out_dir=tmp_path,
        add_timestamp=False,
    )

    with open(f"{run_dir}/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = {item["label"]: item for item in manifest["items"]}
    assert set(items) == {"array", "payload", "os_hist"}
    assert items["array"]["kind"] == "ndarray"
    assert items["payload"]["kind"] == "pickle"
    assert items["os_hist"]["kind"] == "orbital_state_list"

    assert (tmp_path / "manifest_case" / "arrays.npz").exists()
    assert (tmp_path / "manifest_case" / "objects.pkl").exists()
    assert (tmp_path / "manifest_case" / "orbital_states.pkl").exists()


def test_save_data_uses_explicit_path_over_name_and_out_dir(tmp_path):
    explicit = tmp_path / "custom" / "run_dir"

    run_dir = save_data(
        "ignored_name",
        np.arange(3.0),
        labels=["array"],
        out_dir=tmp_path / "unused",
        path=explicit,
        add_timestamp=True,
    )

    assert run_dir == str(explicit.resolve())
    assert explicit.exists()


def test_save_data_rejects_missing_name_when_path_not_provided(tmp_path):
    with pytest.raises(ValueError):
        save_data(None, np.arange(3.0), labels=["array"], out_dir=tmp_path)


def test_save_data_rejects_label_mismatch(tmp_path):
    with pytest.raises(ValueError):
        save_data(
            "bad_labels",
            np.zeros(3),
            np.zeros((2, 2)),
            labels=["only_one_label"],
            out_dir=tmp_path,
            add_timestamp=False,
        )


def test_save_data_supports_default_labels(tmp_path):
    run_dir = save_data(
        "default_labels",
        np.arange(2.0),
        {"x": 1},
        out_dir=tmp_path,
        add_timestamp=False,
    )

    with open(f"{run_dir}/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert [item["label"] for item in manifest["items"]] == ["obj0", "obj1"]


def test_save_data_can_write_uncompressed_arrays(tmp_path):
    run_dir = save_data(
        "uncompressed",
        np.arange(5.0),
        labels=["array"],
        out_dir=tmp_path,
        add_timestamp=False,
        compress=False,
    )

    loaded = np.load(f"{run_dir}/arrays.npz", allow_pickle=False)
    assert np.allclose(loaded["array"], np.arange(5.0))


def test_load_orbital_states_returns_raw_dicts_without_ephemeris(tmp_path):
    run_dir = save_data(
        "raw_orbital_states",
        make_orbital_states(),
        labels=["os_hist"],
        out_dir=tmp_path,
        add_timestamp=False,
    )

    os_raw = load_orbital_states(run_dir, label="os_hist", ephem=None)

    assert isinstance(os_raw, list)
    assert isinstance(os_raw[0], dict)


def test_load_orbital_states_reconstructs_objects_with_ephemeris(tmp_path):
    os_hist = make_orbital_states()
    run_dir = save_data(
        "reconstruct_orbital_states",
        os_hist,
        labels=["os_hist"],
        out_dir=tmp_path,
        add_timestamp=False,
    )

    loaded = load_orbital_states(run_dir, label="os_hist", ephem=Ephemeris(), fast=True)

    assert isinstance(loaded, list)
    assert isinstance(loaded[0], Orbital_State)
    assert np.isclose(loaded[0].J2000, os_hist[0].J2000)
    assert np.allclose(loaded[0].R, os_hist[0].R)
    assert np.allclose(loaded[0].V, os_hist[0].V)


def test_load_orbital_states_returns_mapping_for_multiple_histories(tmp_path):
    run_dir = save_data(
        "multi_orbital_state_histories",
        make_orbital_states(),
        make_orbital_states(),
        labels=["primary", "secondary"],
        out_dir=tmp_path,
        add_timestamp=False,
    )

    loaded = load_orbital_states(run_dir, ephem=None)

    assert isinstance(loaded, dict)
    assert set(loaded) == {"primary", "secondary"}


def test_load_data_missing_manifest_raises(tmp_path):
    run_dir = tmp_path / "missing_manifest"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_data(run_dir)


def test_load_data_rejects_unknown_manifest_kind(tmp_path):
    run_dir = tmp_path / "bad_kind"
    run_dir.mkdir()

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "items": [
                    {
                        "label": "bad",
                        "var_name": "bad",
                        "kind": "mystery",
                        "file": "none",
                        "key": "bad",
                        "info": {},
                    }
                ]
            },
            f,
        )

    with pytest.raises(ValueError, match="Unknown kind"):
        load_data(run_dir)


def test_load_orbital_states_raises_for_missing_label(tmp_path):
    run_dir = save_data(
        "missing_label",
        make_orbital_states(),
        labels=["os_hist"],
        out_dir=tmp_path,
        add_timestamp=False,
    )

    with pytest.raises(KeyError):
        load_orbital_states(run_dir, label="does_not_exist", ephem=None)
