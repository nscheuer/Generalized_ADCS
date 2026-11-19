import json
from importlib import import_module

import numpy as np
import pytest

from ADCS.helpers.animations.godot.schema import REQUIRED_CHANNELS
from ADCS.helpers.animations.godot.writer import VizLogWriter, _jsonable


def test_jsonable_maps_numpy_non_finite_scalars_to_null():
    payload = _jsonable({"dt_s": np.float64("nan"), "span": [np.float32("inf"), np.float64(2.5)]})

    assert payload == {"dt_s": None, "span": [None, 2.5]}
    json.dumps(payload, allow_nan=False)


@pytest.mark.local
def test_overwrite_rejects_unrelated_manifest_without_deleting_directory(tmp_path):
    path = tmp_path / "project"
    path.mkdir()
    manifest = path / "manifest.json"
    manifest.write_text('{"project": "user data"}', encoding="utf-8")
    important = path / "important.txt"
    important.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not an ADCS vizlog"):
        VizLogWriter(path, REQUIRED_CHANNELS.values(), 2451545.0, overwrite=True)

    assert manifest.exists()
    assert important.read_text(encoding="utf-8") == "keep me"


@pytest.mark.local
def test_overwrite_replaces_a_valid_vizlog(tmp_path):
    path = tmp_path / "run.vizlog"
    with VizLogWriter(path, REQUIRED_CHANNELS.values(), 2451545.0):
        pass
    (path / "old-data.txt").write_text("old", encoding="utf-8")

    with VizLogWriter(path, REQUIRED_CHANNELS.values(), 2451546.0, overwrite=True):
        pass

    assert not (path / "old-data.txt").exists()
    assert json.loads((path / "manifest.json").read_text(encoding="utf-8"))["epoch"]["jd_tt_whole"] == 2451546.0


@pytest.mark.local
@pytest.mark.parametrize("overwrite", [False, True])
def test_animate_orbit_godot_overwrite_is_explicit(monkeypatch, tmp_path, overwrite):
    animate_module = import_module("ADCS.helpers.animations.godot.animate_orbit_godot")

    captured = {}

    def write_vizlog(results, output, **kwargs):
        captured["overwrite"] = kwargs["overwrite"]
        return output

    monkeypatch.setattr(animate_module, "results_to_vizlog", write_vizlog)
    monkeypatch.setattr(animate_module, "view", lambda *args, **kwargs: 0)

    animate_module.animate_orbit_godot(results=object(), output=tmp_path / "run.vizlog",
                                       overwrite=overwrite, launch=False)

    assert captured["overwrite"] is overwrite
