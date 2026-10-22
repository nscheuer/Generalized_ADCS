"""adcs-vizlog writer/reader round trips and crash recovery."""

import json

import numpy as np
import pytest

from ADCS.viz import VizLog, VizLogWriter
from ADCS.viz.schema import REQUIRED_CHANNELS, channel_spec

N = 7


def _channels(extra=()):
    return [channel_spec(n) for n in REQUIRED_CHANNELS] + list(extra)


def _columns(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {c: rng.standard_normal((n, spec.width)) for c, spec in REQUIRED_CHANNELS.items()}


def _writer(path, **kw):
    return VizLogWriter(path, kw.pop("channels", _channels()), epoch_jd_tt=2459580.5,
                        epoch_iso_utc="2021-12-31T23:58:50.816Z", scene={"spacecraft": {"mass_kg": 4.0}}, **kw)


def test_bulk_round_trip(tmp_path):
    cols = _columns()
    with _writer(tmp_path / "a.vizlog") as w:
        w.write_run(0, cols, dt_s=1.0)
    log = VizLog.open(tmp_path / "a.vizlog")
    assert log.run_ids == ["0"] and log.n_rows(0) == N and log.is_complete(0)
    assert log.epoch_jd_tt == 2459580.5
    assert log.scene["spacecraft"]["mass_kg"] == 4.0
    for name, arr in cols.items():
        np.testing.assert_array_equal(log.read(0, name), arr)
    with pytest.raises(ValueError):
        log.read(0, "r_sc")[0, 0] = 1.0  # memory maps are read-only


def test_missing_optional_columns_are_nan(tmp_path):
    with _writer(tmp_path / "a.vizlog", channels=_channels([channel_spec("r_moon")])) as w:
        w.write_run("r1", _columns())
    assert np.isnan(VizLog.open(tmp_path / "a.vizlog", mmap=False).read("r1", "r_moon")).all()


def test_appended_rows_match_bulk(tmp_path):
    cols = _columns()
    with _writer(tmp_path / "a.vizlog", flush_every=3) as w:
        w.begin_run(0)
        for k in range(N):
            w.append(0, **{c: v[k] for c, v in cols.items()})
    log = VizLog.open(tmp_path / "a.vizlog")
    assert log.n_rows(0) == N and log.is_complete(0)
    for name, arr in cols.items():
        np.testing.assert_array_equal(log.read(0, name), arr)


def test_interrupted_log_exposes_only_whole_rows(tmp_path):
    cols = _columns()
    w = _writer(tmp_path / "a.vizlog", flush_every=2)
    w.begin_run(0)
    for k in range(5):
        w.append(0, **{c: v[k] for c, v in cols.items()})
    # Simulated crash: 4 rows flushed, row 5 still buffered; then a torn write.
    with open(tmp_path / "a.vizlog" / "runs" / "0" / "r_sc.f64", "ab") as f:
        f.write(b"\x00" * 12)
    log = VizLog.open(tmp_path / "a.vizlog")
    assert not log.is_complete(0)
    assert log.n_rows(0) == 4
    np.testing.assert_array_equal(log.read(0, "q_gcrs_body"), cols["q_gcrs_body"][:4])


def test_required_channels_are_enforced(tmp_path):
    with pytest.raises(ValueError, match="missing required"):
        _writer(tmp_path / "a.vizlog", channels=[channel_spec("t_s")])


def test_existing_output_is_protected(tmp_path):
    _writer(tmp_path / "a.vizlog").close()
    with pytest.raises(FileExistsError):
        _writer(tmp_path / "a.vizlog")
    _writer(tmp_path / "a.vizlog", overwrite=True).close()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not a vizlog"):
        _writer(tmp_path / "other", overwrite=True)
    assert (tmp_path / "other" / "keep.txt").exists()


@pytest.mark.parametrize("bad", ["../x", "a b", "", "x" * 65])
def test_run_ids_are_path_safe(tmp_path, bad):
    with _writer(tmp_path / "a.vizlog") as w, pytest.raises(ValueError):
        w.begin_run(bad)


def test_width_mismatch_is_rejected(tmp_path):
    cols = _columns()
    cols["r_sc"] = cols["r_sc"][:, :2]
    with _writer(tmp_path / "a.vizlog") as w, pytest.raises(ValueError, match="width"):
        w.write_run(0, cols)


def test_reader_rejects_other_versions(tmp_path):
    _writer(tmp_path / "a.vizlog").close()
    manifest = tmp_path / "a.vizlog" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["schema_version"] = 99
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema version"):
        VizLog.open(tmp_path / "a.vizlog")


def test_manifest_is_strict_json(tmp_path):
    with _writer(tmp_path / "a.vizlog", metadata={"note": float("nan")}) as w:
        w.write_run(0, _columns())
    text = (tmp_path / "a.vizlog" / "manifest.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["note"] is None


def test_variable_width_channels_need_a_width():
    with pytest.raises(ValueError):
        channel_spec("h_wheel")
    assert channel_spec("h_wheel", 3, labels=["a", "b", "c"]).width == 3
    with pytest.raises(ValueError):
        channel_spec("r_sc", 4)
    with pytest.raises(KeyError):
        channel_spec("made_up")
