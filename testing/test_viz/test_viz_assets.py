"""Asset pipeline: cube-map geometry, sampling, normals, stars, fetch and build (no network)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ADCS.viz.assets import products as assets_build
from ADCS.viz.assets.cubemap import (
    FACES,
    directions_to_face,
    equirect_to_cube,
    face_directions,
    lonlat,
    sample_equirect,
)
from ADCS.viz.assets.fetch import fetch, raw_path
from ADCS.viz.assets.stars import (
    COLUMNS,
    bv_to_temperature,
    magnitude_to_lux,
    parse_bsc5,
    star_table,
    temperature_to_linear_srgb,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------ cube maps
@pytest.mark.parametrize("face", FACES)
def test_face_centre_points_along_its_axis(face):
    d = face_directions(face, 2)
    centre = d.mean(axis=(0, 1))
    centre /= np.linalg.norm(centre)
    axis = np.zeros(3)
    axis["XYZ".index(face[1])] = 1.0 if face[0] == "+" else -1.0
    np.testing.assert_allclose(centre, axis, atol=1e-12)


def test_texel_centres_map_back_to_themselves():
    size = 32
    for i, face in enumerate(FACES):
        d = face_directions(face, size)
        f, r, c = directions_to_face(d, size)
        assert np.all(f == i)
        np.testing.assert_array_equal(r, np.repeat(np.arange(size)[:, None], size, axis=1))
        np.testing.assert_array_equal(c, np.repeat(np.arange(size)[None, :], size, axis=0))


def test_faces_cover_the_sphere_without_gaps():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((20000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    f, r, c = directions_to_face(d, 64)
    # Each direction is close to the texel it was assigned to.
    back = np.array([face_directions(FACES[fi], 64)[ri, ci] for fi, ri, ci in zip(f[:500], r[:500], c[:500])])
    assert np.degrees(np.arccos(np.clip((back * d[:500]).sum(1), -1, 1))).max() < 2.0
    assert set(np.unique(f)) == set(range(6))


def test_bilinear_sampling_wraps_in_longitude():
    img = np.zeros((4, 8), dtype=np.uint8)
    img[:, 0] = 100   # first column (lon -180 + half pixel)
    img[:, -1] = 200  # last column
    # Exactly at lon = +/-180 the sample is halfway between the last and first columns.
    v = sample_equirect(img, np.array([np.pi, -np.pi]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(v, [150.0, 150.0])


def test_marker_lands_on_the_right_texel():
    w, h, size = 360, 180, 64
    img = np.zeros((h, w), dtype=np.uint8)
    lat, lon = 51.5, -0.1  # London
    # 5 x 5 source pixels (5 deg): wider than the 1.4 deg texel spacing, so
    # nearest sampling always hits it.
    r0, c0 = int((90 - lat) / 180 * h), int((lon + 180) / 360 * w)
    img[r0 - 2:r0 + 3, c0 - 2:c0 + 3] = 255

    def fn(d):
        lo, la = lonlat(d)
        return sample_equirect(img, lo, la, nearest=True).astype(np.uint8)

    faces = dict(equirect_to_cube(size, fn))
    target = np.array([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                       np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))])
    f, r, c = directions_to_face(target[None, :], size)
    face = faces[FACES[int(f[0])]]
    hits = np.argwhere(face == 255)
    assert len(hits) > 0
    assert np.abs(hits.mean(axis=0) - [r[0], c[0]]).max() <= 1.0


# ------------------------------------------------------------------ normals
def test_flat_terrain_gives_radial_normals_and_a_slope_tilts_them():
    flat = np.zeros((180, 360), dtype=np.float32)
    fn = assets_build._height_normals(flat, 6378137.0, 1.0, np.pi / 180)
    d = face_directions("+X", 8)
    n = fn(d).astype(float) / 127.5 - 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    assert np.degrees(np.arccos(np.clip((n * d).sum(-1), -1, 1))).max() < 0.4  # 8-bit quantisation

    # Height rising eastward -> normal tilts west (negative east component).
    lon = (np.arange(360) + 0.5) / 360 * 2 * np.pi - np.pi
    ramp = np.tile(np.where(np.abs(lon) < 0.5, 1.0e5 * lon, 0.0), (180, 1)).astype(np.float32)
    fn = assets_build._height_normals(ramp, 6378137.0, 1.0, np.pi / 180)
    d0 = np.array([[[1.0, 0.0, 0.0]]])  # lon 0, lat 0: east = +y
    n0 = fn(d0).astype(float)[0, 0] / 127.5 - 1.0
    expected = np.degrees(np.arctan(1.0e5 / 6378137.0))
    assert n0[1] < 0
    assert np.degrees(np.arctan2(-n0[1], n0[0])) == pytest.approx(expected, abs=0.5)


# ------------------------------------------------------------------ stars
def test_bsc5_parser_reads_known_stars_and_skips_entries_without_position():
    cat = parse_bsc5(FIXTURES / "bsc5_sample.dat")
    by_hr = {int(hr): i for i, hr in enumerate(cat["hr"])}
    assert set(by_hr) == {2061, 2491, 7001}  # the position-less entry is skipped
    s = by_hr[2491]  # Sirius
    assert np.degrees(cat["ra_rad"][s]) == pytest.approx(101.287, abs=0.001)
    assert np.degrees(cat["dec_rad"][s]) == pytest.approx(-16.716, abs=0.001)
    assert cat["vmag"][s] == -1.46 and cat["b_v"][s] == 0.0
    assert cat["b_v"][by_hr[2061]] == pytest.approx(1.85)  # Betelgeuse
    table = star_table(cat)
    assert table.dtype == np.dtype("<f4") and table.shape == (3, len(COLUMNS))
    assert table[0, COLUMNS.index("vmag")] == pytest.approx(-1.46)  # brightest first


def test_star_colours_and_brightness():
    assert bv_to_temperature(0.65) == pytest.approx(5800, rel=0.03)  # the Sun
    hot, sun, cool = temperature_to_linear_srgb([20000.0, 5800.0, 3000.0])
    assert hot[2] == 1.0 and hot[0] < hot[2]
    assert cool[0] == 1.0 and cool[2] < 0.3
    assert sun.min() > 0.6  # near white
    assert magnitude_to_lux(-26.74) == pytest.approx(1.3e5, rel=0.05)
    assert magnitude_to_lux(5.0) / magnitude_to_lux(0.0) == pytest.approx(0.01)


# ------------------------------------------------------------------ fetch and build
def _source(path: Path, **extra):
    data = path.read_bytes()
    return {"url": path.as_uri(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "license": "test", "attribution": "test data", "redistributable": True,
            "requires_acceptance": False, **extra}


def test_fetch_verifies_and_refuses_bad_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ADCS_VIZ_CACHE", str(tmp_path / "cache"))
    f = tmp_path / "data.bin"
    f.write_bytes(b"x" * 100)
    good = {"a": _source(f)}
    (path,) = fetch(sources=good, log=lambda *_: None)
    assert path.read_bytes() == b"x" * 100
    bad = {"b": {**_source(f), "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="sha256"):
        fetch(sources=bad, log=lambda *_: None)
    assert not raw_path("b", bad["b"]).exists()
    locked = {"c": _source(f, requires_acceptance=True)}
    with pytest.raises(ValueError, match="accepted"):
        fetch(sources=locked, log=lambda *_: None)
    fetch(sources=locked, accept=["c"], log=lambda *_: None)


def test_build_is_idempotent_and_keyed_by_recipe(tmp_path, monkeypatch):
    monkeypatch.setenv("ADCS_VIZ_CACHE", str(tmp_path / "cache"))
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:, :32] = (200, 30, 30)  # western hemisphere red
    src_file = tmp_path / "day.png"
    Image.fromarray(img).save(src_file)
    sources = {"bmng_topo_bathy_200407": _source(src_file)}
    logs = []
    out1 = assets_build.build(["earth_day"], overrides={"earth_day": {"face_size": 8}}, sources=sources,
                              log=logs.append)["earth_day"]
    manifest = json.loads((out1 / "asset_manifest.json").read_text())
    assert [f["face"] for f in manifest["files"]] == list(FACES)
    # -Y face looks at lon -90 (western hemisphere) -> red.
    ny = np.asarray(Image.open(out1 / "earth_day_nY.png"))
    assert ny[..., 0].mean() > 150 and ny[..., 2].mean() < 60

    out2 = assets_build.build(["earth_day"], overrides={"earth_day": {"face_size": 8}}, sources=sources,
                              log=logs.append)["earth_day"]
    assert out2 == out1 and any("up to date" in m for m in logs)
    out3 = assets_build.build(["earth_day"], overrides={"earth_day": {"face_size": 16}}, sources=sources,
                              log=logs.append)["earth_day"]
    assert out3 != out1
    assert assets_build.current_products()["earth_day"].endswith(out3.name)

    assert assets_build.verify_processed() == {"earth_day": []}
    (out3 / "earth_day_pX.png").write_bytes(b"tampered")
    assert "checksum mismatch earth_day_pX.png" in assets_build.verify_processed()["earth_day"]
    text = assets_build.write_attribution(sources).read_text()
    assert "test data" in text
