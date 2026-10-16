"""Fast, offline coverage for sky swaths and Hipparcos loading."""

import numpy as np

from ADCS.environment.star_catalogue import BrightStarCatalog, HipparcosCatalog, StarCatalog


def _hip_line(hip_id, magnitude, ra_deg, dec_deg):
    line = list(" " * 450)
    for start, end, value in (
        (8, 14, f"{hip_id:6d}"),
        (41, 46, f"{magnitude:5.2f}"),
        (51, 63, f"{ra_deg:12.8f}"),
        (64, 76, f"{dec_deg:12.8f}"),
    ):
        line[start:end] = value
    return "".join(line) + "\n"


def test_catalogue_cone_and_wrapping_swath():
    catalog = StarCatalog()
    sirius = catalog.stars[0]
    assert sirius in catalog.stars_in_cone(sirius.s_eci, np.deg2rad(0.1))
    assert sirius not in catalog.stars_in_cone(-sirius.s_eci, np.deg2rad(0.1))
    names = [star.name for star in catalog.load_sky_swath(np.deg2rad(340), np.deg2rad(10), np.deg2rad(-40), np.deg2rad(40))]
    assert "Fomalhaut" in names
    assert "Alpheratz" in names


def test_hipparcos_download_is_cached_and_filtered(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, destination, description):
        calls.append(url)
        destination.write_text(
            _hip_line(1, 4.0, 359.0, 5.0)
            + _hip_line(2, 7.0, 1.0, 5.0)
            + _hip_line(3, 2.0, 100.0, -30.0),
            encoding="ascii",
        )

    monkeypatch.setattr("ADCS.environment.star_catalogue.hipparcos.download_with_progress", fake_download)
    catalog = HipparcosCatalog(max_vmag=6.0, kernel_dir=tmp_path)
    assert calls == []  # Download is deferred until the first query.
    assert [star.hip_id for star in catalog.stars] == [3, 1]
    assert len(calls) == 1
    swath = catalog.stars_in_region(np.deg2rad(350), np.deg2rad(10), 0.0, np.deg2rad(10))
    assert [star.hip_id for star in swath] == [1]
    assert [star.hip_id for star in HipparcosCatalog(max_vmag=None, kernel_dir=tmp_path).stars] == [3, 1, 2]
    assert len(calls) == 1


def test_bright_star_catalogue_is_bundled_and_supports_sky_queries():
    catalog = BrightStarCatalog()
    assert catalog.kernel_path.stat().st_size < 30_000_000
    assert len(catalog.stars) == 8404
    sirius = next(star for star in catalog.stars if star.hr_id == 2491)
    assert sirius.hip_id is None
    assert np.isclose(sirius.vmag, -1.46)
    assert np.isclose(np.rad2deg(sirius.ra_rad), 101.287, atol=0.01)
    assert sirius in catalog.stars_in_cone(sirius.s_eci, np.deg2rad(0.1))
    assert sirius in catalog.load_sky_swath(
        np.deg2rad(100), np.deg2rad(102), np.deg2rad(-18), np.deg2rad(-15)
    )
    assert len(BrightStarCatalog(max_vmag=None).stars) == 9096
