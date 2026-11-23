"""Fast, offline coverage for bundled star catalogues and sky swaths."""

import numpy as np

from ADCS.environment.star_catalogue import BrightStarCatalog, StarCatalog


def test_catalogue_cone_and_wrapping_swath():
    catalog = StarCatalog()
    sirius = catalog.stars[0]
    assert sirius in catalog.stars_in_cone(sirius.s_eci, np.deg2rad(0.1))
    assert sirius not in catalog.stars_in_cone(-sirius.s_eci, np.deg2rad(0.1))
    names = [star.name for star in catalog.load_sky_swath(np.deg2rad(340), np.deg2rad(10), np.deg2rad(-40), np.deg2rad(40))]
    assert "Fomalhaut" in names
    assert "Alpheratz" in names


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
