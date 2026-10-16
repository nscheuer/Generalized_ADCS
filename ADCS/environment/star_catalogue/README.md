# Star catalogues

`StarCatalog()` is the bundled, offline set of 30 bright navigation stars.
It is the default used by both star trackers.

`BrightStarCatalog()` loads the bundled Yale Bright Star Catalogue, fifth
revised edition. Its compressed source file is 574 KB in
`ADCS/environment/kernels/bsc5_catalog.gz`, so no download is needed. The
default `max_vmag=6.5` provides 8,404 usable stars. It accepts the same cone,
sky swath, and visibility queries as `StarCatalog`. The records carry Harvard
Revised numbers in `hr_id`; `hip_id` is `None` because this source does not
provide Hipparcos identifiers. Positions are J2000 FK5 and remain fixed in
this implementation.

```python
import ADCS

catalog = ADCS.Environment.StarCatalogue.BrightStar()
environment = ADCS.Environment(star_catalogue=catalog)
```

The bundled file is the [CDS V/50 `catalog.gz`](https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz)
by Hoffleit and Warren (1991). Its field layout and citation are in the
[CDS catalogue ReadMe](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/V/50?format=html&tex=true).
Credit: Hoffleit, D. and Warren, W. H. Jr., *The Bright Star Catalogue*,
fifth revised edition (1991); CDS, Strasbourg.

`HipparcosCatalog()` reads the [CDS Hipparcos main catalogue](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/I/239?format=html&tex=true). On the first query it downloads `hip_main.dat` into `ADCS/environment/kernels/`, if absent, with a progress bar. Subsequent instances reuse that file. The file is about 53 MB and is ignored by Git. By default, only stars at visual magnitude 6 or brighter are loaded into memory; use `max_vmag=None` for all entries with usable positions and magnitudes.

```python
import numpy as np
import ADCS

catalog = ADCS.Environment.StarCatalogue.Hipparcos(max_vmag=6.0)
stars = catalog.stars_in_cone(np.array([1.0, 0.0, 0.0]), np.deg2rad(10.0))
swath = catalog.load_sky_swath(
    np.deg2rad(350), np.deg2rad(10),
    np.deg2rad(-20), np.deg2rad(20),
)
environment = ADCS.Environment(star_catalogue=catalog)
# Pass environment to Orbital_State / simulate, or pass catalog directly as
# star_catalog=... to StarTracker or StarTrackerQuaternion.
```

The Hipparcos sky queries return `NavigationStar` records with Hipparcos ID,
visual magnitude, right ascension, declination, and an ICRS unit vector.
`load_sky_swath` accepts RA bounds that cross zero. `get_visible_stars` also
applies the tracker's Earth, Moon, and Sun visibility checks. Hipparcos
positions are fixed at J1991.25; proper motion, parallax, and aberration are
not applied.
