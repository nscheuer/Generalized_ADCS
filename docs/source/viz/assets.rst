Visualization assets
====================

The 3-D viewer draws Earth, the Moon and the stars from public datasets. The
asset pipeline (:mod:`ADCS.viz.assets`) downloads them, checks every byte
against pinned SHA-256 hashes, and turns them into cube maps and tables the
viewer loads directly. Nothing is stored in the repository or the wheel.

.. code-block:: bash

   pip install "generalized-adcs[viz-assets]"      # Pillow
   python -m ADCS.viz.assets list                  # sources, products, cache state
   python -m ADCS.viz.assets build                 # fetch what is needed and build everything
   python -m ADCS.viz.assets build earth_day --set earth_day.face_size=2048
   python -m ADCS.viz.assets verify                # re-check raw and processed checksums
   python -m ADCS.viz.assets licenses              # write ATTRIBUTION.txt

Cache
-----

``$ADCS_VIZ_CACHE`` if set; otherwise ``%LOCALAPPDATA%\generalized_adcs\viz``
on Windows, ``~/Library/Caches/generalized_adcs/viz`` on macOS and
``~/.cache/generalized_adcs/viz`` on Linux. **Inside WSL the Windows location
is used**, because the viewer runs as a native Windows program.

.. code-block:: text

   raw/<source>/<file>                      downloads, verified against sources.json
   processed/<product>/<recipe hash>/       one build: files + asset_manifest.json
   processed/current.json                   product -> latest build (what the viewer reads)
   ATTRIBUTION.txt

A build is keyed by a hash of its recipe: product, parameters, input
checksums and processing code version. Rebuilding an unchanged recipe does
nothing; changing a parameter writes a new directory and repoints
``current.json``. A build's ``asset_manifest.json`` is written last, so a
directory with a manifest is a complete build.

Sources
-------

All downloads (about 140 MB) are pinned in ``ADCS/viz/assets/sources.json``
with their URL, size and SHA-256 (verified 2026-10-22).

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Source
     - Content
     - Credit / licence
   * - ``bmng_topo_bathy_200407``
     - Blue Marble Next Generation, July 2004, 21600×10800
     - NASA Earth Observatory; free use with credit
   * - ``black_marble_2016_3km_gray``
     - Black Marble 2016 night lights (VIIRS), 13500×6750
     - NASA Earth Observatory; free use with credit
   * - ``clouds_bm2002_8192``
     - Blue Marble 2002 cloud composite, 8192×4096
     - NASA Earth Observatory; free use with credit
   * - ``gebco08_elev_21600``
     - Land elevation, 8-bit, ≈25 m per step (0–6400 m, saturating)
     - NASA Earth Observatory from GEBCO_08 (public domain)
   * - ``gebco08_bath_21600``
     - Ocean depth, paletted; land = 255 (inland lakes are land)
     - NASA Earth Observatory from GEBCO_08 (public domain)
   * - ``bsc5_cds``
     - Yale Bright Star Catalogue, 5th ed. (CDS V/50)
     - Hoffleit & Warren 1991; no explicit licence, treated as public domain
   * - ``svs_moon_color_4k``
     - CGI Moon Kit colour (LROC WAC, LOLA poles), 4096×2048
     - NASA SVS; free use with credit
   * - ``svs_moon_ldem_4``
     - CGI Moon Kit elevation (LOLA), 4 px/deg, km
     - NASA SVS; free use with credit

The two GEBCO encodings above were measured from the files themselves; for
example Lake Victoria reads 45, which at 25.1 m per step is 1130 m against a
true 1134 m. Hipparcos is deliberately not a default source: its licence (CC
BY-NC 3.0 IGO) forbids commercial use.

Products
--------

.. list-table::
   :header-rows: 1
   :widths: 18 18 16 48

   * - Product
     - Default size
     - Frame
     - Encoding
   * - ``earth_day``
     - 4096² per face
     - ITRS
     - sRGB 8-bit RGB
   * - ``earth_night``
     - 2048²
     - ITRS
     - sRGB-encoded 8-bit luminance (stylised brightness, not radiance)
   * - ``earth_clouds``
     - 2048²
     - ITRS
     - 8-bit cloud brightness/opacity
   * - ``earth_ocean_mask``
     - 2048²
     - ITRS
     - 8-bit, 255 = ocean
   * - ``earth_normals``
     - 2048²
     - ITRS
     - object-space normal, RGB = 127.5 (n + 1); relief exaggerated ×8
   * - ``moon_color``
     - 1024²
     - MOON_ME (as published)
     - sRGB 8-bit RGB
   * - ``moon_normals``
     - 512²
     - MOON_ME (as published)
     - object-space normal, RGB = 127.5 (n + 1)
   * - ``stars``
     - V ≤ 7 (9050 stars)
     - ICRS, J2000
     - float32 rows: x, y, z, Vmag, B−V, linear-sRGB colour (max 1),
       illuminance [lux], proper motion [mas/yr]

**Cube maps** use the OpenGL/Vulkan face order and orientation (+X, −X, +Y,
−Y, +Z, −Z; see :mod:`ADCS.viz.assets.cubemap`). A shader therefore samples
them with the raw body-fixed direction, e.g. ``texture(earth_day, n_itrs)``,
and Earth's orientation comes from the vizlog's ``q_itrs_gcrs`` alone. Normals
are stored in object space (not tangent space), which suits the viewer's
object-space shading of scaled far bodies.

**Stars**: B−V → temperature (Ballesteros 2012) → blackbody chromaticity
integrated against the CIE 1931 colour-matching functions (Wyman et al. 2013
fit) → linear sRGB. Illuminance is 2.54×10⁻⁶ × 10^(−0.4 V) lux, which gives the
Sun about 1.3×10⁵ lux. Aberration and proper motion are not applied.

Checks
------

``testing/test_viz/test_viz_assets.py`` (no network) covers cube geometry and
its inverse, sampling across the date line, marker placement, normals on flat
and sloped terrain, the BSC5 parser on real catalogue lines, star colours and
brightness, fetch verification, and build idempotency and tamper detection.
On the real data:

* a cube rebuilt into a map correlates 0.983 with the Blue Marble source
  (0.685 for the same comparison shifted 45°);
* the ocean mask covers 69% of the sphere (71% in reality; inland seas count
  as land in this source);
* normal tilt is 20° in the Himalaya (×8 exaggeration) and 0.2° over the open
  ocean (8-bit quantisation);
* Sirius comes out at RA 101.287°, Dec −16.716°, V −1.46, 9.75×10⁻⁶ lux.

Not yet included
----------------

* Precomputed atmospheric-scattering tables (Phase 6, with the atmosphere
  shader).
* Daily same-date clouds from NASA GIBS, for comparisons against EPIC/GOES
  photographs.
* 500 m streaming tiles; KTX2/BC7 GPU compression (the viewer's importer
  compresses PNGs today).
