# Magnetic field models

Select a model through an `Environment`:

```python
from datetime import datetime
import numpy as np
import ADCS

environment = ADCS.Environment(
    magnetic_field=ADCS.Environment.MagneticField.WMM2025()
)
field_ecef_tesla = environment.magnetic_field.field(
    np.array([6771.2, 0.0, 0.0]), datetime(2026, 1, 1)
)

environment = ADCS.Environment(
    magnetic_field=ADCS.Environment.MagneticField.CHAOS8()
)
```

Pass `environment=environment` to `ADCS.Orbital_State` or `ADCS.orbits.Orbit`.
Legacy string selectors also accept `"wmm"`/`"wmm2025"` and
`"chaos-8"`/`"chaos8"`.
The classes also provide `field_gc(radius_km, colatitude_deg,
longitude_deg, date)`, returning radial, southward, and eastward components in
nanotesla. Both methods accept one position/date or arrays of positions with
one shared date or one date per position.

`WMM2025` uses NOAA's WMM2025 degree 12 coefficients and annual secular
variation. It accepts dates from 2025-01-01 through 2029-12-31. The model
uses geocentric spherical coordinates internally; its `field` input is an
Earth centered Earth fixed Cartesian vector in kilometers.

`CHAOS8` uses the pinned CHAOS-8.7 core field (degree 1–20, piecewise degree 5
time interpolation) and static lithospheric field (degree 21–185). Evaluation
is limited to the SHC coefficient range, approximately 1997.102–2027.101;
the model's scientifically recommended interval can be narrower. The
external and induced components of CHAOS require geomagnetic activity inputs
and are not represented by these internal field SHC files. Set
`CHAOS8(include_static=False)` to evaluate only its time varying core.

The files `WMM2025.COF`, `CHAOS-8.7_core.shc`, and
`CHAOS-8.7_static.shc` are stored in `ADCS/environment/kernels`. They are
downloaded with a progress bar when first needed and are ignored by Git.
Model constructors accept custom kernel paths for offline use. Sources:

- [NOAA World Magnetic Model](https://www.ncei.noaa.gov/products/world-magnetic-model)
- [NOAA WMM2025 test values](https://www.ncei.noaa.gov/sites/default/files/2025-02/WMM2025_TEST_VALUES.txt)
- [CHAOS-8.7 model files](https://zenodo.org/records/22959079)
