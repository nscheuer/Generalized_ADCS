# Ephemeris models

`DE421()` is the default planetary ephemeris. Its kernel is bundled for offline
use. Missing kernels are resolved through `AssetManager`, which asks for
terminal confirmation before downloading into `ADCS/environment/kernels/`.
Set `ADCS_EPHEMERIS_PATH` to select a specific copy; a missing file at that
path also requires confirmation.

```python
import ADCS

ephem = ADCS.Environment.Ephemeris.DE421()
environment = ADCS.Environment(ephemeris=ephem)
sun_eci_km = ephem.sun_position_eci(0.0)  # J2000 epoch
```

The alternative DE442, DE442s, and DE441 models remain available explicitly.
Their kernels are optional and are downloaded into
`ADCS/environment/kernels/` only after terminal confirmation.
