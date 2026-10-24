# Ephemeris models

`DE421()` is the default planetary ephemeris. It looks first in the configured
user cache and then alongside the source package; if no copy exists, it
downloads JPL's `de421.bsp` into the user cache on first use. The kernel is not
included in the Python wheel. Set `ADCS_EPHEMERIS_PATH` to use a specific copy.

```python
import ADCS

ephem = ADCS.Environment.Ephemeris.DE421()
environment = ADCS.Environment(ephemeris=ephem)
sun_eci_km = ephem.sun_position_eci(0.0)  # J2000 epoch
```

The alternative DE442 and DE442s models remain available explicitly.
