# Ephemeris models

`DE442s()` is the default planetary ephemeris and uses JPL's short DE442 kernel offline.
The 32,701,440-byte `de442s.bsp` file is committed in
`ADCS/environment/kernels/` and included in the Python wheel. It covers
1849-12-25 through 2150-01-21. Its planetary bodies and query methods match
`DE442()`, whose larger kernel covers 1549 through 2650.

```python
import ADCS

ephem = ADCS.Environment.Ephemeris.DE442s()
environment = ADCS.Environment(ephemeris=ephem)
sun_eci_km = ephem.sun_position_eci(0.0)  # J2000 epoch
```

The kernel comes from the [JPL NAIF generic planetary SPK archive](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/).
Its MD5 checksum is `cc49327e06088124c0e39d8dde9f0b58`, matching NAIF's
published checksum. If the bundled file is missing, the class downloads it to
the user cache when constructed.
