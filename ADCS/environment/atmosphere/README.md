# Atmospheric models

`SMADAtmosphere` remains the lightweight default. For an empirical thermosphere model, install the optional PyMSIS backend:

```bash
pip install 'Generalized_ADCS[nrlmsise]'
```

Then select NRLMSISE-00 through the environment:

```python
environment = ADCS.Environment(
    atmosphere=ADCS.Environment.Atmosphere.NRLMSISE00()
)
```

`NRLMSISE00` gets UTC epochs and WGS84 latitude, longitude, and altitude from the orbit sampling code. By default PyMSIS retrieves historical F10.7 and Ap indices when required; set `f107`, `f107a`, and `ap` in the model constructor to provide fixed values. The model supports PyMSIS versions 0 (NRLMSISE-00), 2.0, and 2.1; version 0 is the default. `interpolate(altitude_km)` evaluates an altitude profile at the configured reference date and location, which is useful for plots.

PyMSIS notes that MSIS 2.0/2.1 source use may require permission from NRL for commercial use. Check its current licensing terms before selecting those versions.
