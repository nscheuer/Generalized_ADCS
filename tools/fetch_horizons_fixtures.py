"""Fetch JPL Horizons reference values for the vizlog environment tests.

Writes ``testing/test_viz/fixtures/horizons_sun_moon.json``. Run manually
(network access needed); CI only reads the committed file.

For each epoch (TT Julian date) it records:

* Sun and Moon apparent geocentric RA/Dec, referred to the true equator and
  equinox of date (Horizons quantity 2, observer at the geocentre), and the
  Sun's distance (quantity 20);
* the Sun's apparent local hour angle seen from a site at geodetic (0, 0, 0)
  (quantity 42). Its negative is the sub-solar longitude, independent of how
  Earth rotation is modelled on our side.

Epochs stay within 1990-2026 so Earth orientation comes from measured, not
predicted, EOP on both sides.

Usage::

    python tools/fetch_horizons_fixtures.py
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://ssd.jpl.nasa.gov/api/horizons.api"
OUT = Path(__file__).resolve().parents[1] / "testing" / "test_viz" / "fixtures" / "horizons_sun_moon.json"

# 2 per year from 1990 to 2026 would be plenty; 20 spread epochs, including
# solstices and equinoxes, keep the file small.
EPOCHS_JD_TT = [
    2447892.5, 2448500.25, 2449352.75, 2450083.5, 2450814.125,
    2451544.5, 2451623.9, 2452275.6, 2453006.5, 2453736.5,
    2454467.3, 2455197.5, 2455927.8, 2456658.5, 2457388.5,
    2458119.5, 2458849.5, 2459580.5, 2460310.5, 2461041.5,
]


def _query(params: dict) -> list[list[str]]:
    base = {"format": "json", "EPHEM_TYPE": "OBSERVER", "TLIST_TYPE": "JD", "TIME_TYPE": "TT",
            "ANG_FORMAT": "DEG", "CSV_FORMAT": "YES", "EXTRA_PREC": "YES",
            "TLIST": " ".join(f"'{jd}'" for jd in EPOCHS_JD_TT)}
    base.update(params)
    url = API + "?" + urllib.parse.urlencode(base)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.load(resp)
    result = payload.get("result", "")
    if "$$SOE" not in result:
        raise RuntimeError(f"Horizons error for {params}: {payload.get('error') or result[:500]}")
    body = result.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines()
    rows = [[c.strip() for c in line.split(",")] for line in body]
    if len(rows) != len(EPOCHS_JD_TT):
        raise RuntimeError(f"expected {len(EPOCHS_JD_TT)} rows, got {len(rows)}")
    return rows


def main() -> int:
    sun = _query({"COMMAND": "'10'", "CENTER": "'500@399'", "QUANTITIES": "'2,20'"})
    moon = _query({"COMMAND": "'301'", "CENTER": "'500@399'", "QUANTITIES": "'2,20'"})
    ha = _query({"COMMAND": "'10'", "CENTER": "'coord@399'", "COORD_TYPE": "GEODETIC",
                 "SITE_COORD": "'0,0,0'", "QUANTITIES": "'42'"})

    # CSV columns: date, solar-presence flag, lunar-presence flag, then quantities.
    def nums(row: list[str]) -> list[float]:
        return [float(c) for c in row[3:] if c not in ("", "n.a.")]

    records = []
    for jd, s, m, h in zip(EPOCHS_JD_TT, sun, moon, ha):
        s_ra, s_dec, s_delta, _ = nums(s)[:4]
        m_ra, m_dec, m_delta, _ = nums(m)[:4]
        records.append({
            "jd_tt": jd,
            "date_tt": s[0],
            "sun_ra_app_deg": s_ra, "sun_dec_app_deg": s_dec, "sun_delta_au": s_delta,
            "moon_ra_app_deg": m_ra, "moon_dec_app_deg": m_dec, "moon_delta_au": m_delta,
            "sun_local_hour_angle_greenwich_h": nums(h)[0],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "source": "JPL Horizons API (ssd.jpl.nasa.gov/api/horizons.api), ephemeris DE441",
            "notes": "RA/Dec: apparent, geocentric, true equator and equinox of date. "
                     "Hour angle: apparent, site geodetic lon 0 lat 0 alt 0, hours.",
            "records": records,
        }, f, indent=1)
    print(f"wrote {len(records)} epochs to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
