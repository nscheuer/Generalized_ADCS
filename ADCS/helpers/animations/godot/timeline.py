r"""
Timeline content for the 3-D viewer: bookmarks (named moments) and ground
sites (whose passes the viewer shows as spans on the timeline).

Both are stored in the vizlog's ``scene.json`` (``"timeline"``), so a recorded
log and a live stream carry the same set.

.. code-block:: python

   from ADCS.helpers.animations.godot import timeline as tl
   ADCS.helpers.animations.godot.results_to_vizlog(results, "run.vizlog",
       bookmarks=[tl.Bookmark(t_s=840.0, label="detumble done"),
                  tl.Bookmark(utc="2026-03-14T12:41:07Z", label="first contact")],
       sites=[tl.GroundSite("Boston", 42.36, -71.06, min_elev_deg=10.0)])

:func:`~ADCS.simulate` adds a site for every
:class:`~ADCS.CONOPS.goals.vector_goals.coordinate_goal.Coordinate_Goal` it
was given. :func:`ground_passes` computes the passes in Python (the viewer
computes the same spans itself, so they also work live).

A pass is the time the spacecraft spends above the site's ``min_elev_deg``
(geodetic elevation, WGS84, no refraction or terrain).
"""

__all__ = ["Bookmark", "GroundSite", "Pass", "validate_timeline", "sites_from_goals", "site_itrs",
           "elevation_deg", "ground_passes"]

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional

import numpy as np

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563


@dataclass
class Bookmark:
    """A named moment, given as seconds since the log's epoch (``t_s``) or a UTC time (``utc``)."""

    t_s: Optional[float] = None
    utc: Optional[str] = None
    label: str = ""
    note: str = ""
    color: str = "#ffd166"


@dataclass
class GroundSite:
    """A place on the Earth (geodetic WGS84); the timeline shows when the spacecraft is above ``min_elev_deg``."""

    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    min_elev_deg: float = 10.0
    color: str = "#7cc6ff"


class Pass(NamedTuple):
    """One pass over a site [s since the epoch]. ``partial``: the log starts or ends inside it."""

    t_rise: float
    t_set: float
    t_max: float
    max_elev_deg: float
    partial: bool


def site_itrs(site: GroundSite) -> tuple:
    """ITRS position [m] and local vertical (geodetic normal) of a site."""
    lat, lon = np.radians(site.lat_deg), np.radians(site.lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_M / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    pos = np.array([(n + site.alt_m) * np.cos(lat) * np.cos(lon),
                    (n + site.alt_m) * np.cos(lat) * np.sin(lon),
                    (n * (1.0 - e2) + site.alt_m) * np.sin(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return pos, up


def _rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) of scalar-first Hamilton quaternions (N, 4)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def elevation_deg(r_gcrs_km: np.ndarray, q_itrs_gcrs: np.ndarray, site: GroundSite) -> np.ndarray:
    """Elevation of the spacecraft seen from a site [deg], from the vizlog's ``r_sc`` and ``q_itrs_gcrs``."""
    pos, up = site_itrs(site)
    r = np.einsum("nij,nj->ni", _rot(np.atleast_2d(q_itrs_gcrs)), np.atleast_2d(r_gcrs_km) * 1000.0)
    los = r - pos
    return np.degrees(np.arcsin(np.clip(los @ up / np.linalg.norm(los, axis=1), -1.0, 1.0)))


def ground_passes(log: Any, site: GroundSite, run_id: str = "0") -> List[Pass]:
    """
    Passes of a site in one run of a vizlog (a :class:`~ADCS.helpers.animations.godot.VizLog`).

    Rise and set times are interpolated linearly between rows; the maximum is
    the highest row (refined with a parabola through its neighbours).
    """
    t = log.read(run_id, "t_s")[:, 0]
    e = elevation_deg(log.read(run_id, "r_sc"), log.read(run_id, "q_itrs_gcrs"), site) - site.min_elev_deg
    return _spans(t, e, site.min_elev_deg)


def _spans(t: np.ndarray, e: np.ndarray, offset: float) -> List[Pass]:
    above = e >= 0.0
    out = []
    k, n = 0, len(t)
    while k < n:
        if not above[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and above[j + 1]:
            j += 1
        cross = lambda a, b: t[a] + (t[b] - t[a]) * e[a] / (e[a] - e[b])  # linear zero crossing
        t0 = t[k] if k == 0 else cross(k - 1, k)
        t1 = t[j] if j == n - 1 else cross(j, j + 1)
        m = k + int(np.argmax(e[k:j + 1]))
        tm, em = t[m], e[m]
        if 0 < m < n - 1:  # parabola through the peak and its neighbours (uniform spacing assumed locally)
            d = e[m - 1] - 2 * e[m] + e[m + 1]
            if d < 0:
                s = 0.5 * (e[m - 1] - e[m + 1]) / d
                tm = t[m] + s * 0.5 * (t[m + 1] - t[m - 1])
                em = e[m] - 0.25 * (e[m - 1] - e[m + 1]) * s
        out.append(Pass(float(t0), float(t1), float(tm), float(em + offset), k == 0 or j == n - 1))
        k = j + 1
    return out


def sites_from_goals(*goals: Any) -> List[GroundSite]:
    """
    Ground sites for the coordinate goals among ``goals`` (goals, goal lists or ``None``), unique by position:
    ``"target"`` if there is one, else ``"target 1"``, ``"target 2"``, ... in order of first appearance.
    """
    flat: List[Any] = []
    for goal in goals:
        flat += getattr(goal, "goals", None) or ([goal] if goal is not None else [])
    found, seen = [], set()
    for g in flat:
        if not all(hasattr(g, a) for a in ("lat_deg", "lon_deg", "alt_km")):
            continue
        key = (round(float(g.lat_deg), 6), round(float(g.lon_deg), 6))
        if key not in seen:
            seen.add(key)
            found.append(g)
    return [GroundSite("target" if len(found) == 1 else f"target {i + 1}", float(g.lat_deg), float(g.lon_deg),
                       float(g.alt_km) * 1000.0) for i, g in enumerate(found)]


def _utc_to_t_s(utc: str, to_t_s: Callable[[datetime], float]) -> float:
    d = datetime.fromisoformat(utc.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return float(to_t_s(d))


def validate_timeline(bookmarks: Optional[Iterable[Any]], sites: Optional[Iterable[Any]],
                      t_range: Optional[tuple] = None,
                      utc_to_t_s: Optional[Callable[[datetime], float]] = None) -> Dict[str, Any]:
    r"""
    Turn bookmarks and sites (objects or dicts) into the ``scene.json``
    ``"timeline"`` entry.

    :param t_range: ``(t_first, t_last)`` of the log [s]; bookmarks outside it
        are rejected (``None``: not checked, e.g. a live run of unknown length).
    :param utc_to_t_s: Converts an aware UTC datetime to seconds since the log's
        epoch; needed for bookmarks given as ``utc``.
    :raises ValueError: On a bookmark without a time or outside the log, or a
        site with invalid coordinates or a duplicate name.
    """
    out_b = []
    for i, b in enumerate(bookmarks or []):
        d = dict(b.__dict__) if isinstance(b, Bookmark) else dict(b)
        if d.get("t_s") is None:
            if not d.get("utc"):
                raise ValueError(f"bookmark {i} ({d.get('label', '')!r}) needs t_s or utc")
            if utc_to_t_s is None:
                raise ValueError("bookmarks given in UTC need the log's epoch")
            d["t_s"] = _utc_to_t_s(d["utc"], utc_to_t_s)
        d["t_s"] = float(d["t_s"])
        if t_range is not None and not (t_range[0] - 1e-6 <= d["t_s"] <= t_range[1] + 1e-6):
            raise ValueError(f"bookmark {d.get('label', i)!r} at t = {d['t_s']:.3f} s is outside the log "
                             f"[{t_range[0]:.3f}, {t_range[1]:.3f}] s")
        d.setdefault("label", "")
        d.setdefault("note", "")
        d.setdefault("color", Bookmark.color)
        d.pop("utc", None)
        d["id"] = f"b{i}"
        out_b.append(d)
    out_s, names = [], set()
    for s in sites or []:
        site = s if isinstance(s, GroundSite) else GroundSite(**s)
        if not (-90.0 <= site.lat_deg <= 90.0) or not (-360.0 <= site.lon_deg <= 360.0):
            raise ValueError(f"site {site.name!r}: latitude/longitude out of range")
        if not (-90.0 < site.min_elev_deg < 90.0):
            raise ValueError(f"site {site.name!r}: min_elev_deg must be between -90 and 90")
        if site.name in names:
            raise ValueError(f"duplicate site name {site.name!r}")
        names.add(site.name)
        pos, up = site_itrs(site)
        out_s.append({**site.__dict__, "itrs_m": pos.tolist(), "up_itrs": up.tolist()})
    return {"bookmarks": sorted(out_b, key=lambda d: d["t_s"]), "sites": out_s}
