"""Phase 6: render the Earth as DSCOVR/EPIC saw it and compare with the photograph.

For one EPIC "natural" image (metadata from epic.gsfc.nasa.gov/api):

1. **Geolocation (numeric):** the sub-DSCOVR point implied by *our* Earth
   orientation at the image time vs EPIC's own ``centroid_coordinates``.
2. **Render:** the viewer, from ``dscovr_j2000_position`` looking at Earth's
   centre with Earth's north up (L1b images are north-up), at several time
   offsets.
3. **Compare** after fitting the limb in both images (centre, radius):
   disk overlap (IoU), the time offset that best matches the surface pattern
   (Earth orientation; compared with the time implied by EPIC's centroid,
   because EPIC's two published timestamps differ by minutes), and CIEDE2000 colour difference on pixels
   that are cloud-free in both images (clouds are a static 2002 composite in
   the viewer, so cloudy pixels are excluded).

Usage::

    python tools/viz_validation/run_epic_check.py [--date 2022-01-01] [--lon 20]

Writes tools/viz_validation/EPIC_CHECK.md and comparison images to the work dir.
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_viewer_checks import Runner, find_godot  # noqa: E402

from ADCS.helpers.math_helpers import rot_mat  # noqa: E402
from ADCS.orbits.ephemeris import Ephemeris  # noqa: E402
from ADCS.viz.assets.fetch import cache_dir  # noqa: E402
from ADCS.viz.derive import derive_environment  # noqa: E402
from ADCS.viz.schema import REQUIRED_CHANNELS, channel_spec  # noqa: E402
from ADCS.viz.writer import VizLogWriter  # noqa: E402

HERE = Path(__file__).resolve().parent
API = "https://epic.gsfc.nasa.gov/api/natural/date/{date}"
ARCHIVE = "https://epic.gsfc.nasa.gov/archive/natural/{y}/{m}/{d}/png/{image}.png"
OFFSETS_S = [-1800, -600, -300, -120, -60, 0, 60, 120, 180, 240, 300, 360, 420, 600, 1800]
EARTH_ROTATION_DEG_S = 360.0 / 86164.0905
#: EPIC star-tracker pointing error before L1b re-centring (Frontiers 2021): +-0.05 deg
#: at 1.5e6 km is ~1300 km on the ground, so the metadata centroid (where the camera
#: pointed) may differ from the sub-satellite point (the L1b image centre) by ~1 deg.
CENTROID_TOL_DEG = 2.0
FOV_DEG = 0.62


# ------------------------------------------------------------------ data
def fetch_epic(date: str, lon: float) -> tuple[dict, Path]:
    raw = cache_dir() / "raw" / "epic" / date
    raw.mkdir(parents=True, exist_ok=True)
    meta_path = raw / "api.json"
    if not meta_path.exists():
        with urllib.request.urlopen(API.format(date=date), timeout=60) as r:
            meta_path.write_bytes(r.read())
    items = json.loads(meta_path.read_text())
    item = min(items, key=lambda e: abs(((e["centroid_coordinates"]["lon"] - lon + 180) % 360) - 180))
    png = raw / f"{item['image']}.png"
    if not png.exists():
        y, m, d = date.split("-")
        with urllib.request.urlopen(ARCHIVE.format(y=y, m=m, d=d, image=item["image"]), timeout=300) as r:
            png.write_bytes(r.read())
    return item, png


def build_log(item: dict, path: Path) -> tuple[float, dict]:
    """vizlog spanning the image time +-30 min (60 s rows); returns (t of the image, geometry)."""
    ephem = Ephemeris()
    t_img = datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    ts = ephem.ts.from_datetime(t_img)
    t = np.arange(-1800.0, 1801.0, 60.0)
    jd = ts.tt + t / 86400.0
    env = derive_environment(ephem, jd)
    pos = np.array([item["dscovr_j2000_position"][k] for k in "xyz"])  # km, J2000 ~ GCRS
    n = len(t)
    cols = {"t_s": t - t[0], "r_sc": np.tile(pos, (n, 1)), "v_sc": np.zeros((n, 3)),
            "q_gcrs_body": np.tile([1.0, 0, 0, 0], (n, 1)), "w_body": np.zeros((n, 3)),
            "r_sun": env["r_sun"], "q_itrs_gcrs": env["q_itrs_gcrs"], "shadow_factor": np.ones(n), "r_moon": env["r_moon"]}
    with VizLogWriter(path, [channel_spec(c) for c in REQUIRED_CHANNELS] + [channel_spec("r_moon")],
                      epoch_jd_tt=float(jd[0]), epoch_iso_utc=None, scene={}, overwrite=True) as w:
        w.write_run(0, cols)
    k = n // 2  # the image time
    R = rot_mat(env["q_itrs_gcrs"][k])
    sub = R @ pos
    lat = np.degrees(np.arctan2(sub[2], np.hypot(sub[0], sub[1])))
    geom = {"lat_geocentric": float(lat), "lon": float(np.degrees(np.arctan2(sub[1], sub[0]))),
            "north_gcrs": (R.T @ np.array([0.0, 0.0, 1.0])).tolist(), "pos_m": (pos * 1000.0).tolist(),
            "sun_dir": (env["r_sun"][k] / np.linalg.norm(env["r_sun"][k])).tolist()}
    return float(cols["t_s"][k]), geom


# ------------------------------------------------------------------ image analysis
def load(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def limb_fit(img: np.ndarray) -> tuple[float, float, float]:
    """Circle (cx, cy, r) through the sharp (lit) limb: boundary pixels of the
    disk whose inward neighbourhood is bright. The soft terminator is excluded."""
    from scipy import ndimage

    lum = img.mean(axis=2)
    mask = ndimage.binary_opening(lum > 0.04, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    edge = mask & ~ndimage.binary_erosion(mask)
    inner = ndimage.uniform_filter(lum, 7)
    ys, xs = np.nonzero(edge & (inner > 0.10))
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    (a1, a2, a3), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = a1 / 2, a2 / 2
    return float(cx), float(cy), float(np.sqrt(a3 + cx * cx + cy * cy))


def align(src: np.ndarray, src_fit, dst_fit, shape) -> np.ndarray:
    """Resample src so its fitted disk lands on dst's (scale + shift)."""
    from scipy import ndimage

    s = src_fit[2] / dst_fit[2]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    sx = src_fit[0] + (xx - dst_fit[0]) * s
    sy = src_fit[1] + (yy - dst_fit[1]) * s
    return np.stack([ndimage.map_coordinates(src[..., c], [sy, sx], order=1, mode="constant") for c in range(3)], -1)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma et al. 2005)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                                                  np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm**7 / (Cpm**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def land_mask(img: np.ndarray) -> np.ndarray:
    """Cloud-free land: redder than blue (deserts, vegetation); ocean is bluer."""
    return (img[..., 0] > img[..., 2] + 0.02) & (img.max(axis=2) > 0.10)


def cloud_mask(img: np.ndarray) -> np.ndarray:
    return (img.min(axis=2) > 0.42) & (img.max(axis=2) - img.min(axis=2) < 0.18)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2022-01-01")
    ap.add_argument("--lon", type=float, default=20.0, help="prefer the image whose centroid is nearest this longitude")
    ap.add_argument("--godot")
    args = ap.parse_args()

    item, png = fetch_epic(args.date, args.lon)
    runner = Runner(find_godot(args.godot), 600.0)
    log = runner.work / "epic.vizlog"
    t_img, geom = build_log(item, log)
    c = item["centroid_coordinates"]
    lat_geodetic = np.degrees(np.arctan(np.tan(np.radians(geom["lat_geocentric"])) / (1 - 1 / 298.257223563) ** 2))
    dlon = (geom["lon"] - c["lon"] + 180) % 360 - 180
    print(f"[epic] {item['image']} at {item['date']} UTC; centroid EPIC ({c['lat']:.3f}, {c['lon']:.3f}) "
          f"ours ({lat_geodetic:.3f} geodetic / {geom['lat_geocentric']:.3f} geocentric, {geom['lon']:.3f})")

    shots = [{"name": f"epic_t{dt:+d}", "t": t_img + dt, "cam_pos": geom["pos_m"], "look_at": [0, 0, 0],
              "up": geom["north_gcrs"], "fov_deg": FOV_DEG, "hide": ["craft"]} for dt in OFFSETS_S]
    shots.append({"name": "epic_linear", "t": t_img, "cam_pos": geom["pos_m"], "look_at": [0, 0, 0],
                  "up": geom["north_gcrs"], "fov_deg": FOV_DEG, "hide": ["craft"], "tonemap": "linear"})
    r = runner.run("epic", "pose", log, {"shots": shots}, extra=("--res", "2048x2048"))
    if "error" in r:
        print("[epic] viewer error:", r["error"], r.get("_errors"))
        return 1

    photo = load(png)
    pfit = limb_fit(photo)
    out = runner.work / "epic"
    renders = {dt: load(out / f"epic_t{dt:+d}.png") for dt in OFFSETS_S}
    rfit = limb_fit(renders[0])
    implied_fov = FOV_DEG * rfit[2] / pfit[2]
    yy, xx = np.mgrid[0:photo.shape[0], 0:photo.shape[1]]
    rad = np.hypot(xx - pfit[0], yy - pfit[1]) / pfit[2]
    inner = rad < 0.9
    # Orientation: correlation of the R-B chromatic contrast (land vs ocean;
    # clouds are grey, a uniform haze offset does not change a correlation) on
    # pixels cloud-free in both images, over a sweep of time offsets.
    rb_p = photo[..., 0] - photo[..., 2]
    score = {}
    aligned = {}
    for dt, img in renders.items():
        a = align(img, limb_fit(img), pfit, photo.shape)
        aligned[dt] = a
        clear = (rad < 0.85) & ~cloud_mask(photo) & ~cloud_mask(a)
        score[dt] = float(np.corrcoef(rb_p[clear], (a[..., 0] - a[..., 2])[clear])[0, 1])
    best = max(score, key=score.get)
    corr = score
    # Sub-sample peak: parabola through the best offset and its neighbours.
    ks = sorted(score)
    i = ks.index(best)
    if 0 < i < len(ks) - 1:
        x, y = np.array(ks[i - 1:i + 2], float), np.array([score[k] for k in ks[i - 1:i + 2]])
        a2, a1, _ = np.polyfit(x, y, 2)
        t_pattern = float(-a1 / (2 * a2)) if a2 < 0 else float(best)
    else:
        t_pattern = float(best)
    # Time at which our sub-DSCOVR longitude equals EPIC's centroid longitude (Earth turns east).
    t_centroid = float(dlon / EARTH_ROTATION_DEG_S)
    ident = datetime.strptime(item["identifier"], "%Y%m%d%H%M%S")
    t_ident = (ident - datetime.strptime(item["date"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    a0 = aligned[0]
    pm = photo.mean(2) > 0.04
    rm = a0.mean(2) > 0.04
    iou = float((pm & rm).sum() / (pm | rm).sum())
    clear = inner & ~cloud_mask(photo) & ~cloud_mask(a0)
    de = ciede2000(srgb_to_lab(photo[clear]), srgb_to_lab(a0[clear]))
    # Photometric: linear tone mapping, one best-fit exposure (least squares in linear light).
    lin = load(out / "epic_linear.png")
    al = align(lin, limb_fit(lin), pfit, photo.shape)
    to_lin = lambda x: np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    to_srgb = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)
    pl, rl = to_lin(photo[clear]), to_lin(al[clear])
    k = float((pl * rl).sum() / max((rl * rl).sum(), 1e-12))
    de_lin = ciede2000(srgb_to_lab(photo[clear]), srgb_to_lab(np.clip(to_srgb(rl * k), 0, 1)))
    from PIL import Image
    side = np.concatenate([photo, a0], axis=1)
    Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).resize((2048, 1024)).save(out / "epic_vs_render.png")

    geo_err = float(np.hypot(dlon * np.cos(np.radians(c["lat"])), lat_geodetic - c["lat"]))
    result = {
        "image": item["image"], "date_utc": item["date"],
        "centroid_epic": c, "centroid_ours_geodetic": {"lat": lat_geodetic, "lon": geom["lon"]},
        "centroid_error_deg": geo_err,
        "disk_radius_px": {"photo": pfit[2], "render_at_0.62deg": rfit[2]}, "implied_epic_fov_deg": implied_fov,
        "disk_iou": iou, "rb_correlation_by_offset": corr,
        "time_from_pattern_s": t_pattern, "time_from_centroid_s": t_centroid, "identifier_time_s": t_ident,
        "cloud_free_fraction": float(clear.sum() / inner.sum()),
        "ciede2000_cloud_free_agx": {"mean": float(de.mean()), "median": float(np.median(de)), "p90": float(np.percentile(de, 90))},
        "ciede2000_cloud_free_linear_best_exposure": {"exposure": k, "mean": float(de_lin.mean()),
                                                       "median": float(np.median(de_lin)), "p90": float(np.percentile(de_lin, 90))},
    }
    checks = {
        "centroid": (geo_err < CENTROID_TOL_DEG, f"sub-DSCOVR point vs EPIC's pointing centroid within its pointing "
                     f"error ({geo_err:.2f} deg < {CENTROID_TOL_DEG} deg)"),
        "orientation": (abs(t_pattern - t_centroid) <= 60 and -60 <= t_pattern <= t_ident + 60,
                        f"time implied by the surface pattern ({t_pattern:+.0f} s) and by EPIC's centroid "
                        f"({t_centroid:+.0f} s) agree within 60 s (0.25 deg of rotation) and lie within EPIC's "
                        f"timestamp window (date +0 s .. identifier {t_ident:+.0f} s)"),
        "disk": (iou > 0.99, f"disk + night-side sliver IoU > 0.99 ({iou:.4f})"),
        "colour": (float(np.median(de_lin)) < 10.0, f"median CIEDE2000 on cloud-free pixels < 10, linear render at "
                   f"best exposure ({np.median(de_lin):.1f}; as displayed with AgX: {np.median(de):.1f})"),
    }
    # Accumulate per-image results; the report lists every image checked.
    store = HERE / "results" / "epic_results.json"
    store.parent.mkdir(exist_ok=True)
    allres = json.loads(store.read_text()) if store.exists() else {}
    allres[item["image"]] = {"checks": {k: [bool(ok), text] for k, (ok, text) in checks.items()}, "result": result}
    store.write_text(json.dumps(allres, indent=1, default=float))
    lines = ["# EPIC comparison", "",
             "DSCOVR/EPIC photographs vs the viewer rendered from DSCOVR's position at the image time. "
             "Generated by `tools/viz_validation/run_epic_check.py`.", "",
             "Note on time: EPIC publishes two timestamps per image (`date` and the identifier, ~288 s apart; the "
             "natural-colour image is a composite of bands exposed over minutes). The surface pattern and EPIC's "
             "centroid both place the image a few minutes after `date`; our Earth orientation itself agrees with JPL "
             "Horizons to < 1 arcsec (testing/test_viz/test_vizlog_environment.py).", ""]
    for image, entry in allres.items():
        rr = entry["result"]
        lines += [f"## `{image}` ({rr['date_utc']} UTC)", "", "| Check | Result | Detail |", "|---|---|---|"]
        for name, (ok, text) in entry["checks"].items():
            lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {text} |")
        lines += ["", f"Implied EPIC field of view {rr['implied_epic_fov_deg']:.4f} deg; "
                  f"cloud-free fraction compared {rr['cloud_free_fraction']:.2f}.", ""]
    (HERE / "EPIC_CHECK.md").write_text("\n".join(lines))
    for name, (ok, text) in checks.items():
        print(f"[epic] {name}: {'PASS' if ok else 'FAIL'} - {text}")
    print(json.dumps({k: result[k] for k in ('implied_epic_fov_deg', 'time_from_pattern_s', 'time_from_centroid_s', 'cloud_free_fraction',
                                              'ciede2000_cloud_free_agx', 'ciede2000_cloud_free_linear_best_exposure')}, indent=1))
    return 0 if all(ok for ok, _ in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
