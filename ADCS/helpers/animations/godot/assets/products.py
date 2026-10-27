r"""
Build viewer-ready assets from the raw sources.

Each product is a recipe (inputs + parameters + code version). Its outputs go
to ``processed/<product>/<recipe hash>/`` with an ``asset_manifest.json``;
``processed/current.json`` maps each product to its latest build, which is
what the viewer reads. Building a product whose recipe hash already exists is
a no-op.
"""

__all__ = ["PRODUCTS", "build", "recipe_hash", "current_products", "verify_processed"]

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ADCS.helpers.animations.godot.assets.cubemap import FACES, equirect_to_cube, lonlat, sample_equirect
from ADCS.helpers.animations.godot.assets.fetch import cache_dir, fetch, file_sha256, load_sources
from ADCS.helpers.animations.godot.assets.stars import COLUMNS, parse_bsc5, star_table

#: Bump when processing code changes in a way that alters outputs.
CODE_VERSION = 1

R_EARTH_M = 6378137.0
R_MOON_M = 1737400.0
GEBCO_ELEV_M_PER_STEP = 6400.0 / 255.0  # 8-bit land elevation, saturating at 6400 m


@dataclass(frozen=True)
class Product:
    name: str
    inputs: List[str]
    params: Dict[str, Any]
    kind: str            # "cube" or "stars"
    description: str
    frame: str = "-"
    encoding: str = "-"
    builder: Optional[Callable[..., Any]] = field(default=None, compare=False)


# ------------------------------------------------------------------ image I/O
def _open_image(path: Path, mode: Optional[str] = None) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # trusted, checksum-verified inputs of known size
    with Image.open(path) as im:
        if mode is not None and im.mode != mode:
            im = im.convert(mode)
        return np.asarray(im)


def _open_paletted_indices(path: Path) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        if im.mode != "P" and im.mode != "L":
            raise ValueError(f"{path}: expected a paletted image, got {im.mode}")
        return np.asarray(im)


def _save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path, compress_level=6)


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ texel functions
def _color_texels(img: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def fn(d: np.ndarray) -> np.ndarray:
        lon, lat = lonlat(d)
        return _to_u8(sample_equirect(img, lon, lat))
    return fn


def _height_normals(height_m: np.ndarray, radius_m: float, exaggeration: float, step_rad: float
                    ) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Object-space (body-fixed) normals from an equirectangular height field,
    encoded as RGB = 127.5 * (n + 1). Central differences on the sphere:
    east/north slopes -> local up-tilted normal -> body frame.
    """
    def fn(d: np.ndarray) -> np.ndarray:
        lon, lat = lonlat(d)
        h_e = sample_equirect(height_m, lon + step_rad, lat)
        h_w = sample_equirect(height_m, lon - step_rad, lat)
        h_n = sample_equirect(height_m, lon, np.minimum(lat + step_rad, np.pi / 2))
        h_s = sample_equirect(height_m, lon, np.maximum(lat - step_rad, -np.pi / 2))
        coslat = np.maximum(np.cos(lat), 1e-3)
        dh_de = exaggeration * (h_e - h_w) / (2.0 * step_rad * radius_m * coslat)
        dh_dn = exaggeration * (h_n - h_s) / (2.0 * step_rad * radius_m)
        n_enu = np.stack([-dh_de, -dh_dn, np.ones_like(dh_de)], axis=-1)
        n_enu /= np.linalg.norm(n_enu, axis=-1, keepdims=True)
        sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
        east = np.stack([-so, co, np.zeros_like(so)], axis=-1)
        north = np.stack([-sl * co, -sl * so, cl], axis=-1)
        up = np.stack([cl * co, cl * so, sl], axis=-1)
        n = n_enu[..., :1] * east + n_enu[..., 1:2] * north + n_enu[..., 2:] * up
        return _to_u8(127.5 * (n + 1.0))
    return fn


# ------------------------------------------------------------------ builders
def _build_cube(product: Product, inputs: Dict[str, Path], out: Path) -> List[Dict[str, Any]]:
    p = product.params
    size = int(p["face_size"])
    name = product.name
    if name in ("earth_day", "moon_color"):
        fn = _color_texels(_open_image(inputs[product.inputs[0]], "RGB"))
    elif name in ("earth_night", "earth_clouds"):
        fn = _color_texels(_open_image(inputs[product.inputs[0]], "L"))
    elif name == "earth_ocean_mask":
        idx = _open_paletted_indices(inputs[product.inputs[0]])
        ocean = np.where(idx < 255, 255, 0).astype(np.uint8)
        del idx

        def fn(d: np.ndarray) -> np.ndarray:
            lon, lat = lonlat(d)
            return sample_equirect(ocean, lon, lat, nearest=True).astype(np.uint8)
    elif name == "earth_normals":
        h = _open_image(inputs[product.inputs[0]], "L").astype(np.float32) * GEBCO_ELEV_M_PER_STEP
        fn = _height_normals(h, R_EARTH_M, p["exaggeration"], np.pi / h.shape[0])
    elif name == "moon_normals":
        h = _open_image(inputs[product.inputs[0]]).astype(np.float32) * 1000.0  # km -> m
        fn = _height_normals(h, R_MOON_M, p["exaggeration"], np.pi / h.shape[0])
    else:
        raise ValueError(f"no cube builder for {name}")
    files = []
    for face, arr in equirect_to_cube(size, fn):
        fname = f"{name}_{face.replace('+', 'p').replace('-', 'n')}.png"
        _save_png(arr, out / fname)
        files.append({"name": fname, "face": face, "width": size, "height": size,
                      "channels": 1 if arr.ndim == 2 else arr.shape[2], "sha256": file_sha256(out / fname)})
    return files


def _build_stars(product: Product, inputs: Dict[str, Path], out: Path) -> List[Dict[str, Any]]:
    cat = parse_bsc5(inputs[product.inputs[0]])
    table = star_table(cat)
    keep = table[:, COLUMNS.index("vmag")] <= product.params["max_vmag"]
    table = np.ascontiguousarray(table[keep])
    (out / "stars.f32").write_bytes(table.tobytes())
    return [{"name": "stars.f32", "rows": int(len(table)), "columns": list(COLUMNS), "dtype": "<f4",
             "sha256": file_sha256(out / "stars.f32")}]


def _build_atmosphere(product: Product, inputs: Dict[str, Path], out: Path) -> List[Dict[str, Any]]:
    from ADCS.helpers.animations.godot.assets.atmosphere import AtmosphereParams, build_tables, params_json

    p = AtmosphereParams()
    tables = build_tables(p)
    files = []
    for name, arr in tables.items():
        path = out / f"{name}.f32"
        path.write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        files.append({"name": path.name, "shape": list(arr.shape), "dtype": "<f4", "sha256": file_sha256(path)})
    (out / "atmosphere.json").write_text(json.dumps(params_json(p), indent=1))
    files.append({"name": "atmosphere.json", "sha256": file_sha256(out / "atmosphere.json")})
    return files


def _products() -> Dict[str, Product]:
    defs = [
        Product("earth_day", ["bmng_topo_bathy_200407"], {"face_size": 4096}, "cube",
                "Daytime surface colour (July)", "ITRS", "sRGB 8-bit RGB"),
        Product("earth_night", ["black_marble_2016_3km_gray"], {"face_size": 2048}, "cube",
                "City lights", "ITRS", "sRGB-encoded 8-bit luminance (stylised brightness, not radiance)"),
        Product("earth_clouds", ["clouds_bm2002_8192"], {"face_size": 2048}, "cube",
                "Cloud cover (static 2002 composite)", "ITRS", "8-bit cloud brightness/opacity"),
        Product("earth_ocean_mask", ["gebco08_bath_21600"], {"face_size": 2048}, "cube",
                "Ocean mask for specular reflection (inland lakes excluded)", "ITRS", "8-bit, 255 = ocean"),
        Product("earth_normals", ["gebco08_elev_21600"], {"face_size": 2048, "exaggeration": 8.0}, "cube",
                "Terrain normals (land relief, exaggerated)", "ITRS",
                "object-space normal, RGB = 127.5 (n + 1)"),
        Product("moon_color", ["svs_moon_color_4k"], {"face_size": 1024}, "cube",
                "Lunar surface colour", "MOON_ME (as published by SVS)", "sRGB 8-bit RGB"),
        Product("moon_normals", ["svs_moon_ldem_4"], {"face_size": 512, "exaggeration": 1.0}, "cube",
                "Lunar terrain normals from LOLA", "MOON_ME (as published by SVS)",
                "object-space normal, RGB = 127.5 (n + 1)"),
        Product("atmosphere", [], {"model": "hillaire2020", "transmittance": [256, 64], "multiscattering": [32, 32]},
                "atmosphere", "Precomputed atmosphere tables (transmittance, multiple scattering)",
                "-", "little-endian float32 (rows, cols, RGB); see atmosphere.json"),
        Product("stars", ["bsc5_cds"], {"max_vmag": 7.0}, "stars",
                "Naked-eye star table from BSC5", "ICRS (J2000 equinox, epoch 2000.0)",
                "little-endian float32 rows, see columns"),
    ]
    return {p.name: p for p in defs}


PRODUCTS = _products()


# ------------------------------------------------------------------ orchestration
def recipe_hash(product: Product, sources: Dict[str, Dict[str, Any]], input_sha: Dict[str, str]) -> str:
    recipe = {"product": product.name, "params": product.params, "code_version": CODE_VERSION,
              "inputs": {sid: input_sha[sid] for sid in product.inputs}}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def current_products() -> Dict[str, str]:
    path = cache_dir() / "processed" / "current.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _write_current(entries: Dict[str, str]) -> None:
    path = cache_dir() / "processed" / "current.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1, sort_keys=True))
    os.replace(tmp, path)


def build(
    names: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    sources: Optional[Dict[str, Dict[str, Any]]] = None,
    log=print,
) -> Dict[str, Path]:
    r"""
    Fetch inputs as needed and build products.

    :param names: Products to build (default: all).
    :param overrides: Per-product parameter overrides, e.g.
        ``{"earth_day": {"face_size": 2048}}``.
    :param sources: Source table (default: packaged ``sources.json``).
    :return: ``{product: output directory}``.
    """
    sources = sources or load_sources()
    names = names or list(PRODUCTS)
    overrides = overrides or {}
    current = current_products()
    out_dirs = {}
    for name in names:
        base = PRODUCTS[name]
        product = Product(base.name, base.inputs, {**base.params, **overrides.get(name, {})}, base.kind,
                          base.description, base.frame, base.encoding)
        paths = dict(zip(product.inputs, fetch(product.inputs, sources=sources, log=log))) if product.inputs else {}
        input_sha = {sid: file_sha256(p) for sid, p in paths.items()}
        rh = recipe_hash(product, sources, input_sha)
        out = cache_dir() / "processed" / name / rh[:16]
        manifest_path = out / "asset_manifest.json"
        if manifest_path.exists():
            log(f"build {name}: up to date ({rh[:16]})")
        else:
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            log(f"build {name}: {product.params} ...")
            builder = {"stars": _build_stars, "atmosphere": _build_atmosphere}.get(product.kind, _build_cube)
            files = builder(product, paths, out)
            manifest = {
                "product": name, "description": product.description, "kind": product.kind,
                "recipe_hash": rh, "params": product.params, "code_version": CODE_VERSION,
                "inputs": {sid: {"sha256": input_sha[sid], "url": sources[sid]["url"]} for sid in product.inputs},
                "frame": product.frame, "encoding": product.encoding,
                "cube_convention": "OpenGL/Vulkan cube map; see ADCS/viz/assets/cubemap.py" if product.kind == "cube" else None,
                "face_order": list(FACES) if product.kind == "cube" else None,
                "attribution": [sources[sid]["attribution"] for sid in product.inputs],
                "files": files, "build_seconds": round(time.perf_counter() - t0, 1),
            }
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest, indent=1))
            os.replace(tmp, manifest_path)  # written last: a manifest means a complete build
            log(f"build {name}: done in {manifest['build_seconds']} s -> {out}")
        current[name] = str(out.relative_to(cache_dir() / "processed"))
        _write_current(current)
        out_dirs[name] = out
    return out_dirs


def verify_processed() -> Dict[str, List[str]]:
    r"""Re-hash every current product's files. Returns ``{product: [problems]}``."""
    problems: Dict[str, List[str]] = {}
    root = cache_dir() / "processed"
    for name, rel in current_products().items():
        issues = []
        manifest_path = root / rel / "asset_manifest.json"
        if not manifest_path.exists():
            problems[name] = ["missing asset_manifest.json"]
            continue
        for f in json.loads(manifest_path.read_text())["files"]:
            p = root / rel / f["name"]
            if not p.exists():
                issues.append(f"missing {f['name']}")
            elif file_sha256(p) != f["sha256"]:
                issues.append(f"checksum mismatch {f['name']}")
        problems[name] = issues
    return problems


def write_attribution(sources: Optional[Dict[str, Dict[str, Any]]] = None) -> Path:
    r"""Write ``ATTRIBUTION.txt`` for the sources behind the current products."""
    sources = sources or load_sources()
    used = sorted({sid for name in current_products() if name in PRODUCTS for sid in PRODUCTS[name].inputs})
    lines = ["Visualization assets built by Generalized ADCS from the following sources.", ""]
    for sid in used:
        s = sources[sid]
        lines += [f"{sid}: {s['attribution']}", f"  {s['url']}", f"  Licence: {s['license']}", ""]
    path = cache_dir() / "ATTRIBUTION.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
