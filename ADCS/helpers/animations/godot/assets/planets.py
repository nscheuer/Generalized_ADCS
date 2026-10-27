"""Optional planetary textures, downloaded only by an explicit command."""

from pathlib import Path
import sys

from ADCS.helpers.asset_manager import Asset, AssetManager

from .fetch import cache_dir

SOURCE = "https://www.solarsystemscope.com/textures/download/"
FILES = {
    "sun": {"2k": "2k_sun.jpg", "8k": "8k_sun.jpg"},
    "mercury": {"2k": "2k_mercury.jpg", "8k": "8k_mercury.jpg"},
    "venus": {"2k": "2k_venus_atmosphere.jpg", "8k": "4k_venus_atmosphere.jpg"},
    "earth": {"2k": "2k_earth_daymap.jpg", "8k": "8k_earth_daymap.jpg"},
    "earth_night": {"2k": "2k_earth_nightmap.jpg", "8k": "8k_earth_nightmap.jpg"},
    "earth_normal": {"2k": "2k_earth_normal_map.tif", "8k": "8k_earth_normal_map.tif"},
    "earth_specular": {"2k": "2k_earth_specular_map.tif", "8k": "8k_earth_specular_map.tif"},
    "moon": {"2k": "2k_moon.jpg", "8k": "8k_moon.jpg"},
    "mars": {"2k": "2k_mars.jpg", "8k": "8k_mars.jpg"},
    "jupiter": {"2k": "2k_jupiter.jpg", "8k": "8k_jupiter.jpg"},
    "saturn": {"2k": "2k_saturn.jpg", "8k": "8k_saturn.jpg"},
    # Solar System Scope does not publish 8K maps for Uranus or Neptune.
    "uranus": {"2k": "2k_uranus.jpg", "8k": "2k_uranus.jpg"},
    "neptune": {"2k": "2k_neptune.jpg", "8k": "2k_neptune.jpg"},
}


def _manager(lod: str, names: tuple[str, ...], root: Path | None = None) -> tuple[AssetManager, dict[str, str]]:
    root = (root or cache_dir()) / "planet_textures"
    registry = {}
    result_names = {}
    for name in names:
        filename = FILES[name][lod]
        asset_name = f"{lod}_{name}"
        registry[asset_name] = Asset(asset_name, SOURCE + filename, filename)
        result_names[asset_name] = name
    return AssetManager(registry=registry, locations=[], cache_dir=root), result_names


def missing_planet_textures(lod="2k", names=None, *, root: Path | None = None) -> dict[str, Path]:
    """Find missing files without prompting or accessing the network."""
    if lod not in {"2k", "8k"}:
        raise ValueError("lod must be '2k' or '8k'")
    names = tuple(FILES if names is None else names)
    unknown = set(names) - set(FILES)
    if unknown:
        raise ValueError(f"unknown planet textures: {sorted(unknown)}")
    root = root or cache_dir()
    manager, result_names = _manager(lod, names, root)
    missing = {}
    for asset_name in result_names:
        try:
            found = manager.get(asset_name, interactive=False)
            if found.suffix.lower() in {".tif", ".tiff"} and not found.with_suffix(".png").exists():
                raise FileNotFoundError(found.with_suffix(".png"))
        except FileNotFoundError:
            missing[result_names[asset_name]] = root / "planet_textures" / FILES[result_names[asset_name]][lod]
    return missing


def fetch_planets(names=None, lod="2k", *, accepted=False, root: Path | None = None) -> dict[str, Path]:
    """Fetch a requested texture tier through AssetManager.

    By default each asset uses AssetManager's terminal y/n prompt. Set
    ``accepted=True`` after an aggregate confirmation to download the batch.
    """
    if lod not in {"2k", "8k"}:
        raise ValueError("lod must be '2k' or '8k'")
    names = tuple(FILES if names is None else names)
    unknown = set(names) - set(FILES)
    if unknown:
        raise ValueError(f"unknown planet textures: {sorted(unknown)}")
    manager, result_names = _manager(lod, names, root)
    out = {}
    for asset_name, name in result_names.items():
        path = manager.get(asset_name, interactive=not accepted, allow_download=accepted)
        if path.suffix.lower() in {".tif", ".tiff"}:
            from PIL import Image
            png = path.with_suffix(".png")
            if not png.exists():
                with Image.open(path) as image:
                    image.convert("RGB").save(png)
            path = png
        out[name] = path
    return out


def ensure_planet_textures(lod="2k", *, root: Path | None = None) -> bool:
    """Prompt once in the terminal for all missing maps in a resolution tier."""
    root = root or cache_dir()
    missing = missing_planet_textures(lod, root=root)
    if not missing:
        return True
    names = tuple(missing)
    if not sys.stdin.isatty():
        print(f"Missing {lod} visualization textures. Run `python -m ADCS.helpers.animations.godot.assets fetch-planets --lod {lod}` in a terminal.")
        return False
    print(f"Missing {len(names)} {lod} Solar System Scope textures:", flush=True)
    for name, path in missing.items():
        print(f"  {name}: {path.name}", flush=True)
    answer = input(f"Download these textures to {root / 'planet_textures'}? [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print(f"Skipping {lod} textures; the viewer will use its built-in colors or other local maps.")
        return False
    fetch_planets(names, lod=lod, accepted=True, root=root)
    return True
