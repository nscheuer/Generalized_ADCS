"""Resolve Solar System Scope textures and fetch missing ones with consent."""

import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from tqdm.auto import tqdm

SOURCE = "https://www.solarsystemscope.com/textures/download/"
NASA_STAR_MAP_BASE = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/"
BUNDLED = Path(__file__).resolve().parents[1] / "plotting" / "textures"


def _cache_dir() -> Path:
    override = os.environ.get("ADCS_TEXTURE_DIR")
    if override:
        return Path(override).expanduser()
    if os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path.home() / ".cache"
    return base / "generalized_adcs" / "textures"


def texture_path(name: str) -> Path:
    """Return a cached or checkout texture path, without network access."""
    if Path(name).name != name:
        raise ValueError(f"Invalid texture name: {name}")
    if name.endswith('.tif'):
        png_name = name.replace('.tif', '.png')
        for path in (_cache_dir() / png_name, BUNDLED / png_name):
            if path.is_file():
                return path
        for source in (_cache_dir() / name, BUNDLED / name):
            if source.is_file():
                from PIL import Image
                target = _cache_dir() / png_name
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as original:
                    clean = Image.frombytes(original.mode, original.size, original.tobytes())
                    clean.save(target)
                return target
    for path in (_cache_dir() / name, BUNDLED / name):
        if path.is_file():
            return path
    return _cache_dir() / name


def ensure_textures(names, on_progress=None) -> dict[str, Path]:
    """Prompt once with the total size, then download the missing textures."""
    names = tuple(dict.fromkeys(names))
    missing = [name for name in names if not texture_path(name).is_file()]
    sizes = {}
    for name in missing:
        url = NASA_STAR_MAP_BASE + name if name.startswith('starmap_2020_') and name.endswith('.exr') else SOURCE + name
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise RuntimeError(f"Texture provider did not report a file size: {name}")
            sizes[name] = int(length)
    if missing:
        total = sum(sizes.values())
        if on_progress is not None:
            on_progress(0, total, 'Preparing graphics')
        print(f"Graphics textures: {len(missing)} files, {total / 1048576:.1f} MiB total")
        sys.stdout.flush()
        if not sys.stdin.isatty():
            raise RuntimeError("Textures are missing. Run the viewer in a terminal to approve the download, or set ADCS_TEXTURE_DIR.")
        if input("Download now? [y/N] ").strip().lower() not in ("y", "yes"):
            raise RuntimeError("Texture download declined")
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                  desc='Downloading textures', dynamic_ncols=True) as progress:
            for name in missing:
                temporary = cache / (name + ".part")
                progress.set_postfix_str(name, refresh=False)
                downloaded = 0
                try:
                    url = NASA_STAR_MAP_BASE + name if name.startswith('starmap_2020_') and name.endswith('.exr') else SOURCE + name
                    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urlopen(request, timeout=60) as source, temporary.open("wb") as target:
                        while block := source.read(1024 * 1024):
                            target.write(block)
                            downloaded += len(block)
                            progress.update(len(block))
                            if on_progress is not None:
                                on_progress(progress.n, total, name)
                    if downloaded != sizes[name]:
                        raise IOError(f"Incomplete texture download: {name} ({downloaded} of {sizes[name]} bytes)")
                    temporary.replace(cache / name)
                    if name.endswith('.tif'):
                        from PIL import Image
                        with Image.open(cache / name) as original:
                            clean = Image.frombytes(original.mode, original.size, original.tobytes())
                            clean.save(cache / name.replace('.tif', '.png'))
                finally:
                    temporary.unlink(missing_ok=True)
        if on_progress is not None:
            on_progress(total, total, 'Finishing graphics')
    return {name: texture_path(name) for name in names}


PLANETS_2K = ("2k_mercury.jpg", "2k_venus_atmosphere.jpg", "2k_moon.jpg",
              "2k_mars.jpg", "2k_jupiter.jpg", "2k_saturn.jpg",
              "2k_uranus.jpg", "2k_neptune.jpg", "2k_sun.jpg")
EARTH_ADVANCED_2K = ("2k_earth_nightmap.jpg", "2k_earth_clouds.jpg",
                     "2k_earth_normal_map.tif", "2k_earth_specular_map.tif")


def tier_textures(tier: str) -> tuple[str, ...]:
    """All textures needed by a tier, for a single consent prompt."""
    if tier not in ("low", "medium", "high"):
        raise ValueError(f"Unknown graphics tier: {tier}")
    if tier == "low":
        return ("starmap_2020_4k.exr", "2k_earth_daymap.jpg", *PLANETS_2K,
                "2k_saturn_ring_alpha.png")
    if tier == "medium":
        return ("starmap_2020_4k.exr", "2k_earth_daymap.jpg", *EARTH_ADVANCED_2K,
                *PLANETS_2K, "2k_saturn_ring_alpha.png")
    return ("starmap_2020_16k.exr", "2k_earth_daymap.jpg", *EARTH_ADVANCED_2K,
            *PLANETS_2K, "2k_saturn_ring_alpha.png",
            "8k_earth_daymap.jpg", "8k_earth_nightmap.jpg",
            "8k_earth_clouds.jpg", "8k_earth_normal_map.tif",
            "8k_earth_specular_map.tif", "8k_mercury.jpg",
            "4k_venus_atmosphere.jpg",
            "8k_moon.jpg", "8k_mars.jpg", "8k_jupiter.jpg",
            "8k_saturn.jpg", "2k_uranus.jpg", "2k_neptune.jpg",
            "8k_sun.jpg", "8k_saturn_ring_alpha.png")


def tier_name(tier: str, name: str) -> str:
    if tier != "high":
        return name
    if name == "2k_venus_atmosphere.jpg":
        return "4k_venus_atmosphere.jpg"
    if name in ("2k_uranus.jpg", "2k_neptune.jpg"):
        return name
    return name.replace("2k_", "8k_", 1)
