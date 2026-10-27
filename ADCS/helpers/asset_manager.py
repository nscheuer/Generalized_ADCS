"""Locate large data assets and request consent before downloading them."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm.auto import tqdm


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    filename: str
    sha256: str | None = None


KERNEL_DIR = Path(__file__).resolve().parents[1] / "environment" / "kernels"
NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
ASSETS = {
    name: Asset(name, NAIF + name + ".bsp", name + ".bsp")
    for name in ("de421", "de442", "de442s", "de441_part-1", "de441_part-2")
}
ASSETS.update({
    "bsc5": Asset("bsc5", "https://cdsarc.cds.unistra.fr/ftp/cats/V/50/catalog.gz", "bsc5_catalog.gz"),
    "hipparcos": Asset("hipparcos", "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat", "hip_main.dat"),
    "igrf14": Asset("igrf14", "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt", "IGRF14.shc"),
})


class AssetManager:
    """Resolve registered files from local directories or a user cache."""

    def __init__(self, registry=None, locations=None, cache_dir=None):
        self.registry = ASSETS if registry is None else registry
        self.locations = [KERNEL_DIR] if locations is None else [Path(p) for p in locations]
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else base / "generalized_adcs"

    def get(self, name: str, *, destination=None, interactive=True, allow_download=False) -> Path:
        asset = self.registry[name]
        candidates = ([Path(destination)] if destination is not None else
                      [self.cache_dir / asset.filename] + [p / asset.filename for p in self.locations])
        for path in candidates:
            if path.is_file():
                return path
        target = candidates[0]
        if not allow_download:
            if not interactive or not sys.stdin.isatty():
                raise FileNotFoundError(f"Missing {name}: {target}. Download from {asset.url} with allow_download=True")
            answer = input(f"Download {name} from {asset.url} to {target}? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                raise FileNotFoundError(f"Download declined for {name}: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        try:
            request = Request(asset.url, headers={"User-Agent": "Mozilla/5.0 (compatible; Generalized-ADCS asset downloader)"})
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=name) as bar:
                with urlopen(request) as response, partial.open("wb") as output:
                    length = response.headers.get("Content-Length")
                    if length is not None:
                        bar.total = int(length)
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        bar.update(len(chunk))
            if asset.sha256:
                import hashlib
                if hashlib.sha256(partial.read_bytes()).hexdigest() != asset.sha256:
                    raise ValueError(f"Checksum mismatch for {name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return target
