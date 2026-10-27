__all__ = ["cache_dir", "load_sources", "raw_path", "fetch", "file_sha256"]

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_SOURCES = Path(__file__).with_name("sources.json")


def load_sources(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    r"""Raw-input definitions: ``{id: {url, bytes, sha256, license, attribution, ...}}``."""
    with open(path or _SOURCES, encoding="utf-8") as f:
        return json.load(f)["sources"]


def cache_dir() -> Path:
    """Root for explicitly requested visualization assets."""
    override = os.environ.get("ADCS_VIZ_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "assets"


def raw_path(source_id: str, source: Dict[str, Any]) -> Path:
    name = source["url"].rsplit("/", 1)[-1]
    return cache_dir() / "raw" / source_id / name


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    ids: Optional[Iterable[str]] = None,
    sources: Optional[Dict[str, Dict[str, Any]]] = None,
    accept: Iterable[str] = (),
    log=print,
) -> List[Path]:
    r"""
    Download raw inputs into the cache and verify them.

    A file already in the cache is re-verified, not downloaded again. Every
    download goes to a temporary file and is moved into place only after its
    size and SHA-256 match ``sources.json``. Sources with no pinned hash yet
    are accepted and their hash is reported, so it can be pinned.

    :param ids: Source ids to fetch (default: all).
    :param sources: Source table (default: the packaged ``sources.json``).
    :param accept: Ids whose licence the user has accepted
        (required for sources with ``requires_acceptance``).
    :return: Paths of the verified files.
    :raises ValueError: On a size or checksum mismatch, or an unaccepted licence.
    """
    from ADCS.helpers.asset_manager import Asset, AssetManager

    sources = sources or load_sources()
    ids = list(sources) if ids is None else list(ids)  # [] means nothing, not everything
    accepted = set(accept)
    out = []
    for sid in ids:
        src = sources[sid]
        if src.get("requires_acceptance") and sid not in accepted:
            raise ValueError(f"{sid}: licence must be accepted explicitly ({src['license']}); "
                             f"pass --accept-license {sid}")
        dest = raw_path(sid, src)
        manager = AssetManager(registry={sid: Asset(sid, src["url"], dest.name, src.get("sha256"))},
                               locations=[], cache_dir=dest.parent)
        # This command is explicit, but still asks before each missing download.
        manager.get(sid, destination=dest, interactive=True, allow_download=False)
        _verify(sid, src, dest)
        out.append(dest)
    return out


def _verify(sid: str, src: Dict[str, Any], path: Path) -> str:
    size = path.stat().st_size
    if src.get("bytes") and size != src["bytes"]:
        raise ValueError(f"{sid}: size {size} != expected {src['bytes']}")
    digest = file_sha256(path)
    if src.get("sha256") is None:
        print(f"note: {sid} has no pinned sha256; got {digest}")
    elif digest != src["sha256"]:
        raise ValueError(f"{sid}: sha256 {digest} != pinned {src['sha256']}")
    return digest
