r"""
Locate, download and verify the 3-D viewer binary.

The viewer is a separate native program (a Godot export), published on GitHub
Releases and pinned by version and SHA-256 in ``_viewer_manifest.json``. It is
never bundled in the Python wheel. Resolution order:

1. an explicit path passed by the caller,
2. the ``ADCS_VIEWER_BIN`` environment variable (developers running their own
   build or the Godot editor),
3. a previously installed release in the viewer cache.

Inside WSL the viewer runs as a native Windows program, so the Windows build is
selected and installed under ``%LOCALAPPDATA%``, where Windows can run it at
full GPU speed.
"""

__all__ = ["ViewerNotFoundError", "find_viewer", "install_viewer", "is_wsl", "platform_key", "viewer_manifest"]

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

_MANIFEST = Path(__file__).with_name("_viewer_manifest.json")


class ViewerNotFoundError(FileNotFoundError):
    """No viewer binary could be located."""


def viewer_manifest() -> Dict[str, Any]:
    r"""The pinned viewer release: ``{"version", "assets": {platform: {url, sha256, executable}}}``."""
    with open(_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def is_wsl() -> bool:
    r"""True when running inside Windows Subsystem for Linux."""
    if sys.platform != "linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def platform_key() -> str:
    r"""Release asset key for the viewer this machine should run."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "win32" or is_wsl():
        return f"windows-{arch}"
    if sys.platform == "darwin":
        return "macos-universal"
    return f"linux-{arch}"


def _windows_local_appdata() -> Optional[Path]:
    # From WSL: ask Windows for %LOCALAPPDATA% and translate it to a /mnt path.
    try:
        win = subprocess.run(["cmd.exe", "/c", "echo %LOCALAPPDATA%"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip()
        if not win or "%" in win:
            return None
        return Path(subprocess.run(["wslpath", "-u", win], capture_output=True, text=True,
                                   timeout=10, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def cache_dir() -> Path:
    r"""Where viewer releases are installed (``$ADCS_VIEWER_CACHE`` overrides)."""
    override = os.environ.get("ADCS_VIEWER_CACHE")
    if override:
        return Path(override).expanduser()
    if is_wsl():
        local = _windows_local_appdata()
        if local is not None:
            return local / "generalized_adcs" / "viewer"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "generalized_adcs" / "viewer"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "generalized_adcs" / "viewer"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "generalized_adcs" / "viewer"


def _installed(manifest: Dict[str, Any], key: str) -> Optional[Path]:
    asset = manifest.get("assets", {}).get(key)
    if not manifest.get("version") or not asset:
        return None
    exe = cache_dir() / manifest["version"] / key / asset["executable"]
    return exe if exe.exists() else None


def find_viewer(explicit: Optional[str | Path] = None) -> Path:
    r"""
    Resolve the viewer executable.

    :param explicit: Path given by the caller; takes precedence.
    :raises ViewerNotFoundError: If no viewer is available, with instructions.
    """
    for candidate in (explicit, os.environ.get("ADCS_VIEWER_BIN")):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.exists():
                raise ViewerNotFoundError(f"viewer not found at {path}")
            return path
    manifest = viewer_manifest()
    installed = _installed(manifest, platform_key())
    if installed is not None:
        return installed
    if not manifest.get("version"):
        raise ViewerNotFoundError(
            "no viewer release is published yet. Run the viewer from the Godot editor and "
            "connect it to `python -m ADCS.viz view <log> --no-launch`, or set ADCS_VIEWER_BIN.")
    raise ViewerNotFoundError(
        f"viewer {manifest['version']} is not installed; run `python -m ADCS.viz install-viewer`.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"archive member escapes the install directory: {member.filename}")
    archive.extractall(dest)


def install_viewer(key: Optional[str] = None, manifest: Optional[Dict[str, Any]] = None) -> Path:
    r"""
    Download, verify (SHA-256) and unpack the pinned viewer release.

    :param key: Platform asset key; defaults to :func:`platform_key`.
    :param manifest: Release manifest; defaults to the packaged one.
    :return: Path to the installed executable.
    :raises ViewerNotFoundError: If no release is published for this platform.
    :raises ValueError: If the download's checksum does not match.
    """
    manifest = manifest or viewer_manifest()
    key = key or platform_key()
    asset = manifest.get("assets", {}).get(key)
    if not manifest.get("version") or not asset:
        raise ViewerNotFoundError(f"no viewer release published for {key}")
    dest = cache_dir() / manifest["version"] / key
    exe = dest / asset["executable"]
    if exe.exists():
        return exe

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "viewer.zip"
        with urllib.request.urlopen(asset["url"], timeout=300) as resp, open(archive_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        digest = _sha256(archive_path)
        if digest != asset["sha256"].lower():
            raise ValueError(f"viewer download checksum mismatch: got {digest}, expected {asset['sha256']}")
        staging = Path(tmp) / "unpacked"
        with zipfile.ZipFile(archive_path) as zf:
            _safe_extract(zf, staging)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(staging), str(dest))
    if not exe.exists():
        raise ViewerNotFoundError(f"release archive does not contain {asset['executable']}")
    if not key.startswith("windows"):
        exe.chmod(exe.stat().st_mode | 0o111)
    return exe
