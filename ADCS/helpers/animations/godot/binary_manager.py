r"""
Locate, download and verify the 3-D viewer binary.

The Linux x86_64 Godot executable is bundled in the Python wheel as a
checksum-pinned ZIP and extracted to ``animations/assets/bin`` on first use.
Resolution order:

1. an explicit path passed by the caller,
2. the ``ADCS_VIEWER_BIN`` environment variable (developers running their own
   build or the Godot editor),
3. the bundled executable,
4. a previously installed release in the viewer cache.

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
_GODOT_VERSION = "4.7.2"
_BUNDLED_ARCHIVE_SHA256 = "cadd3204e728a35d3f13adb7fd0d7902636b79f6b95c40c265eb73b6c35329e4"
_BUNDLED_BINARY_SHA256 = "8d106cbe6144c2dc7e881d61d2429c1a8a76e6b22ef48bd5e48dcf934953f71e"


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

    # On WSL, prefer native Windows Godot so WSLg does not route the 3-D
    # viewport through the Linux software renderer (LLVMpipe). The project is
    # copied to Windows by launcher.viewer_project().
    if is_wsl():
        windows_godot = (cache_dir() / _GODOT_VERSION / "windows-x86_64"
                         / f"Godot_v{_GODOT_VERSION}-stable_win64.exe")
        if windows_godot.is_file():
            return windows_godot

    manifest = viewer_manifest()
    installed = _installed(manifest, platform_key())
    if installed is not None:
        return installed

    if sys.platform == "linux" and not is_wsl() and platform.machine().lower() in ("x86_64", "amd64"):
        archive = Path(__file__).with_name("bin") / "godot-linux-x86_64.zip"
        if archive.is_file():
            target = Path(__file__).resolve().parents[1] / "assets" / "bin" / "godot"
            if target.is_file() and _sha256(target) == _BUNDLED_BINARY_SHA256:
                return target
            if _sha256(archive) != _BUNDLED_ARCHIVE_SHA256:
                raise ValueError("bundled Godot archive checksum mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name("godot.part")
            with zipfile.ZipFile(archive) as zf, zf.open("Godot_v4.7.2-stable_linux.x86_64") as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out)
            if _sha256(temporary) != _BUNDLED_BINARY_SHA256:
                temporary.unlink(missing_ok=True)
                raise ValueError("bundled Godot executable checksum mismatch")
            temporary.chmod(0o755)
            os.replace(temporary, target)
            return target
    if not manifest.get("version"):
        if is_wsl():
            raise ViewerNotFoundError(
                "no native Windows Godot viewer found in the WSL cache; install Godot for Windows or set ADCS_VIEWER_BIN")
        raise ViewerNotFoundError(
            "no viewer release is published yet. Run the viewer from the Godot editor and "
            "connect it to `python -m ADCS.helpers.animations.godot view <log> --no-launch`, or set ADCS_VIEWER_BIN.")
    raise ViewerNotFoundError(
        f"viewer {manifest['version']} is not installed; run `python -m ADCS.helpers.animations.godot install-viewer`.")


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
