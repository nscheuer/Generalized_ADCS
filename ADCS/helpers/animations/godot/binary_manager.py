"""Locate and fetch the Godot viewer for the current operating system."""

__all__ = ["ViewerNotFoundError", "find_viewer", "install_viewer", "is_wsl", "platform_key", "viewer_manifest"]

import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

from ADCS.helpers.asset_manager import Asset, AssetManager

_GODOT_VERSION = "4.7.2"
_BIN_DIR = Path(__file__).with_name("bin")
# Godot's official release archive suffix and executable path for each build.
_BUILDS = {
    "linux-x86_64": ("linux.x86_64", "Godot_v{version}-stable_linux.x86_64"),
    "linux-arm64": ("linux.arm64", "Godot_v{version}-stable_linux.arm64"),
    "windows-x86_64": ("win64.exe", "Godot_v{version}-stable_win64.exe"),
    "macos-universal": ("macos.universal", "Godot.app/Contents/MacOS/Godot"),
}


class ViewerNotFoundError(FileNotFoundError):
    """No viewer binary could be located or fetched."""


def viewer_manifest():
    """Describe the pinned Godot release and its platform archives."""
    return {"version": _GODOT_VERSION, "assets": {
        key: {"url": _asset_url(key), "executable": exe.format(version=_GODOT_VERSION)}
        for key, (_, exe) in _BUILDS.items()
    }}


def _asset_url(key: str) -> str:
    suffix, _ = _BUILDS[key]
    filename = f"Godot_v{_GODOT_VERSION}-stable_{suffix}.zip"
    return f"https://github.com/godotengine/godot/releases/download/{_GODOT_VERSION}-stable/{filename}"


def is_wsl() -> bool:
    """True when running inside Windows Subsystem for Linux."""
    if sys.platform != "linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _windows_local_appdata() -> Optional[Path]:
    """Resolve Windows %LOCALAPPDATA% as a WSL path for viewer work files."""
    try:
        win = subprocess.run(["cmd.exe", "/c", "echo %LOCALAPPDATA%"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip()
        if not win or "%" in win:
            return None
        return Path(subprocess.run(["wslpath", "-u", win], capture_output=True, text=True,
                                   timeout=10, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def platform_key() -> str:
    """Return the release asset key for this machine."""
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    if sys.platform == "win32" or is_wsl():
        return f"windows-{arch}"
    if sys.platform == "darwin":
        return "macos-universal"
    return f"linux-{arch}"


def _executable(key: str) -> Path:
    return _BIN_DIR / _BUILDS[key][1].format(version=_GODOT_VERSION)


def _download_and_unpack(key: str, *, interactive: bool) -> Path:
    archive_name = f"godot-{_GODOT_VERSION}-{key}.zip"
    asset = Asset(f"Godot {_GODOT_VERSION} ({key})", _asset_url(key), archive_name)
    manager = AssetManager(registry={key: asset}, locations=[], cache_dir=_BIN_DIR)
    archive = manager.get(key, destination=_BIN_DIR / archive_name, interactive=interactive)
    exe = _executable(key)
    if not exe.is_file():
        member = _BUILDS[key][1].format(version=_GODOT_VERSION)
        try:
            with zipfile.ZipFile(archive) as zf:
                # Extract only the known executable path from the official archive.
                exe.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, exe.open("wb") as output:
                    output.write(source.read())
        except (KeyError, zipfile.BadZipFile) as exc:
            exe.unlink(missing_ok=True)
            raise ViewerNotFoundError(f"Godot archive does not contain {member}") from exc
    if not key.startswith("windows"):
        exe.chmod(exe.stat().st_mode | 0o111)
    return exe


def _wsl_windows_executable(exe: Path) -> Path:
    """Copy a downloaded Windows build to a Windows-visible path under WSL."""
    base = _windows_local_appdata()
    if base is None:
        raise ViewerNotFoundError("could not resolve Windows %LOCALAPPDATA% from WSL")
    destination = base / "generalized_adcs" / "viewer" / _GODOT_VERSION / exe.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != exe.stat().st_size:
        shutil.copy2(exe, destination)
    return destination


def find_viewer(explicit: Optional[str | Path] = None) -> Path:
    """Resolve an explicit/environment binary or fetch the platform build on demand."""
    for candidate in (explicit, os.environ.get("ADCS_VIEWER_BIN")):
        if candidate:
            path = Path(candidate).expanduser()
            if not path.exists():
                raise ViewerNotFoundError(f"viewer not found at {path}")
            return path
    key = platform_key()
    if key not in _BUILDS:
        raise ViewerNotFoundError(f"no Godot build is configured for {key}; set ADCS_VIEWER_BIN")
    exe = _executable(key)
    if exe.is_file():
        return _wsl_windows_executable(exe) if is_wsl() and key.startswith("windows") else exe
    try:
        exe = _download_and_unpack(key, interactive=True)
        return _wsl_windows_executable(exe) if is_wsl() and key.startswith("windows") else exe
    except FileNotFoundError as exc:
        raise ViewerNotFoundError(str(exc)) from exc


def install_viewer(key: Optional[str] = None, manifest=None) -> Path:
    """Fetch the pinned Godot build for this platform via AssetManager."""
    key = key or platform_key()
    if key not in _BUILDS:
        raise ViewerNotFoundError(f"no viewer release published for {key}")
    return _download_and_unpack(key, interactive=True)
