"""Headless tests for the Godot viewer's GDScript core."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ADCS.helpers.animations.godot.binary_manager import install_viewer, platform_key


def _godot_executable() -> Path:
    configured = os.environ.get("ADCS_VIEWER_BIN")
    if configured:
        return Path(configured).expanduser()
    on_path = shutil.which("godot") or shutil.which("godot4")
    if on_path:
        return Path(on_path)
    # The GUI launcher prefers Windows under WSL, but this headless test needs
    # a native Linux executable.
    if sys.platform == "linux":
        machine = os.uname().machine
        key = "linux-arm64" if machine in ("aarch64", "arm64") else "linux-x86_64"
    else:
        key = platform_key()
    # This test is explicitly selected with `-m local`, so fetching the pinned
    # engine is an intentional part of running it when no executable is present.
    return install_viewer(key, allow_download=True)


@pytest.mark.local
def test_godot_viewer_gdscript_suite():
    """Run the viewer tests when explicitly selected for a local run."""
    viewer_dir = Path(__file__).resolve().parents[2] / "ADCS/helpers/animations/godot/viewer"
    godot = _godot_executable()
    result = subprocess.run(
        [str(godot), "--headless", "--path", str(viewer_dir), "--script", "res://tests/run_tests.gd"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"Godot test suite exited {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ALL PASSED" in result.stdout
