"""Runs the viewer's headless GDScript unit tests (viewer/tests/run_tests.gd) when Godot is available.

Godot is looked up in $ADCS_GODOT_BIN, then `godot`/`godot4` on PATH, then (on
WSL) the Windows build installed by the Phase 0 tooling. Skipped otherwise.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ADCS.viz.binary_manager import _windows_local_appdata, is_wsl
from ADCS.viz.launcher import viewer_project
from ADCS.viz.schema import REQUIRED_CHANNELS, channel_spec
from ADCS.viz.writer import VizLogWriter

REPO = Path(__file__).resolve().parents[2]


def _godot():
    cand = os.environ.get("ADCS_GODOT_BIN") or shutil.which("godot") or shutil.which("godot4")
    if cand:
        return Path(cand)
    if is_wsl():
        local = _windows_local_appdata()
        if local is not None:
            p = local / "generalized_adcs" / "tools" / "godot-4.7.2" / "Godot_v4.7.2-stable_win64_console.exe"
            if p.exists():
                return p
    return None


GODOT = _godot()
pytestmark = pytest.mark.skipif(GODOT is None, reason="Godot not available")


def _host(p: Path) -> str:
    if GODOT.suffix.lower() == ".exe" and is_wsl():
        return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


def _fixture(path: Path) -> None:
    t = np.arange(121, dtype=float)
    n = np.sqrt(398600.4418 / 7000.0**3)
    r = 7000.0 * np.column_stack([np.cos(n * t), np.sin(n * t), 0 * t])
    v = 7000.0 * n * np.column_stack([-np.sin(n * t), np.cos(n * t), 0 * t])
    q = np.tile([1.0, 0, 0, 0], (len(t), 1))
    cols = {"t_s": t, "r_sc": r, "v_sc": v, "q_gcrs_body": q, "w_body": np.zeros((len(t), 3)),
            "r_sun": np.tile([1.5e8, 0, 0], (len(t), 1)), "q_itrs_gcrs": q, "shadow_factor": np.ones(len(t))}
    with VizLogWriter(path, [channel_spec(c) for c in REQUIRED_CHANNELS], epoch_jd_tt=2459580.5, overwrite=True) as w:
        w.write_run(0, cols)


def test_viewer_unit_tests(tmp_path):
    work = tmp_path
    if GODOT.suffix.lower() == ".exe" and is_wsl():
        work = _windows_local_appdata() / "generalized_adcs" / "viewer_unit_tests"
        work.mkdir(parents=True, exist_ok=True)
    fixture = work / "fixture.vizlog"
    _fixture(fixture)
    project = viewer_project(GODOT, REPO / "viewer")
    proc = subprocess.run([str(GODOT), "--headless", "--path", _host(project), "--script", "res://tests/run_tests.gd",
                           "--", "--fixture", _host(fixture)], capture_output=True, text=True, timeout=300,
                          check=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "ALL PASSED" in out, out[-3000:]
