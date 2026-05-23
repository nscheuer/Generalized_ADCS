"""
Regression guard for the C++ planner-extension loader
(ADCS/controller/helpers/optional_dependencies.py).

trajectory_planner's `pysat` and SALTRO's `saltro_py` are independently built
pybind11 extensions that BOTH register a C++ type named `Satellite`. pybind11
keeps a single process-global type registry, so importing both into one
process makes the second raise:

    ImportError: generic_type: type "Satellite" is already registered!

(reproduced directly: import order pysat->saltro_py and saltro_py->pysat both
fail symmetrically -- whichever loads second).

Before this fix BOTH loaders funnelled *every* exception through
`*_missing_reason()`, so a registry clash was reported as
"Optional add-on ... is not available. Build ... into <path>. See
docs/Install_*.md" -- telling the user to (re)build an add-on that is in fact
already built. That misdiagnosis is the bug under test here. These tests do
NOT require the compiled .so files: they monkeypatch the loader's immediate
import step to raise the exact pybind11 message, so they run in CI / a
worktree without the C++ build. A real double-import integration check is
included but skipped unless both extensions are actually built.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from ADCS.controller.helpers import optional_dependencies as od

PYBIND_CLASH = ImportError('generic_type: type "Satellite" is already registered!')

# Phrases the *missing/unbuilt* message uses to send the user to (re)build.
# A registry clash must NOT be reported with any of these.
_REBUILD_GUIDANCE = ("is not available", "Build ", "docs/Install")


def _raise(exc):
    def _f(*_a, **_k):
        raise exc
    return _f


@pytest.mark.parametrize(
    ("loader", "patch_attr"),
    [
        (od.get_saltro_module, "import_module"),
        (od.get_trajectory_planner_modules, "_load_so_under_qualname"),
    ],
)
def test_genuine_missing_keeps_build_guidance(loader, patch_attr, monkeypatch):
    """A truly absent/unbuilt extension must still get the actionable
    'build it / read the install docs' message (behaviour preserved)."""
    monkeypatch.setattr(od, patch_attr, _raise(ModuleNotFoundError("No module named 'x'")))
    with pytest.raises(ImportError) as ei:
        loader()
    msg = str(ei.value)
    assert "docs/Install" in msg, msg
    assert "build" in msg.lower(), msg
    # It is genuinely missing, so it must NOT claim a same-process clash.
    assert "same Python process" not in msg, msg


@pytest.mark.parametrize(
    ("loader", "patch_attr"),
    [
        (od.get_saltro_module, "import_module"),
        (od.get_trajectory_planner_modules, "_load_so_under_qualname"),
    ],
)
def test_registry_clash_is_diagnosed_not_misreported(loader, patch_attr, monkeypatch):
    """The pybind11 'Satellite already registered' clash must be reported as
    a mutual-exclusivity / one-planner-per-process problem, NOT as a
    missing/unbuilt add-on the user should go (re)build."""
    monkeypatch.setattr(od, patch_attr, _raise(PYBIND_CLASH))
    with pytest.raises(ImportError) as ei:
        loader()
    msg = str(ei.value)
    # Accurate diagnosis present.
    assert "same Python process" in msg, msg
    assert "registry" in msg or "registers" in msg, msg
    # And the misleading rebuild guidance is gone (RED on origin/main: the
    # clash was wrapped in `*_missing_reason()` which contains all of these).
    for bad in _REBUILD_GUIDANCE:
        assert bad not in msg, f"clash message must not contain {bad!r}: {msg}"


def test_clash_detected_through_cause_chain(monkeypatch):
    """pybind11 raises from inside importlib, so the marker can sit on a
    chained cause/context, not the top exception. The classifier must walk
    the chain."""
    try:
        try:
            raise PYBIND_CLASH
        except ImportError as inner:
            raise ImportError("trajectory_planner.build.pysat failed") from inner
    except ImportError as chained:
        assert od._is_pybind_registry_clash(chained) is True
    assert od._is_pybind_registry_clash(ModuleNotFoundError("nope")) is False


def _both_extensions_built() -> bool:
    def _has_so(d):
        return os.path.isdir(d) and any(f.endswith(".so") for f in os.listdir(d))
    return _has_so(od.trajectory_planner_build_path()) and _has_so(od.saltro_build_path())


@pytest.mark.skipif(not _both_extensions_built(),
                    reason="needs both pysat/tplaunch and saltro_py compiled")
def test_real_double_import_clash_message():
    """End-to-end: in a fresh process, load trajectory_planner then saltro_py.
    The real pybind11 clash must surface as the accurate clash message, not
    'Build SALTRO into ...'. External reference: the actual extension import,
    not the loader compared to itself."""
    script = textwrap.dedent(
        """
        from ADCS.controller.helpers.optional_dependencies import (
            get_trajectory_planner_modules, get_saltro_module)
        get_trajectory_planner_modules()
        try:
            get_saltro_module()
            print("NO_CLASH")
        except ImportError as e:
            print("MSG:" + str(e).replace(chr(10), " "))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = od._repo_root()
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=env, cwd=od._repo_root()).stdout
    assert "MSG:" in out, out
    line = [l for l in out.splitlines() if l.startswith("MSG:")][-1]
    assert "same Python process" in line, line
    assert "Build SALTRO into" not in line, line
