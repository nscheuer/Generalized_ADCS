from __future__ import annotations

import importlib.util
import os
import sys
from importlib import import_module
from types import ModuleType
from typing import Tuple


# When the trajectory_planner add-on lived in-tree, the two compiled
# pybind modules were discoverable as ``trajectory_planner.build.pysat``
# and ``trajectory_planner.build.tplaunch``. The C++ side of
# ``tplaunch`` *hard-codes* that qualified name:
#
#     // PyPlanner.cpp
#     PYBIND11_MODULE(tplaunch, m) {
#         py::module_::import("trajectory_planner.build.pysat");
#         ...
#
# That import bootstraps the ``Satellite`` type registration in
# pybind11's process-wide type table *before* ``tplaunch``'s own
# bindings reference the type. If ``tplaunch`` is loaded under a
# different qualified name, the hard-coded import fails silently and
# the next ``import pysat`` triggers a ``generic_type: type "Satellite"
# is already registered`` clash from pybind11.
#
# We now ship the actual ``.so`` files in the OldPlanner submodule
# (``OldPlanner/build/``), but to keep PyPlanner.cpp's ABI contract
# intact we *load* them under the historical qualified names. That's
# what the constants and ``_load_so_under_qualname`` calls below
# encode.
_LEGACY_TP_PKG = "trajectory_planner.build"
_TP_MODULE_NAME = f"{_LEGACY_TP_PKG}.tplaunch"
_PYSAT_MODULE_NAME = f"{_LEGACY_TP_PKG}.pysat"


def _repo_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../.."))


def _ensure_path(path: str) -> None:
    if path not in sys.path:
        sys.path.append(path)


def trajectory_planner_build_path() -> str:
    """Path to the compiled ``tplaunch`` / ``pysat`` pybind modules.

    The trajectory-planner add-on lives in the ``OldPlanner`` submodule
    (https://github.com/patrickmckeen/OldPlanner), built out-of-tree
    into ``OldPlanner/build/``.
    """
    return os.path.join(_repo_root(), "OldPlanner", "build")


def saltro_build_path() -> str:
    return os.path.join(_repo_root(), "SALTRO", "build")


def trajectory_planner_missing_reason() -> str:
    build_path = trajectory_planner_build_path()
    return (
        "Optional add-on trajectory_planner (OldPlanner) is not available. "
        f"Initialise the submodule and build the C++ bindings into "
        f"{build_path}. See docs/Install_Trajectory_Planner.md."
    )


def saltro_missing_reason() -> str:
    build_path = saltro_build_path()
    return (
        "Optional add-on saltro_py is not available. "
        f"Build SALTRO into {build_path}. "
        "See docs/Install_SALTRO.md."
    )


# trajectory_planner's `pysat` and SALTRO's `saltro_py` are independently
# built pybind11 extensions that BOTH register a C++ type named `Satellite`.
# pybind11 keeps a single process-global type registry, so the second of the
# two to import raises `generic_type: type "Satellite" is already registered!`.
# This is NOT a missing/unbuilt extension, so we must not report it via
# *_missing_reason() and send users down a rebuild dead end.
_PYBIND_CLASH_MARKER = "already registered"


def _is_pybind_registry_clash(exc: BaseException) -> bool:
    """True iff ``exc`` or its cause/context chain is the pybind11 clash."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _PYBIND_CLASH_MARKER in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def planner_extension_clash_reason() -> str:
    return (
        "trajectory_planner (pysat/tplaunch) and saltro_py each register a C++ "
        "`Satellite` type via pybind11, which keeps a single process-global "
        "type registry. They CANNOT be imported into the same Python process: "
        "whichever loads second raises "
        '`generic_type: type "Satellite" is already registered!`. This is NOT '
        "a missing or unbuilt extension -- rebuilding will not help. Use only "
        "one planner family per process (for example, run SALTRO and "
        "plan_and_track controllers in separate processes)."
    )


def _so_filename(stem: str) -> str:
    """Return the platform-specific filename for a compiled pybind module.

    pybind11 emits e.g. ``tplaunch.cpython-312-darwin.so`` /
    ``tplaunch.cpython-312-x86_64-linux-gnu.so`` -- match by stem.
    """
    build_dir = trajectory_planner_build_path()
    if os.path.isdir(build_dir):
        for fname in os.listdir(build_dir):
            if fname.startswith(stem + ".") and (fname.endswith(".so") or fname.endswith(".pyd")):
                return fname
    return stem + ".so"


def _ensure_parent_namespace_modules(qualname: str) -> None:
    """Insert dummy parent ``ModuleType`` entries into ``sys.modules``.

    For ``qualname == "trajectory_planner.build.pysat"`` this seeds
    ``sys.modules["trajectory_planner"]`` and
    ``sys.modules["trajectory_planner.build"]`` with stub module
    objects, so that pybind11's internal lookup of the parent module
    name succeeds when loading the compiled extension.
    """
    parts = qualname.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            stub = ModuleType(parent)
            stub.__path__ = []  # namespace-package-style
            sys.modules[parent] = stub


def _load_so_under_qualname(qualname: str, so_filename: str) -> ModuleType:
    """Load a compiled pybind11 module under a chosen qualified name."""
    if qualname in sys.modules:
        return sys.modules[qualname]
    so_path = os.path.join(trajectory_planner_build_path(), so_filename)
    if not os.path.exists(so_path):
        raise ImportError(f"Compiled module not found: {so_path}")
    _ensure_parent_namespace_modules(qualname)
    spec = importlib.util.spec_from_file_location(qualname, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[qualname]
        raise
    return module


def get_trajectory_planner_modules() -> Tuple[ModuleType, ModuleType]:
    """Return ``(tplaunch, pysat)`` from the OldPlanner submodule build dir.

    Load order matters: ``pysat`` must be loaded first so its
    ``Satellite`` type binding is registered in pybind11's global
    type table before ``tplaunch``'s PYBIND11_MODULE runs (which
    internally does ``py::module_::import("trajectory_planner.build.pysat")``
    and would otherwise either re-register or fail).
    """
    try:
        pysat = _load_so_under_qualname(_PYSAT_MODULE_NAME, _so_filename("pysat"))
        tplaunch = _load_so_under_qualname(_TP_MODULE_NAME, _so_filename("tplaunch"))
    except Exception as exc:
        if _is_pybind_registry_clash(exc):
            raise ImportError(planner_extension_clash_reason()) from exc
        raise ImportError(f"{trajectory_planner_missing_reason()} Original error: {exc}") from exc

    return tplaunch, pysat


def trajectory_planner_available() -> bool:
    try:
        get_trajectory_planner_modules()
    except ImportError:
        return False
    return True


def get_saltro_module() -> ModuleType:
    _ensure_path(saltro_build_path())
    try:
        return import_module("saltro_py")
    except Exception as exc:
        if _is_pybind_registry_clash(exc):
            raise ImportError(planner_extension_clash_reason()) from exc
        raise ImportError(f"{saltro_missing_reason()} Original error: {exc}") from exc


def saltro_available() -> bool:
    try:
        get_saltro_module()
    except ImportError:
        return False
    return True
