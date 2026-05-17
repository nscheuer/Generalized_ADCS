from __future__ import annotations

from importlib import import_module
import os
import sys
from types import ModuleType
from typing import Tuple


def _repo_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../.."))


def _ensure_path(path: str) -> None:
    if path not in sys.path:
        sys.path.append(path)


def trajectory_planner_build_path() -> str:
    return os.path.join(_repo_root(), "trajectory_planner", "build")


def saltro_build_path() -> str:
    return os.path.join(_repo_root(), "SALTRO", "build")


def trajectory_planner_missing_reason() -> str:
    build_path = trajectory_planner_build_path()
    return (
        "Optional add-on trajectory_planner is not available. "
        f"Build the C++ bindings into {build_path}. "
        "See docs/Install_WSL.md or docs/Install_Windows.md."
    )


def saltro_missing_reason() -> str:
    build_path = saltro_build_path()
    return (
        "Optional add-on saltro_py is not available. "
        f"Build SALTRO into {build_path}. "
        "See docs/Install_WSL.md or docs/Install_Windows.md."
    )


# trajectory_planner's `pysat` and `SALTRO`'s `saltro_py` are independently
# built pybind11 extensions that BOTH register a C++ type named `Satellite`.
# pybind11 keeps a single process-global type registry, so the second of the
# two to import raises `generic_type: type "Satellite" is already registered!`.
# This is NOT a missing/unbuilt extension -- rebuilding does not help -- so it
# must NOT be reported via *_missing_reason() (which tells the user to build
# the add-on and read the install docs, sending them down a dead end).
_PYBIND_CLASH_MARKER = "already registered"


def _is_pybind_registry_clash(exc: BaseException) -> bool:
    """True iff ``exc`` (or any exception in its cause/context chain) is the
    pybind11 process-global type-registry clash between ``pysat`` and
    ``saltro_py`` (both register a C++ ``Satellite``)."""
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
        "one planner family per process (e.g. run the SALTRO and "
        "plan_and_track controllers in separate processes / Monte-Carlo runs)."
    )


def get_trajectory_planner_modules() -> Tuple[ModuleType, ModuleType]:
    _ensure_path(trajectory_planner_build_path())
    try:
        tplaunch = import_module("trajectory_planner.build.tplaunch")
        pysat = import_module("trajectory_planner.build.pysat")
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
