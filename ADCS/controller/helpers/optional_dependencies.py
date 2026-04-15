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


def get_trajectory_planner_modules() -> Tuple[ModuleType, ModuleType]:
    _ensure_path(trajectory_planner_build_path())
    try:
        tplaunch = import_module("trajectory_planner.build.tplaunch")
        pysat = import_module("trajectory_planner.build.pysat")
    except Exception as exc:
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
        raise ImportError(f"{saltro_missing_reason()} Original error: {exc}") from exc


def saltro_available() -> bool:
    try:
        get_saltro_module()
    except ImportError:
        return False
    return True
