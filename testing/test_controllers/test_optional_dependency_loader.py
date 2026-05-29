import os
import subprocess
import sys
import textwrap

import pytest

from ADCS.controller.helpers import optional_dependencies as od


PYBIND_CLASH = ImportError('generic_type: type "Satellite" is already registered!')


def raise_exception(exc):
    def inner(*_args, **_kwargs):
        raise exc

    return inner


@pytest.mark.parametrize(
    ("loader", "patch_attr"),
    [
        (od.get_saltro_module, "import_module"),
        (od.get_trajectory_planner_modules, "_load_so_under_qualname"),
    ],
)
def test_missing_optional_dependency_keeps_build_guidance(loader, patch_attr, monkeypatch):
    monkeypatch.setattr(od, patch_attr, raise_exception(ModuleNotFoundError("No module named 'x'")))
    with pytest.raises(ImportError) as exc_info:
        loader()
    message = str(exc_info.value)
    assert "docs/Install" in message
    assert "build" in message.lower()
    assert "same Python process" not in message


@pytest.mark.parametrize(
    ("loader", "patch_attr"),
    [
        (od.get_saltro_module, "import_module"),
        (od.get_trajectory_planner_modules, "_load_so_under_qualname"),
    ],
)
def test_registry_clash_reports_process_conflict(loader, patch_attr, monkeypatch):
    monkeypatch.setattr(od, patch_attr, raise_exception(PYBIND_CLASH))
    with pytest.raises(ImportError) as exc_info:
        loader()
    message = str(exc_info.value)
    assert "same Python process" in message
    assert "registry" in message or "registers" in message


@pytest.mark.parametrize("bad_phrase", ["is not available", "Build ", "docs/Install"])
@pytest.mark.parametrize(
    ("loader", "patch_attr"),
    [
        (od.get_saltro_module, "import_module"),
        (od.get_trajectory_planner_modules, "_load_so_under_qualname"),
    ],
)
def test_registry_clash_omits_rebuild_guidance(loader, patch_attr, monkeypatch, bad_phrase):
    monkeypatch.setattr(od, patch_attr, raise_exception(PYBIND_CLASH))
    with pytest.raises(ImportError) as exc_info:
        loader()
    assert bad_phrase not in str(exc_info.value)


def test_registry_clash_is_detected_through_exception_chain():
    try:
        try:
            raise PYBIND_CLASH
        except ImportError as inner:
            raise ImportError("trajectory_planner.build.pysat failed") from inner
    except ImportError as chained:
        assert od._is_pybind_registry_clash(chained) is True


def test_non_clash_exception_is_not_detected_as_registry_conflict():
    assert od._is_pybind_registry_clash(ModuleNotFoundError("nope")) is False


def both_extensions_built() -> bool:
    def has_shared_object(path: str) -> bool:
        return os.path.isdir(path) and any(name.endswith(".so") for name in os.listdir(path))

    return has_shared_object(od.trajectory_planner_build_path()) and has_shared_object(od.saltro_build_path())


@pytest.mark.skipif(not both_extensions_built(), reason="needs both pysat/tplaunch and saltro_py compiled")
def test_real_double_import_clash_message():
    script = textwrap.dedent(
        """
        from ADCS.controller.helpers.optional_dependencies import (
            get_trajectory_planner_modules, get_saltro_module)
        get_trajectory_planner_modules()
        try:
            get_saltro_module()
            print("NO_CLASH")
        except ImportError as exc:
            print("MSG:" + str(exc).replace(chr(10), " "))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = od._repo_root()
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=od._repo_root(),
    )
    lines = completed.stdout.splitlines()
    clash_lines = [line for line in lines if line.startswith("MSG:")]
    assert clash_lines, completed.stdout
    assert "same Python process" in clash_lines[-1]
    assert "Build SALTRO into" not in clash_lines[-1]
