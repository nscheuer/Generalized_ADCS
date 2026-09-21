"""Controllers must accept a plain ``Satellite``.

Controllers only read ``Satellite``-level members, so a plain ``Satellite``
models the perfect-state-knowledge case (a control cycle with no estimator in
the loop). Annotating ``est_sat`` as ``EstimatedSatellite`` is a stricter
contract than the code needs, and because the package ships ``py.typed`` that
strictness becomes a type error in *user* code that is actually correct.

See issue #122.
"""

import inspect
import typing

import pytest

import ADCS.controller as controller_module
from ADCS.controller.controller import Controller
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite


def _controller_classes():
    seen = {}
    for name in dir(controller_module):
        obj = getattr(controller_module, name)
        if inspect.isclass(obj) and issubclass(obj, Controller):
            seen[obj.__qualname__] = obj
    return sorted(seen.values(), key=lambda c: c.__qualname__)


def test_controller_classes_are_discoverable():
    assert _controller_classes(), "no Controller subclasses exported from ADCS.controller"


@pytest.mark.parametrize("method_name", ["__init__", "find_u"])
def test_est_sat_is_annotated_satellite_not_estimated_satellite(method_name):
    offenders = []
    for cls in _controller_classes():
        method = getattr(cls, method_name, None)
        if method is None:
            continue
        try:
            hints = typing.get_type_hints(method)
        except Exception:
            continue
        annotation = hints.get("est_sat")
        if annotation is EstimatedSatellite:
            offenders.append(f"{cls.__qualname__}.{method_name}")

    assert not offenders, (
        "these accept a plain Satellite at runtime but declare EstimatedSatellite, "
        "which makes correct user code fail type checking (issue #122): "
        + ", ".join(offenders)
    )


def test_controllers_do_not_use_estimated_satellite_only_members():
    """The contract above is only safe while this holds."""
    estimator_only = set(dir(EstimatedSatellite)) - set(dir(Satellite))
    assert estimator_only, "expected EstimatedSatellite to add members over Satellite"

    import pathlib

    offenders = []
    root = pathlib.Path(controller_module.__file__).parent
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for member in estimator_only:
            if f"est_sat.{member}" in text:
                offenders.append(f"{path.relative_to(root)}: est_sat.{member}")

    assert not offenders, (
        "controllers now use EstimatedSatellite-only members, so the Satellite "
        "annotation is no longer sound: " + ", ".join(offenders)
    )
