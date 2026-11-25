from importlib import import_module
from typing import Any

import pytest


@pytest.fixture(scope="module")
def rotation_backends() -> Any:
    result = {}
    for name in ("python", "cpp", "rust"):
        try:
            root = f"GenADCS.backends.{name}.helpers.math.rotations"
            module = import_module(root)
            module.quaternion = import_module(root + ".quaternion")
            module.dcm = import_module(root + ".dcm")
            module.DCM = module.dcm.DCM
            module.mrp = import_module(root + ".mrp")
            module.mrp_to_quaternion = module.mrp.mrp_to_quaternion
            module.quaternion_to_mrp = module.mrp.quaternion_to_mrp
            module.mrp_shadow_set = module.mrp.mrp_shadow_set
            module.rodrigues = import_module(root + ".rodrigues")
            module.rodrigues_to_quaternion = module.rodrigues.rodrigues_to_quaternion
            module.quaternion_to_rodrigues = module.rodrigues.quaternion_to_rodrigues
            module.slerp = module.quaternion.slerp
            module.quaternion_to_rotation_vector = (
                module.quaternion.quaternion_to_rotation_vector
            )
            module.rotation_vector_to_quaternion = (
                module.quaternion.rotation_vector_to_quaternion
            )
            result[name] = module
        except ImportError as exc:
            pytest.fail(f"build {name} backend before differential tests: {exc}")
    return result
