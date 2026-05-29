import warnings

import numpy as np
import pytest

from ADCS.helpers import math_helpers as H


RNG = np.random.default_rng(1)


def noncontiguous_view(length: int) -> np.ndarray:
    values = RNG.standard_normal(4 * length)
    view = values[1 : 1 + 2 * length : 2]
    assert view.shape == (length,)
    assert not view.flags["C_CONTIGUOUS"]
    return view


def make_cases():
    vector = noncontiguous_view(3)
    quaternion = noncontiguous_view(4)
    quaternion /= np.linalg.norm(quaternion)
    cases = [
        ("normalize", H.normalize, vector),
        ("norm", H.norm, vector),
        ("skewsym", H.skewsym, vector),
        ("rot_mat", H.rot_mat, quaternion),
        ("quat_inv", H.quat_inv, quaternion),
        ("quat_mult", lambda values: H.quat_mult(values, quaternion), quaternion),
        ("mrp_to_quat", H.mrp_to_quat, vector),
        ("cayley_to_quat", H.cayley_to_quat, vector),
    ]
    for mode in (0, 1, 2, 6):
        cases.append((f"quat_to_vec3_mode_{mode}", lambda values, mm=mode: H.quat_to_vec3(values, mm), quaternion))
        cases.append((f"vec3_to_quat_mode_{mode}", lambda values, mm=mode: H.vec3_to_quat(values, mm), vector))
    return cases


@pytest.mark.parametrize("name, function, argument", make_cases(), ids=lambda value: value if isinstance(value, str) else "")
def test_jit_kernel_matches_contiguous_reference(name, function, argument):
    contiguous = np.ascontiguousarray(argument)
    assert not argument.flags["C_CONTIGUOUS"]
    assert contiguous.flags["C_CONTIGUOUS"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contiguous_result = np.asarray(function(contiguous), dtype=float)
        strided_result = np.asarray(function(argument), dtype=float)
    np.testing.assert_allclose(
        strided_result,
        contiguous_result,
        atol=1e-10,
        rtol=1e-9,
        err_msg=f"{name}: non-contiguous result differs from contiguous reference",
    )
