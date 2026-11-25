"""Philox stream mapping used by compiled sensor sampling."""

import numpy as np
import pytest

from GenADCS.backends.python.helpers.random.philox import (
    _philox4x32,  # pyright: ignore[reportPrivateUsage] -- verifies a published internal test vector.
    standard_normals,
    validate_seed,
)


def test_philox4x32_10_reference_vector() -> None:
    # Random123 Philox4x32-10, zero counter and zero key.
    assert tuple(int(word) for word in _philox4x32(np.uint64(0), 0, 0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_normals_are_repeatable_and_counter_separated() -> None:
    seed = validate_seed(123456789)
    a = standard_normals(seed, 7, 2, 6)
    np.testing.assert_array_equal(a, standard_normals(seed, 7, 2, 6))
    assert not np.array_equal(a, standard_normals(seed, 8, 2, 6))
    assert not np.array_equal(a, standard_normals(seed, 7, 3, 6))
    assert not np.array_equal(a, standard_normals(validate_seed(123456790), 7, 2, 6))


@pytest.mark.parametrize("bad_seed", [-1, 2**64, 1.5, True])
def test_invalid_seed_is_rejected(bad_seed: float | bool) -> None:
    with pytest.raises(ValueError):
        validate_seed(bad_seed)  # pyright: ignore[reportArgumentType] -- exercises runtime rejection of non-integer values.
