"""Tests for HoganSchaubManagement desaturation."""

import numpy as np
import pytest

from GenADCS.backends.python.hardware.ADCS_subsystem.controllers.desaturation import (
    HoganSchaubManagement,
    NoDesaturation,
    hogan_schaub_u0,
)


def test_hogan_schaub_validation_and_u0() -> None:
    with pytest.raises(ValueError):
        HoganSchaubManagement(gain=0.0)
    with pytest.raises(ValueError):
        HoganSchaubManagement(gain=0.5, target_momentum=float("nan"))

    hs = HoganSchaubManagement(gain=0.5, target_momentum=0.1, mtq_gain=0.2)
    assert np.allclose(hs.to_params(), [0.5, 0.1, 0.2])
    assert np.allclose(NoDesaturation().to_params(), [0.0, 0.0, 0.0])

    u0 = hogan_schaub_u0(np.array([0.3, -0.1, 0.1]), 0.5, 0.1)
    assert np.allclose(u0, [0.1, -0.1, 0.0])
