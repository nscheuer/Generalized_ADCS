import numpy as np
import pytest

from GenADCS.backends.python.environment.magnetic_field_model.igrf14 import IGRF14
from GenADCS.backends.python.helpers.math.time import calendar_to_jd


def test_igrf14_loads_bundled_coefficients_and_returns_realistic_field() -> None:
    model = IGRF14()
    field = model.field_inertial_tesla(
        np.array((6_778_137.0, 0.0, 0.0)), calendar_to_jd(2020, 1, 1)
    )

    assert field.shape == (3,)
    assert np.all(np.isfinite(field))
    assert 1e-5 < np.linalg.norm(field) < 7e-5


@pytest.mark.parametrize("year", (1899, 2031))
def test_igrf14_rejects_epochs_outside_coefficient_range(year: int) -> None:
    with pytest.raises(ValueError, match="cover 1900-01-01 to 2030-01-01"):
        IGRF14().field((6_778_137.0, 0.0, 0.0), calendar_to_jd(year, 1, 1))


def test_igrf14_rejects_zero_position() -> None:
    with pytest.raises(ValueError, match="positive radius"):
        IGRF14().field((0.0, 0.0, 0.0), calendar_to_jd(2020, 1, 1))
