import numpy as np
import pytest

from GenADCS.backends.python.environment.magnetic_field_model.chaos8 import CHAOS8
from GenADCS.backends.python.helpers.math.time import calendar_to_jd


def test_chaos8_core_only_field_is_realistic() -> None:
    model = CHAOS8(include_static=False)
    field = model.field((6_778_137.0, 0.0, 0.0), calendar_to_jd(2020, 1, 1))

    assert field.shape == (3,)
    assert np.all(np.isfinite(field))
    assert 1e-5 < np.linalg.norm(field) < 7e-5


def test_chaos8_static_field_loads_and_changes_the_result() -> None:
    position = (6_778_137.0, 0.0, 0.0)
    epoch = calendar_to_jd(2020, 1, 1)
    core = CHAOS8(include_static=False).field(position, epoch)
    full = CHAOS8(include_static=True).field(position, epoch)

    assert np.all(np.isfinite(full))
    assert np.linalg.norm(full - core) > 0.0


def test_chaos8_rejects_epoch_outside_core_coefficients() -> None:
    with pytest.raises(ValueError, match="outside CHAOS-8.7"):
        CHAOS8(include_static=False).field(
            (6_778_137.0, 0.0, 0.0), calendar_to_jd(1800, 1, 1)
        )
