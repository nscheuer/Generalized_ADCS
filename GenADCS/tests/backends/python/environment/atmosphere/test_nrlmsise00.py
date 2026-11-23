import numpy as np
import pytest

from GenADCS.backends.python.environment.atmosphere.NRMLSISE00 import NRLMSISE00
from GenADCS.backends.python.helpers.math.time import calendar_to_jd


def test_nrlmsise00_density_is_realistic_at_low_earth_orbit():
    model = NRLMSISE00()
    position_m = np.array((6_778_137.0, 0.0, 0.0))
    density = model.density(position_m, calendar_to_jd(2020, 1, 1))

    assert np.isfinite(density)
    assert 1e-13 < density < 1e-9


def test_nrlmsise00_density_changes_with_altitude():
    model = NRLMSISE00()
    epoch = calendar_to_jd(2020, 1, 1)
    low = model.density(np.array((6_778_137.0, 0.0, 0.0)), epoch)
    high = model.density(np.array((7_000_000.0, 0.0, 0.0)), epoch)

    assert low > high > 0.0


def test_nrlmsise00_rejects_ap_history_with_wrong_length():
    with pytest.raises(ValueError, match="seven Ap values"):
        NRLMSISE00.density(
            np.array((6_778_137.0, 0.0, 0.0)),
            calendar_to_jd(2020, 1, 1),
            ap_array=np.ones(6),
        )
