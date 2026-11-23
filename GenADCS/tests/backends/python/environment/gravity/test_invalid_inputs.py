import numpy as np
import pytest

from GenADCS.backends.python.environment.gravity import JGravityModel


@pytest.mark.parametrize("degrees", ((1,), (7,), (2, 2.5)))
def test_j_gravity_rejects_invalid_degrees(degrees):
    with pytest.raises(ValueError, match="degrees must be selected"):
        JGravityModel(degrees=degrees)


def test_j_gravity_rejects_short_coefficient_vector():
    with pytest.raises(ValueError, match="five coefficients"):
        JGravityModel(coefficients=np.ones(4))


def test_j_gravity_rejects_centre_position():
    with pytest.raises(ValueError, match="body's centre"):
        JGravityModel().evaluate((0.0, 0.0, 0.0))
