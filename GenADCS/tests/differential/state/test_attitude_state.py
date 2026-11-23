import numpy as np
import pytest


@pytest.mark.parametrize("orientation,rates", [
    ([2, 0, 0, 0], [1, 2, 3]),
    ([0, -4, 0, 0], [0.1, -0.2, 0.3]),
    ([0.5, 0.5, 0.5, 0.5], [0, 0, 0]),
])
def test_quaternion_and_rates_agree(state_backends, orientation, rates):
    results = [module.AttitudeState(orientation, rates) for module in state_backends.values()]
    for result in results[1:]:
        np.testing.assert_allclose(result.quaternion_array, results[0].quaternion_array, atol=1e-14)
        np.testing.assert_allclose(result.rates, results[0].rates, atol=0)


def test_invalid_rates_raise_same_error_category(state_backends):
    for module in state_backends.values():
        with pytest.raises(ValueError):
            module.AttitudeState([1, 0, 0, 0], [np.nan, 0, 0])
