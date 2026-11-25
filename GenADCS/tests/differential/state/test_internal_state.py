from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


@pytest.mark.parametrize("momentum", [[], [0], [1.25, -2.5, 0.125]])
def test_momentum_values_agree(state_backends: Any, momentum: Any) -> None:
    results = [module.InternalState(momentum) for module in state_backends.values()]
    for result in results[1:]:
        np.testing.assert_array_equal(result.wheel_momentum, results[0].wheel_momentum)


def test_wheel_source_values_agree(state_backends: Any) -> None:
    source = SimpleNamespace(
        reaction_wheels=[SimpleNamespace(h=1.5), SimpleNamespace(h=-0.25)]
    )
    results = [module.WheelMomentumState(source) for module in state_backends.values()]
    for result in results[1:]:
        np.testing.assert_array_equal(result.wheel_momentum, results[0].wheel_momentum)


def test_nonfinite_momentum_rejected_everywhere(state_backends: Any) -> None:
    for module in state_backends.values():
        with pytest.raises(ValueError):
            module.InternalState([np.inf])
