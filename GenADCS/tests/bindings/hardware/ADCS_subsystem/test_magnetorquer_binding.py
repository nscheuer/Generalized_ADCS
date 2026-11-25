"""Python binding behavior for the native magnetorquer type."""

from typing import Any

from GenADCS.backends.python.bias import Bias
from GenADCS.backends.python.noise import Noise


def test_magnetorquer_effects_are_handled_at_the_binding_boundary(subsystem_backend: Any) -> None:
    class AddEffect:
        def __init__(self, amount: float, calls: list[tuple[Any, float, Any]]) -> None:
            self.amount = amount
            self.calls = calls

        def apply(self, value: float, dt: float, rng: Any = None) -> float:
            self.calls.append((value, dt, rng))
            return value + self.amount

    noise, bias = Noise(), Bias()
    torquer = subsystem_backend.Magnetorquer(
        (1, 2, 3),
        (0, 0, 1),
        0.25,
        effects=[noise, bias],
    )

    assert torquer.position == (1.0, 2.0, 3.0)
    assert torquer.orientation == (0.0, 0.0, 1.0)
    assert torquer.noises == [noise]
    assert torquer.biases == [bias]

    calls = []
    rng = object()
    torquer.effects = [AddEffect(2.0, calls), AddEffect(-0.5, calls)]
    assert torquer.apply_errors(1.0, 0.125, rng) == 2.5
    assert calls == [(1.0, 0.125, rng), (3.0, 0.125, rng)]
