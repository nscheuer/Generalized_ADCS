from typing import Any

import numpy as np
import pytest


def make_state(module: Any) -> Any:
    return module.State(
        module.AttitudeState([2, 0, 0, 0], [0.1, 0.2, 0.3]),
        module.OrbitalState.from_cartesian([7e6, 0, 0], [0, 7500, 0], 2451545.0),
        module.InternalState([1.25, -0.5]),
    )


def test_structured_records_and_roundtrips_agree(state_backends: Any) -> None:
    records = [make_state(module).to_record() for module in state_backends.values()]
    for record in records[1:]:
        assert record.dtype == records[0].dtype
        for field in record.dtype.names:
            np.testing.assert_allclose(
                record[field], records[0][field], rtol=0, atol=1e-14
            )
    rebuilt = [
        module.State.from_record(record, 2451545.0)
        for module, record in zip(state_backends.values(), records)
    ]
    for state in rebuilt[1:]:
        for field in records[0].dtype.names:
            np.testing.assert_allclose(
                state.to_record()[field], rebuilt[0].to_record()[field]
            )


def test_wheel_count_error_category_agrees(state_backends: Any) -> None:
    for module in state_backends.values():
        with pytest.raises(ValueError):
            make_state(module).to_record(wheel_count=3)
