import json

import numpy as np

from ADCS.helpers.animations.godot.writer import _jsonable


def test_jsonable_maps_numpy_non_finite_scalars_to_null():
    payload = _jsonable({"dt_s": np.float64("nan"), "span": [np.float32("inf"), np.float64(2.5)]})

    assert payload == {"dt_s": None, "span": [None, 2.5]}
    json.dumps(payload, allow_nan=False)
