import lzma
import pickle

import numpy as np

from ADCS import EstimatorState, State
from ADCS.helpers.simresults import RunResults, SimulationResults
from ADCS.satellite_hardware.satellite import EstimatedSatellite, Satellite


def test_simulation_results_roundtrip_structured_states(tmp_path):
    run = RunResults(
        satellite=Satellite(),
        est_satellite=EstimatedSatellite(),
        state_hist=[State(w=[1, 2, 3], q=[1, 0, 0, 0])],
        est_state_hist=[EstimatorState(w=[4, 5, 6], q=[1, 0, 0, 0])],
    )

    path = SimulationResults(runs=[run]).save("state", out_dir=tmp_path)
    loaded = SimulationResults.load(path)

    assert isinstance(loaded.first().state_hist[0], State)
    assert isinstance(loaded.first().est_state_hist[0], EstimatorState)
    np.testing.assert_array_equal(loaded.first().state_hist[0].w, [1, 2, 3])
    np.testing.assert_array_equal(loaded.first().est_state_hist[0].w, [4, 5, 6])


def test_simulation_results_loads_legacy_state_matrices(tmp_path):
    satellite = Satellite()
    estimated_satellite = EstimatedSatellite()
    payload = {
        "runs": [
            {
                "satellite": satellite,
                "est_satellite": estimated_satellite,
                "state_hist": np.array([[1, 2, 3, 1, 0, 0, 0]], dtype=float),
                "est_state_hist": np.array([[4, 5, 6, 1, 0, 0, 0]], dtype=float),
            }
        ]
    }
    path = tmp_path / "legacy.sim"
    with lzma.open(path, "wb") as stream:
        pickle.dump(payload, stream)

    loaded = SimulationResults.load(path)

    assert isinstance(loaded.first().state_hist[0], State)
    assert isinstance(loaded.first().est_state_hist[0], EstimatorState)
    np.testing.assert_array_equal(loaded.first().state_hist[0].as_array(), payload["runs"][0]["state_hist"][0])
