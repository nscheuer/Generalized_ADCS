import numpy as np
import pytest

from ADCS.CONOPS.goals import Coordinate_Goal, ECI_Goal
from ADCS.helpers.math_helpers import normalize

from testing.test_goals._helpers import make_multi_boresight_satellite, make_test_orbital_state


def test_multi_boresight_satellite_returns_named_axes():
    sat = make_multi_boresight_satellite()

    assert np.allclose(sat.get_boresight("camera"), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(sat.get_boresight("solar_panel"), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(sat.get_boresight("antenna"), np.array([0.0, 1.0, 0.0]))


def test_get_boresight_defaults_to_first_defined_entry():
    sat = make_multi_boresight_satellite()

    assert np.allclose(sat.get_boresight(None), sat.get_boresight("camera"))


def test_get_boresight_raises_for_unknown_name():
    sat = make_multi_boresight_satellite()

    with pytest.raises(KeyError):
        sat.get_boresight("missing")


def test_switching_boresights_changes_goal_error():
    sat = make_multi_boresight_satellite()
    goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    os0 = make_test_orbital_state()
    q = np.array([1.0, 0.0, 0.0, 0.0])

    camera_error = goal.error(q=q, body_boresight=sat.get_boresight("camera"), os0=os0)
    panel_error = goal.error(q=q, body_boresight=sat.get_boresight("solar_panel"), os0=os0)

    assert not np.allclose(camera_error, panel_error)


def test_eci_goal_can_align_a_selected_boresight_exactly():
    sat = make_multi_boresight_satellite()
    goal = ECI_Goal(np.array([1.0, 0.0, 0.0]), boresight_name="camera")
    q_body_to_eci = normalize(np.array([1.0, 0.0, 1.0, 0.0]))

    error = goal.error(
        q=q_body_to_eci,
        body_boresight=sat.get_boresight("camera"),
        os0=make_test_orbital_state(),
    )

    assert np.allclose(error, np.zeros(3), atol=1e-12)


def test_coordinate_goal_keeps_selected_boresight_name():
    goal = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0, boresight_name="antenna")

    assert goal.boresight_name == "antenna"
