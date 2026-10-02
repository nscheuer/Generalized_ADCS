import numpy as np

from ADCS.CONOPS.goals import Coordinate_Goal
from ADCS.helpers.math_helpers import normalize

from testing.test_goals._helpers import make_test_orbital_state


def test_coordinate_goal_geodetic_to_ecef_matches_equator_reference():
    goal = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0)

    assert np.allclose(goal.target_ecef, np.array([6378.137, 0.0, 0.0]), atol=1e-6)


def test_coordinate_goal_to_ref_matches_relative_geometry():
    os0 = make_test_orbital_state()
    goal = Coordinate_Goal(lat=0.0, lon=0.0, alt=0.0, boresight_name="camera")

    r_ref, w_ref = goal.to_ref(os0)

    target_eci = os0.ecef_to_eci(goal.target_ecef)
    v_target_eci = np.cross(np.array([0.0, 0.0, 7.2921159e-5]), target_eci)
    r_rel = target_eci - os0.R
    v_rel = v_target_eci - os0.V

    assert np.isnan(r_ref[0])
    assert np.allclose(r_ref[1:], normalize(r_rel))
    assert np.allclose(w_ref, np.cross(r_rel, v_rel) / np.dot(r_rel, r_rel))


def test_coordinate_goal_preserves_requested_boresight_name():
    goal = Coordinate_Goal(lat=10.0, lon=-30.0, alt=0.2, boresight_name="antenna")

    assert goal.boresight_name == "antenna"
