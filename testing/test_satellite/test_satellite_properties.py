import numpy as np
import pytest

from ADCS.helpers.math_constants import MathConstants
from ADCS.satellite_hardware.satellite import Satellite

from testing.test_satellite._helpers import make_rws


def test_satellite_inertia_defaults_without_reaction_wheels():
    sat = Satellite(mass=1.0, COM=np.zeros(3), J_0=np.diagflat([0.1, 100.0, 5.0]))

    assert np.allclose(sat.J_0, np.diagflat([0.1, 100.0, 5.0]))
    assert np.allclose(sat.invJ_0, np.diagflat([10.0, 0.01, 0.2]))
    assert np.allclose(sat.J_noRW, sat.J_0)
    assert np.allclose(sat.invJ_noRW, sat.invJ_0)


def test_satellite_inertia_subtracts_reaction_wheel_contributions():
    sat = Satellite(J_0=np.diagflat([0.1, 100.0, 5.0]), actuators=make_rws())

    assert np.allclose(sat.J_noRW, np.diagflat([0.099, 99.998, 4.5]))
    assert np.allclose(sat.invJ_noRW, np.diagflat([1.0 / 0.099, 1.0 / 99.998, 2.0 / 9.0]))


def test_satellite_center_of_mass_parallel_axis_shift():
    one = np.array([1.0, 0.0, 0.0])
    JA = np.eye(3)
    JB = np.eye(3) + (np.eye(3) * 4.0 - 4.0 * np.outer(one, one))
    sat = Satellite(COM=one, mass=2.0, J_0=JA + JB)

    assert np.allclose(sat.J_0, JA + JB)
    assert np.allclose(sat.J_COM, 2.0 * np.diagflat([1.0, 2.0, 2.0]))


def test_update_rwhs_accepts_full_state_or_direct_momenta():
    sat = Satellite(actuators=make_rws())
    new_h = np.array([0.03, -0.04, 0.05])

    sat.update_RWhs(new_h)
    assert np.allclose(sat.RWhs(), new_h)

    x = np.concatenate([0.01 * MathConstants.unitvecs[0], MathConstants.zeroquat, new_h])
    sat.update_RWhs(x)
    assert np.allclose(sat.RWhs(), new_h)


def test_update_rwhs_rejects_wrong_length():
    sat = Satellite(actuators=make_rws())

    with pytest.raises(ValueError):
        sat.update_RWhs(np.array([0.1, 0.2]))


def test_get_boresight_supports_default_named_and_missing_lookup():
    sat = Satellite(
        boresight={
            "camera": np.array([0.0, 0.0, 1.0]),
            "panel": np.array([1.0, 0.0, 0.0]),
        }
    )

    assert np.allclose(sat.get_boresight(None), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(sat.get_boresight("panel"), np.array([1.0, 0.0, 0.0]))

    with pytest.raises(KeyError):
        sat.get_boresight("missing")
