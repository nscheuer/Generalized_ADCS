"""The _skip_jacobians placeholders must have the shapes of the Jacobians they replace."""

import numpy as np

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.sensors import MTM
from ADCS.state import State


def _orbital_state() -> Orbital_State:
    return Orbital_State(ephem=Ephemeris(), J2000=0.22, R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]))


def test_skip_jacobians_placeholders_have_the_real_shapes():
    quaternion = np.array([0.9, 0.2, -0.3, 0.1])
    state = State(w=np.zeros(3), q=quaternion / np.linalg.norm(quaternion))
    full = _orbital_state().get_state_vector(x=state)
    skipping = _orbital_state()
    skipping._skip_jacobians = True
    placeholders = skipping.get_state_vector(x=state)
    for key in ("db", "ds", "dv", "dr", "ddb", "dds", "ddv", "ddr"):
        assert placeholders[key].shape == full[key].shape, key  # used to be transposed (3, 4) vs (4, 3)
        assert not np.any(placeholders[key])
    # a sensor Jacobian can be assembled from them (used to raise a shape error)
    jacobian = MTM(axis=np.array([1.0, 0.0, 0.0])).basestate_jac(state, skipping)
    assert jacobian.shape == (7, 1) and not np.any(jacobian)
