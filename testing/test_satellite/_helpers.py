import numpy as np

from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.errors import Bias, Noise


def make_orbital_state(B=None):
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([1.0e-5, 0.0, 0.0]) if B is None else np.asarray(B, dtype=float),
        fast=True,
    )


def make_mtqs():
    return [
        MTQ(axis=axis, max_torque=1.0, bias=Bias(), noise=Noise())
        for axis in MathConstants.unitvecs
    ]


def make_rws():
    max_torques = [0.01, 0.05, 0.02]
    rw_inertias = [0.001, 0.002, 0.5]
    h0 = [0.1, 0.0, 0.0]
    h_max = [0.1, 0.1, 0.1]
    return [
        RW(
            axis=MathConstants.unitvecs[j],
            max_torque=max_torques[j],
            J=rw_inertias[j],
            h=h0[j],
            h_max=h_max[j],
            bias=Bias(bias=0.0, std_bias=0.0),
            noise=Noise(noise=0.0, std_noise=0.0),
        )
        for j in range(3)
    ]


def expected_quat_dot(w, q):
    return 0.5 * np.concatenate([[-np.dot(q[1:], w)], q[0] * w + np.cross(q[1:], w)])


def rotated_inertia(J_diag, q):
    R = rot_mat(normalize(q))
    return R @ J_diag @ R.T
