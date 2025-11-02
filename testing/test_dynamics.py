import sys
import os
import numpy as np
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, rot_mat, random_n_unit_vec
from ADCS.helpers.math_constants import MathConstants

def test_dynamics_plain():
    sat = Satellite()
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))
    
    w = np.array([0.01, 0, 0])
    q = np.array([1, 0, 0, 0])
    x = np.hstack((w, q))
    u = np.ndarray([])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,0,0,0,0.005,0,0]))

    w = np.array([0.01, 0, 0])
    q = np.array([0, 0, 1, 0])
    x = np.hstack((w, q))
    u = np.ndarray([])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,0,0,0,0,0,-0.005]))

    qJ = random_n_unit_vec(4)
    J_0 = np.diagflat([2, 3, 10])
    RJ = rot_mat(qJ)
    J_body = RJ@J_0@RJ.T

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    J_ECI = R@J_body@R.T

    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = R@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI

    exp_wd = -R.T@np.linalg.inv(J_ECI)@np.cross(w_ECI, H_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])

    sat = Satellite(J_0=J_body)
    x = np.concatenate((w0, q0))
    u = np.ndarray([])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)

    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),dx))


def test_dynamics_MTQ():
    mtqs = [MTQ(axis=j, max_torque=1, bias=Bias()) for j in MathConstants.unitvecs]
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=1e-5*np.array([1, 0, 0]))

    sat = Satellite(actuators=[mtqs[0]])
    w = np.array([0.01, 0, 0])
    q = np.array([1, 0, 0, 0])
    x = np.hstack((w, q))
    u = np.array([1])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,0,0,0,0.005,0,0]))

    sat = Satellite(actuators=[mtqs[1]])
    u = np.array([1])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,0,-1e-5,0,0.005,0,0]))

    sat = Satellite(actuators=[mtqs[2]])
    u = np.array([1])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,1e-5,0,0,0.005,0,0]))

    sat = Satellite(actuators=mtqs)
    u = np.array([1, 0, 0])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,0,0,0,0.005,0,0]))

    u = np.array([0, 1, 0])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,0,-1e-5,0,0.005,0,0]))

    u = np.array([0, 0, 1])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(dx == np.array([0,1e-5,0,0,0.005,0,0]))

    qJ = random_n_unit_vec(4)
    m0 = random_n_unit_vec(3)
    B_ECI = 1e-5*random_n_unit_vec(3)
    J_0 = np.diagflat([2.0, 3.0, 10.0])
    RJ = rot_mat(qJ)
    J_body = RJ@J_0@RJ.T

    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    J_ECI = R@J_body@R.T

    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = R@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    m_ECI = R@m0

    torq_ECI = np.cross(B_ECI, m_ECI)
    exp_wd = -R.T@np.linalg.inv(J_ECI)@(np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])

    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)
    sat = Satellite(J_0=J_body, actuators=mtqs)
    x = np.concatenate((w0, q0))
    dx = sat.dynamics_core(x=x, u=m0, orbital_state=os)

    expected_dx = np.concatenate([exp_wd,exp_qd])
    assert np.all(np.isclose(expected_dx ,dx))



    

if __name__ == "__main__":
    test_dynamics_MTQ()
