import sys
import os
import numpy as np
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GeometryConfig
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_constants import MathConstants

def test_J():
    mass = 1
    COM = np.array([0, 0, 0])
    J_0 = np.diagflat([0.1, 100, 5])
    sat = Satellite(mass=mass, COM=COM, J_0=J_0)
    assert np.all(sat.J_0 == np.array([[0.1,0,0],[0,100,0],[0,0,5]]))
    assert np.all(sat.invJ_0 == np.array([[10,0,0],[0,0.01,0],[0,0,0.2]]))
    assert np.all(sat.J_noRW == np.array([[0.1,0,0],[0,100,0],[0,0,5]]))
    assert np.all(sat.invJ_noRW == np.array([[10,0,0],[0,0.01,0],[0,0,0.2]]))

def test_J_with_RW():
    Js = [0.001, 0.002, 0.5]
    unitvecs = MathConstants.unitvecs
    acts: List[Actuator] = [RW(axis=unitvecs[j], max_torque=0.1, J=Js[j], h=0, h_max=0.1, bias=None, noise=None, estimate_bias=False) for j in range(3)]
    
    J_0 = np.diagflat([0.1, 100, 5])
    sat = Satellite(J_0=J_0, actuators=acts)
    assert np.all(sat.J_0 == np.array([[0.1,0,0],[0,100,0],[0,0,5]]))
    assert np.all(sat.invJ_0 == np.array([[10,0,0],[0,0.01,0],[0,0,0.2]]))
    assert np.all(sat.J_noRW == np.array([[0.099,0,0],[0,99.998,0],[0,0,4.5]]))
    assert np.all(sat.invJ_noRW == np.array([[1/0.099,0,0],[0,1/99.998,0],[0,0,2/9]]))

def test_COM_J():
    JA = np.eye(3)
    one = np.array([[1], [0], [0]]).squeeze()
    JB = np.eye(3) + 1*(np.eye(3)*4 - 4*np.outer(one, one))
    m = 2
    COM = one
    sat = Satellite(COM = COM, mass = m, J_0 = JA+JB)

    assert np.all(sat.J_0 == JA+JB)
    assert np.all(sat.J_COM == 2*np.diagflat([1,2,2]))

def test_update_RWhs_from_state():
    unitvecs = MathConstants.unitvecs
    zeroquat = MathConstants.zeroquat

    # Create Satellite
    max_torque = [0.03, 0.05, 0.02]
    rw_J = [0.001, 0.002, 0.5]
    h = [0.1, 0.0, 0.0]
    h_max = [0.1, 0.1, 0.1]

    bias_center = [-0.001, 0.05, 0]
    bias_std = [0.3, 0.3, 0.3]

    acts = [
        RW(
            axis=unitvecs[j],
            max_torque=max_torque[j],
            J=rw_J[j],
            h=h[j],
            h_max=h_max[j],
            bias=Bias(bias_center[j], bias_std[j])
        )
        for j in range(3)
    ]
    sat = Satellite(actuators=acts)

    # Create Orbital State
    ephem = Ephemeris()
    t = 0.22
    R = np.array([-0.001, 0.05, 0])
    V = np.array([0, 8, 0])
    B = np.array([1, 0, 0])*1e-5
    os = Orbital_State(ephem=ephem, J2000=t, R=R, V=V, B=B)

    # Dynamics
    x = np.concatenate([0.01*unitvecs[0], zeroquat, h])
    u = np.array([0.021, -0.05, 0.0])
    dx = sat.dynamics_core(x=x, u=u, orbital_state=os)

    new_h = [np.random.uniform(-0.1, 0.1) for j in range(3)]
    nh = np.copy(new_h)
    sat.update_RWhs(new_h)
    assert np.all(sat.RWhs() == nh)
    assert np.all([sat.actuators[j].h for j in sat.momentum_inds] == nh)

    new_h = [np.random.uniform(-0.1, 0.1) for j in range(3)]
    nh = np.copy(new_h)
    x_new = np.concatenate([x[0:7], new_h])
    sat.update_RWhs(x_new)
    assert np.all(sat.RWhs() == nh)
    assert np.all([sat.actuators[j].h for j in sat.momentum_inds] == nh)

def test_srp():
    faces = [{"index": 0, "area": 0.1, "centroid": np.array([1.0, 0.2, 0.0]), "normal": np.array([1, 0, 0]), "eta_s": 0.0, "eta_d": 0.5, "eta_a": 0.5, "cd": 2},
    {"index": 1, "area": 0.03, "centroid": np.array([-0.05, 0.1, 0.3]), "normal": np.array([0, 1, 0]), "eta_s": 0.1, "eta_d": 0.2, "eta_a": 0.1, "cd": 0.1},
    {"index": 2, "area": 10, "centroid": np.array([0.25,-0.01,-0.7]), "normal": np.array([0, 0, 1]), "eta_s": 0.3, "eta_d": 0.1, "eta_a": 0.6, "cd": 0.3}]

    config = GeometryConfig(geometry=faces)
    dist = [SRP_Disturbance(config=config)]
    sat = Satellite(disturbances=dist)

    assert np.all(sat.disturbances[0].eta_s == [0.0,0.1,0.3])
    assert np.all(sat.disturbances[0].eta_d == [0.5,0.2,0.1])
    assert np.all(sat.disturbances[0].eta_a == [0.5,0.1,0.6])
    assert np.all(sat.disturbances[0].areas == [0.1,0.03,10])
    assert np.all([sat.disturbances[0].normals[j] == MathConstants.unitvecs[j] for j in range(3)])
    assert np.all([sat.disturbances[0].centroids[j] == faces[j]["centroid"] for j in range(3)])
    
def test_drag():
    faces = [{"index": 0, "area": 0.1, "centroid": np.array([1.0, 0.2, 0.0]), "normal": np.array([1, 0, 0]), "eta_s": 0.0, "eta_d": 0.5, "eta_a": 0.5, "cd": 2},
    {"index": 1, "area": 0.03, "centroid": np.array([-0.05, 0.1, 0.3]), "normal": np.array([0, 1, 0]), "eta_s": 0.1, "eta_d": 0.2, "eta_a": 0.1, "cd": 0.1},
    {"index": 2, "area": 10, "centroid": np.array([0.25,-0.01,-0.7]), "normal": np.array([0, 0, 1]), "eta_s": 0.3, "eta_d": 0.1, "eta_a": 0.6, "cd": 0.3}]
    
    config = GeometryConfig(geometry=faces)
    dist = [Drag_Disturbance(config=config)]
    sat = Satellite(disturbances=dist)

    assert np.all(sat.disturbances[0].areas == [0.1,0.03,10])
    assert np.all([sat.disturbances[0].centroids[j] == [np.array([1,0.2,0]),np.array([-0.05,0.1,0.3]),np.array([0.25,-0.01,-0.7])][j] for j in range(3)])
    assert np.all([sat.disturbances[0].normals[j] == [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])][j] for j in range(3)])
    assert np.all(sat.disturbances[0].CDs == [2,0.1,0.3])

def test_prop():
    nominal_torque = np.array([1, 2, 4])
    noise = Noise()
    dist = [Prop_Disturbance(nominal_torque, noise)]
    sat = Satellite(disturbances=dist)

    assert np.all(sat.disturbances[0].torque_nominal == np.array([1,2,4]))
    assert np.all(sat.disturbances[0].torque() == np.array([1,2,4]))

def test_resdipole():
    nominal_dipole = np.array([0.1, -0.1, 0.5])
    noise = Noise()
    dist = [Dipole_Disturbance(nominal_dipole, noise)]
    sat = Satellite(disturbances=dist)

    assert np.all(sat.disturbances[0].torque_nominal == np.array([0.1,-0.1,0.5]))
    assert np.all(sat.disturbances[0].noise.std_noise == 0.0)

if __name__ == "__main__":
    test_update_RWhs_from_state()





