import sys
import os
import numpy as np
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat
from ADCS.helpers.math_constants import MathConstants

def test_torque_mag_dist():
    B_ECI = B = np.array([1,0,0])*1e-5
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)
    q0 = MathConstants.zeroquat

    res_dipole = Dipole_Disturbance(MathConstants.unitvecs[0])
    sat = Satellite(disturbances=[res_dipole])

    x = np.array([0.01,0,0,1,0,0,0])
    u = np.array([])
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.array([0,0,0,0,0.005,0,0]) == xd)
    assert np.all(0 == res_dipole.torque(x=x, os=os))

    res_dipole = Dipole_Disturbance(MathConstants.unitvecs[1])
    sat = Satellite(disturbances=[res_dipole])
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.array([0,0,-1e-5,0,0.005,0,0]) == xd)
    assert np.all(MathConstants.unitvecs[2]*-1e-5 == res_dipole.torque(x=x, os=os))

    res_dipole = Dipole_Disturbance(MathConstants.unitvecs[2])
    sat = Satellite(disturbances=[res_dipole])
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.array([0,1e-5,0,0,0.005,0,0]) == xd)
    assert np.all(MathConstants.unitvecs[1]*1e-5 == res_dipole.torque(x=x, os=os))

    qJ = random_n_unit_vec(4)
    dip = random_n_unit_vec(3)
    B_ECI = 1e-1*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ)
    J_body = RJ@J0@RJ.T

    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0

    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    dip_ECI = Rm@dip

    torq_ECI = -np.cross(B_ECI, dip_ECI)
    exp_wd = Rm.T@np.linalg.inv(J_ECI)@(-np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])
    res_dipole = Dipole_Disturbance(dipole_torque=dip)
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)
    sat = Satellite(J_0=J_body, disturbances=[res_dipole])
    x = np.concatenate([w0, q0])
    u = np.array([])

    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(Rm.T@torq_ECI,res_dipole.torque(x=x, os=os))
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))


def test_torque_prop_dist():
    t0 = 3.2*random_n_unit_vec(3)
    prop_torq = Prop_Disturbance(t0)

    qJ = random_n_unit_vec(4)
    dip = random_n_unit_vec(3)
    B_ECI = 1e-5*random_n_unit_vec(3)
    J0 = np.diagflat([2,3,10])
    RJ = rot_mat(qJ) # RJ@v_PA=v_body where v_PA is the principal axis frame
    J_body = RJ@J0@RJ.T
    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T
    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@w0
    H_body = J_body@w0
    H_ECI = J_ECI@w_ECI
    torq_ECI = Rm@t0
    exp_wd = Rm.T@np.linalg.inv(J_ECI)@(-np.cross(w_ECI,H_ECI)+torq_ECI)
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:],w0)],q0[0]*w0 + np.cross(q0[1:],w0)])

    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)
    sat = Satellite(J_0=J_body, disturbances=[prop_torq])
    x = np.concatenate([w0, q0])
    u = np.array([])

    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(Rm.T@torq_ECI,prop_torq.torque(x=x, os=os))
    assert np.allclose(t0,prop_torq.torque(x=x, os=os))
    assert np.all(np.isclose(np.concatenate([exp_wd,exp_qd]),xd))


def test_torque_drag_dist():
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), CD=2.2)
    config = GeometryConfig(geometry_faces=[face])
    drag = Drag_Disturbance(config=config)
    ephem = Ephemeris()
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 5e-12
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = -0.5*rho*1.2*2.2*8000*8000*MathConstants.unitvecs[2]
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]), xd)


    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, -1, 0]), CD=2.2)
    config = GeometryConfig(geometry_faces=[face])
    drag = Drag_Disturbance(config=config)
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 5e-12
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = np.array([0, 0, 0])
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]) == xd)


    faces = [GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), CD=2.2), GeometryFace(area=1.2, centroid=np.array([-1, 0, 0]), normal=np.array([0, 1, 0]), CD=2.2)]
    config = GeometryConfig(geometry_faces=faces)
    drag = Drag_Disturbance(config=config)
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 5e-12
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = np.array([0, 0, 0])
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]) == xd)


    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([1, 0, 0]), CD=2.2)
    config = GeometryConfig(geometry_faces=[face])
    drag = Drag_Disturbance(config=config)
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 5e-12
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = np.array([0, 0, 0])
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]) == xd)


    face = GeometryFace(area=1.2, centroid=np.array([0, 1, 0]), normal=np.array([0, 1, 0]), CD=2.2)
    config = GeometryConfig(geometry_faces=[face])
    drag = Drag_Disturbance(config=config)
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 5e-12
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = np.array([0, 0, 0])
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]) == xd)


    faces = [GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([1, 0, 0]), CD=2.2), 
             GeometryFace(area=1.2, centroid=np.array([-1, 0, 0]), normal=np.array([-1, 0, 0]), CD=2.2),
             GeometryFace(area=1.2, centroid=np.array([0, 1, 0]), normal=np.array([0, 1, 0]), CD=2.2), 
             GeometryFace(area=1.2, centroid=np.array([0, -1, 0]), normal=np.array([0, -1, 0]), CD=2.2),
             GeometryFace(area=1.2, centroid=np.array([0, 0, 1]), normal=np.array([0, 0, 1]), CD=2.2),
             GeometryFace(area=1.2, centroid=np.array([0, 0, -1]), normal=np.array([0, 0, -1]), CD=2.2)]
    
    config = GeometryConfig(geometry_faces=faces)
    drag = Drag_Disturbance(config=config)
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 5e-12
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = np.array([0, 0, 0])
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]) == xd)


def test_torque_srp_dist():
    # Common setup bits
    ephem = Ephemeris()
    solar_constant = EarthConstants.solar_constant
    c = EarthConstants.c
    q0 = MathConstants.zeroquat
    w0 = 0.01 * MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    # 1) Single face, +Y normal, purely absorptive, S along +Y
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), eta_a=1, eta_d=0, eta_s=0)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([0, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = -solar_constant * 1.2 / c * MathConstants.unitvecs[2]
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]) == xd)

    # 2) Single face, -Y normal, purely absorptive, S along +Y -> shadow / back-face, zero
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, -1, 0]), eta_a=1, eta_d=0, eta_s=0)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([0, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = np.array([0, 0, 0])
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]) == xd)

    # 3) Single face, +Y normal, mixed surface (eta_a, eta_d, eta_s), S along +Y
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([0, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = -solar_constant * 1.2 / c * (0.05 + (5.0/3.0)*0.25 + 2*0.7) * MathConstants.unitvecs[2]
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 4) Single face, -Y normal, mixed surface, S along +Y -> zero
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, -1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([0, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = np.array([0, 0, 0])
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]) == xd)

    # 5) Single face, +Y normal, specular only (eta_s=1), S along +X+Y (45 deg in XY-plane)
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), eta_a=0, eta_d=0, eta_s=1)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([1e12, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    cg = 1/np.sqrt(2)
    expected_torque = -solar_constant * (1.2/c) * (MathConstants.unitvecs[2] * (2 * cg * cg))
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 6) Single face, +Y normal, purely absorptive (eta_a=1), S along +X+Y
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), eta_a=1, eta_d=0, eta_s=0)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([1e12, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    cg = 1/np.sqrt(2)
    expected_torque = -solar_constant * (1.2/c) * (cg * MathConstants.unitvecs[2] / np.sqrt(2))
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 7) Single face, +Y normal, mixed surface, S along +X+Y
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([1e12, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    cg = 1/np.sqrt(2)
    expected_torque = -solar_constant * (1.2/c) * (
        MathConstants.unitvecs[2]*(2*0.25*cg/3 + 2*cg*cg*0.7) + cg*(0.05 + 0.25)*MathConstants.unitvecs[2]/np.sqrt(2)
    )
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 8) Single face, -Y normal, mixed surface, S along +Y (repeat zero case)
    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, -1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = 1e12 * MathConstants.unitvecs[1]
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = np.array([0, 0, 0])
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]) == xd)

    # 9) Single face, centroid along +Z, +Y normal, mixed surface, S along +X+Y
    face = GeometryFace(area=1.2, centroid=np.array([0, 0, 1]), normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([1e12, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    cg = 1/np.sqrt(2)
    expected_torque = solar_constant * (1.2/c) * (
        MathConstants.unitvecs[0]*(2*0.25*cg/3 + 2*cg*cg*0.7) +
        cg*(0.05 + 0.25)*(MathConstants.unitvecs[0] - MathConstants.unitvecs[1]) / np.sqrt(2)
    )
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 10) Single face, centroid along +Y, +Y normal, mixed surface, S along +Y -> zero
    face = GeometryFace(area=1.2, centroid=np.array([0, 1, 0]), normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([0, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = 0 * MathConstants.unitvecs[2]
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 11) Same face as 10), S along +X+Y -> nonzero Z torque from absorption+diffuse cross-terms
    face = GeometryFace(area=1.2, centroid=np.array([0, 1, 0]), normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([1e12, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    cg = 1/np.sqrt(2)
    expected_torque = solar_constant * (1.2/c) * (cg * (0.05 + 0.25) * MathConstants.unitvecs[2] / np.sqrt(2))
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)

    # 12) Symmetric pair of faces at +/-Z, +Y normals, mixed surface, S along +X+Y -> cancellation
    face1 = GeometryFace(area=1.2, centroid=np.array([0, 0, 1]),  normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    face2 = GeometryFace(area=1.2, centroid=np.array([0, 0, -1]), normal=np.array([0, 1, 0]), eta_a=0.05, eta_d=0.25, eta_s=0.7)
    config = GeometryConfig(geometry_faces=[face1, face2])
    srp = SRP_Disturbance(config=config)
    sat = Satellite(disturbances=[srp])
    S_ECI = np.array([1e12, 1e12, 0])
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), S=S_ECI)
    expected_torque = np.zeros(3)
    real_torque = srp.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_torque, real_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([expected_torque, [0], 0.005 * MathConstants.unitvecs[0]]), xd)


def test_torque_gg_dist():
    rng = np.random.default_rng(42)

    # ----- Build a physically consistent satellite -----
    Nmass = 5
    point_masses = [
        (
            float(rng.uniform(0, 3)),
            random_n_unit_vec(3) * float(rng.uniform(0, 2)),
        )
        for _ in range(Nmass)
    ]

    # Central solid sphere at the reference origin
    msphere = 1.0
    radius = 0.1
    Jsphere = (2.0 / 5.0) * msphere * radius**2 * np.eye(3)

    # Total mass and total COM (include the sphere at origin)
    m_points = sum(m for m, _ in point_masses)
    m_total = msphere + m_points
    COM_total = sum(m * r for m, r in point_masses) / m_total  # sphere contributes 0

    # Inertia about the **reference origin** (needed for Satellite.J_0)
    J_points_about_origin = sum(
        m * (np.eye(3) * np.dot(r, r) - np.outer(r, r)) for m, r in point_masses
    )
    J0_origin = Jsphere + J_points_about_origin

    # Expected inertia about COM (parallel-axis check)
    J_COM_expected = (
        J0_origin
        - m_total
        * (np.eye(3) * np.dot(COM_total, COM_total) - np.outer(COM_total, COM_total))
    )

    # ----- Satellite & disturbance -----
    gg = GG_Disturbance()
    sat = Satellite(J_0=J0_origin, disturbances=[gg], mass=m_total, COM=COM_total)

    # Structural sanity checks
    assert np.allclose(sat.COM, COM_total)
    assert np.allclose(sat.J_0, J0_origin)
    assert np.allclose(sat.J_COM, J_COM_expected)
    assert np.allclose(np.linalg.inv(sat.invJ_noRW), sat.J_noRW)  # invJ_noRW is actually inverse

    # ----- Orbit & attitude -----
    ephem = Ephemeris()
    os = Orbital_State(
        ephem=ephem,
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),  # km
        V=np.array([0.0, 8.0, 0.0]),     # km/s
    )

    q0 = random_n_unit_vec(4)
    w0 = np.zeros(3)
    x = np.concatenate([w0, q0])
    u = np.array([])

    # ----- Expected GG torque (match implementation details) -----
    # Use the same body-frame vector the model uses
    vecs = os.get_state_vector(x=x)
    R_B = vecs["r"]
    rhat_B = R_B / np.linalg.norm(R_B)
    nadir = -rhat_B

    mu = EarthConstants.mu_e
    factor = 3.0 * mu / (np.linalg.norm(R_B) ** 3)
    expected_tau = factor * np.cross(nadir, nadir @ sat.J_0)

    # Compare against disturbance output
    real_tau = gg.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_tau, real_tau)

    # ----- Check injection into dynamics (no actuators, w=0) -----
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    # Code uses right-multiplication by invJ_noRW and invJ_noRW is symmetric -> equivalent to J_noRW^{-1} * tau
    expected_wdot = real_tau @ sat.invJ_noRW
    assert np.allclose(xd[:3], expected_wdot)
    # With w=0, quaternion kinematics should be zero too
    assert np.allclose(xd[3:7], np.zeros(4))

    # ----- Edge cases -----

    # (a) Isotropic inertia -> torque must be zero for any attitude
    J_iso = 2.0 * np.eye(3)
    sat_iso = Satellite(J_0=J_iso, disturbances=[gg], mass=1.0, COM=np.zeros(3))
    tau_iso = gg.torque(sat=sat_iso, x=x, os=os)
    assert np.allclose(tau_iso, np.zeros(3))

    # (b) Principal-axis alignment (rhat_B along an eigenvector) -> torque = 0
    # Identity quaternion aligns body +X with ECI +X; with R along +X_ECI, rhat_B = [1,0,0]
    q_align = MathConstants.zeroquat  # identity quaternion
    x_align = np.concatenate([w0, q_align])
    J_diag = np.diag([10.0, 5.0, 3.0])
    sat_diag = Satellite(J_0=J_diag, disturbances=[gg], mass=1.0, COM=np.zeros(3))
    tau_align = gg.torque(sat=sat_diag, x=x_align, os=os)
    assert np.allclose(tau_align, np.zeros(3))


if __name__ == "__main__":
    test_torque_drag_dist()