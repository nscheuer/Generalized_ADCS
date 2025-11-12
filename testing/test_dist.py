import sys
import os
import numpy as np
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
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
    rho = 0.0034
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI, rho=rho)

    q0 = MathConstants.zeroquat
    w0 = 0.01*MathConstants.unitvecs[0]
    x = np.concatenate([w0, q0])
    u = np.array([])

    expected_torque = -0.5*rho*1.2*2.2*8000*8000*MathConstants.unitvecs[2]
    real_torque = drag.torque(sat=sat, x=x, os=os)
    assert np.allclose(real_torque, expected_torque)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    assert np.all(np.concatenate([expected_torque, [0], 0.005*MathConstants.unitvecs[0]]) == xd)


    face = GeometryFace(area=1.2, centroid=np.array([1, 0, 0]), normal=np.array([0, -1, 0]), CD=2.2)
    config = GeometryConfig(geometry_faces=[face])
    drag = Drag_Disturbance(config=config)
    sat = Satellite(disturbances=[drag])
    B_ECI = np.array([1, 0, 0])*1e-5
    rho = 0.0034
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
    rho = 0.0034
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
    rho = 0.0034
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
    rho = 0.0034
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
    rho = 0.0034
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
    Nmass = 5
    masses = [(float(rng.uniform(0, 3)),
               random_n_unit_vec(3) * float(rng.uniform(0, 2)))
              for _ in range(Nmass)]
    msphere = 1.0
    radius = 0.1
    Jsphere = (2.0/5.0) * msphere * radius**2 * np.eye(3)

    # Totals about inertial origin using parallel-axis contributions
    m_point = sum(m for m, _ in masses)
    m0 = m_point + msphere
    com0 = sum(m * r for m, r in masses) / m_point

    # Inertia about COM: central sphere + shifted point masses
    J_points_about_COM = sum(
        m * (np.eye(3) * np.dot(r - com0, r - com0) - np.outer(r - com0, r - com0))
        for m, r in masses
    )
    J0 = Jsphere + J_points_about_COM 

    # --- Orbit & satellite setup (new framework) ---
    ephem = Ephemeris()
    gg = GG_Disturbance()
    sat = Satellite(J_0=J0, disturbances=[gg], mass=m0, COM=com0)

    # Sanity: satellite stored structural properties correctly
    assert np.all(sat.COM == com0)
    assert np.all(sat.J_0 == J0)
    assert sat.mass == m0
    # And sat.J matches what we constructed
    assert np.allclose(sat.J_0, J0)

    # Orbital state (km, km/s). GG uses os.R magnitude and direction.
    os = Orbital_State(ephem=ephem, J2000=0.22,
                       R=np.array([7000.0, 0.0, 0.0]),
                       V=np.array([0.0, 8.0, 0.0]))

    # Random attitude; zero body rates
    q0 = random_n_unit_vec(4)
    w0 = np.zeros(3)
    x = np.concatenate([w0, q0])
    u = np.array([])

    # --- Expected GG torque (body frame) ---
    # Rotate ECI->Body the same way the codebase does elsewhere: rot_mat(q).T is ECI->Body
    C_be = rot_mat(q0).T
    r_hat_ECI = os.R / np.linalg.norm(os.R)
    r_hat_B = C_be @ r_hat_ECI

    mu = EarthConstants.mu_e  # or ephem.mu, depending on your constants wiring
    factor = 3.0 * mu / (np.linalg.norm(os.R)**3)
    expected_tau = factor * np.cross(r_hat_B, sat.J_0 @ r_hat_B)

    # --- Compare with disturbance output ---
    real_tau = gg.torque(sat=sat, x=x, os=os)
    assert np.allclose(expected_tau, real_tau)

    # Also check it’s injected into dynamics properly (no other torques, so only GG shows up)
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)
    expected_wdot = np.linalg.inv(sat.J_0) @ real_tau
    # xd = [tau (3), 0 (scalar bias?), attitude-rate RHS (3)] -> keep your structure;
    # here we only assert the torque portion matches.
    assert np.allclose(xd[:3], expected_wdot)

    # --- Two useful edge cases ---

    # (a) Isotropic inertia => torque must be zero for any attitude
    J_iso = 2.0 * np.eye(3)
    sat_iso = Satellite(J=J_iso, disturbances=[gg], mass=m0, COM=com0)
    tau_iso = gg.torque(sat=sat_iso, x=x, os=os)
    assert np.allclose(tau_iso, np.zeros(3))

    # (b) Principal-axis alignment (r_hat_B along an eigenvector) => tau = 0
    # Align attitude so +X_body points to Earth
    q_align = MathConstants.zeroquat  # identity makes body X == ECI X with our conventions
    x_align = np.concatenate([w0, q_align])
    C_be_align = rot_mat(q_align).T
    r_hat_B_align = C_be_align @ r_hat_ECI
    # r_hat_B_align is [1,0,0]; if J is diagonal in body frame, cross product vanishes
    J_diag = np.diag([10.0, 5.0, 3.0])
    sat_diag = Satellite(J=J_diag, disturbances=[gg], mass=m0, COM=com0)
    tau_align = gg.torque(sat=sat_diag, x=x_align, os=os)
    assert np.allclose(tau_align, np.zeros(3))


def test_prop_dist_update():
    noise = Noise(noise=0.0, std_noise=0.2)
    t0 = np.array([0.1, -0.1, 0.5])
    dist = Prop_Disturbance(torque_nominal=t0, noise=noise)

    assert np.all(dist.torque_nominal == t0)
    assert np.all(dist.noise.std_noise == 0.2)

    dist.update()



if __name__ == "__main__":
    test_torque_gg_dist()