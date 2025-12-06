import sys
import os
import numpy as np
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GG_Disturbance, GeometryConfig, GeometryFace
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunSensor, SunPair, GPS
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import EarthConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat
from ADCS.helpers.math_constants import MathConstants


def test_sat_sensors_bias():
    bias_mtq = 0.2*random_n_unit_vec(3)
    bias_rw = 0.2*random_n_unit_vec(3)
    h_rw = 1.0*random_n_unit_vec(3)
    B_ECI = random_n_unit_vec(3)

    bias_mtm = 0.3*random_n_unit_vec(3)
    bias_gyro = 0.1*random_n_unit_vec(3)
    bias_sun = 30*random_n_unit_vec(9)
    bias_gps = np.concatenate([random_n_unit_vec(3)*60, random_n_unit_vec(3)*1])
    sun_eff = 0.3

    mtqs = [MTQ(axis=j, max_torque=1, bias=Bias(bias=bias_mtq, std_bias=0)) for j in MathConstants.unitvecs]
    rws = [RW(axis=j, max_torque=1, J=0.1, h=np.dot(h_rw, j), h_max=2.0, bias=Bias(bias=np.dot(bias_rw, j), std_bias=0)) for j in MathConstants.unitvecs]

    mtms = [MTM(axis=j, bias=Bias(bias=np.dot(bias_mtm, j), std_bias=0)) for j in MathConstants.unitvecs]
    gyros = [Gyro(axis=j, bias=Bias(bias=np.dot(bias_gyro, j), std_bias=0)) for j in MathConstants.unitvecs]
    suns1 = [SunSensor(axis=j, efficiency=sun_eff, bias=Bias(bias=np.dot(bias_sun[0:3], j), std_bias=0)) for j in MathConstants.unitvecs]
    suns2 = [SunSensor(axis=-j, efficiency=sun_eff, bias=Bias(bias=np.dot(bias_sun[3:6], j), std_bias=0)) for j in MathConstants.unitvecs]
    suns3 = [SunPair(axis=-j, efficiency=sun_eff, bias=Bias(bias=np.dot(bias_sun[6:], j), std_bias=0)) for j in MathConstants.unitvecs]
    gps = [GPS(bias=Bias(bias=bias_gps, std_bias=0))]

    qJ = random_n_unit_vec(4)
    J0 = np.diagflat([2, 3, 10])
    RJ = rot_mat(qJ)
    J_body = RJ@J0@RJ.T

    q0 = random_n_unit_vec(4)
    Rm = rot_mat(q0)
    J_ECI = Rm@J_body@Rm.T

    w0 = 0.05*random_n_unit_vec(3)
    w_ECI = Rm@J_body@Rm.T
    H_body = J_body@w0 + h_rw
    H_ECI = J_ECI@w_ECI + Rm@h_rw

    acts = mtqs+rws
    sensors = mtms+gyros+suns1+suns2+suns3+gps

    u = 5*random_n_unit_vec(6)
    R_ECI = random_n_unit_vec(3)*np.random.uniform(6900, 7800)
    V_ECI = random_n_unit_vec(3)*np.random.uniform(6, 15)
    S_ECI = random_n_unit_vec(3)*np.random.uniform(1e12, 1e14)
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=R_ECI, V=V_ECI, S=S_ECI, B=B_ECI)
    sat = Satellite(J_0=J_body, actuators=acts, sensors=sensors)
    state = np.concatenate([w0, q0, h_rw])

    exp_wd = -np.linalg.inv(sat.J_noRW)@np.cross(w0, H_body) + np.linalg.inv(sat.J_noRW)@sum([acts[j].torque(u=u[j], x=state, os=os) for j in range(6)], np.zeros(3))
    exp_qd = 0.5*np.concatenate([[-np.dot(q0[1:], w0)], q0[0]*w0 + np.cross(q0[1:], w0)])
    exp_hd = sum([acts[j].torque(u=u[j], x=state, os=os) for j in range(6) if not isinstance(acts[j], RW)], np.zeros(3) - sat.J_0@exp_wd - np.cross(w0, H_body))

    xd = sat.dynamics_core(x=state, u=u, orbital_state=os)
    assert np.allclose(np.concatenate([exp_wd, exp_qd, exp_hd]), xd)


if __name__ == "__main__":
    test_sat_sensors_bias()

