import sys
import os
import numpy as np
import numdifftools as nd
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, rot_mat, random_n_unit_vec, drotmatTvecdq, ddrotmatTvecdqdq
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


def test_MTQ_torque():
    mtqs = [MTQ(axis=j, max_torque=1, bias=Bias()) for j in MathConstants.unitvecs]
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=1e-5*np.array([1, 0, 0]))

    w = np.array([0.01, 0, 0])
    q = np.array([1, 0, 0, 0])
    x = np.hstack((w, q))

    for i in range(3):
        assert np.all(mtqs[i].torque(u=0, x=x, os=os) == np.zeros(3))
        assert np.all(mtqs[i].torque(u=1, x=x, os=os) == 1e-5*np.cross(MathConstants.unitvecs[i], MathConstants.unitvecs[0]))

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

    sat = Satellite(J_0=J_body, actuators=mtqs)
    x = np.concatenate((w0, q0))
    dx = sat.dynamics_core(x=x, u=m0, orbital_state=os)

    expected_dx = np.concatenate([exp_wd,exp_qd])
    assert np.all(np.isclose(expected_dx ,dx))

    B = B_ECI
    rmat_ECI2B = rot_mat(q0).T
    B_B = rmat_ECI2B@B
    exp_torq = [np.cross(i, B_B) for i in MathConstants.unitvecs]
    for i in range(3):
        assert np.all(mtqs[i].torque(u=1, x=x, os=os) == exp_torq[i])
    

def test_MTQ_setup():
    ax = random_n_unit_vec(3)*3
    max_torque = 4.51

    std_noise = 0.243
    noise = Noise(noise=0, std_noise=std_noise)

    e_bias = random_n_unit_vec(3)[1]*0.1
    bsr = 0.03
    bias = Bias(bias=e_bias, std_bias=bsr)

    mtq = MTQ(axis=ax, max_torque=max_torque, bias=bias, noise=noise, estimate_bias=False)

    assert np.all(np.isclose(ax/3, mtq.axis))
    assert np.all(bias==mtq.bias)
    assert mtq.u_max == max_torque
    assert mtq.noise.noise == 0
    assert mtq.noise.std_noise == std_noise
    assert mtq.bias.bias == e_bias
    assert mtq.bias.std_bias == bsr


def test_MTQ_torque_clean():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_torque = 4.51
    mtq = MTQ(axis=ax, max_torque=max_torque)
    mtqs = [mtq]

    m0 = random_n_unit_vec(3)[0]
    B_ECI = 1e-5*random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=mtqs)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)

    expected_torque = np.cross(ax/3*(m0), B_B)
    actual_torque = mtq.torque(u=m0, x=x0, os=os)
    assert np.all(np.isclose(expected_torque, actual_torque))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}
    
    ufun = lambda c: mtq.torque(u=c, x=x0, os=os)
    xfun = lambda c: mtq.torque(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
    hfun = lambda c: mtq.torque(u=m0, x=x0, os=os)
    bfun = lambda c: MTQ(axis=ax, max_torque=max_torque).torque(u=m0, x=x0, os=os)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    expected_Jxfun = mtq.dtorq__dbasestate(command=m0, q=q0, os=os)
    Jufun = np.array(nd.Jacobian(ufun)(m0)).T
    expected_Jufun = mtq.dtorq__du(command=m0, q=q0, os=os)
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T
    expected_Jbfun = mtq.dtorq__dbias(command=m0, q=q0, os=os)
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T
    expected_Jhfun = mtq.dtorq__dh(command=m0, q=q0, os=os)

    assert np.allclose(Jxfun, expected_Jxfun)
    assert np.allclose(Jufun, expected_Jufun)
    assert np.allclose(Jbfun, expected_Jbfun)
    assert np.allclose(Jhfun, expected_Jhfun)


    for j in MathConstants.unitvecs:
        fun_hj = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque).torque(command=c[0], x=np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]), os=os))

        ufunjju = lambda c: np.dot(mtq.dtorq__du(command=c, x=x0, os=os), j).item()
        ufunjjb = lambda c: np.dot(mtq.dtorq__dbias(command=c, x=x0, os=os),j)
        ufunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(command=c, x=x0, os=os),j)
        ufunjjh = lambda c: np.dot(mtq.dtorq__dh(command=c, x=x0, os=os),j)

        xfunjju = lambda c: np.dot(mtq.dtorq__du(command=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j).item()
        xfunjjb = lambda c: np.dot(mtq.dtorq__dbias(command=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)
        xfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(command=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)
        xfunjjh = lambda c: np.dot(mtq.dtorq__dh(command=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)

        hfunjju = lambda c: np.dot(mtq.dtorq__du(command=m0, x=x0, os=os),j).item()
        hfunjjb = lambda c: np.dot(mtq.dtorq__dbias(command=m0, x=x0, os=os),j)
        hfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(command=m0, x=x0, os=os),j)
        hfunjjh = lambda c: np.dot(mtq.dtorq__dh(command=m0, x=x0, os=os),j)

        bfunjju = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque).dtorq__du(command=m0, q=q0, os=os), j).item()
        bfunjjx = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque).dtorq__dbasestate(command=m0, q=q0, os=os), j)

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))

        assert np.allclose(Jxfunjju, np.dot(mtq.ddtorq__dudbasestate(command=m0, q=q0, os=os), j))
        assert np.allclose(Jxfunjjx, np.dot(mtq.ddtorq__dbasestatedbasestate(command=m0, q=q0, os=os), j))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(m0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(m0))

        assert np.allclose(Jufunjju, np.dot(mtq.ddtorq__dudu(command=m0, q=q0, os=os), j))
        assert np.allclose(Jufunjjx.T, np.dot(mtq.ddtorq__dudbasestate(command=m0, q=q0, os=os), j))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))

        assert np.allclose(Jbfunjju, np.dot(mtq.ddtorq__dudbias(command=m0, q=q0, os=os), j))
        assert np.allclose(Jbfunjjx.T, np.dot(mtq.ddtorq__dbiasdbasestate(command=m0, q=q0, os=os), j))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))

        assert np.allclose(Jhfunjju, np.dot(mtq.ddtorq__dudh(command=m0, q=q0, os=os), j))
        assert np.allclose(Jhfunjjx, np.dot(mtq.ddtorq__dbasestatedh(command=m0, q=q0, os=os), j))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[m0], x0, [500.2]]).flatten().tolist()))
        Hguess = np.block([
            [
                mtq.ddtorq__dudu(command=m0, q=q0, os=os) @ j,
                mtq.ddtorq__dudbias(command=m0, q=q0, os=os) @ j,
                mtq.ddtorq__dudbasestate(command=m0, q=q0, os=os) @ j,
                mtq.ddtorq__dudh(command=m0, q=q0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudbias(command=m0, q=q0, os=os) @ j).T,
                mtq.ddtorq__dbiasdbias(command=m0, q=q0, os=os) @ j,
                mtq.ddtorq__dbiasdbasestate(command=m0, q=q0, os=os) @ j,
                mtq.ddtorq__dbiasdh(command=m0, q=q0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudbasestate(command=m0, q=q0, os=os) @ j).T,
                (mtq.ddtorq__dbiasdbasestate(command=m0, q=q0, os=os) @ j).T,
                mtq.ddtorq__dbasestatedbasestate(command=m0, q=q0, os=os) @ j,
                mtq.ddtorq__dbasestatedh(command=m0, q=q0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudh(command=m0, q=q0, os=os) @ j).T,
                (mtq.ddtorq__dbiasdh(command=m0, q=q0, os=os) @ j).T,
                (mtq.ddtorq__dbasestatedh(command=m0, q=q0, os=os) @ j).T,
                mtq.ddtorq__dhdh(command=m0, q=q0, os=os) @ j
            ]
        ])

        assert np.allclose(Hfun[8,:],0)
        assert np.allclose(Hfun[:,8],0)
        assert np.allclose(Hfun[0:8, 0:8], Hguess)

    # Torque and first-order derivatives
    assert np.all(np.isclose(mtq.dtorq__du(command=m0, q=q0, os=os), np.cross(ax/3, B_B)))
    assert np.all(np.isclose(mtq.dtorq__dbias(command=m0, q=q0, os=os), np.cross(ax/3, B_B)))
    assert np.all(
        np.isclose(
            mtq.dtorq__dbasestate(command=m0, q=q0, os=os),
            np.vstack([np.zeros((3, 3)), np.cross(ax/3 * (m0), drotmatTvecdq(q0, B_ECI))]),
        )
    )

    # Derivative wrt h (should be zero for MTQ)
    assert np.all(np.isclose(mtq.dtorq__dh(command=m0, q=q0, os=os), np.zeros((0, 3))))
    assert mtq.dtorq__dh(command=m0, q=q0, os=os).shape == (0, 3)

    # Second-order derivatives (Hessians)
    assert np.all(np.isclose(mtq.ddtorq__dudu(command=m0, q=q0, os=os), np.zeros((1, 1, 3))))
    assert mtq.ddtorq__dudu(command=m0, q=q0, os=os).shape == (1, 1, 3)

    assert np.all(np.isclose(mtq.ddtorq__dudbias(command=m0, q=q0, os=os), np.zeros((1, 0, 3))))
    assert mtq.ddtorq__dudbias(command=m0, q=q0, os=os).shape == (1, 0, 3)

    assert np.all(
        np.isclose(
            mtq.ddtorq__dudbasestate(command=m0, q=q0, os=os),
            np.expand_dims(np.vstack([np.zeros((3, 3)), np.cross(ax/3, drotmatTvecdq(q0, B_ECI))]), 0),
        )
    )
    assert mtq.ddtorq__dudbasestate(command=m0, q=q0, os=os).shape == (1, 7, 3)

    assert np.all(np.isclose(mtq.ddtorq__dudh(command=m0, q=q0, os=os), np.zeros((1, 0, 3))))
    assert mtq.ddtorq__dudh(command=m0, q=q0, os=os).shape == (1, 0, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbiasdbias(command=m0, q=q0, os=os), np.zeros((0, 0, 3))))
    assert mtq.ddtorq__dbiasdbias(command=m0, q=q0, os=os).shape == (0, 0, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbiasdbasestate(command=m0, q=q0, os=os), np.zeros((0, 7, 3))))
    assert mtq.ddtorq__dbiasdbasestate(command=m0, q=q0, os=os).shape == (0, 7, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbiasdh(command=m0, q=q0, os=os), np.zeros((0, 0, 3))))
    assert mtq.ddtorq__dbiasdh(command=m0, q=q0, os=os).shape == (0, 0, 3)

    # Base state Hessian
    dxdx = np.zeros((7, 7, 3))
    dxdx[3:7, 3:7, :] = np.cross(ax/3 * (m0), ddrotmatTvecdqdq(q0, B_ECI))
    assert np.all(np.isclose(mtq.ddtorq__dbasestatedbasestate(command=m0, q=q0, os=os), dxdx))
    assert mtq.ddtorq__dbasestatedbasestate(command=m0, q=q0, os=os).shape == (7, 7, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbasestatedh(command=m0, q=q0, os=os), np.zeros((7, 0, 3))))
    assert mtq.ddtorq__dbasestatedh(command=m0, q=q0, os=os).shape == (7, 0, 3)

    assert np.all(np.isclose(mtq.ddtorq__dhdh(command=m0, q=q0, os=os), np.zeros((0, 0, 3))))
    assert mtq.ddtorq__dhdh(command=m0, q=q0, os=os).shape == (0, 0, 3)

    # Momentum storage torque (MTQ has none)
    assert np.all(mtq.storage_torque(command=m0, j2000=os.J2000) == np.zeros(0))
    assert mtq.storage_torque(command=m0, j2000=os.J2000).shape == (0,)

    # First-order derivatives of storage torque
    assert np.all(np.isclose(mtq.dstor_torq__du(command=m0, q=q0, os=os), np.zeros((1, 0))))
    assert mtq.dstor_torq__du(command=m0, q=q0, os=os).shape == (1, 0)

    assert np.all(np.isclose(mtq.dstor_torq__dbias(command=m0, q=q0, os=os), np.zeros((0, 0))))
    assert mtq.dstor_torq__dbias(command=m0, q=q0, os=os).shape == (0, 0)

    assert np.all(np.isclose(mtq.dstor_torq__dbasestate(command=m0, q=q0, os=os), np.zeros((7, 0))))
    assert mtq.dstor_torq__dbasestate(command=m0, q=q0, os=os).shape == (7, 0)

    assert np.all(np.isclose(mtq.dstor_torq__dh(command=m0, q=q0, os=os), np.zeros((0, 0))))
    assert mtq.dstor_torq__dh(command=m0, q=q0, os=os).shape == (0, 0)

    # Second-order derivatives of storage torque
    assert np.all(np.isclose(mtq.ddstor_torq__dudu(command=m0, q=q0, os=os), np.zeros((1, 1, 0))))
    assert mtq.ddstor_torq__dudu(command=m0, q=q0, os=os).shape == (1, 1, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dudbias(command=m0, q=q0, os=os), np.zeros((1, 0, 0))))
    assert mtq.ddstor_torq__dudbias(command=m0, q=q0, os=os).shape == (1, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dudbasestate(command=m0, q=q0, os=os), np.zeros((1, 7, 0))))
    assert mtq.ddstor_torq__dudbasestate(command=m0, q=q0, os=os).shape == (1, 7, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dudh(command=m0, q=q0, os=os), np.zeros((1, 0, 0))))
    assert mtq.ddstor_torq__dudh(command=m0, q=q0, os=os).shape == (1, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbiasdbias(command=m0, q=q0, os=os), np.zeros((0, 0, 0))))
    assert mtq.ddstor_torq__dbiasdbias(command=m0, q=q0, os=os).shape == (0, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbiasdbasestate(command=m0, q=q0, os=os), np.zeros((0, 7, 0))))
    assert mtq.ddstor_torq__dbiasdbasestate(command=m0, q=q0, os=os).shape == (0, 7, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbiasdh(command=m0, q=q0, os=os), np.zeros((0, 0, 0))))
    assert mtq.ddstor_torq__dbiasdh(command=m0, q=q0, os=os).shape == (0, 0, 0)

    dxdx = np.zeros((7, 7, 0))
    assert np.all(np.isclose(mtq.ddstor_torq__dbasestatedbasestate(command=m0, q=q0, os=os), dxdx))
    assert mtq.ddstor_torq__dbasestatedbasestate(command=m0, q=q0, os=os).shape == (7, 7, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbasestatedh(command=m0, q=q0, os=os), np.zeros((7, 0, 0))))
    assert mtq.ddstor_torq__dbasestatedh(command=m0, q=q0, os=os).shape == (7, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dhdh(command=m0, q=q0, os=os), np.zeros((0, 0, 0))))
    assert mtq.ddstor_torq__dhdh(command=m0, q=q0, os=os).shape == (0, 0, 0)





if __name__ == "__main__":
    test_MTQ_torque_clean()
