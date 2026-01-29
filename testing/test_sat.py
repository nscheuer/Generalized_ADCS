import sys
import os
import numpy as np
import pytest
from typing import List

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.disturbances import SRP_Disturbance, Drag_Disturbance, Prop_Disturbance, Dipole_Disturbance, GeometryConfig, GeometryFace
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import random_n_unit_vec, rot_mat
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


def test_dynamics_plain():
    r"""
    Validate the core rigid-body attitude dynamics and quaternion kinematics used
    by :meth:`Satellite.dynamics_core` against closed-form expressions.

    The state is ordered as
    :math:`\mathbf{x} = \begin{bmatrix}\boldsymbol{\omega}^\top & q_0 & \mathbf{q}_v^\top\end{bmatrix}^\top`,
    where :math:`\boldsymbol{\omega}\in\mathbb{R}^3` is the body angular rate,
    :math:`q_0\in\mathbb{R}` is the scalar part of the quaternion, and
    :math:`\mathbf{q}_v\in\mathbb{R}^3` is the vector part (scalar-first convention).

    The test covers two scenarios:

    1. **Simple kinematics checks** with small angular rate and special quaternions,
       where the expected quaternion time derivative reduces to a simple form and
       the angular acceleration is zero under the chosen configuration.

    2. **General torque-free rigid-body dynamics** with a rotated inertia tensor,
       where the expected angular acceleration and quaternion rate are computed in
       closed form and compared to :meth:`Satellite.dynamics_core`.

    **Quaternion kinematics**

    With scalar-first quaternion :math:`q = \begin{bmatrix}q_0 & \mathbf{q}_v^\top\end{bmatrix}^\top`
    and body rate :math:`\boldsymbol{\omega}`, the kinematics are

    .. math::

       \dot{q}
       \;=\;
       \tfrac{1}{2}
       \begin{bmatrix}
         -\,\mathbf{q}_v^\top \boldsymbol{\omega} \\
         q_0\,\boldsymbol{\omega} + \mathbf{q}_v \times \boldsymbol{\omega}
       \end{bmatrix}.

    The test asserts, for :math:`\boldsymbol{\omega} = [0.01,\,0,\,0]^\top`:

    - If :math:`q = [1,\,0,\,0,\,0]^\top`, then
      :math:`\dot{q} = \tfrac{1}{2}[0,\,0.01,\,0,\,0]^\top`,
      so :math:`\dot{q} = [0,\,0.005,\,0,\,0]^\top`.

    - If :math:`q = [0,\,0,\,1,\,0]^\top` (i.e. :math:`q_0=0,\ \mathbf{q}_v = [0,\,1,\,0]^\top`), then

      .. math::

         \dot{q}
         \;=\;
         \tfrac{1}{2}
         \begin{bmatrix}
           -\,\mathbf{q}_v^\top \boldsymbol{\omega} \\
           \mathbf{q}_v \times \boldsymbol{\omega}
         \end{bmatrix}
         \;=\;
         \tfrac{1}{2}
         \begin{bmatrix}
           0 \\[3pt]
           0 \\[2pt]
           0 \\[2pt]
           -0.01
         \end{bmatrix}
         \;=\;
         \begin{bmatrix}
           0 \\[2pt]
           0 \\[2pt]
           0 \\[2pt]
           -0.005
         \end{bmatrix}.

    **Torque-free rigid-body dynamics**

    For a rigid body with body-frame inertia :math:`\mathbf{J}_b` and no external torques,
    the torque-free Euler equation in the body frame is

    .. math::

       \dot{\boldsymbol{\omega}}
       \;=\;
       \mathbf{J}_b^{-1}\!\left(\,-\,\boldsymbol{\omega} \times (\mathbf{J}_b \boldsymbol{\omega})\,\right).

    In the test, we equivalently compute the expected angular acceleration via a
    body↔ECI transformation. Let :math:`\mathbf{R}(q)` be the body-to-ECI rotation from
    the quaternion :math:`q`, and define

    .. math::

       \boldsymbol{\omega}_{\text{ECI}} \;=\; \mathbf{R}\,\boldsymbol{\omega},\qquad
       \mathbf{J}_{\text{ECI}} \;=\; \mathbf{R}\,\mathbf{J}_b\,\mathbf{R}^\top,\qquad
       \mathbf{H}_{\text{ECI}} \;=\; \mathbf{J}_{\text{ECI}}\,\boldsymbol{\omega}_{\text{ECI}}.

    Then the expected body-frame angular acceleration used in the assertion is

    .. math::

       \dot{\boldsymbol{\omega}}_{\text{exp}}
       \;=\;
       -\,\mathbf{R}^\top\,\mathbf{J}_{\text{ECI}}^{-1}\!\left(
         \boldsymbol{\omega}_{\text{ECI}} \times \mathbf{H}_{\text{ECI}}
       \right),

    which is algebraically equivalent to the standard body-frame expression above.

    The expected quaternion derivative in this general case is computed from the
    same kinematic equation using the randomly drawn :math:`q_0` and
    :math:`\boldsymbol{\omega}`:

    .. math::

       \dot{q}_{\text{exp}}
       \;=\;
       \tfrac{1}{2}
       \begin{bmatrix}
         -\,\mathbf{q}_{0,v}^\top \boldsymbol{\omega} \\
         q_{0,0}\,\boldsymbol{\omega} + \mathbf{q}_{0,v} \times \boldsymbol{\omega}
       \end{bmatrix}.

    **Assertions checked**

    #. For specific :math:`(q,\boldsymbol{\omega})`, the quaternion derivative
       returned by :meth:`Satellite.dynamics_core` matches the closed-form
       kinematics (values :math:`[0,\,0.005,\,0,\,0]^\top` and
       :math:`[0,\,0,\,0,\,-0.005]^\top` respectively), and the angular
       acceleration is zero under the chosen setup.

    #. For random :math:`\mathbf{J}_b,\ q_0,\ \boldsymbol{\omega}`, the concatenated
       derivative :math:`\begin{bmatrix}\dot{\boldsymbol{\omega}}_{\text{exp}}^\top & \dot{q}_{\text{exp}}^\top\end{bmatrix}^\top`
       equals the output of :meth:`Satellite.dynamics_core` within numerical tolerance.

    The orbital environment (:class:`~ADCS.orbits.orbital_state.Orbital_State`) is
    initialized but exerts no external control torque in this test
    (i.e., :math:`\mathbf{u}=\varnothing`).

    """
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
    r"""
    Validate satellite dynamics under **magnetorquer (MTQ)** actuation using
    :meth:`Satellite.dynamics_core`, for both axis-aligned sanity checks and a
    fully general randomized case.

    The state is ordered as
    :math:`\mathbf{x} = \begin{bmatrix}\boldsymbol{\omega}^\top & q_0 & \mathbf{q}_v^\top\end{bmatrix}^\top`
    (scalar-first quaternion). The **body angular rate** is
    :math:`\boldsymbol{\omega}\in\mathbb{R}^3`, and
    :math:`q=\begin{bmatrix}q_0 & \mathbf{q}_v^\top\end{bmatrix}^\top`.

    **Magnetorquer model**

    For a single-axis magnetorquer aligned with body-frame unit vector
    :math:`\mathbf{a}_{\text{axis}}` and commanded dipole magnitude :math:`u`
    (A·m\ :sup:`2`), the **magnetic dipole** in the body frame is

    .. math::

       \mathbf{m}_b \;=\; \mathbf{a}_{\text{axis}}\,u.

    The **magnetic torque** in the body frame is

    .. math::

       \mathbf{T}_b \;=\; -\,\mathbf{B}_b \times \mathbf{m}_b
       \;=\; -\,\mathbf{B}_b \times \big(\mathbf{a}_{\text{axis}}\,u\big),

    where :math:`\mathbf{B}_b` is the geomagnetic field in the body frame.
    With attitude quaternion :math:`q`, the body-to-ECI rotation
    :math:`\mathbf{R}(q)` yields

    .. math::

       \mathbf{B}_b \;=\; \mathbf{R}(q)^\top \mathbf{B}_{\text{ECI}},\qquad
       \mathbf{m}_{\text{ECI}} \;=\; \mathbf{R}(q)\,\mathbf{m}_b,
       \qquad
       \mathbf{T}_{\text{ECI}} \;=\; \mathbf{B}_{\text{ECI}} \times \mathbf{m}_{\text{ECI}}.

    **Rigid-body dynamics & kinematics**

    Let :math:`\mathbf{J}_b` be the body inertia and
    :math:`\mathbf{J}_{\text{ECI}}=\mathbf{R}\mathbf{J}_b\mathbf{R}^\top`.
    Define :math:`\mathbf{H}_b=\mathbf{J}_b\boldsymbol{\omega}`,
    :math:`\boldsymbol{\omega}_{\text{ECI}}=\mathbf{R}\boldsymbol{\omega}`,
    :math:`\mathbf{H}_{\text{ECI}}=\mathbf{J}_{\text{ECI}}\boldsymbol{\omega}_{\text{ECI}}`.
    The test checks the **body-frame angular acceleration** computed via ECI:

    .. math::

       \dot{\boldsymbol{\omega}}_{\text{exp}}
       \;=\;
       -\,\mathbf{R}^\top\,\mathbf{J}_{\text{ECI}}^{-1}
       \!\left(\,
           \boldsymbol{\omega}_{\text{ECI}} \times \mathbf{H}_{\text{ECI}}
           \;+\; \mathbf{T}_{\text{ECI}}
       \right),

    which is equivalent to the classical body-frame form
    :math:`\dot{\boldsymbol{\omega}}=\mathbf{J}_b^{-1}\big(\mathbf{T}_b-\boldsymbol{\omega}\times\mathbf{H}_b\big)`.

    The quaternion kinematics (scalar-first) are

    .. math::

       \dot{q}
       \;=\;
       \tfrac{1}{2}
       \begin{bmatrix}
         -\,\mathbf{q}_v^\top \boldsymbol{\omega} \\
         q_0\,\boldsymbol{\omega} + \mathbf{q}_v \times \boldsymbol{\omega}
       \end{bmatrix}.
    
    **Axis-aligned sanity checks**

    With :math:`q=[1,0,0,0]^\top` so that :math:`\mathbf{R}=\mathbf{I}` and thus
    :math:`\mathbf{B}_b=\mathbf{B}_{\text{ECI}}`, set
    :math:`\mathbf{B}_{\text{ECI}}=10^{-5}[1,0,0]^\top` (Tesla),
    :math:`\boldsymbol{\omega}=[0.01,0,0]^\top` (rad/s), and **unit** inertia by default.

    - **MTQ along** :math:`\hat{\mathbf{x}}`:
      :math:`\mathbf{m}_b=[u,0,0]^\top \Rightarrow \mathbf{T}_b=\mathbf{0}`.
      Dynamics reduce to pure kinematics, yielding
      :math:`\dot{q}=\tfrac12[0,\,0.01,\,0,\,0]^\top=[0,\,0.005,\,0,\,0]^\top`
      and :math:`\dot{\boldsymbol{\omega}}=\mathbf{0}`.

    - **MTQ along** :math:`\hat{\mathbf{y}}`:
      :math:`\mathbf{m}_b=[0,u,0]^\top \Rightarrow \mathbf{T}_b
      = -\,\mathbf{B}_b \times \mathbf{m}_b
      = -\,[1,0,0]^\top \times [0,u,0]^\top
      = [0,0,-u]^\top\,10^{-5}`.
      With :math:`\mathbf{J}_b=\mathbf{I}`, we check
      :math:`\dot{\boldsymbol{\omega}}=[0,0,-10^{-5}]^\top`.

    - **MTQ along** :math:`\hat{\mathbf{z}}`:
      \;analogously
      :math:`\mathbf{T}_b=[0,10^{-5}u,0]^\top`, hence
      :math:`\dot{\boldsymbol{\omega}}=[0,10^{-5},0]^\top`.

    When **multiple MTQs** are present, the net torque is the vector sum of each MTQ's
    contribution, so the checks confirm that selecting a single MTQ via the input
    vector :math:`\mathbf{u}` reproduces the corresponding single-axis result.

    **General randomized case**

    Randomly draw :math:`q_J` and set
    :math:`\mathbf{J}_0=\mathrm{diag}(2,3,10)`, with
    :math:`\mathbf{J}_b=\mathbf{R}(q_J)\mathbf{J}_0\mathbf{R}(q_J)^\top`.
    Draw :math:`q_0,\ \boldsymbol{\omega}_0,\ \mathbf{m}_0,\ \mathbf{B}_{\text{ECI}}` and form

    .. math::

       \mathbf{R}=\mathbf{R}(q_0),\quad
       \boldsymbol{\omega}_{\text{ECI}}=\mathbf{R}\boldsymbol{\omega}_0,\quad
       \mathbf{J}_{\text{ECI}}=\mathbf{R}\mathbf{J}_b\mathbf{R}^\top,\quad
       \mathbf{H}_{\text{ECI}}=\mathbf{J}_{\text{ECI}}\boldsymbol{\omega}_{\text{ECI}},\quad
       \mathbf{m}_{\text{ECI}}=\mathbf{R}\mathbf{m}_0,\quad
       \mathbf{T}_{\text{ECI}}=\mathbf{B}_{\text{ECI}}\times \mathbf{m}_{\text{ECI}}.

    The test constructs and compares

    .. math::

       \dot{\boldsymbol{\omega}}_{\text{exp}}
       \;=\;
       -\,\mathbf{R}^\top\,\mathbf{J}_{\text{ECI}}^{-1}
       \!\left(\,
         \boldsymbol{\omega}_{\text{ECI}} \times \mathbf{H}_{\text{ECI}}
         + \mathbf{T}_{\text{ECI}}
       \right),\qquad
       \dot{q}_{\text{exp}}
       \;=\;
       \tfrac{1}{2}
       \begin{bmatrix}
         -\,\mathbf{q}_{0,v}^\top \boldsymbol{\omega}_0 \\
         q_{0,0}\,\boldsymbol{\omega}_0 + \mathbf{q}_{0,v} \times \boldsymbol{\omega}_0
       \end{bmatrix},

    and verifies

    .. math::

       \begin{bmatrix}
         \dot{\boldsymbol{\omega}}_{\text{exp}} \\[2pt]
         \dot{q}_{\text{exp}}
       \end{bmatrix}
       \;\approx\;
       \mathrm{dynamics\_core}\!\left(\mathbf{x}=\begin{bmatrix}\boldsymbol{\omega}_0\\ q_0\end{bmatrix},
                                       \ \mathbf{u}=\mathbf{m}_0,\ \text{Orbital\_State}\right),

    within numerical tolerance.

    The initial simple cases also re-validate that with :math:`q=[1,0,0,0]^\top` and
    :math:`\boldsymbol{\omega}=[0.01,0,0]^\top`, the quaternion derivative satisfies
    :math:`\dot{q}=[0,\,0.005,\,0,\,0]^\top` in all MTQ configurations above.
    """
    mtqs = [MTQ(axis=j, max_moment=1, bias=Bias()) for j in MathConstants.unitvecs]
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


def test_dynamics_RW():
    max_torques = [0.01, 0.05, 0.02]
    rwj = [0.001, 0.002, 0.5]
    h0 = [0.1, 0.0, 0.0]
    maxh = [0.1, 0.1, 0.1]

    e_bias = [-0.001, 0.05, 0.0]

    acts = [RW(axis=MathConstants.unitvecs[j], max_torque=max_torques[j], J=rwj[j], h=h0[j], h_max=maxh[j], bias=Bias(bias=e_bias[j], std_bias=0.3)) for j in range(3)]
    sat = Satellite(actuators=acts)
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]))

    x = np.concatenate([0.01*MathConstants.unitvecs[0], MathConstants.zeroquat, h0])
    u = np.array([0.021, -0.05, 0.0])
    xd = sat.dynamics_core(x=x, u=u, orbital_state=os)

    assert np.allclose(np.array([0.02/(1-0.001),0,0,0,0.005,0,0,-0.02/(1-0.001),0,0]), xd,rtol = 1e-8,atol=1e-8)



def test_srp():
    faces = [
        GeometryFace(
            area=0.1,
            centroid=np.array([1.0, 0.2, 0.0]),
            normal=np.array([1, 0, 0]),
            eta_s=0.0, eta_d=0.5, eta_a=0.5,
            CD=2
        ),
        GeometryFace(
            area=0.03,
            centroid=np.array([-0.05, 0.1, 0.3]),
            normal=np.array([0, 1, 0]),
            eta_s=0.1, eta_d=0.2, eta_a=0.1,
            CD=0.1
        ),
        GeometryFace(
            area=10.0,
            centroid=np.array([0.25, -0.01, -0.7]),
            normal=np.array([0, 0, 1]),
            eta_s=0.3, eta_d=0.1, eta_a=0.6,
            CD=0.3
        ),
    ]
    config = GeometryConfig(geometry_faces=faces)
    dist = [SRP_Disturbance(config=config)]
    sat = Satellite(disturbances=dist)

    assert np.all(sat.disturbances[0].eta_s == [0.0,0.1,0.3])
    assert np.all(sat.disturbances[0].eta_d == [0.5,0.2,0.1])
    assert np.all(sat.disturbances[0].eta_a == [0.5,0.1,0.6])
    assert np.all(sat.disturbances[0].areas == [0.1,0.03,10])
    assert np.all([
      np.allclose(sat.disturbances[0].normals[j], MathConstants.unitvecs[j])
      for j in range(3)
    ])

    assert np.all([
      np.allclose(sat.disturbances[0].centroids[j], faces[j].centroid)
      for j in range(3)
    ])
    
def test_drag():
    faces = [
        GeometryFace(
            area=0.1,
            centroid=np.array([1.0, 0.2, 0.0]),
            normal=np.array([1, 0, 0]),
            eta_s=0.0, eta_d=0.5, eta_a=0.5,
            CD=2
        ),
        GeometryFace(
            area=0.03,
            centroid=np.array([-0.05, 0.1, 0.3]),
            normal=np.array([0, 1, 0]),
            eta_s=0.1, eta_d=0.2, eta_a=0.1,
            CD=0.1
        ),
        GeometryFace(
            area=10.0,
            centroid=np.array([0.25, -0.01, -0.7]),
            normal=np.array([0, 0, 1]),
            eta_s=0.3, eta_d=0.1, eta_a=0.6,
            CD=0.3
        ),
    ]

    config = GeometryConfig(geometry_faces=faces)
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

    ephem = Ephemeris()
    t = 0.22
    R = np.array([-0.001, 0.05, 0])
    V = np.array([0, 8, 0])
    B = np.array([1, 0, 0])*1e-5
    os = Orbital_State(ephem=ephem, J2000=t, R=R, V=V, B=B)

    w0 = random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    x = np.concatenate([w0, q0])

    assert np.all(sat.disturbances[0].torque(x=x, os=os) == np.array([1,2,4]))

def test_resdipole():
    nominal_dipole = np.array([0.1, -0.1, 0.5])
    noise = Noise()
    dist = [Dipole_Disturbance(nominal_dipole, noise)]
    sat = Satellite(disturbances=dist)

    assert np.all(sat.disturbances[0].torque_nominal == np.array([0.1,-0.1,0.5]))
    assert np.all(sat.disturbances[0].noise.std_noise == 0.0)

if __name__ == "__main__":
    test_J_with_RW()





