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
    r"""
    Validate **magnetorquer (MTQ)** torque generation in both axis-aligned and
    general settings, and verify that :meth:`Satellite.dynamics_core` matches
    the expected rigid-body dynamics with magnetic torque included.

    **State convention**

    The state is
    :math:`\mathbf{x}=\begin{bmatrix}\boldsymbol{\omega}^\top & q_0 & \mathbf{q}_v^\top\end{bmatrix}^\top`,
    where the quaternion is scalar-first :math:`q=[q_0,\ \mathbf{q}_v^\top]^\top`.

    **Magnetorquer model**

    For a single-axis MTQ aligned with body-frame unit axis
    :math:`\mathbf{a}_{\text{axis}}` and commanded dipole magnitude :math:`u`
    (A·m\ :sup:`2`), the **body-frame dipole** is

    .. math::

       \mathbf{m}_b = \mathbf{a}_{\text{axis}}\,u.

    The **magnetic torque** in the body frame is modeled as

    .. math::

       \mathbf{T}_b
       \;=\;
       -\,\mathbf{B}_b \times \mathbf{m}_b
       \;=\;
       -\,\mathbf{B}_b \times \big(\mathbf{a}_{\text{axis}}\,u\big),

    where :math:`\mathbf{B}_b` is the geomagnetic field expressed in the body
    frame. With body-to-ECI rotation :math:`\mathbf{R}(q)`,

    .. math::

       \mathbf{B}_b = \mathbf{R}(q)^\top \mathbf{B}_{\text{ECI}},
       \qquad
       \mathbf{m}_{\text{ECI}} = \mathbf{R}(q)\,\mathbf{m}_b,
       \qquad
       \mathbf{T}_{\text{ECI}} = \mathbf{B}_{\text{ECI}} \times \mathbf{m}_{\text{ECI}}.

    **What this test checks**

    1) **Axis-aligned sanity checks (body aligned with ECI).**

       With :math:`q=[1,0,0,0]^\top \Rightarrow \mathbf{R}=\mathbf{I}` and
       :math:`\mathbf{B}_{\text{ECI}} = 10^{-5}[1,\,0,\,0]^\top` (Tesla):

       - For MTQ axis :math:`\hat{\mathbf{x}}` and inputs :math:`u\in\{0,1\}`,

         .. math::
            \mathbf{m}_b = [u,0,0]^\top,
            \qquad
            \mathbf{T}_b = -\,\mathbf{B}_b \times \mathbf{m}_b = \mathbf{0}.

         The assertions verify
         :math:`\mathbf{T}_b=\mathbf{0}` for :math:`u=0` and for :math:`u=1`.

       - For MTQ axis :math:`\hat{\mathbf{y}}`,

         .. math::
            \mathbf{m}_b=[0,u,0]^\top,
            \qquad
            \mathbf{T}_b
            = -\,[1,0,0]^\top \times [0,u,0]^\top \cdot 10^{-5}
            = [0,0,-u]^\top 10^{-5}.

         The test checks that :math:`\mathbf{T}_b` equals that value when
         :math:`u=1`.

       - For MTQ axis :math:`\hat{\mathbf{z}}`,

         .. math::
            \mathbf{m}_b=[0,0,u]^\top,
            \qquad
            \mathbf{T}_b
            = -\,[1,0,0]^\top \times [0,0,u]^\top \cdot 10^{-5}
            = [0,\,u,\,0]^\top 10^{-5},

         verified in the assertions for :math:`u=1`.

       These are also re-checked when **all three MTQs** are present in the
       satellite and the input vector selects one axis at a time (superposition).

    2) **General randomized dynamics with magnetic torque.**

       Randomly draw an attitude :math:`q_J` and form a rotated body inertia

       .. math::
          \mathbf{J}_0 = \mathrm{diag}(2,3,10), \qquad
          \mathbf{J}_b = \mathbf{R}(q_J)\,\mathbf{J}_0\,\mathbf{R}(q_J)^\top.

       Draw :math:`q_0,\,\boldsymbol{\omega}_0,\,\mathbf{m}_0,\,\mathbf{B}_{\text{ECI}}`
       and define

       .. math::
          \mathbf{R}=\mathbf{R}(q_0),\quad
          \boldsymbol{\omega}_{\text{ECI}}=\mathbf{R}\boldsymbol{\omega}_0,\quad
          \mathbf{J}_{\text{ECI}}=\mathbf{R}\mathbf{J}_b\mathbf{R}^\top,\\
          \mathbf{H}_{\text{ECI}}=\mathbf{J}_{\text{ECI}}\boldsymbol{\omega}_{\text{ECI}},\quad
          \mathbf{m}_{\text{ECI}}=\mathbf{R}\mathbf{m}_0,\quad
          \mathbf{T}_{\text{ECI}}=\mathbf{B}_{\text{ECI}}\times \mathbf{m}_{\text{ECI}}.

       The **expected body angular acceleration** is constructed via ECI:

       .. math::
          \dot{\boldsymbol{\omega}}_{\text{exp}}
          \;=\;
          -\,\mathbf{R}^\top\,\mathbf{J}_{\text{ECI}}^{-1}
          \!\left(\,
              \boldsymbol{\omega}_{\text{ECI}} \times \mathbf{H}_{\text{ECI}}
              + \mathbf{T}_{\text{ECI}}
          \right),

       which is equivalent to the standard body-frame Euler equation
       :math:`\dot{\boldsymbol{\omega}}=\mathbf{J}_b^{-1}\big(\mathbf{T}_b-\boldsymbol{\omega}\times\mathbf{H}_b\big)`.

       The **quaternion kinematics** (scalar-first) used for the expected value are

       .. math::
          \dot{q}_{\text{exp}}
          \;=\;
          \tfrac{1}{2}
          \begin{bmatrix}
            -\,\mathbf{q}_{0,v}^\top \boldsymbol{\omega}_0 \\
            q_{0,0}\,\boldsymbol{\omega}_0 + \mathbf{q}_{0,v} \times \boldsymbol{\omega}_0
          \end{bmatrix}.

       The test concatenates
       :math:`\begin{bmatrix}\dot{\boldsymbol{\omega}}_{\text{exp}}^\top & \dot{q}_{\text{exp}}^\top\end{bmatrix}^\top`
       and asserts numerical equality (within tolerance) to the output of
       :meth:`Satellite.dynamics_core(\mathbf{x}=[\boldsymbol{\omega}_0;q_0],\ \mathbf{u}=\mathbf{m}_0)`.

    3) **Body-frame torque from ECI quantities for direct MTQ calls.**

       For the same random :math:`q_0` and field :math:`\mathbf{B}_{\text{ECI}}`,
       the test computes

       .. math::
          \mathbf{B}_b = \mathbf{R}(q_0)^\top \mathbf{B}_{\text{ECI}},\qquad
          \mathbf{T}_b^{(i)} = \mathbf{a}_{\text{axis}}^{(i)} \times \mathbf{B}_b
          \quad\text{(with sign convention used in the implementation)},

       and checks that direct calls
       :math:`\mathrm{MTQ}_i.\mathrm{torque}(u=1,\ \mathbf{x},\ \text{Orbital\_State})`
       match these closed-form values for each actuator axis
       :math:`i\in\{x,y,z\}`.

    The collection of assertions therefore validates:
    (a) the MTQ torque direction/magnitude in simple axis-aligned cases,
    (b) the coupling of magnetic torque into rigid-body dynamics, and
    (c) consistency between body/ECI transformations and the MTQ's torque API.
    """
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
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)
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
    r"""
    Verify correct initialization of the :class:`MTQ` actuator and its
    associated :class:`Bias` and :class:`Noise` models.

    **What is being checked**

    1) **Axis normalization**

       The MTQ stores a *unit* axis. If the constructor is called with
       :math:`\mathbf{a} \in \mathbb{R}^3`, then

       .. math::

          \widehat{\mathbf{a}}
          \;=\;
          \frac{\mathbf{a}}{\lVert \mathbf{a} \rVert_2}.

       In this test, :math:`\mathbf{a}` is chosen so that
       :math:`\lVert \mathbf{a} \rVert_2 = 3`, hence
       :math:`\widehat{\mathbf{a}} = \mathbf{a}/3`. The assertion

       .. code-block:: python

          assert np.all(np.isclose(ax/3, mtq.axis))

       confirms that the internal axis is normalized as expected.

    2) **Bias object passthrough**

       The constructor receives a :class:`Bias` instance with parameters
       :math:`\mathbf{b}` (bias) and :math:`\sigma_b` (bias standard deviation).
       The MTQ should keep a reference to this object, not copy it, so that

       .. math::

          \text{mtq.bias.bias} = \mathbf{b}, \qquad
          \text{mtq.bias.std\_bias} = \sigma_b.

       The test asserts identity/equality:

       .. code-block:: python

          assert np.all(bias == mtq.bias)
          assert mtq.bias.bias == e_bias
          assert mtq.bias.std_bias == bsr

    3) **Noise object passthrough**

       Similarly, the :class:`Noise` instance with mean :math:`\mu_n` (here 0)
       and standard deviation :math:`\sigma_n` is stored directly:

       .. math::

          \text{mtq.noise.noise} = \mu_n, \qquad
          \text{mtq.noise.std\_noise} = \sigma_n.

       Verified by:

       .. code-block:: python

          assert mtq.noise.noise == 0
          assert mtq.noise.std_noise == std_noise

    4) **Maximum torque parameter**

       The MTQ enforces a maximum commanded magnitude
       :math:`u_{\max} \in \mathbb{R}_{\ge 0}`. The constructor argument
       is passed through to the field ``u_max`` unchanged:

       .. code-block:: python

          assert mtq.u_max == max_torque

    5) **Estimator flag**

       Although not asserted here, the ``estimate_bias`` flag is provided to the
       constructor and should be stored for later use by estimation routines.

    Together, these checks confirm that the MTQ constructor:
    (i) normalizes the axis direction,
    (ii) retains the provided :class:`Bias` and :class:`Noise` models intact,
    and (iii) preserves scalar configuration like ``u_max`` (maximum torque).
    """
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
    expected_Jxfun = mtq.dtorq__dbasestate(u=m0, x=x0, os=os)
    Jufun = np.array(nd.Jacobian(ufun)(m0)).T
    expected_Jufun = mtq.dtorq__du(u=m0, x=x0, os=os)
    Jbfun = np.array(nd.Jacobian(bfun)(20000)).T
    expected_Jbfun = mtq.dtorq__dbias(u=m0, x=x0, os=os)
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T
    expected_Jhfun = mtq.dtorq__dh(u=m0, x=x0, os=os)

    assert np.allclose(Jxfun, expected_Jxfun)
    assert np.allclose(Jufun, expected_Jufun)
    assert np.allclose(Jbfun, expected_Jbfun)
    assert np.allclose(Jhfun, expected_Jhfun)


    for j in MathConstants.unitvecs:
        fun_hj = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque).torque(u=c[0], x=np.array([c[1],c[2],c[3],c[4],c[5],c[6],c[7]]), os=os), j)

        ufunjju = lambda c: np.dot(mtq.dtorq__du(u=c, x=x0, os=os), j).item()
        ufunjjb = lambda c: np.dot(mtq.dtorq__dbias(u=c, x=x0, os=os),j)
        ufunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(u=c, x=x0, os=os),j)
        ufunjjh = lambda c: np.dot(mtq.dtorq__dh(u=c, x=x0, os=os),j)

        xfunjju = lambda c: np.dot(mtq.dtorq__du(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j).item()
        xfunjjb = lambda c: np.dot(mtq.dtorq__dbias(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)
        xfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)
        xfunjjh = lambda c: np.dot(mtq.dtorq__dh(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)

        hfunjju = lambda c: np.dot(mtq.dtorq__du(u=m0, x=x0, os=os),j).item()
        hfunjjb = lambda c: np.dot(mtq.dtorq__dbias(u=m0, x=x0, os=os),j)
        hfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(u=m0, x=x0, os=os),j)
        hfunjjh = lambda c: np.dot(mtq.dtorq__dh(u=m0, x=x0, os=os),j)

        bfunjju = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque).dtorq__du(u=m0, x=x0, os=os), j).item()
        bfunjjx = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque).dtorq__dbasestate(u=m0, x=x0, os=os), j)

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))

        assert np.allclose(Jxfunjju, np.dot(mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os), j))
        assert np.allclose(Jxfunjjx, np.dot(mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os), j))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(m0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(m0))

        assert np.allclose(Jufunjju, np.dot(mtq.ddtorq__dudu(u=m0, x=x0, os=os), j))
        assert np.allclose(Jufunjjx.T, np.dot(mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os), j))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))

        assert np.allclose(Jbfunjju, np.dot(mtq.ddtorq__dudbias(u=m0, x=x0, os=os), j))
        assert np.allclose(Jbfunjjx.T, np.dot(mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os), j))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))

        assert np.allclose(Jhfunjju, np.dot(mtq.ddtorq__dudh(u=m0, x=x0, os=os), j))
        assert np.allclose(Jhfunjjx, np.dot(mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os), j))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[m0], x0, [500.2]]).flatten().tolist()))
        Hguess = np.block([
            [
                mtq.ddtorq__dudu(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dudbias(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dudh(u=m0, x=x0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudbias(u=m0, x=x0, os=os) @ j).T,
                mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os) @ j).T,
                (mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os) @ j).T,
                mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudh(u=m0, x=x0, os=os) @ j).T,
                (mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os) @ j).T,
                (mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os) @ j).T,
                mtq.ddtorq__dhdh(u=m0, x=x0, os=os) @ j
            ]
        ])

        assert np.allclose(Hfun[8,:],0)
        assert np.allclose(Hfun[:,8],0)
        assert np.allclose(Hfun[0:8, 0:8], Hguess)

    # Torque and first-order derivatives
    assert np.all(np.isclose(mtq.dtorq__du(u=m0, x=x0, os=os), np.cross(ax/3, B_B)))
    assert np.all(np.isclose(mtq.dtorq__dbias(u=m0, x=x0, os=os), np.cross(ax/3, B_B)))
    assert np.all(
        np.isclose(
            mtq.dtorq__dbasestate(u=m0, x=x0, os=os),
            np.vstack([np.zeros((3, 3)), np.cross(ax/3 * (m0), drotmatTvecdq(q=q0, v=B_ECI))]),
        )
    )

    # Derivative wrt h (should be zero for MTQ)
    assert np.all(np.isclose(mtq.dtorq__dh(u=m0, x=x0, os=os), np.zeros((0, 3))))
    assert mtq.dtorq__dh(u=m0, x=x0, os=os).shape == (0, 3)

    # Second-order derivatives (Hessians)
    assert np.all(np.isclose(mtq.ddtorq__dudu(u=m0, x=x0, os=os), np.zeros((1, 1, 3))))
    assert mtq.ddtorq__dudu(u=m0, x=x0, os=os).shape == (1, 1, 3)

    assert np.all(np.isclose(mtq.ddtorq__dudbias(u=m0, x=x0, os=os), np.zeros((1, 0, 3))))
    assert mtq.ddtorq__dudbias(u=m0, x=x0, os=os).shape == (1, 0, 3)

    assert np.all(
        np.isclose(
            mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os),
            np.expand_dims(np.vstack([np.zeros((3, 3)), np.cross(ax/3, drotmatTvecdq(q=q0, v=B_ECI))]), 0),
        )
    )
    assert mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os).shape == (1, 7, 3)

    assert np.all(np.isclose(mtq.ddtorq__dudh(u=m0, x=x0, os=os), np.zeros((1, 0, 3))))
    assert mtq.ddtorq__dudh(u=m0, x=x0, os=os).shape == (1, 0, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os), np.zeros((0, 0, 3))))
    assert mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os).shape == (0, 0, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os), np.zeros((0, 7, 3))))
    assert mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os).shape == (0, 7, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os), np.zeros((0, 0, 3))))
    assert mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os).shape == (0, 0, 3)

    # Base state Hessian
    dxdx = np.zeros((7, 7, 3))
    dxdx[3:7, 3:7, :] = np.cross(ax/3 * (m0), ddrotmatTvecdqdq(x0, B_ECI))
    assert np.all(np.isclose(mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os), dxdx))
    assert mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os).shape == (7, 7, 3)

    assert np.all(np.isclose(mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os), np.zeros((7, 0, 3))))
    assert mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os).shape == (7, 0, 3)

    assert np.all(np.isclose(mtq.ddtorq__dhdh(u=m0, x=x0, os=os), np.zeros((0, 0, 3))))
    assert mtq.ddtorq__dhdh(u=m0, x=x0, os=os).shape == (0, 0, 3)

    # Momentum storage torque (MTQ has none)
    assert np.all(mtq.storage_torque(u=m0, j2000=os.J2000) == np.zeros(0))
    assert mtq.storage_torque(u=m0, j2000=os.J2000).shape == (0,)

    # First-order derivatives of storage torque
    assert np.all(np.isclose(mtq.dstor_torq__du(u=m0, x=x0, os=os), np.zeros((1, 0))))
    assert mtq.dstor_torq__du(u=m0, x=x0, os=os).shape == (1, 0)

    assert np.all(np.isclose(mtq.dstor_torq__dbias(u=m0, x=x0, os=os), np.zeros((0, 0))))
    assert mtq.dstor_torq__dbias(u=m0, x=x0, os=os).shape == (0, 0)

    assert np.all(np.isclose(mtq.dstor_torq__dbasestate(u=m0, x=x0, os=os), np.zeros((7, 0))))
    assert mtq.dstor_torq__dbasestate(u=m0, x=x0, os=os).shape == (7, 0)

    assert np.all(np.isclose(mtq.dstor_torq__dh(u=m0, x=x0, os=os), np.zeros((0, 0))))
    assert mtq.dstor_torq__dh(u=m0, x=x0, os=os).shape == (0, 0)

    # Second-order derivatives of storage torque
    assert np.all(np.isclose(mtq.ddstor_torq__dudu(u=m0, x=x0, os=os), np.zeros((1, 1, 0))))
    assert mtq.ddstor_torq__dudu(u=m0, x=x0, os=os).shape == (1, 1, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dudbias(u=m0, x=x0, os=os), np.zeros((1, 0, 0))))
    assert mtq.ddstor_torq__dudbias(u=m0, x=x0, os=os).shape == (1, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dudbasestate(u=m0, x=x0, os=os), np.zeros((1, 7, 0))))
    assert mtq.ddstor_torq__dudbasestate(u=m0, x=x0, os=os).shape == (1, 7, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dudh(u=m0, x=x0, os=os), np.zeros((1, 0, 0))))
    assert mtq.ddstor_torq__dudh(u=m0, x=x0, os=os).shape == (1, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbiasdbias(u=m0, x=x0, os=os), np.zeros((0, 0, 0))))
    assert mtq.ddstor_torq__dbiasdbias(u=m0, x=x0, os=os).shape == (0, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbiasdbasestate(u=m0, x=x0, os=os), np.zeros((0, 7, 0))))
    assert mtq.ddstor_torq__dbiasdbasestate(u=m0, x=x0, os=os).shape == (0, 7, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbiasdh(u=m0, x=x0, os=os), np.zeros((0, 0, 0))))
    assert mtq.ddstor_torq__dbiasdh(u=m0, x=x0, os=os).shape == (0, 0, 0)

    dxdx = np.zeros((7, 7, 0))
    assert np.all(np.isclose(mtq.ddstor_torq__dbasestatedbasestate(u=m0, x=x0, os=os), dxdx))
    assert mtq.ddstor_torq__dbasestatedbasestate(u=m0, x=x0, os=os).shape == (7, 7, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dbasestatedh(u=m0, x=x0, os=os), np.zeros((7, 0, 0))))
    assert mtq.ddstor_torq__dbasestatedh(u=m0, x=x0, os=os).shape == (7, 0, 0)

    assert np.all(np.isclose(mtq.ddstor_torq__dhdh(u=m0, x=x0, os=os), np.zeros((0, 0, 0))))
    assert mtq.ddstor_torq__dhdh(u=m0, x=x0, os=os).shape == (0, 0, 0)


def test_MTQ_torque_bias():
    ax = random_n_unit_vec(3)*3
    ax = ax.copy()
    max_torque = 4.51

    biasv = random_n_unit_vec(3)[1]*0.1
    biast = biasv.copy()
    std_bias=0.03

    bias = Bias(bias=biasv, std_bias=std_bias)
    mtq = MTQ(axis=ax, max_torque=max_torque, bias=bias)
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

    expected_torque = np.cross(ax/3*(m0 + biast), B_B)
    actual_torque = mtq.torque(u=m0, x=x0, os=os)
    assert np.all(np.isclose(expected_torque, actual_torque))

    vecsxfun = lambda c: {"b":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@B,"r":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@R,"s":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@S,"v":rot_mat(np.array([c[3],c[4],c[5],c[6]])).T@V,"rho":rho,"db":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),B),"ds":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),S),"dv":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),V),"dr":drotmatTvecdq(np.array([c[3],c[4],c[5],c[6]]),R),\
                "ddb":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),B),"dds":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),S),"ddv":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),V),"ddr":ddrotmatTvecdqdq(np.array([c[3],c[4],c[5],c[6]]),R)}
    
    ufun = lambda c: mtq.torque(u=c, x=x0, os=os)
    xfun = lambda c: mtq.torque(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
    hfun = lambda c: mtq.torque(u=m0, x=x0, os=os)
    bfun = lambda c: MTQ(axis=ax, max_torque=max_torque, bias=Bias(bias=c, std_bias=std_bias)).torque(u=m0, x=x0, os=os)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    expected_Jxfun = mtq.dtorq__dbasestate(u=m0, x=x0, os=os)
    Jufun = np.array(nd.Jacobian(ufun)(m0)).T
    expected_Jufun = mtq.dtorq__du(u=m0, x=x0, os=os)
    Jbfun = np.array(nd.Jacobian(bfun)(biast)).T
    expected_Jbfun = mtq.dtorq__dbias(u=m0, x=x0, os=os)
    Jhfun = np.array(nd.Jacobian(hfun)(500.2)).T
    expected_Jhfun = mtq.dtorq__dh(u=m0, x=x0, os=os)

    assert np.allclose(Jxfun, expected_Jxfun)
    assert np.allclose(Jufun, expected_Jufun)
    assert np.allclose(Jbfun, expected_Jbfun)
    assert np.allclose(Jhfun, expected_Jhfun)

    for j in MathConstants.unitvecs:
        fun_hj = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque, bias=Bias(bias=c[1], std_bias=std_bias)).torque(u=c[0], x=np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]), os=os), j)

        ufunjju = lambda c: np.dot(mtq.dtorq__du(u=c, x=x0, os=os), j).item()
        ufunjjb = lambda c: np.dot(mtq.dtorq__dbias(u=c, x=x0, os=os),j)
        ufunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(u=c, x=x0, os=os),j)
        ufunjjh = lambda c: np.dot(mtq.dtorq__dh(u=c, x=x0, os=os),j)

        xfunjju = lambda c: np.dot(mtq.dtorq__du(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j).item()
        xfunjjb = lambda c: np.dot(mtq.dtorq__dbias(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j).item()
        xfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)
        xfunjjh = lambda c: np.dot(mtq.dtorq__dh(u=m0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os),j)

        hfunjju = lambda c: np.dot(mtq.dtorq__du(u=m0, x=x0, os=os),j).item()
        hfunjjb = lambda c: np.dot(mtq.dtorq__dbias(u=m0, x=x0, os=os),j)
        hfunjjx = lambda c: np.dot(mtq.dtorq__dbasestate(u=m0, x=x0, os=os),j)
        hfunjjh = lambda c: np.dot(mtq.dtorq__dh(u=m0, x=x0, os=os),j)

        bfunjju = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque, bias=Bias(bias=c, std_bias=std_bias)).dtorq__du(u=m0, x=x0, os=os), j).item()
        bfunjjb = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque, bias=Bias(bias=c, std_bias=std_bias)).dtorq__dbias(u=m0, x=x0, os=os), j).item()
        bfunjjx = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque, bias=Bias(bias=c, std_bias=std_bias)).dtorq__dbasestate(u=m0, x=x0, os=os), j)
        bfunjjh = lambda c: np.dot(MTQ(axis=ax, max_torque=max_torque, bias=Bias(bias=c, std_bias=std_bias)).dtorq__dh(u=m0, x=x0, os=os), j)

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjb = np.array(nd.Jacobian(xfunjjb)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))

        assert np.allclose(Jxfunjju, np.dot(mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os), j))
        assert np.allclose(Jxfunjjb, np.dot(mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os), j))
        assert np.allclose(Jxfunjjx, np.dot(mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os), j))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(m0))
        Jufunjjb = np.array(nd.Jacobian(ufunjjb)(m0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(m0))

        assert np.allclose(Jufunjju, np.dot(mtq.ddtorq__dudu(u=m0, x=x0, os=os), j))
        assert np.allclose(Jufunjjb, np.dot(mtq.ddtorq__dudbias(u=m0, x=x0, os=os), j))
        assert np.allclose(Jufunjjx.T, np.dot(mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os), j))

        Jbfunjju = np.array(nd.Jacobian(bfunjju)(biast))
        Jbfunjjb = np.array(nd.Jacobian(bfunjjb)(biast))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(biast))

        assert np.allclose(Jbfunjju, np.dot(mtq.ddtorq__dudbias(u=m0, x=x0, os=os), j))
        assert np.allclose(Jbfunjju, np.dot(mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os), j))
        assert np.allclose(Jbfunjjx.T, np.dot(mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os), j))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(500.2))
        Jhfunjjb = np.array(nd.Jacobian(hfunjjb)(500.2))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(500.2))

        assert np.allclose(Jhfunjju, np.dot(mtq.ddtorq__dudh(u=m0, x=x0, os=os), j))
        assert np.allclose(Jhfunjjb, np.dot(mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os), j))
        assert np.allclose(Jhfunjjx, np.dot(mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os), j))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[m0], [biast], x0, [500.2]]).flatten().tolist()))
        Hguess = np.block([
            [
                mtq.ddtorq__dudu(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dudbias(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dudh(u=m0, x=x0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudbias(u=m0, x=x0, os=os) @ j).T,
                mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os) @ j).T,
                (mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os) @ j).T,
                mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os) @ j,
                mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os) @ j
            ],
            [
                (mtq.ddtorq__dudh(u=m0, x=x0, os=os) @ j).T,
                (mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os) @ j).T,
                (mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os) @ j).T,
                mtq.ddtorq__dhdh(u=m0, x=x0, os=os) @ j
            ]
        ])


        assert np.allclose(Hfun[0:9,0:9],Hguess)
        assert np.allclose(Hfun[0:9,9],0)
        assert np.allclose(Hfun[9,0:9],0)
        assert np.allclose(Hfun[9,9],0)



if __name__ == "__main__":
    test_MTQ_torque_bias()
