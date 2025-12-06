import sys
import os
import numpy as np
import numdifftools as nd
import pytest
from typing import List
from scipy.stats import kstest, ks_2samp
from asciichartpy import plot

# === Import project modules ===
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import Actuator, RW, MTQ, Noise, Bias
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.helpers.math_helpers import normalize, rot_mat, random_n_unit_vec, drotmatTvecdq, ddrotmatTvecdqdq
from ADCS.helpers.math_constants import MathConstants


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
    r"""
    Validate the magnetorquer (MTQ) torque model and all provided analytic
    first- and second-order derivatives against finite-difference baselines.

    **Physical model**

    The MTQ produces a magnetic dipole moment ``m`` aligned with a fixed body-axis
    unit vector ``a`` and scaled by a scalar command ``u`` (dimensionless):

    .. math::

       \mathbf{m}(u) = u\,\mathbf{a}, \qquad \|\mathbf{a}\|=1.

    The geomagnetic field expressed in the body frame is

    .. math::

       \mathbf{B}_\mathcal{B}(\mathbf{q}) \;=\; \mathbf{C}(\mathbf{q})^\top \,\mathbf{B}_\text{ECI},

    where :math:`\mathbf{q}` is the body attitude quaternion and
    :math:`\mathbf{C}(\mathbf{q})\in\mathbb{R}^{3\times 3}` is the body-to-ECI
    direction-cosine matrix (so :math:`\mathbf{C}^\top` maps ECI to body).
    The MTQ torque in the body frame is

    .. math::

       \boldsymbol{\tau}(u,\mathbf{q})
       \;=\;
       \mathbf{m}(u) \times \mathbf{B}_\mathcal{B}(\mathbf{q})
       \;=\;
       u\,\mathbf{a} \times \big(\mathbf{C}(\mathbf{q})^\top \mathbf{B}_\text{ECI}\big).

    In the test, the axis vector ``ax`` is generated with norm 3 and then normalized
    via ``a = ax/3`` so that :math:`\mathbf{a}` is unit. The base state is
    :math:`\mathbf{x}=[\,\boldsymbol{\omega}^\top,\ \mathbf{q}^\top\,]^\top\in\mathbb{R}^7`
    (body rates :math:`\boldsymbol{\omega}` and quaternion :math:`\mathbf{q}`), and the
    command is a scalar :math:`u\in\mathbb{R}`.

    **Derivatives used by the test**

    Let :math:`D_\mathbf{q}\mathbf{B}_\mathcal{B}(\mathbf{q})\in\mathbb{R}^{4\times 3}` denote the
    Jacobian of :math:`\mathbf{B}_\mathcal{B}` w.r.t. the four quaternion components, arranged
    row-wise (the helper ``drotmatTvecdq(q, v)`` returns exactly this object for a given
    :math:`\mathbf{v}`, here :math:`\mathbf{B}_\text{ECI}`). Let
    :math:`D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_\mathcal{B}(\mathbf{q})\in\mathbb{R}^{4\times 4\times 3}`
    be the corresponding Hessian (from ``ddrotmatTvecdqdq``). Define also the skew matrix
    :math:`\mathbf{a}_\times` such that :math:`\mathbf{a}_\times\mathbf{v}=\mathbf{a}\times\mathbf{v}`.

    * First derivatives:

      - W.r.t. the input :math:`u`:

        .. math:: \frac{\partial\boldsymbol{\tau}}{\partial u}
                  \;=\; \mathbf{a} \times \mathbf{B}_\mathcal{B}(\mathbf{q})
                  \;=\; \mathbf{a}_\times\,\mathbf{B}_\mathcal{B}(\mathbf{q})
                  \;\in\; \mathbb{R}^{1\times 3}.

      - W.r.t. the base state :math:`\mathbf{x}=[\boldsymbol{\omega},\mathbf{q}]`:

        .. math::
           \frac{\partial\boldsymbol{\tau}}{\partial \boldsymbol{\omega}}
           \;=\; \mathbf{0}_{3\times 3},\qquad
           \frac{\partial\boldsymbol{\tau}}{\partial \mathbf{q}}
           \;=\; \big[\, (u\,\mathbf{a})\times \partial_{\!q_i}\mathbf{B}_\mathcal{B}(\mathbf{q}) \,\big]_{i=0}^3
           \;\in\; \mathbb{R}^{3\times 4}.

        The test stacks these to obtain :math:`\partial\boldsymbol{\tau}/\partial\mathbf{x}\in\mathbb{R}^{3\times 7}`.

      - W.r.t. bias and storage state:

        The MTQ model under test has no additive bias and no momentum storage states, so

        .. math::
           \frac{\partial\boldsymbol{\tau}}{\partial \text{bias}}=\mathbf{0}_{0\times 3},\qquad
           \frac{\partial\boldsymbol{\tau}}{\partial \mathbf{h}}=\mathbf{0}_{0\times 3}.
        
        (Shapes with zero leading dimension are preserved by the implementation; comparisons
        to nonzero vectors are vacuously true under NumPy's broadcasting and are followed by
        explicit shape checks.)

    * Second derivatives (Hessians of :math:`\boldsymbol{\tau}` componentwise):

      - Pure input:

        .. math:: \frac{\partial^2\boldsymbol{\tau}}{\partial u^2}=\mathbf{0}_{1\times 1\times 3}.

      - Mixed input–state:

        .. math::
           \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial \boldsymbol{\omega}}
           \;=\; \mathbf{0}_{1\times 3\times 3},\qquad
           \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial \mathbf{q}}
           \;=\; \big[\, \mathbf{a}\times \partial_{\!q_i}\mathbf{B}_\mathcal{B}(\mathbf{q}) \,\big]_{i=0}^3
           \;\in\; \mathbb{R}^{1\times 4\times 3}.

      - Pure state:

        .. math::
           \frac{\partial^2\boldsymbol{\tau}}{\partial \boldsymbol{\omega}\,\partial \boldsymbol{\omega}}
           \;=\; \mathbf{0}_{3\times 3\times 3},\qquad
           \frac{\partial^2\boldsymbol{\tau}}{\partial \boldsymbol{\omega}\,\partial \mathbf{q}}
           \;=\; \mathbf{0}_{3\times 4\times 3},\\[2pt]
           \frac{\partial^2\boldsymbol{\tau}}{\partial \mathbf{q}\,\partial \mathbf{q}}
           \;=\; \big[\, (u\,\mathbf{a})\times \partial^2_{\!q_i q_j}\mathbf{B}_\mathcal{B}(\mathbf{q}) \,\big]_{i,j=0}^3
           \;\in\; \mathbb{R}^{4\times 4\times 3}.

      - Any second derivative involving bias or storage state is identically zero with the
        appropriate (possibly zero) leading dimensions.

    **Directional Hessian check**

    For each coordinate basis vector :math:`\mathbf{j}\in\mathbb{R}^3`, the test forms
    the scalar function :math:`f_{\mathbf{j}}(u,\mathbf{x},\mathbf{h})=\mathbf{j}^\top\boldsymbol{\tau}`
    and verifies (via ``numdifftools``) that

    .. math::

       \nabla^2 f_{\mathbf{j}} \;=\;
       \begin{bmatrix}
         0 & 0 & \mathbf{A} & 0\\[2pt]
         0 & 0 & 0 & 0\\[2pt]
         \mathbf{A}^\top & 0 & \mathbf{Q} & 0\\[2pt]
         0 & 0 & 0 & 0
       \end{bmatrix},\qquad
       \mathbf{A} = \big(\mathbf{a}_\times D_\mathbf{q}\mathbf{B}_\mathcal{B}\big)\mathbf{j},
       \quad
       \mathbf{Q} = \big(u\,\mathbf{a}_\times D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_\mathcal{B}\big)\mathbf{j},

    where the block rows/columns correspond to :math:`(u,\ \text{bias},\ \mathbf{x},\ \mathbf{h})`
    and the :math:`\mathbf{x}` block is further partitioned as
    :math:`(\boldsymbol{\omega},\ \mathbf{q})`. The last row/column (storage state) is zero.

    **What this test asserts**

    1. The torque implementation matches :math:`\boldsymbol{\tau}=u\,\mathbf{a}\times\mathbf{B}_\mathcal{B}`.
    2. Analytic Jacobians :math:`\partial\boldsymbol{\tau}/\partial u`,
       :math:`\partial\boldsymbol{\tau}/\partial \mathbf{x}`, bias, and storage agree with
       finite-difference Jacobians and with the closed forms above.
    3. All second-order blocks (pure, mixed, and pure-state) match finite-difference Hessians,
       with zeros in the expected locations (no dependence on :math:`\boldsymbol{\omega}`, bias,
       or storage).
    4. The MTQ has no momentum storage torque; all derivatives of the storage torque are zero
       with correct shapes.

    **Shapes (as implemented)**

    - :math:`\boldsymbol{\tau}\in\mathbb{R}^{3}`.
    - :math:`\partial\boldsymbol{\tau}/\partial u\in\mathbb{R}^{1\times 3}`,
      :math:`\partial\boldsymbol{\tau}/\partial \mathbf{x}\in\mathbb{R}^{7\times 3}`,
      :math:`\partial\boldsymbol{\tau}/\partial \text{bias}\in\mathbb{R}^{0\times 3}`,
      :math:`\partial\boldsymbol{\tau}/\partial \mathbf{h}\in\mathbb{R}^{0\times 3}`.
    - Second-order arrays follow the same leading-dimension convention, e.g.
      :math:`\partial^2\boldsymbol{\tau}/\partial u^2\in\mathbb{R}^{1\times 1\times 3}`,
      :math:`\partial^2\boldsymbol{\tau}/\partial u\,\partial \mathbf{x}\in\mathbb{R}^{1\times 7\times 3}`,
      :math:`\partial^2\boldsymbol{\tau}/\partial \mathbf{x}\,\partial \mathbf{x}\in\mathbb{R}^{7\times 7\times 3}`,
      and so on (zero-size leading dimensions for bias/storage).

    **Assumptions & conventions**

    - ``rot_mat(q)`` maps body :math:`\rightarrow` ECI, hence ``rot_mat(q).T`` maps
      ECI :math:`\rightarrow` body.
    - The quaternion :math:`\mathbf{q}` is treated as unit length at evaluation.
    - The MTQ is ideal (no saturation or rate limits); ``max_torque`` does not enter the
      torque expression in this test.
    - ``drotmatTvecdq`` and ``ddrotmatTvecdqdq`` return derivatives of
      :math:`\mathbf{C}(\mathbf{q})^\top \mathbf{v}` w.r.t. the quaternion components.

    The test raises :class:`AssertionError` if any equality, tolerance, or shape check fails.
    """
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
    assert np.all(mtq.storage_torque(u=m0, x=x0, os=os) == np.zeros(0))
    assert mtq.storage_torque(u=m0, x=x0, os=os).shape == (0,)

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

   assert np.all(np.isclose(mtq.dtorq__du(u=m0, x=x0, os=os), np.cross(ax/3, B_B)))
   assert np.all(np.isclose( mtq.dtorq__dbias(u=m0, x=x0, os=os) , np.cross(ax/3,B_B) ))
   assert np.all(np.isclose( mtq.dtorq__dbasestate(u=m0, x=x0, os=os) , np.vstack([np.zeros((3,3)),np.cross(ax/3*(m0+biast),drotmatTvecdq(q0,B_ECI))]) ))

   assert np.all(np.isclose( mtq.dtorq__dh(u=m0, x=x0, os=os) , np.zeros((0,3))))
   assert np.all(mtq.dtorq__dh(u=m0, x=x0, os=os).shape==(0,3))

   assert np.all(np.isclose( mtq.ddtorq__dudu(u=m0, x=x0, os=os) ,np.zeros((1,1,3)) ))
   assert np.all(mtq.ddtorq__dudu(u=m0, x=x0, os=os).shape==(1,1,3))
   assert np.all(np.isclose( mtq.ddtorq__dudbias(u=m0, x=x0, os=os) ,np.zeros((1,1,3)) ))
   assert np.all(mtq.ddtorq__dudbias(u=m0, x=x0, os=os).shape==(1,1,3))
   assert np.all(np.isclose( mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os) , np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]) ,0) ))
   assert np.all(mtq.ddtorq__dudbasestate(u=m0, x=x0, os=os).shape==(1,7,3))
   assert np.all(np.isclose( mtq.ddtorq__dudh(u=m0, x=x0, os=os) ,np.zeros((1,0,3)) ))
   assert np.all(mtq.ddtorq__dudh(u=m0, x=x0, os=os).shape==(1,0,3))

   assert np.all(np.isclose( mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os) ,np.zeros((1,1,3)) ))
   assert np.all(mtq.ddtorq__dbiasdbias(u=m0, x=x0, os=os).shape==(1,1,3))
   assert np.all(np.isclose( mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os) , np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
   assert np.all(mtq.ddtorq__dbiasdbasestate(u=m0, x=x0, os=os).shape==(1,7,3))
   assert np.all(np.isclose( mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os) ,np.zeros((1,0,3)) ))
   assert np.all(mtq.ddtorq__dbiasdh(u=m0, x=x0, os=os).shape==(1,0,3))

   dxdx = np.zeros((7,7,3))
   dxdx[3:7,3:7,:] = np.cross(ax/3*(m0+biast),ddrotmatTvecdqdq(q0,B_ECI))

   assert np.all(np.isclose( mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os) , dxdx))
   assert np.all(mtq.ddtorq__dbasestatedbasestate(u=m0, x=x0, os=os).shape==(7,7,3))
   assert np.all(np.isclose( mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os) ,np.zeros((7,0,3)) ))
   assert np.all(mtq.ddtorq__dbasestatedh(u=m0, x=x0, os=os).shape==(7,0,3))
   assert np.all(np.isclose( mtq.ddtorq__dhdh(u=m0, x=x0, os=os) ,np.zeros((0,0,3)) ))
   assert np.all(mtq.ddtorq__dhdh(u=m0, x=x0, os=os).shape==(0,0,3))

   assert np.all(mtq.storage_torque(u=m0, x=x0, os=os)  == np.zeros(0))
   assert np.all(mtq.storage_torque(u=m0, x=x0, os=os).shape == (0,))
   assert np.all(np.isclose( mtq.dstor_torq__du(u=m0, x=x0, os=os) , np.zeros((1,0)) ))
   assert np.all(mtq.dstor_torq__du(u=m0, x=x0, os=os).shape == (1,0))
   assert np.all(np.isclose( mtq.dstor_torq__dbias(u=m0, x=x0, os=os) , np.zeros((1,0)) ))
   assert np.all(mtq.dstor_torq__dbias(u=m0, x=x0, os=os).shape == (1,0))
   assert np.all(np.isclose( mtq.dstor_torq__dbasestate(u=m0, x=x0, os=os) , np.zeros((7,0))))
   assert np.all(mtq.dstor_torq__dbasestate(u=m0, x=x0, os=os).shape == (7,0))
   assert np.all(np.isclose( mtq.dstor_torq__dh(u=m0, x=x0, os=os) , np.zeros((0,0))))
   assert np.all(mtq.dstor_torq__dh(u=m0, x=x0, os=os).shape==(0,0))

   assert np.all(np.isclose( mtq.ddstor_torq__dudu(u=m0, x=x0, os=os) ,np.zeros((1,1,0)) ))
   assert np.all(mtq.ddstor_torq__dudu(u=m0, x=x0, os=os).shape==(1,1,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dudbias(u=m0, x=x0, os=os) ,np.zeros((1,1,0)) ))
   assert np.all(mtq.ddstor_torq__dudbias(u=m0, x=x0, os=os).shape==(1,1,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dudbasestate(u=m0, x=x0, os=os) ,np.zeros((1,7,0))))
   assert np.all(mtq.ddstor_torq__dudbasestate(u=m0, x=x0, os=os).shape==(1,7,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dudh(u=m0, x=x0, os=os) ,np.zeros((1,0,0)) ))
   assert np.all(mtq.ddstor_torq__dudh(u=m0, x=x0, os=os).shape==(1,0,0))

   assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbias(u=m0, x=x0, os=os) ,np.zeros((1,1,0)) ))
   assert np.all(mtq.ddstor_torq__dbiasdbias(u=m0, x=x0, os=os).shape==(1,1,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dbiasdbasestate(u=m0, x=x0, os=os) , np.zeros((1,7,0))))
   assert np.all(mtq.ddstor_torq__dbiasdbasestate(u=m0, x=x0, os=os).shape==(1,7,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dbiasdh(u=m0, x=x0, os=os) ,np.zeros((1,0,0)) ))
   assert np.all(mtq.ddstor_torq__dbiasdh(u=m0, x=x0, os=os).shape==(1,0,0))

   dxdx = np.zeros((7,7,0))

   assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedbasestate(u=m0, x=x0, os=os) , dxdx))
   assert np.all(mtq.ddstor_torq__dbasestatedbasestate(u=m0, x=x0, os=os).shape==(7,7,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dbasestatedh(u=m0, x=x0, os=os) ,np.zeros((7,0,0)) ))
   assert np.all(mtq.ddstor_torq__dbasestatedh(u=m0, x=x0, os=os).shape==(7,0,0))
   assert np.all(np.isclose( mtq.ddstor_torq__dhdh(u=m0, x=x0, os=os) ,np.zeros((0,0,0)) ))
   assert np.all(mtq.ddstor_torq__dhdh(u=m0, x=x0, os=os).shape==(0,0,0))

def test_MTQ_torque_bias_KS():
   ax = random_n_unit_vec(3)
   max_torque = 4.51

   biasv = random_n_unit_vec(3)[1]*0.1
   biast = biasv.copy()
   std_bias = 0.03

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

   # No change to bias if time in Orbital_State does not progress
   N = 1000
   test_torq = mtq.torque(u=m0, x=x0, os=os)
   opts = [mtq.torque(u=m0, x=x0, os=os) for j in range(N)]
   assert np.all([np.allclose(test_torq, j) for j in opts])

   # Change to bias
   N = 1000
   test_torq = mtq.torque(u=m0, x=x0, os=os)
   torq_drift = []
   for j in range(N):
      os.J2000 += 0.5*TimeConstants.sec2cent
      torque1 = mtq.torque(u=m0, x=x0, os=os)
      os.J2000 += 0.5*TimeConstants.sec2cent
      torque2 = mtq.torque(u=m0, x=x0, os=os)
      torq_drift.append(torque1 - torque2)

   exp_dist = [np.cross(ax, B_B)*np.random.normal(0, std_bias*np.sqrt(0.5)) for j in range(N)]
   
   ks0 = kstest([j[0] for j in torq_drift],[j[0] for j in exp_dist])
   ks1 = kstest([j[1] for j in torq_drift],[j[1] for j in exp_dist])
   ks2 = kstest([j[2] for j in torq_drift],[j[2] for j in exp_dist])

   ind = 0
   data_a = torq_drift
   data_b = exp_dist
   hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
   hist_edges = hist[1]
   hist_a = np.cumsum(hist[0]).tolist()
   hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
   graph_data = [hist_a,hist_b]
   print(plot(graph_data,{'height':20}))
   assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
   ind = 1
   data_a = torq_drift
   data_b = exp_dist
   hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
   hist_edges = hist[1]
   hist_a = np.cumsum(hist[0]).tolist()
   hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
   graph_data = [hist_a,hist_b]
   print(plot(graph_data,{'height':20}))
   assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
   ind = 2
   data_a = torq_drift
   data_b = exp_dist
   hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
   hist_edges = hist[1]
   hist_a = np.cumsum(hist[0]).tolist()
   hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
   graph_data = [hist_a,hist_b]
   print(plot(graph_data,{'height':20}))
   assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))


def test_MTQ_noise_KS():
   ax = random_n_unit_vec(3)
   ax = ax.copy()
   max_torque = 4.51

   noisev = 0.0
   std_noise = 0.243
   noise = Noise(noise=noisev, std_noise=std_noise)

   mtq = MTQ(axis=ax, max_torque=max_torque, noise=noise)
   mtqs = [mtq]

   m0 = random_n_unit_vec(3)[0]*3
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

   # Change to bias
   N = 1000
   test_torq = mtq.torque(u=m0, x=x0, os=os)
   torq_drift = []
   for j in range(N):
      os.J2000 += 0.5*TimeConstants.sec2cent
      torque1 = mtq.torque(u=m0, x=x0, os=os)
      os.J2000 += 0.5*TimeConstants.sec2cent
      torque2 = mtq.torque(u=m0, x=x0, os=os)
      torq_drift.append(torque1 - torque2)

   torq_drift = np.stack(torq_drift, axis=0)   # shape (N, 3)

   exp_dist = np.random.normal(0.0, np.sqrt(2)*std_noise, size=(N, 1)) * np.ones((1, 3))
   exp_dist = np.asarray(exp_dist)             # shape (N, 3)

   # Now KS works
   ks0 = ks_2samp(torq_drift[:, 0], exp_dist[:, 0])
   ks1 = ks_2samp(torq_drift[:, 1], exp_dist[:, 1])
   ks2 = ks_2samp(torq_drift[:, 2], exp_dist[:, 2])

   threshold = np.sqrt((1/N) * -0.5 * np.log(0.5 * 1e-5))
   assert ks0.pvalue > 0.05 or abs(ks0.statistic) < threshold
   assert ks1.pvalue > 0.05 or abs(ks1.statistic) < threshold
   assert ks2.pvalue > 0.05 or abs(ks2.statistic) < threshold

def test_MTQ_bias_noise_KS():
    ax = random_n_unit_vec(3)
    max_torque = 4.51

    # --- Bias and noise parameters ---
    std_bias = 0.03
    std_noise = 0.243
    bias = Bias(bias=0.0, std_bias=std_bias)
    noise = Noise(noise=0.0, std_noise=std_noise)

    mtq = MTQ(axis=ax, max_torque=max_torque, bias=bias, noise=noise)
    mtqs = [mtq]

    m0 = random_n_unit_vec(3)[0]*3
    B_ECI = 1e-5*random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05*random_n_unit_vec(3)
    sat = Satellite(actuators=mtqs)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    # Body-frame B for expected direction
    rmat_ECI2B = rot_mat(q0).T
    B_B = rmat_ECI2B @ B_ECI

    # Direction scaling for the bias term; noise term is scalar added to all components
    d = -np.cross(B_B, ax)  # matches torque = d*(u + b) + n

    # --- Collect paired differences over Δt ---
    N = 1000
    torq_drift = []
    for _ in range(N):
        # advance 0.5 s, take first torque
        os.J2000 += 0.5 * TimeConstants.sec2cent
        tau1 = mtq.torque(u=m0, x=x0, os=os)
        # advance another 0.5 s, take second torque
        os.J2000 += 0.5 * TimeConstants.sec2cent
        tau2 = mtq.torque(u=m0, x=x0, os=os)
        torq_drift.append(tau1 - tau2)

    torq_drift = np.stack(torq_drift, axis=0)  # (N,3)

    # --- Reference distribution: Δτ = d*ζ + 1⃗*η ---
    # Bias RW step std over Δt = 0.5 s:
    dt_sec = (0.5 * TimeConstants.sec2cent) * TimeConstants.cent2sec  # = 0.5 if constants are consistent
    std_bias_step = std_bias * np.sqrt(dt_sec)

    # Sample ζ and η, then form reference vectors
    zeta = np.random.normal(0.0, std_bias_step, size=(N, 1))  # (N,1)
    eta  = np.random.normal(0.0, np.sqrt(2)*std_noise,     size=(N, 1))  # (N,1)
    ones = np.ones((1, 3))
    d_row = d.reshape(1, 3)

    exp_dist = zeta @ d_row + eta @ ones  # (N,3)

    # --- KS tests per component (two-sample) ---
    ksx = ks_2samp(torq_drift[:, 0], exp_dist[:, 0])
    ksy = ks_2samp(torq_drift[:, 1], exp_dist[:, 1])
    ksz = ks_2samp(torq_drift[:, 2], exp_dist[:, 2])

    threshold = np.sqrt((1/N) * -0.5 * np.log(0.5 * 1e-5))
    assert ksx.pvalue > 0.05 or abs(ksx.statistic) < threshold
    assert ksy.pvalue > 0.05 or abs(ksy.statistic) < threshold
    assert ksz.pvalue > 0.05 or abs(ksz.statistic) < threshold


if __name__ == "__main__":
   test_MTQ_torque_bias()
