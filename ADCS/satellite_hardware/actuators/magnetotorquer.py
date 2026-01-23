__all__ = ["MTQ"]

import numpy as np
import warnings
from typing import Union, Dict

from ADCS.satellite_hardware.actuators.actuator import Actuator
from ADCS.satellite_hardware.errors.bias import Bias
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.satellite_hardware.disturbances.helpers.disturbance_mode import DisturbanceMode
from ADCS.orbits.orbital_state import Orbital_State

class MTQ(Actuator):
    r"""
    **Magnetorquer (MTQ) actuator model**

    This class models a single-axis magnetorquer that generates body torque from the
    interaction between a commanded magnetic dipole moment and the local geomagnetic
    field.

    The model extends :class:`~ADCS.satellite_hardware.actuators.actuator.Actuator`.

    Physical model
    --------------

    The torque produced by a magnetic dipole :math:`\mathbf{m}` in a magnetic field
    :math:`\mathbf{B}_{\mathcal{B}}` (expressed in the body frame) is

    .. math::

        \boldsymbol{\tau}_{\mathrm{MTQ}}
        = -\,\mathbf{B}_{\mathcal{B}} \times \mathbf{m}.

    The dipole is constrained to a fixed body-axis :math:`\mathbf{a}` (unit vector) and
    commanded with a scalar magnitude :math:`u` (with optional scalar bias :math:`b`):

    .. math::

        \mathbf{m} = \mathbf{a}\,(u + b).

    Combining the two gives the actuation model used by :meth:`~ADCS.satellite_hardware.actuators.magnetotorquer.MTQ.torque`:

    .. math::

        \boldsymbol{\tau}_{\mathrm{MTQ}}
        = -(\mathbf{B}_{\mathcal{B}} \times \mathbf{a})\,(u+b) + \mathbf{n},

    where :math:`\mathbf{n}` is an additive noise term (if enabled).

    Symbols
    --------

    .. list-table::
       :header-rows: 1
       :widths: 30 70

       * - Symbol
         - Meaning
       * - :math:`\mathbf{B}_{\mathcal{B}}`
         - Earth's magnetic field expressed in the body frame [T]
       * - :math:`\mathbf{m}`
         - Magnetic dipole moment vector [A·m²]
       * - :math:`\mathbf{a}`
         - Magnetorquer unit axis in the body frame
       * - :math:`u`
         - Commanded dipole magnitude [A·m²]
       * - :math:`b`
         - Bias in the commanded dipole (modeled by :class:`~ADCS.satellite_hardware.errors.bias.Bias`)
       * - :math:`\mathbf{n}`
         - Additive actuator torque noise (modeled by :class:`~ADCS.satellite_hardware.errors.noise.Noise`)

    State dependence
    -----------------

    The body-frame field depends on the attitude quaternion :math:`\mathbf{q}` via the
    direction cosine matrix :math:`\mathbf{C}(\mathbf{q})`:

    .. math::

        \mathbf{B}_{\mathcal{B}}(\mathbf{q})
        = \mathbf{C}(\mathbf{q})^\top \mathbf{B}_{\mathrm{ECI}}.

    The orbital-state provider :class:`~ADCS.orbits.orbital_state.Orbital_State` supplies
    :math:`\mathbf{B}_{\mathcal{B}}`, its Jacobian :math:`D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}`,
    and Hessian :math:`D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_{\mathcal{B}}` through
    :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.

    :param axis: Unit vector defining the magnetorquer alignment in the body frame.
    :type axis: numpy.ndarray

    :param max_torque: Maximum allowable commanded dipole magnitude (used as the actuator
        saturation limit).
    :type max_torque: float

    :param bias: Bias model representing constant or slowly varying dipole offset.
    :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` | None

    :param noise: Noise model representing stochastic actuation uncertainty.
    :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` | None

    :param estimate_bias: If ``True``, includes this actuator's bias term in the estimator state vector.
    :type estimate_bias: bool
    """

    def __init__(
        self,
        axis: np.ndarray,
        max_torque: float,
        bias: Bias = None,
        noise: Noise = None,
        estimate_bias: bool = False,
    ) -> None:
        r"""
        Initialize a magnetorquer actuator model.

        This constructor configures the physical axis, saturation limits, and optional
        stochastic error models. All parameters are passed through to the base
        :class:`~ADCS.satellite_hardware.actuators.actuator.Actuator` class.

        :param axis: Unit vector defining the physical alignment of the magnetorquer
            in the spacecraft body frame.
        :type axis: numpy.ndarray, shape ``(3,)``

        :param max_torque: Maximum allowable magnetic dipole magnitude.
        :type max_torque: float

        :param bias: Optional bias model applied additively to the commanded dipole.
        :type bias: :class:`~ADCS.satellite_hardware.errors.bias.Bias` | None

        :param noise: Optional noise model applied additively to the generated torque.
        :type noise: :class:`~ADCS.satellite_hardware.errors.noise.Noise` | None

        :param estimate_bias: If ``True``, the bias is appended to the estimator state.
        :type estimate_bias: bool

        :return: None
        :rtype: None
        """
        super().__init__(axis=axis, u_max=max_torque, bias=bias, noise=noise, estimate_bias=estimate_bias)

    def torque(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State | Dict[str, np.ndarray],
        dmode: DisturbanceMode = None
    ) -> np.ndarray:
        r"""
        Compute the torque generated by the magnetorquer.

        The applied body-frame torque is given by:

        .. math::

            \boldsymbol{\tau}
            = -(\mathbf{B}_{\mathcal{B}} \times \mathbf{a})\,(u + b) + \mathbf{n}

        where the magnetic field and its derivatives are obtained from
        :class:`~ADCS.orbits.orbital_state.Orbital_State`.

        Bias and noise terms are conditionally applied according to
        :class:`~ADCS.satellite_hardware.disturbances.disturbance_mode.DisturbanceMode`.

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Full spacecraft state vector
            :math:`[\boldsymbol{\omega};\mathbf{q}]`.
        :type x: numpy.ndarray, shape ``(7,)``

        :param os: Orbital state providing geomagnetic field vectors and derivatives.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State` | dict[str, numpy.ndarray]

        :param dmode: Disturbance mode specifying whether bias and noise are applied
            and/or updated.
        :type dmode: :class:`~ADCS.satellite_hardware.disturbances.disturbance_mode.DisturbanceMode` | None

        :return: Body-frame torque vector.
        :rtype: numpy.ndarray, shape ``(3,)``
        """
        if abs(u) > self.u_max:
            warnings.warn("requested torque exceeds actuation limit")

        vecs = os.get_state_vector(x=x)
        b_body = vecs["b"]

        if dmode is None:
            dmode = DisturbanceMode(add_bias=True, add_noise=True, update_bias=True, update_noise=True)

        if self.bias and dmode.add_bias:
            u += self.bias.get_bias(j2000=os.J2000)
        if dmode.update_bias:
            self.bias._update_bias(j2000=os.J2000)

        torque = -np.cross(b_body, self.axis) * u

        if self.noise and dmode.add_noise:
            torque += self.noise.get_noise()
        if dmode.update_noise:
            self.noise._update_noise()

        return torque
    
    def storage_torque(self, u: float, x: np.ndarray, os: Orbital_State, dmode: DisturbanceMode = None) -> float:
        r"""
        Storage torque contribution of the actuator.

        Magnetorquers do **not** store angular momentum. Consequently, this method
        always returns an empty vector.

        :param u: Commanded dipole magnitude.
        :type u: float

        :param x: Spacecraft state vector.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :param dmode: Disturbance mode (unused).
        :type dmode: :class:`~ADCS.satellite_hardware.disturbances.disturbance_mode.DisturbanceMode` | None

        :return: Empty storage-torque vector.
        :rtype: numpy.ndarray, shape ``(0,)``
        """
        return np.zeros((0,))

    def jacobians(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State,
    ) -> Union[np.ndarray, np.ndarray]:
        r"""
        Compute first-order derivatives (Jacobians) of the torque.

        Let :math:`\mathbf{a}=\texttt{self.axis}` and let :math:`b` be the (optional) scalar bias.
        The torque model (without noise) is

        .. math::

            \boldsymbol{\tau}(u,\mathbf{q})
            = -(\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a})\,(u+b).

        Jacobian w.r.t. control input
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Since :math:`\boldsymbol{\tau}` is affine in :math:`u`:

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial u}
            = -(\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}).

        Jacobian w.r.t. base state
        ~~~~~~~~~~~~~~~~~~~~~~~~~~

        The base state is :math:`\mathbf{x}=[\boldsymbol{\omega};\mathbf{q}]\in\mathbb{R}^7`.
        The torque depends on :math:`\mathbf{q}` only through :math:`\mathbf{B}_{\mathcal{B}}(\mathbf{q})`
        and does not depend on :math:`\boldsymbol{\omega}`:

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial\boldsymbol{\omega}} = \mathbf{0}_{3\times 3}.

        With :math:`D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q})\in\mathbb{R}^{4\times 3}` (provided as ``"db"``),
        the quaternion Jacobian is

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial\mathbf{q}}
            = -\bigl(D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr)\,(u+b),

        where the matrix–vector cross is applied row-wise:
        :math:`( \mathbf{M}\times\mathbf{v})_{i:}=\mathbf{M}_{i:}\times\mathbf{v}`.

        The returned base-state Jacobian stacks these blocks as

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial\mathbf{x}}
            =
            \begin{bmatrix}
            \mathbf{0}_{3\times 3}\\
            -\bigl(D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr)\,(u+b)
            \end{bmatrix}.

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state :math:`[\boldsymbol{\omega};\mathbf{q}]`.
        :type x: numpy.ndarray

        :param os: Orbital state providing :math:`\mathbf{B}_{\mathcal{B}}` and :math:`D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}`
            via :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: ``(dT_du, dT_dx)`` where:

            * ``dT_du`` has shape ``(1,3)`` and equals :math:`\partial\boldsymbol{\tau}/\partial u`.
            * ``dT_dx`` has shape ``(7,3)`` and equals :math:`\partial\boldsymbol{\tau}/\partial \mathbf{x}`.

        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        vecs = os.get_state_vector(x=x)
        b_body = vecs["b"]
        db_body__dq = vecs["db"]

        dtorq__du = -np.cross(b_body, self.axis).reshape((1, 3))

        biased_command = u + self.bias.get_bias(os.J2000)
        dtorq__dbasestate = np.vstack(
            [np.zeros((3, 3)), -np.cross(db_body__dq, self.axis) * biased_command]
        )

        return dtorq__du, dtorq__dbasestate

    def hessians(
        self,
        u: float,
        x: np.ndarray,
        os: Orbital_State,
    ) -> Union[np.ndarray, np.ndarray]:
        r"""
        Compute second-order derivatives (Hessians) of the torque.

        Using the noise-free model

        .. math::

            \boldsymbol{\tau}(u,\mathbf{q})
            = -(\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a})\,(u+b),

        and defining

        * :math:`D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q})\in\mathbb{R}^{4\times 3}` (provided as ``"db"``),
        * :math:`D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q})\in\mathbb{R}^{4\times 4\times 3}` (provided as ``"ddb"``),

        the only nonzero second derivatives involve the quaternion.

        Mixed control–state Hessian
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~

        The mixed derivative :math:`\partial^2\boldsymbol{\tau}/(\partial u\,\partial\mathbf{x})`
        has zero angular-velocity block and a quaternion block:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\boldsymbol{\omega}}
            = \mathbf{0}_{1\times 3\times 3},

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\mathbf{q}}
            = -\bigl(D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr).

        In stacked form (matching the base state :math:`[\boldsymbol{\omega};\mathbf{q}]`):

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\mathbf{x}}
            =
            \begin{bmatrix}
            \mathbf{0}_{3\times 3}\\
            -\bigl(D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr)
            \end{bmatrix}.

        Pure base-state Hessian
        ~~~~~~~~~~~~~~~~~~~~~~~

        All second derivatives involving :math:`\boldsymbol{\omega}` vanish, and the only nonzero block is
        the quaternion–quaternion block:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial\mathbf{q}\,\partial\mathbf{q}}
            = -\bigl(D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr)\,(u+b).

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state providing :math:`D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}` and
            :math:`D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_{\mathcal{B}}` via
            :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: ``(ddT_du_dx, ddT_dx2)`` where:

            * ``ddT_du_dx`` has shape ``(1,7,3)`` and equals :math:`\partial^2\boldsymbol{\tau}/(\partial u\,\partial\mathbf{x})`.
            * ``ddT_dx2`` has shape ``(7,7,3)`` and equals :math:`\partial^2\boldsymbol{\tau}/(\partial\mathbf{x}\,\partial\mathbf{x})`.

        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        vecs = os.get_state_vector(x=x)
        db_body__dq = vecs["db"]
        ddb_body__dqdq = vecs["ddb"]

        ddtorq__dudbasestate = np.vstack(
            [np.zeros((3, 3)), -np.cross(db_body__dq, self.axis)]
        ).reshape((1, 7, 3))

        biased_command = u + self.bias.get_bias(j2000=os.J2000)
        ddtorq__dbasestatedbasestate = np.zeros((7, 7, 3))
        ddtorq__dbasestatedbasestate[3:7, 3:7, :] = (
            -np.cross(ddb_body__dqdq, self.axis) * biased_command
        )

        return ddtorq__dudbasestate, ddtorq__dbasestatedbasestate
    

    def dtorq__du(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque with respect to the control input.

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial u}
            = \mathbf{a} \times \mathbf{B}_{\mathcal{B}}
            = -\,\mathbf{B}_{\mathcal{B}} \times \mathbf{a}

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft state vector.
        :type x: numpy.ndarray

        :param os: Orbital state providing the body-frame magnetic field.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Row Jacobian of torque w.r.t. control input.
        :rtype: numpy.ndarray, shape ``(1,3)``
        """
        vecs = os.get_state_vector(x=x)
        b_body = vecs["b"]
        return -np.cross(b_body, self.axis).reshape((1, 3))

    def dtorq__dbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        First derivative of torque with respect to the base spacecraft state.

        The torque does not depend on angular velocity, and depends on attitude only
        through the magnetic field transformation.

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft state vector
            :math:`[\boldsymbol{\omega};\mathbf{q}]`.
        :type x: numpy.ndarray, shape ``(7,)``

        :param os: Orbital state providing magnetic field Jacobians.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Jacobian of torque w.r.t. base state.
        :rtype: numpy.ndarray, shape ``(7,3)``
        """
        vecs = os.get_state_vector(x=x)
        db_body__dq = vecs["db"]

        biased_command = u + self.bias.get_bias(os.J2000)
        return np.vstack(
            [np.zeros((3, 3)), -np.cross(db_body__dq, self.axis) * biased_command]
        )

    def ddtorq__dudbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/(\partial u\,\partial\mathbf{x})`.

        With

        .. math::

            \boldsymbol{\tau}(u,\mathbf{q})
            = -(\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a})\,(u+b),

        the mixed derivative is independent of :math:`u` and has zero angular-velocity block:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\boldsymbol{\omega}}
            = \mathbf{0}_{1\times 3\times 3},\qquad
            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\mathbf{q}}
            = -\bigl(D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr).

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state providing ``"db"`` via :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Mixed Hessian tensor stacked by :math:`(\boldsymbol{\omega},\mathbf{q})` with a leading singleton
            control dimension.
        :rtype: numpy.ndarray
        """
        vecs = os.get_state_vector(x=x)
        db_body__dq = vecs["db"]

        return np.vstack(
            [np.zeros((3, 3)), -np.cross(db_body__dq, self.axis)]
        ).reshape((1, 7, 3))

    def ddtorq__dbasestatedbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Base-state Hessian :math:`\partial^2\boldsymbol{\tau}/(\partial\mathbf{x}\,\partial\mathbf{x})`.

        With the noise-free model

        .. math::

            \boldsymbol{\tau}(u,\mathbf{q})
            = -(\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a})\,(u+b),

        all second derivatives involving :math:`\boldsymbol{\omega}` vanish. The only nonzero block is the
        quaternion–quaternion block, using :math:`D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q})` (``"ddb"``):

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial\mathbf{q}\,\partial\mathbf{q}}
            = -\bigl(D^2_{\mathbf{q}\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr)\,(u+b).

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state providing ``"ddb"`` via :meth:`~ADCS.orbits.orbital_state.Orbital_State.get_state_vector`.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Base-state Hessian tensor (only the quaternion block is nonzero).
        :rtype: numpy.ndarray
        """
        vecs = os.get_state_vector(x=x)
        ddb_body__dqdq = vecs["ddb"]

        biased_command = u + self.bias.get_bias(j2000=os.J2000)
        ddtorq__dbasestatedbasestate = np.zeros((7, 7, 3))
        ddtorq__dbasestatedbasestate[3:7, 3:7, :] = (
            -np.cross(ddb_body__dqdq, self.axis) * biased_command
        )

        return ddtorq__dbasestatedbasestate

    def ddtorq__dbiasdbias(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Second derivative :math:`\partial^2\boldsymbol{\tau}/\partial b^2`.

        The bias :math:`b` enters the torque linearly through :math:`(u+b)`, so:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial b^2} = \mathbf{0}.

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Zero tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddtorq__dudu(u=u, x=x, os=os)
        else:
            return np.zeros((0, 0, 3))

    def ddtorq__dbiasdbasestate(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/(\partial b\,\partial\mathbf{x})`.

        Since :math:`b` enters identically to :math:`u` in :math:`(u+b)`, the mixed derivative equals the
        control–state mixed derivative:

        .. math::

            \frac{\partial^2\boldsymbol{\tau}}{\partial b\,\partial\mathbf{x}}
            = \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\mathbf{x}}
            =
            \begin{bmatrix}
            \mathbf{0}_{3\times 3}\\
            -\bigl(D_{\mathbf{q}}\mathbf{B}_{\mathcal{B}}(\mathbf{q}) \times \mathbf{a}\bigr)
            \end{bmatrix}.

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Mixed Hessian tensor if a bias model exists; otherwise an empty tensor consistent with no bias state.
        :rtype: numpy.ndarray
        """
        if self.bias:
            return self.ddtorq__dudbasestate(u=u, x=x, os=os)
        else:
            return np.zeros((0, 7, 3))

    def ddtorq__dudh(self, u: float, x: np.ndarray, os: Orbital_State) -> np.ndarray:
        r"""
        Mixed second derivative :math:`\partial^2\boldsymbol{\tau}/(\partial u\,\partial\mathbf{h})`.

        This actuator has no momentum-storage state :math:`\mathbf{h}` in the model, so all derivatives
        w.r.t. :math:`\mathbf{h}` are empty:

        .. math::

            \frac{\partial\boldsymbol{\tau}}{\partial\mathbf{h}}=\mathbf{0}
            \quad\Longrightarrow\quad
            \frac{\partial^2\boldsymbol{\tau}}{\partial u\,\partial\mathbf{h}}=\mathbf{0}.

        :param u: Commanded magnetic dipole magnitude.
        :type u: float

        :param x: Spacecraft base state.
        :type x: numpy.ndarray

        :param os: Orbital state.
        :type os: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Empty tensor along the storage-state dimension (shape ``(1,0,3)``).
        :rtype: numpy.ndarray
        """
        return np.zeros((1, 0, 3))



