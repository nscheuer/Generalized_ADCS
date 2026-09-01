__all__ = ["EstimatedSatellite"]
import copy
import numpy as np
from functools import cached_property

from ADCS.covariance import Covariance
from ADCS.state import State

from typing import List, Dict, Union, Tuple, Any, Optional, Sequence
from scipy.linalg import block_diag

from .satellite import Satellite

import ADCS.orbits.universal_constants as uc
from ADCS.helpers.math_helpers import *
from ADCS.satellite_hardware.disturbances import Disturbance, SRP_Disturbance, General_Disturbance, Prop_Disturbance
from ADCS.satellite_hardware.sensors import Sensor, GPS
from ADCS.satellite_hardware.actuators import Actuator, RW
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.state import EstimatorState

class EstimatedSatellite(Satellite):
    r"""
    Satellite model with estimator-synchronized biases and disturbance parameters.

    :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite` extends
    :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite` by tracking additional
    estimator-driven quantities:

    * actuator bias states (for actuators with ``estimate_bias=True``),
    * attitude sensor bias states (for sensors with ``estimate_bias=True``),
    * disturbance parameter states (for disturbances with ``estimate_dist=True``).

    These extra states typically live in an estimator state vector, while the physical simulation
    maintains internal subsystem objects. This class provides a **synchronization bridge** between
    them via :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.

    **Augmented estimator state layout**

    The estimator is assumed to output a stacked vector

    .. math::

        \mathbf{x}_{\mathrm{est}} \;=\;
        \begin{bmatrix}
        \mathbf{x} \\
        \mathbf{b}_{act} \\
        \mathbf{b}_{sens} \\
        \boldsymbol{\theta}_{dist}
        \end{bmatrix},

    where

    * :math:`\mathbf{x}\in\mathbb{R}^{n_x}` is the base satellite state used by
      :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite`,
    * :math:`\mathbf{b}_{act}\in\mathbb{R}^{n_{ab}}` concatenates actuator bias parameters,
    * :math:`\mathbf{b}_{sens}\in\mathbb{R}^{n_{sb}}` concatenates sensor bias parameters,
    * :math:`\boldsymbol{\theta}_{dist}\in\mathbb{R}^{n_{dp}}` concatenates disturbance parameters.

    The total expected estimator dimension is:

    .. math::

        n_{\mathrm{tot}} \;=\; n_x + n_{ab} + n_{sb} + n_{dp}.

    **Bias and parameter covariance mapping**

    The estimator provides a covariance :math:`\mathbf{P}` and an integrated covariance
    :math:`\int \mathbf{P}\,dt` (stored as ``int_cov``). :meth:`match_estimate` extracts the diagonal
    blocks corresponding to each bias/parameter group and maps them to per-subsystem standard deviations
    (square roots of block diagonals).

    .. note::
        During matching, noise is disabled in actuators/sensors and time variation is disabled in
        disturbances to ensure deterministic consistency when comparing propagation to estimation.

    :raises ValueError:
        If an estimator state vector with incompatible dimension is provided to
        :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.match_estimate`.
    """

    def __init__(self, mass: float = 1.0, COM: Optional[np.ndarray] = None, J_0: Optional[np.ndarray] = None, disturbances: Sequence[Disturbance] = [], sensors: Sequence[Sensor] = [], actuators: Sequence[Actuator] = [], boresight: Optional[dict[str, np.ndarray] | np.ndarray] = None) -> None:
        r"""
        Construct an estimator-augmented satellite.

        Calls :class:`~ADCS.satellite_hardware.satellite.satellite.Satellite` initialization and then
        detects which subsystems include estimated parameters by reading:

        * ``actuator.estimate_bias`` for each :class:`~ADCS.satellite_hardware.actuators.Actuator`,
        * ``sensor.estimate_bias`` for each attitude :class:`~ADCS.satellite_hardware.sensors.Sensor`
          (excluding :class:`~ADCS.satellite_hardware.sensors.GPS`),
        * ``disturbance.estimate_dist`` for each :class:`~ADCS.satellite_hardware.disturbances.Disturbance`.

        The detected indices are stored in:

        * ``act_bias_inds`` and total length ``act_bias_len``,
        * ``att_sens_bias_inds`` and total length ``att_sens_bias_len``,
        * ``dist_param_inds`` and total length ``dist_param_len``.

        :param mass: Total satellite mass including angular momentum storage hardware (kg).
        :type mass: float

        :param COM: Center of mass vector in the body/reference frame, shape ``(3,)``.
            If ``None``, defaults to ``[0,0,0]``.
        :type COM: numpy.ndarray | None

        :param J_0: Inertia tensor about the reference origin, shape ``(3,3)`` (kg·m²).
            If ``None``, defaults to identity.
        :type J_0: numpy.ndarray | None

        :param disturbances: Disturbance model list.
        :type disturbances: list[:class:`~ADCS.satellite_hardware.disturbances.Disturbance`]

        :param sensors: Sensor model list.
        :type sensors: list[:class:`~ADCS.satellite_hardware.sensors.Sensor`]

        :param actuators: Actuator model list.
        :type actuators: list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]

        :param boresight: Named boresight vectors dict or single vector, shape ``(3,)``.
        :type boresight: dict[str, numpy.ndarray] | numpy.ndarray | None

        :return: ``None``.
        :rtype: None
        """
        super().__init__(mass, COM, J_0, disturbances, sensors, actuators, boresight)

        # Add estimated states
        self.act_bias_inds = [j for j in range(len(self.actuators)) if self.actuators[j].estimate_bias] # Indices with actuator bias
        self.act_bias_len = sum([self.actuators[j].input_len for j in self.act_bias_inds]) # Number of actuators with biases
        self.att_sens_bias_inds = [j for j in range(len(self.attitude_sensors)) if self.attitude_sensors[j].estimate_bias] # Indices with sensor bias
        self.att_sens_bias_len = sum([self.attitude_sensors[j].output_length for j in self.att_sens_bias_inds]) # Number of sensors with bias
        self.dist_param_inds = [j for j in range(len(self.disturbances)) if self.disturbances[j].estimate_dist] # Indices with sensor disturbaces
        self.dist_param_len = sum([self.disturbances[j].estimated_vector_length for j in self.dist_param_inds]) # Number of sensors with bias


    @classmethod
    def from_satellite(cls, sat: Satellite) -> "EstimatedSatellite":
        """
        Create an EstimatedSatellite by cloning a Satellite.

        The sensor / actuator / disturbance objects MUST be deep-copied, not
        shared by reference: they carry mutable state (``Bias``/``Noise``
        evolution, reaction-wheel momentum and the shared
        ``_effective_command`` cache, disturbance parameters). When the
        estimated satellite is used by an estimator (e.g. inside
        :func:`~ADCS.simulate.simulate`, which auto-builds it via this
        method), the filter repeatedly evaluates ``sensor_readings`` /
        ``noiseless_rk4`` / actuator torques on its sigma points; if those
        objects were the *same* instances as the true satellite's, the
        filter's internal predictions would silently corrupt the truth's
        sensor/actuator/disturbance state (and vice versa) within and across
        timesteps. Deep-copying makes the estimate model independent of the
        plant, as the "cloning" contract promises.
        """
        est = cls(
            mass=sat.mass,
            COM=sat.COM.copy(),
            J_0=sat.J_0.copy(),
            disturbances=copy.deepcopy(sat.disturbances),
            sensors=copy.deepcopy(sat.sensors),
            actuators=copy.deepcopy(sat.actuators),
            boresight=sat.boresight.copy(),
        )

        return est


    def match_estimate(self, est_state: EstimatorState, dt: float) -> None:
        r"""
        Synchronize the satellite instance with the estimator output.

        This method maps the estimator output into the simulation model by:

        1. Extracting wheel momenta from the base state portion and updating wheel objects via
           :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.update_RWhs`.
        2. Writing actuator biases into each actuator's bias object and reporting posterior
           standard deviations separately from configured random-walk rates.
        3. Writing sensor biases into each attitude sensor's bias object and reporting posterior
           standard deviations separately from configured random-walk rates.
        4. Writing disturbance parameters into each disturbance model (for active disturbances) and
           setting their parameter standard deviations from covariance blocks.
        5. For deterministic matching, disabling:
           * ``actuator.use_noise = False`` for all actuators,
           * ``sensor.use_noise = False`` for all sensors,
           * ``disturbance.time_varying = False`` for all disturbances.

        **Estimator vector partition**

        With

        .. math::

            \mathbf{x}_{\mathrm{est}}
            =
            \begin{bmatrix}
            \mathbf{x} \\
            \mathbf{b}_{act} \\
            \mathbf{b}_{sens} \\
            \boldsymbol{\theta}_{dist}
            \end{bmatrix},

        indices are:

        .. math::

            \begin{aligned}
            \mathbf{x} &\in \mathbb{R}^{n_x},\\
            \mathbf{b}_{act} &\in \mathbb{R}^{n_{ab}},\\
            \mathbf{b}_{sens} &\in \mathbb{R}^{n_{sb}},\\
            \boldsymbol{\theta}_{dist} &\in \mathbb{R}^{n_{dp}},
            \end{aligned}

        where

        .. math::

            n_x=\texttt{self.state\_len},\;
            n_{ab}=\texttt{self.act\_bias\_len},\;
            n_{sb}=\texttt{self.att\_sens\_bias\_len},\;
            n_{dp}=\texttt{self.dist\_param\_len}.

        **Covariance-to-standard-deviation mapping**

        For each extracted block covariance :math:`\mathbf{P}_{block}`, per-element standard deviations are
        assigned as:

        .. math::

            \boldsymbol{\sigma} = \sqrt{\operatorname{diag}(\mathbf{P}_{block})}.

        The estimator may provide ``int_cov`` in full quaternion coordinates or
        reduced tangent coordinates. Each bias/parameter block is selected
        through :class:`~ADCS.state.EstimatorState`'s named layout, so the
        quaternion dimension does not leak into block-index arithmetic.

        :param est_state: Estimator output container providing the attitude state,
            bias/parameter blocks, covariance, and integrated covariance.
        :type est_state: :class:`~ADCS.state.EstimatorState`

        :param dt: Estimator propagation step (s). Used by some pipelines to normalize integrated covariance.
            (The current implementation partitions ``int_cov`` directly.)
        :type dt: float

        :return: ``None``.
        :rtype: None

        :raises ValueError:
            If ``est_state.as_estimator_array()`` does not have length
            ``self.state_len + self.act_bias_len + self.att_sens_bias_len + self.dist_param_len``.
        """
        if not isinstance(est_state, EstimatorState):
            raise TypeError(f"est_state must be an EstimatorState, got {type(est_state).__name__}")
        est_state.validate_layout(
            wheel_momentum=self.number_RW,
            actuator_bias=self.act_bias_len,
            sensor_bias=self.att_sens_bias_len,
            disturbance_parameter=self.dist_param_len,
        )

        # Covariance blocks use the same semantic layout as the state.  The
        # attitude block alone changes width between full and tangent
        # coordinates, so no global off-by-one adjustment is needed.
        covariance_coordinates = (
            "tangent" if est_state.uses_reduced_quaternion_covariance else "full"
        )
        int_cov = est_state.int_cov

        def process_block(name: str) -> np.ndarray:
            block_slice = est_state.slice(name, coordinates=covariance_coordinates)
            return int_cov[block_slice, block_slice]

        act_bias = est_state.block("actuator_bias")
        act_bias_ic = process_block("actuator_bias")
        sens_bias = est_state.block("sensor_bias")
        sens_bias_ic = process_block("sensor_bias")
        dist_param = est_state.block("disturbance_parameter")
        dist_param_ic = process_block("disturbance_parameter")

        # --- Update satellite dynamic state ---
        self.update_RWhs(est_state)

        # --- Disable noise sources for deterministic matching ---
        for a in self.actuators:
            a.use_noise = False
        for s in self.sensors:
            s.use_noise = False
        for d in self.disturbances:
            d.time_varying = False

        # --- Update actuator biases ---
        ind = 0
        for j in self.act_bias_inds:
            act = self.actuators[j]
            l = act.input_len
            act.bias.bias = act_bias[ind : ind + l]
            act.bias.estimated_std_bias = np.sqrt(
                np.diag(act_bias_ic[ind : ind + l, ind : ind + l])
            )
            ind += l

        # --- Update sensor biases ---
        ind = 0
        for j in self.att_sens_bias_inds:
            sens = self.attitude_sensors[j]
            l = sens.output_length
            sens.bias.bias = sens_bias[ind : ind + l]
            sens.bias.estimated_std_bias = np.sqrt(
                np.diag(sens_bias_ic[ind : ind + l, ind : ind + l])
            )
            ind += l

        # --- Update disturbance parameters ---
        ind = 0
        for j in self.dist_param_inds:
            dist = self.disturbances[j]
            if getattr(dist, "active", True):  # Only update active ones
                l = dist.main_param.size
                dist.main_param = dist_param[ind : ind + l]
                dist.std = np.sqrt(
                    np.diag(dist_param_ic[ind : ind + l, ind : ind + l])
                )
                ind += l

    def dist_torques_jacobian(self, x: State, vecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Jacobians of total disturbance torque w.r.t. state and disturbance parameters.

        This overrides :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.dist_torques_jacobian`
        by additionally returning the derivative of the total disturbance torque w.r.t. estimated disturbance
        parameters.

        Let the total disturbance torque be

        .. math::

            \boldsymbol{\tau}_d(\mathbf{x},\boldsymbol{\theta}_d)
            =
            \sum_{i=1}^{N_d}\boldsymbol{\tau}_{d,i}(\mathbf{x},\boldsymbol{\theta}_{d,i}).

        The method returns:

        .. math::

            \mathbf{J}_x = \frac{\partial \boldsymbol{\tau}_d}{\partial \mathbf{x}},
            \qquad
            \mathbf{J}_{\theta} = \frac{\partial \boldsymbol{\tau}_d}{\partial \boldsymbol{\theta}_d}.

        **State Jacobian structure**

        Only attitude (quaternion) affects rotated environment vectors and thus torque for the provided
        disturbance interface; therefore:

        .. math::

            \mathbf{J}_x[3:7,:] =
            \sum_i \frac{\partial \boldsymbol{\tau}_{d,i}}{\partial \mathbf{q}},\qquad
            \mathbf{J}_x[k,:]=0\;\text{for}\;k\notin\{3,4,5,6\}.

        **Parameter Jacobian assembly**

        If ``self.dist_param_len > 0``, the parameter Jacobian is formed by vertically stacking each
        estimated disturbance's parameter Jacobian:

        .. math::

            \mathbf{J}_{\theta} =
            \begin{bmatrix}
            \frac{\partial \boldsymbol{\tau}_{d,i_1}}{\partial \boldsymbol{\theta}_{d,i_1}} \\
            \vdots \\
            \frac{\partial \boldsymbol{\tau}_{d,i_m}}{\partial \boldsymbol{\theta}_{d,i_m}}
            \end{bmatrix}.

        Each estimated disturbance is expected to implement:

        * ``torque_qjac(self, vecs)`` → :math:`\partial \boldsymbol{\tau}/\partial\mathbf{q}`,
        * ``torque_valjac(self, vecs)`` → :math:`\partial \boldsymbol{\tau}/\partial\boldsymbol{\theta}`.

        :param x: Current base spacecraft state vector, shape ``(state_len,)``.
        :type x: ADCS.state.State

        :param vecs: Dictionary containing body-frame environment vectors and their quaternion derivatives,
            typically including keys like ``"b"``, ``"r"``, ``"s"``, ``"v"``, ``"rho"``, and derivative keys
            like ``"db"``, ``"dr"``, ``"ds"``, ``"dv"``.
        :type vecs: dict[str, numpy.ndarray]

        :return: Tuple ``(ddist_torq__dx, ddist_torq__ddmp)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        .. rubric:: Return shapes

        +--------------------+------------------------------------+
        | Name               | Shape                              |
        +====================+====================================+
        | ``ddist_torq__dx`` | ``(state_len, 3)``                 |
        +--------------------+------------------------------------+
        | ``ddist_torq__ddmp`` | ``(dist_param_len, 3)``          |
        +--------------------+------------------------------------+
        """
        ddist_torq__dx = np.zeros((self.state_len,3))
        ddist_torq__dx[3:7,:] = sum([j.torque_qjac(self,x,vecs["os"]) for j in self.disturbances],np.zeros((4,3)))
        ddist_torq__ddmp = np.zeros((0,3))
        if self.dist_param_len>0:
            ddist_torq__ddmp = np.vstack([self.disturbances[j].torque_valjac(self,x,vecs["os"]) for j in self.dist_param_inds])
        return ddist_torq__dx,ddist_torq__ddmp

    def dist_torque_hess(self, x: State, vecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Hessians of total disturbance torque w.r.t. state and disturbance parameters.

        This overrides :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.dist_torque_hess`
        by populating parameter–parameter and quaternion–parameter second derivatives for disturbances
        with estimated parameters.

        Let :math:`\boldsymbol{\tau}_d(\mathbf{x},\boldsymbol{\theta}_d)` be the total disturbance torque.
        This method returns:

        .. math::

            \mathbf{H}_{xx} = \frac{\partial^2 \boldsymbol{\tau}_d}{\partial \mathbf{x}^2},
            \qquad
            \mathbf{H}_{x\theta} = \frac{\partial^2 \boldsymbol{\tau}_d}{\partial \mathbf{x}\,\partial \boldsymbol{\theta}_d},
            \qquad
            \mathbf{H}_{\theta\theta} = \frac{\partial^2 \boldsymbol{\tau}_d}{\partial \boldsymbol{\theta}_d^2}.

        **State Hessian sparsity**

        As in the base class, only quaternion components contribute to state second derivatives:

        .. math::

            \mathbf{H}_{xx}[3:7,3:7,:]
            =
            \sum_i \frac{\partial^2 \boldsymbol{\tau}_{d,i}}{\partial \mathbf{q}^2}.

        **Parameter blocks**

        For each estimated disturbance :math:`i` with parameter vector size :math:`\ell_i`, the method places:

        .. math::

            \mathbf{H}_{\theta\theta}[k:k+\ell_i,\,k:k+\ell_i,:]
            =
            \frac{\partial^2 \boldsymbol{\tau}_{d,i}}{\partial \boldsymbol{\theta}_{d,i}^2},

        and

        .. math::

            \mathbf{H}_{x\theta}[3:7,\,k:k+\ell_i,:]
            =
            \frac{\partial^2 \boldsymbol{\tau}_{d,i}}{\partial \mathbf{q}\,\partial \boldsymbol{\theta}_{d,i}}.

        Each estimated disturbance is expected to implement:

        * ``torque_qqhess(self, vecs)`` → :math:`\partial^2 \boldsymbol{\tau}/\partial\mathbf{q}^2`,
        * ``torque_qvalhess(self, vecs)`` → :math:`\partial^2 \boldsymbol{\tau}/\partial\mathbf{q}\partial\boldsymbol{\theta}`,
        * ``torque_valvalhess(self, vecs)`` → :math:`\partial^2 \boldsymbol{\tau}/\partial\boldsymbol{\theta}^2`.

        :param x: Current base spacecraft state vector, shape ``(state_len,)``.
        :type x: ADCS.state.State

        :param vecs: Dictionary containing body-frame environment vectors and their derivatives.
        :type vecs: dict[str, numpy.ndarray]

        :return: Tuple ``(dddist_torq__dxdx, dddist_torq__dxddmp, dddist_torq__ddmpddmp)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]

        .. rubric:: Return shapes

        +---------------------------+-----------------------------------------+
        | Name                      | Shape                                   |
        +===========================+=========================================+
        | ``dddist_torq__dxdx``     | ``(state_len, state_len, 3)``           |
        +---------------------------+-----------------------------------------+
        | ``dddist_torq__dxddmp``   | ``(state_len, dist_param_len, 3)``      |
        +---------------------------+-----------------------------------------+
        | ``dddist_torq__ddmpddmp`` | ``(dist_param_len, dist_param_len, 3)`` |
        +---------------------------+-----------------------------------------+
        """
        dddist_torq__dxdx = np.zeros((self.state_len,self.state_len,3))
        dddist_torq__dxdx[3:7,3:7,:] = sum([j.torque_qqhess(self,x,vecs["os"]) for j in self.disturbances],np.zeros((4,4,3)))
        dddist_torq__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,3))
        dddist_torq__dxddmp = np.zeros((self.state_len,self.dist_param_len,3))
        ind = 0
        for j in self.dist_param_inds:
            l = self.disturbances[j].main_param.size
            dddist_torq__ddmpddmp[ind:ind+l,ind:ind+l,:] = self.disturbances[j].torque_valvalhess(self,x,vecs["os"])
            dddist_torq__dxddmp[3:7,ind:ind+l,:] = self.disturbances[j].torque_qvalhess(self,x,vecs["os"])
            ind += l
        return dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp
    
    def dynJacCore(self, x: State, u: np.ndarray, orbital_state: Orbital_State) -> Union[np.ndarray, np.ndarray]:
        r"""
        Jacobians of augmented dynamics w.r.t. state, inputs, and estimated parameters.

        This method extends the base linearization from
        :meth:`~ADCS.satellite_hardware.satellite.satellite.Satellite.dynJacCore` by additionally
        returning Jacobians of :math:`\dot{\mathbf{x}}` w.r.t.:

        * actuator bias vector :math:`\mathbf{b}_{act}`,
        * sensor bias vector :math:`\mathbf{b}_{sens}` (typically zeros for the process model),
        * disturbance parameter vector :math:`\boldsymbol{\theta}_{dist}`.

        The local first-order model is:

        .. math::

            \delta\dot{\mathbf{x}}
            \approx
            \mathbf{A}\,\delta\mathbf{x}
            + \mathbf{B}\,\delta\mathbf{u}
            + \mathbf{G}_{ab}\,\delta\mathbf{b}_{act}
            + \mathbf{G}_{sb}\,\delta\mathbf{b}_{sens}
            + \mathbf{G}_{dp}\,\delta\boldsymbol{\theta}_{dist}.

        **Returned blocks**

        The method returns a list:

        .. math::

            \left[
            \frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{x}},
            \frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{u}},
            \frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{b}_{act}},
            \frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{b}_{sens}},
            \frac{\partial \dot{\mathbf{x}}}{\partial \boldsymbol{\theta}_{dist}}
            \right].

        Where the last three blocks are assembled via actuator- and disturbance-provided derivatives:

        * actuators implement ``dtorq__dbias`` and (if wheels) storage-torque bias derivatives,
        * disturbances provide parameter Jacobians via
          :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.dist_torques_jacobian`.

        .. note::
            Sensor bias Jacobians are typically **zero in the process model** because sensor biases affect
            measurements, not rotational dynamics; therefore ``dxdot__dsb`` is returned as zeros.

        :param x: Current base state vector, shape ``(state_len,)``.
        :type x: ADCS.state.State

        :param u: Current control vector, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param orbital_state: Orbital/environmental state supplying ECI vectors and scalars used by torque models.
        :type orbital_state: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: List ``[dxdot__dx, dxdot__du, dxdot__dab, dxdot__dsb, dxdot__ddmp]``.
        :rtype: list[numpy.ndarray]

        .. rubric:: Return shapes

        +----------------+--------------------------------------+
        | Name           | Shape                                |
        +================+======================================+
        | ``dxdot__dx``  | ``(state_len, state_len)``           |
        +----------------+--------------------------------------+
        | ``dxdot__du``  | ``(control_len, state_len)``         |
        +----------------+--------------------------------------+
        | ``dxdot__dab`` | ``(act_bias_len, state_len)``        |
        +----------------+--------------------------------------+
        | ``dxdot__dsb`` | ``(att_sens_bias_len, state_len)``   |
        +----------------+--------------------------------------+
        | ``dxdot__ddmp``| ``(dist_param_len, state_len)``      |
        +----------------+--------------------------------------+
        """
        R = orbital_state.R # Position in ECI [km]
        V = orbital_state.V # Velocity in body frame [km/s]
        B = orbital_state.B # Magnetic field in ECI [T]
        S = orbital_state.S # Sun Vector in ECI [km]
        rho = orbital_state.rho # Atmospheric density [kg/m^3]

        w = x.w
        q = x.q
        RWhs = x.h
        # Must match the inertia used in dynamics_core (J_COM, not J_0) so the
        # estimated-model Jacobian remains the correct linearization when COM
        # is offset from the reference origin.
        J = self.J_COM
        invJ_noRW = self.invJ_noRW

        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V

        dR_B__dq = drotmatTvecdq(q,R)
        dB_B__dq = drotmatTvecdq(q,B)
        dV_B__dq = drotmatTvecdq(q,V)
        dS_B__dq = drotmatTvecdq(q,S)
        vecs: Dict[str, Any] = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"os":orbital_state}
        com = self.COM

        ddist_torq__dx,ddist_torq__ddmp = self.dist_torques_jacobian(x,vecs)
        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],x,orbital_state) for j in range(len(self.actuators))],np.zeros((7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],x,orbital_state) for j in range(len(self.actuators))])

        dxdot__dx = np.zeros((self.state_len,self.state_len))
        dxdot__du = np.zeros((self.control_len,self.state_len))
        dxdot__dx[3,4:7] = 0.5*w
        dxdot__dx[4:7,3] = -0.5*w
        dxdot__dx[4:7,4:7] = 0.5*skewsym(w)
        dxdot__dx[0:3,3:7] = 0.5*Wmat(q).T
        dxdot__du[:,0:3] = dact_torq__du@invJ_noRW

        dxdot__dx[:,0:3] += ddist_torq__dx@invJ_noRW
        dxdot__dx[0:7,0:3] += dact_torq__dbase@invJ_noRW
        dxdot__dx[0:3,0:3] += (-skewsym(w@J)+J@skewsym(w))@invJ_noRW

        # Reaction Wheels
        if self.number_RW>0:
            dact_torq__dh = np.vstack([self.actuators[j].dtorq__dh(u[j],x,orbital_state) for j in range(len(self.actuators))])
            RWjs = np.array([rw.J for rw in self.rw_actuators])
            RWaxes = np.vstack([rw.axis for rw in self.rw_actuators])
            mRWjs = np.diagflat(RWjs)
            dxdot__dx[0:3,0:3] += -skewsym(RWhs@RWaxes)@invJ_noRW
            dxdot__dx[7:,0:3] += (dact_torq__dh+np.cross(RWaxes,w))@invJ_noRW
            dxdot__du[:,7:] = block_diag(*[self.actuators[j].dstor_torq__du(u[j],x,orbital_state) for j in range(len(self.actuators))])
            dxdot__du[:,7:] -= dxdot__du[:,0:3]@RWaxes.T@mRWjs
            
            dxdot__dx[0:7,7:] = np.hstack([act.dstor_torq__dbasestate(u[j],x,orbital_state) for j,act in enumerate(self.actuators)])
            dxdot__dx[7:,7:] = np.diagflat([rw.dstor_torq__dh(u[self.momentum_inds[i]],x,orbital_state) for i,rw in enumerate(self.rw_actuators)])
            dxdot__dx[:,7:] -= dxdot__dx[:,0:3]@RWaxes.T@mRWjs
        dxdot__dab = np.zeros((self.act_bias_len,self.state_len))
        dxdot__dsb = np.zeros((self.att_sens_bias_len,self.state_len))
        dxdot__ddmp = np.zeros((self.dist_param_len,self.state_len))
        if self.act_bias_len>0:
            dact_torq__dab = np.vstack([self.actuators[j].dtorq__dbias(u[j],x,orbital_state) for j in self.act_bias_inds])
        else:
            dact_torq__dab = np.zeros((0,3))

        dxdot__dab[:,0:3] = dact_torq__dab@invJ_noRW

        dxdot__ddmp[:,0:3] = ddist_torq__ddmp@invJ_noRW

        if self.number_RW>0:
            dxdot__dab[:,7:] = block_diag(*[self.actuators[j].dstor_torq__dbias(u[j],x,orbital_state).T for j in self.act_bias_inds]).T
            dxdot__dab[:,7:] -= dxdot__dab[:,0:3]@RWaxes.T@mRWjs
            dxdot__ddmp[:,7:] -= dxdot__ddmp[:,0:3]@RWaxes.T@mRWjs

        return [dxdot__dx,dxdot__du,dxdot__dab,dxdot__dsb,dxdot__ddmp]
    

    def dynamics_Hessians(self, x: State, u: np.ndarray, orbital_state: Orbital_State) -> List[List[np.ndarray]]:
        r"""
        Second-order derivatives (Hessians) of augmented dynamics.

        This method computes second derivatives of the base rotational dynamics and extends them to
        include curvature terms w.r.t. actuator biases and disturbance parameters. The Hessians are
        organized as a block matrix over the stacked variable vector

        .. math::

            \mathbf{z} =
            \begin{bmatrix}
            \mathbf{x} \\
            \mathbf{u} \\
            \mathbf{b}_{act} \\
            \mathbf{b}_{sens} \\
            \boldsymbol{\theta}_{dist}
            \end{bmatrix}.

        For each state-derivative component :math:`\dot{x}_k`, the Hessian stores

        .. math::

            \frac{\partial^2 \dot{x}_k}{\partial z_i\,\partial z_j}.

        **Returned structure**

        If reaction wheels are present (and extended estimation terms are active), the method returns:

        .. code-block:: text

            [
              [ddxdot__dxdx,  ddxdot__dxdu,  ddxdot__dxdab, ddxdot__dxdsb, ddxdot__dxddmp],
              [ddxdot__dxdu.T, ddxdot__dudu, ddxdot__dudab, ddxdot__dudsb, ddxdot__duddmp],
              [0,              0,           ddxdot__dabdab, ddxdot__dabdsb, ddxdot__dabddmp],
              [0,              0,           0,             ddxdot__dsbdsb,  ddxdot__dsbddmp],
              [0,              0,           0,             0,              ddxdot__ddmpddmp]
            ]

        If extended estimation terms are not used, it falls back to the base 2×2 structure:

        .. code-block:: text

            [
              [ddxdot__dxdx, ddxdot__dxdu],
              [ddxdot__dxdu.T, ddxdot__dudu]
            ]

        **Key analytic contributions**

        * Quaternion kinematics curvature terms from :math:`\dot{\mathbf{q}} = \tfrac12 W(\mathbf{q})^\top\boldsymbol{\omega}`.
        * Rigid-body dynamics curvature from the gyroscopic term
          :math:`-\boldsymbol{\omega}\times(\mathbf{J}\boldsymbol{\omega} + \mathbf{H}_{RW})`.
        * Disturbance torque curvature terms from
          :meth:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.dist_torque_hess`.
        * Actuator torque curvature terms from actuator-provided second derivatives, including bias couplings.

        :param x: Current base state vector, shape ``(state_len,)``.
        :type x: ADCS.state.State

        :param u: Current control vector, shape ``(control_len,)``.
        :type u: numpy.ndarray

        :param orbital_state: Orbital/environmental state providing ECI vectors and scalars for torque models.
        :type orbital_state: :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return: Nested list of Hessian tensors (see structure above).
        :rtype: list[list[numpy.ndarray]]

        .. rubric:: Common tensor shapes (extended case)

        +--------------------+------------------------------------------------------+
        | Tensor             | Shape                                                |
        +====================+======================================================+
        | ``ddxdot__dxdx``   | ``(state_len, state_len, state_len)``                |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dxdu``   | ``(state_len, control_len, state_len)``              |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dudu``   | ``(control_len, control_len, state_len)``            |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dxdab``  | ``(state_len, act_bias_len, state_len)``             |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dxdsb``  | ``(state_len, att_sens_bias_len, state_len)``        |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dxddmp`` | ``(state_len, dist_param_len, state_len)``           |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dabdab`` | ``(act_bias_len, act_bias_len, state_len)``          |
        +--------------------+------------------------------------------------------+
        | ``ddxdot__dsbdsb`` | ``(att_sens_bias_len, att_sens_bias_len, state_len)``|
        +--------------------+------------------------------------------------------+
        |``ddxdot__ddmpddmp``| ``(dist_param_len, dist_param_len, state_len)``      |
        +--------------------+------------------------------------------------------+
        """
        w = x.w
        q = x.q
        RWhs = x.h
        invJ_noRW = self.invJ_noRW
        # Match the COM-based rotational dynamics and avoid relying on the
        # undefined self.J attribute.
        J = self.J_COM

        R = orbital_state.R
        V = orbital_state.V
        B = orbital_state.B
        S = orbital_state.S
        rho = orbital_state.rho

        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V
        dR_B__dq = drotmatTvecdq(q,R)
        dB_B__dq = drotmatTvecdq(q,B)
        dV_B__dq = drotmatTvecdq(q,V)
        dS_B__dq = drotmatTvecdq(q,S)
        ddR_B__dqdq = ddrotmatTvecdqdq(q,R)
        ddB_B__dqdq = ddrotmatTvecdqdq(q,B)
        ddV_B__dqdq = ddrotmatTvecdqdq(q,V)
        ddS_B__dqdq = ddrotmatTvecdqdq(q,S)
        vecs = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"db":dB_B__dq,"ds":dS_B__dq,"dv":dV_B__dq,"dr":dR_B__dq,"ddb":ddB_B__dqdq,"dds":ddS_B__dqdq,"ddv":ddV_B__dqdq,"ddr":ddR_B__dqdq,"os":orbital_state}
        com = self.COM

        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],x,orbital_state) for j in range(len(self.actuators))],np.zeros((7,3)))
        ddact_torq__dbasedbase = sum([self.actuators[j].ddtorq__dbasestatedbasestate(u[j],x,orbital_state) for j in range(len(self.actuators))],np.zeros((7,7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],x,orbital_state) for j in range(len(self.actuators))])
        ddact_torq__dudu = np.zeros((self.control_len,self.control_len,3))
        ddact_torq__dudbase = np.zeros((self.control_len,7,3))
        for j in range(len(self.actuators)):
            ddact_torq__dudu[j,j,:] = self.actuators[j].ddtorq__dudu(u[j],x,orbital_state)
            ddact_torq__dudbase[j,:,:] = self.actuators[j].ddtorq__dudbasestate(u[j],x,orbital_state)


        ddxdot__dxdx = np.zeros((self.state_len,self.state_len,self.state_len))
        ddxdot__dudu = np.zeros((self.control_len,self.control_len,self.state_len))
        ddxdot__dxdu = np.zeros((self.state_len,self.control_len,self.state_len))

        dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp = self.dist_torque_hess(x,vecs)

        ddxdot__dxdx[3,0:3,4:7]  = 0.5*np.eye(3)
        ddxdot__dxdx[4:7,0:3,3]  = 0.5*-np.eye(3)
        ddxdot__dxdx[4:7,0:3,4:7] = 0.5*-np.cross(np.expand_dims(np.eye(3),0),np.expand_dims(np.eye(3),1))
        ddxdot__dxdx[0:3,3:7,3:7] = np.transpose(ddxdot__dxdx[3:7,0:3,3:7],(1,0,2))

        ddxdot__dudu[:,:,0:3] = ddact_torq__dudu@invJ_noRW
        ddxdot__dxdu[0:7,:,0:3] = np.transpose(ddact_torq__dudbase,(1,0,2))@invJ_noRW
        ddxdot__dxdx[:,:,0:3] += dddist_torq__dxdx@invJ_noRW
        ddxdot__dxdx[0:7,0:7,0:3] += ddact_torq__dbasedbase@invJ_noRW
      
        JxI = np.cross(np.expand_dims(J,0),np.expand_dims(np.eye(3),1))
     
        ddxdot__dxdx[0:3,0:3,0:3] += (JxI + np.transpose( JxI,(1,0,2)))@invJ_noRW
        if self.number_RW>0:
            ddact_torq__dudh = np.zeros((self.control_len,self.number_RW,3))
            ddact_torq__dhdh = np.zeros((self.number_RW,self.number_RW,3))
            ddact_torq__dbasedh =  np.zeros((7,self.number_RW,3))
            ind = 0
            for ind in range(self.number_RW):
                j = self.momentum_inds[ind]
                ddact_torq__dudh[j,ind,:] = self.actuators[j].ddtorq__dudh(u[j],x,orbital_state)
                ddact_torq__dhdh[ind,ind,:] = self.actuators[j].ddtorq__dhdh(u[j],x,orbital_state)
                ddact_torq__dbasedh[:,ind,:] = np.squeeze(self.actuators[j].ddtorq__dbasestatedh(u[j],x,orbital_state))

            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])

            mRWjs = np.diagflat(RWjs)

            ddxdot__dxdu[7:,:,0:3] += np.transpose(ddact_torq__dudh,(1,0,2))@invJ_noRW
            ddxdot__dxdx[7:,0:7,0:3] += np.transpose(ddact_torq__dbasedh,(1,0,2))@invJ_noRW ###
            ddxdot__dxdx[0:7,7:,0:3] +=  ddact_torq__dbasedh@invJ_noRW


            AxI = -np.cross(np.expand_dims(RWaxes,1),np.expand_dims(np.eye(3),0))
            ddxdot__dxdx[7:,0:3,0:3] += -AxI@invJ_noRW
            ddxdot__dxdx[0:3,7:,0:3] += -np.transpose(AxI,(1,0,2))@invJ_noRW
            ddxdot__dxdx[7:,7:,0:3] += (ddact_torq__dhdh)@invJ_noRW

            ind = 0
            for ind in range(self.number_RW):
                j = self.momentum_inds[ind]
                ddxdot__dxdu[0:7,j,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dudbasestate(u[j],x,orbital_state),(1,0,2)))
                ddxdot__dxdu[7+ind,j,7+ind] += np.transpose(self.actuators[j].ddstor_torq__dudh(u[j],x,orbital_state),(1,0,2))
                ddxdot__dudu[j,j,7+ind] = self.actuators[j].ddstor_torq__dudu(u[j],x,orbital_state)
                ddxdot__dxdx[0:7,0:7,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedbasestate(u[j],x,orbital_state))
                ddxdot__dxdx[7+ind,0:7,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dbasestatedh(u[j],x,orbital_state),(1,0,2)))
                ddxdot__dxdx[0:7,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedh(u[j],x,orbital_state))
                ddxdot__dxdx[7+ind,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dhdh(u[j],x,orbital_state))

            ddxdot__dxdu[:,:,7:] -= ddxdot__dxdu[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dudu[:,:,7:] -= ddxdot__dudu[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dxdx[:,:,7:] -= ddxdot__dxdx[:,:,0:3]@RWaxes.T@mRWjs

        ddxdot__dxdab = np.zeros((self.state_len,self.act_bias_len,self.state_len))
        ddxdot__dudab = np.zeros((self.control_len,self.act_bias_len,self.state_len))

        # Estimator Part
        ddxdot__dxdsb = np.zeros((self.state_len,self.att_sens_bias_len,self.state_len))
        ddxdot__dudsb = np.zeros((self.control_len,self.att_sens_bias_len,self.state_len))

        ddxdot__dxddmp = np.zeros((self.state_len,self.dist_param_len,self.state_len))
        ddxdot__duddmp = np.zeros((self.control_len,self.dist_param_len,self.state_len))

        ddxdot__dabdab = np.zeros((self.act_bias_len,self.act_bias_len,self.state_len))
        ddxdot__dsbdsb = np.zeros((self.att_sens_bias_len,self.att_sens_bias_len,self.state_len))
        ddxdot__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,self.state_len))

        ddxdot__dabdsb = np.zeros((self.act_bias_len,self.att_sens_bias_len,self.state_len))
        ddxdot__dabddmp = np.zeros((self.act_bias_len,self.dist_param_len,self.state_len))
        ddxdot__dsbddmp = np.zeros((self.att_sens_bias_len,self.dist_param_len,self.state_len))

        ddact_torq__dudab = np.zeros((self.control_len,self.act_bias_len,3))
        ddact_torq__dabdab = np.zeros((self.act_bias_len,self.act_bias_len,3))
        ddact_torq__dbasedab = np.zeros((7,self.act_bias_len,3))
        ind = 0
        for j in range(len(self.act_bias_inds)):
            actind = self.act_bias_inds[j]
            l = self.actuators[actind].input_len
            ddact_torq__dabdab[ind:ind+l,ind:ind+l,:] = self.actuators[actind].ddtorq__dbiasdbias(u[actind],self,x,vecs)
            ddact_torq__dudab[actind,ind:ind+l,:] = self.actuators[actind].ddtorq__dudbias(u[actind],self,x,vecs)
            ddact_torq__dbasedab[:,ind:ind+l,:] = np.transpose(self.actuators[actind].ddtorq__dbiasdbasestate(u[actind],self,x,vecs),(1,0,2))
            ind+=l

        ddxdot__dabdab[:,:,0:3] = ddact_torq__dabdab@invJ_noRW
        ddxdot__dudab[:,:,0:3] = ddact_torq__dudab@invJ_noRW
        ddxdot__dxdab[0:7,:,0:3] = ddact_torq__dbasedab@invJ_noRW

        ddxdot__ddmpddmp[:,:,0:3] = dddist_torq__ddmpddmp@invJ_noRW
        ddxdot__dxddmp[:,:,0:3] = dddist_torq__dxddmp@invJ_noRW

        if self.number_RW>0:
            ddact_torq__dabdh = np.zeros((self.act_bias_len,self.number_RW,3))
            if ind in range(len(self.act_bias_inds)):
                actind = self.act_bias_inds[ind]
                if ind in self.momentum_inds:
                    j = np.where(self.momentum_inds==ind)
                    ddact_torq__dabdh[ind,j,:] = self.actuators[actind].ddtorq__dbiasdh(u[actind],self,x,vecs)

            ddxdot__dxdab[7:,:,0:3] += np.transpose(ddact_torq__dabdh,(1,0,2))@invJ_noRW
            if ind in range(len(self.act_bias_inds)):
                actind = self.act_bias_inds[ind]
                l = self.actuators[actind].input_len
                if ind in self.momentum_inds:
                    j = np.where(self.momentum_inds==ind)
                    ddxdot__dxdab[0:7,ind:ind+l,7+j] += np.squeeze(np.transpose(self.actuators[actind].ddstor_torq__dbiasdbasestate(u[actind],self,x,vecs),(1,0,2)))
                    ddxdot__dxdab[7+j,ind:ind+l,7+j] += np.transpose(self.actuators[actind].ddstor_torq__dbiasdh(u[actind],self,x,vecs),(1,0,2))
                    ddxdot__dabdab[ind:ind+l,ind:ind+l,7+j] = self.actuators[actind].ddstor_torq__dbiasdbias(u[actind],self,x,vecs)
                    ddxdot__dudab[actind,ind:ind+l,7+j] = self.actuators[actind].ddstor_torq__dudbias(u[actind],self,x,vecs)

            ddxdot__dxdab[:,:,7:] -= ddxdot__dxdab[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dudab[:,:,7:] -= ddxdot__dudab[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dabdab[:,:,7:] -= ddxdot__dabdab[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__dxddmp[:,:,7:] -= ddxdot__dxddmp[:,:,0:3]@RWaxes.T@mRWjs
            ddxdot__ddmpddmp[:,:,7:] -= ddxdot__ddmpddmp[:,:,0:3]@RWaxes.T@mRWjs

            return [[ddxdot__dxdx,ddxdot__dxdu,ddxdot__dxdab,ddxdot__dxdsb,ddxdot__dxddmp],[ddxdot__dxdu.T,ddxdot__dudu,ddxdot__dudab,ddxdot__dudsb,ddxdot__duddmp],[0,0,ddxdot__dabdab,ddxdot__dabdsb,ddxdot__dabddmp],[0,0,0,ddxdot__dsbdsb,ddxdot__dsbddmp],[0,0,0,0,ddxdot__ddmpddmp]]
        return [[ddxdot__dxdx,ddxdot__dxdu],[ddxdot__dxdu.T,ddxdot__dudu]]

    def sensor_bias_slice(self, att_sensor_index: int) -> Optional[slice]:
        r"""
        Slice for an attitude sensor's bias block in the full estimator state vector.

        The full estimator state is assumed to be ordered as:

        .. math::

            \mathbf{x}_{\mathrm{est}} =
            \begin{bmatrix}
            \mathbf{x} \\
            \mathbf{b}_{act} \\
            \mathbf{b}_{sens} \\
            \boldsymbol{\theta}_{dist}
            \end{bmatrix},

        where the sensor-bias block starts at offset:

        .. math::

            k_0 = n_x + n_{ab}
            \;=\;
            \texttt{self.state\_len} + \texttt{self.act\_bias\_len}.

        This method returns the Python ``slice(start, stop)`` selecting the bias elements for
        ``self.attitude_sensors[att_sensor_index]`` **within** :math:`\mathbf{x}_{\mathrm{est}}`.
        If the specified sensor does not have an estimated bias (i.e., not in ``att_sens_bias_inds``),
        the method returns ``None``.

        :param att_sensor_index: Index into ``self.attitude_sensors``.
        :type att_sensor_index: int

        :return: Slice selecting the sensor's bias portion in the estimator vector, or ``None``.
        :rtype: slice | None

        :raises RuntimeError:
            If the sensor index is listed as bias-estimated but the slice cannot be constructed.
        """
        if att_sensor_index not in self.att_sens_bias_inds:
            return None

        # Start of the sensor–bias block in the full state vector
        base = self.state_len + self.act_bias_len

        offset = 0
        for j in self.att_sens_bias_inds:
            out_len = self.attitude_sensors[j].output_length
            if j == att_sensor_index:
                start = base + offset
                return slice(start, start + out_len)
            offset += out_len

        # Should never be reached
        raise RuntimeError(
            f"Sensor index {att_sensor_index} is in att_sens_bias_inds, "
            "but its bias slice could not be constructed."
        )
    
    def control_covariance(self, *, form: str = "full") -> Covariance:
        """Return block-diagonal control-noise covariance for all actuators."""
        return Covariance.block_diagonal(
            (actuator.control_covariance() for actuator in self.actuators),
            form=form,
            coordinates="control",
        )

    def actuator_bias_process_psd(self, *, form: str = "full") -> Covariance:
        """Return continuous random-walk PSD for estimated actuator biases."""
        return Covariance.block_diagonal(
            (self.actuators[index].bias_process_psd() for index in self.act_bias_inds),
            form=form,
            coordinates="actuator_bias_rate",
        )

    def sensor_bias_process_psd(self, *, form: str = "full") -> Covariance:
        """Return continuous random-walk PSD for estimated sensor biases."""
        return Covariance.block_diagonal(
            (
                self.attitude_sensors[index].bias_process_psd()
                for index in self.att_sens_bias_inds
            ),
            form=form,
            coordinates="sensor_bias_rate",
        )

    def disturbance_parameter_process_psd(self, *, form: str = "full") -> Covariance:
        """Return continuous random-walk PSD for estimated disturbance parameters."""
        return Covariance.block_diagonal(
            (
                self.disturbances[index].parameter_process_psd()
                for index in self.dist_param_inds
            ),
            form=form,
            coordinates="disturbance_parameter_rate",
        )

    def measurement_covariance(
        self,
        which_sensors: Sequence[bool] | None = None,
        *,
        form: str = "full",
    ) -> Covariance:
        """Return selected sensor and all wheel measurement covariances.

        This compatibility API remains for existing callers. The estimator
        rewire should use :attr:`measurement_stack`, whose covariance mask
        covers both sensors and wheels and whose quaternion blocks are in
        residual coordinates.
        """
        if which_sensors is None:
            which_sensors = [True for _ in self.attitude_sensors]
        if len(which_sensors) != len(self.attitude_sensors):
            raise ValueError("which_sensors must match the number of attitude sensors")
        blocks = [
            sensor.measurement_covariance()
            for sensor, selected in zip(self.attitude_sensors, which_sensors)
            if selected
        ]
        blocks.extend(
            wheel.momentum_measurement_covariance() for wheel in self.rw_actuators
        )
        return Covariance.block_diagonal(
            blocks, form=form, coordinates="measurement"
        )

    @cached_property
    def measurement_stack(self):
        """Canonical estimator measurement order and model for this satellite."""
        from ADCS.estimators.measurement_stack import MeasurementStack

        return MeasurementStack(self)

    def control_cov(self) -> np.ndarray:
        r"""
        Block-diagonal covariance of actuator input noise.

        For each actuator :math:`j`, let its input noise covariance be
        :math:`\mathbf{R}_{u_j}` returned by ``actuator.noise.cov()``.
        This method constructs:

        .. math::

            \mathbf{R}_u = \mathrm{blkdiag}\left(\mathbf{R}_{u_1}, \mathbf{R}_{u_2}, \dots, \mathbf{R}_{u_{N_u}}\right).

        This covariance is commonly used as the control-noise covariance in stochastic propagation or
        estimator process-noise modeling.

        :return: Block-diagonal actuator noise covariance matrix.
        :rtype: numpy.ndarray
        """
        blocks = [actuator.noise.cov() for actuator in self.actuators]
        return np.array(block_diag(*blocks))

    def control_srcov(self) -> np.ndarray:
        r"""
        Block-diagonal square-root covariance of actuator input noise.

        For each actuator :math:`j`, let its square-root covariance be
        :math:`\mathbf{S}_{u_j}` returned by ``actuator.noise.srcov()`` such that:

        .. math::

            \mathbf{R}_{u_j} = \mathbf{S}_{u_j}\mathbf{S}_{u_j}^\top.

        This method constructs:

        .. math::

            \mathbf{S}_u = \mathrm{blkdiag}\left(\mathbf{S}_{u_1}, \mathbf{S}_{u_2}, \dots, \mathbf{S}_{u_{N_u}}\right).

        :return: Block-diagonal actuator noise square-root covariance matrix.
        :rtype: numpy.ndarray
        """
        blocks = [actuator.noise.srcov() for actuator in self.actuators]
        return np.array(block_diag(*blocks))

    def sensor_cov(self, which_sensors: Sequence[bool]) -> np.ndarray:
        r"""
        Block-diagonal covariance of attitude sensor noise (and wheel momentum measurement noise).

        This is retained for the current UKF/SRUKF implementations. The
        estimator rewire should replace it with ``measurement_stack.covariance``;
        unlike this legacy method, the stack mask also controls wheel entries.

        For each attitude sensor :math:`i` with covariance :math:`\mathbf{R}_{y_i}` (from
        ``sensor.noise.cov()``), the attitude-sensor covariance is:

        .. math::

            \mathbf{R}_y = \mathrm{blkdiag}\left(\mathbf{R}_{y_{i_1}}, \dots, \mathbf{R}_{y_{i_m}}\right),

        where the set :math:`\{i_1,\dots,i_m\}` is selected by ``which_sensors``.

        If reaction wheels are present, each wheel contributes an additional measurement covariance
        for its momentum measurement model (e.g., ``rw.h_meas_noise.cov()``), appended to the block diagonal.

        :param which_sensors: Boolean selector list of length ``len(self.attitude_sensors)`` indicating
            which attitude sensors to include. If ``None`` is passed in the implementation, all are included.
        :type which_sensors: list[bool]

        :return: Block-diagonal measurement covariance matrix. If no sensors are selected, returns a ``(0,0)`` array.
        :rtype: numpy.ndarray
        """
        if which_sensors is None:
            which_sensors = [True for j in self.attitude_sensors]
        blocks = []
        for j, attitude_sensor in enumerate(self.attitude_sensors):
            if which_sensors[j]:
                blocks.append(attitude_sensor.noise.cov())

        # Reaction wheel sensors (if any)
        for rw_sensor in self.rw_actuators:
            blocks.append(np.atleast_2d(rw_sensor.h_meas_noise.cov()))

        return block_diag(*blocks) if blocks else np.zeros((0, 0))

    def sensor_srcov(self, which_sensors: Sequence[bool]) -> np.ndarray:
        r"""
        Block-diagonal square-root covariance of attitude sensor noise (and wheel momentum measurement noise).

        For each attitude sensor :math:`i` with square-root covariance :math:`\mathbf{S}_{y_i}` (from
        ``sensor.noise.srcov()``) such that :math:`\mathbf{R}_{y_i}=\mathbf{S}_{y_i}\mathbf{S}_{y_i}^\top`,
        this method constructs

        .. math::

            \mathbf{S}_y = \mathrm{blkdiag}\left(\mathbf{S}_{y_{i_1}}, \dots, \mathbf{S}_{y_{i_m}}\right),

        with selection controlled by ``which_sensors``.

        If reaction wheels are present, each wheel contributes an additional square-root covariance
        block for its momentum measurement model (e.g., ``rw.h_meas_noise.srcov()``).

        :param which_sensors: Boolean selector list of length ``len(self.attitude_sensors)`` indicating
            which attitude sensors to include. If ``None`` is passed in the implementation, all are included.
        :type which_sensors: list[bool]

        :return: Block-diagonal measurement square-root covariance matrix. If no sensors are selected, returns a ``(0,0)`` array.
        :rtype: numpy.ndarray
        """
        if which_sensors is None:
            which_sensors = [True for j in self.attitude_sensors]
        blocks = []
        for j, attitude_sensor in enumerate(self.attitude_sensors):
            if which_sensors[j]:
                blocks.append(attitude_sensor.noise.srcov())

        # Reaction wheel sensors
        for rw_sensor in self.rw_actuators:
            blocks.append(np.atleast_2d(rw_sensor.h_meas_noise.srcov()))

        return block_diag(*blocks) if blocks else np.zeros((0, 0))
