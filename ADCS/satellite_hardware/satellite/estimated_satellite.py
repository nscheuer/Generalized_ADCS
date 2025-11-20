__all__ = ["EstimatedSatellite"]
import numpy as np

from typing import List, Dict, Union, Tuple, Any, Optional
from scipy.linalg import block_diag

from .satellite import Satellite

import ADCS.orbits.universal_constants as uc
from ADCS.helpers.math_helpers import *
from ADCS.satellite_hardware.disturbances import Disturbance, SRP_Disturbance, General_Disturbance, Prop_Disturbance
from ADCS.satellite_hardware.sensors import Sensor, GPS
from ADCS.satellite_hardware.actuators import Actuator, RW
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.logging.logger import ADCSLogger
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedArray

class EstimatedSatellite(Satellite):
    """
    A satellite model that augments the base :class:`~ADCS.satellite_hardware.satellite.Satellite`
    with estimator-driven state synchronization and bias/parameter tracking.

    This class bridges the simulated physical satellite and an estimation framework
    (e.g., Kalman filter or nonlinear observer). It allows dynamic updates of the
    satellite’s internal state, actuator and sensor biases, and disturbance parameters
    using estimated data.

    The :class:`EstimatedSatellite` is primarily intended for **hardware-in-the-loop**
    or **estimation validation** scenarios, where the true spacecraft dynamics are
    compared or synchronized with the estimated state vector.

    Parameters
    ----------
    mass : float, optional
        Satellite mass in kilograms. Default is ``1.0``.
    COM : numpy.ndarray, optional
        Center of mass vector in body coordinates [m].
    J_0 : numpy.ndarray, optional
        Nominal inertia tensor about the body frame [kg·m²].
    disturbances : list[Disturbance], optional
        List of disturbance models (e.g., SRP, aerodynamic, propulsion)
        that influence satellite dynamics.
    sensors : list[Sensor], optional
        List of sensor objects (e.g., :class:`~ADCS.satellite_hardware.sensors.GPS`).
    actuators : list[Actuator], optional
        List of actuator objects (e.g., :class:`~ADCS.satellite_hardware.actuators.RW`).

    Attributes
    ----------
    act_bias_inds : list[int]
        Indices of actuators that include estimated bias parameters.
    act_bias_len : int
        Total length of actuator bias state components.
    att_sens_bias_inds : list[int]
        Indices of attitude sensors that include estimated bias parameters.
    att_sens_bias_len : int
        Total length of attitude sensor bias state components.
    dist_param_inds : list[int]
        Indices of disturbances that include estimated parameters.
    dist_param_len : int
        Total length of disturbance parameter state components.
    actuators : list[Actuator]
        List of actuator instances associated with the satellite.
    sensors : list[Sensor]
        List of sensor instances associated with the satellite.
    disturbances : list[Disturbance]
        List of disturbance models influencing dynamics.

    Notes
    -----
    - The constructor automatically determines which actuators, sensors, and
      disturbances include **estimated parameters** by checking their
      respective attributes:
      ``.estimated_bias`` or ``.estimated_param``.
    - These indices and lengths are stored to facilitate block-based extraction
      and update of corresponding state segments during estimation.
    - The model can later be synchronized with an estimator output using
      :meth:`match_estimate`, which aligns simulated states and parameters
      with the current estimated values.

    Examples
    --------
    >>> from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
    >>> from ADCS.satellite_hardware.actuators import RW
    >>> from ADCS.satellite_hardware.sensors import GPS
    >>> sat = EstimatedSatellite(
    ...     mass=2.5,
    ...     COM=np.array([0.0, 0.0, 0.02]),
    ...     actuators=[RW(...), RW(...), RW(...)],
    ...     sensors=[GPS()]
    ... )
    >>> print(sat.act_bias_len, sat.att_sens_bias_len)
    3  # number of bias parameters found
    """

    def __init__(self, mass: float = 1.0, COM: np.ndarray = None, J_0: np.ndarray = None, disturbances: List[Disturbance] = None, sensors: List[Sensor] = None, actuators: List[Actuator] = None) -> None:
        super().__init__(mass, COM, J_0, disturbances, sensors, actuators)

        # Add estimated states
        self.act_bias_inds = [j for j in range(len(self.actuators)) if self.actuators[j].estimate_bias] # Indices with actuator bias
        self.act_bias_len = sum([self.actuators[j].input_len for j in self.act_bias_inds]) # Number of actuators with biases
        self.att_sens_bias_inds = [j for j in range(len(self.attitude_sensors)) if self.sensors[j].estimate_bias] # Indices with sensor bias
        self.att_sens_bias_len = sum([self.attitude_sensors[j].output_length for j in self.att_sens_bias_inds]) # Number of sensors with bias
        self.dist_param_inds = [j for j in range(len(self.disturbances)) if self.disturbances[j].estimate_dist] # Indices with sensor disturbaces
        self.dist_param_len = sum([self.disturbances[j].estimated_vector_length for j in self.dist_param_inds]) # Number of sensors with bias

    def match_estimate(self, est_state: EstimatedArray, dt: float) -> None:
        """
        Synchronize the satellite model with the latest estimated state and covariance.

        This method updates the satellite’s internal state, actuator and sensor biases,
        and disturbance parameters using an :class:`EstimatedArray` provided by an estimator.
        It also disables noise and time variation in subsystems to maintain deterministic
        consistency between simulation and estimation models.

        Parameters
        ----------
        est : EstimatedArray
            The estimated full state vector including the satellite’s dynamic state,
            actuator biases, sensor biases, and disturbance parameters. Must include
            covariance (`cov`) and integrated covariance (`int_cov`) matrices.
        dt : float
            Integration time step, used to normalize the integrated covariance.

        Raises
        ------
        ValueError
            If the provided estimated state does not match the expected total length.

        Notes
        -----
        - The method assumes that `est.val` has the following structure:

        .. math::

            [ x_{state},\; b_{act},\; b_{sens},\; p_{dist} ]

        where each segment corresponds to the system state, actuator biases,
        sensor biases, and disturbance parameters.

        - After synchronization, actuator and sensor noise are disabled via
        ``use_noise = False``, and disturbance time variations are frozen by setting
        ``time_varying = False``. This ensures deterministic propagation when
        comparing true and estimated dynamics.

        Examples
        --------
        >>> sat = EstimatedSatellite(...)
        >>> est = EstimatedArray(val, cov, int_cov)
        >>> sat.match_estimate(est, dt=0.1)
        """
        # Extract state
        full_state = est_state.val
        cov = est_state.cov
        int_cov = est_state.int_cov

        # Dimension checks
        expected_len = (self.state_len + self.act_bias_len + self.att_sens_bias_len + self.dist_param_len)
        if np.size(full_state) != expected_len:
            raise ValueError(f"Estimator state has wrong size (expected {expected_len}, got {np.size(full_state)})")
        
        # Adjustment if integrated covariance is off by one (e.g. quaternion handling)
        adj = -1 if int_cov.shape[0] + 1 == expected_len else 0

        # --- Partition the full state vector ---
        idx = self.state_len
        act_bias = full_state[idx : idx + self.act_bias_len]
        act_bias_ic = int_cov[
            idx + adj : idx + self.act_bias_len + adj,
            idx + adj : idx + self.act_bias_len + adj,
        ]

        idx += self.act_bias_len
        sens_bias = full_state[idx : idx + self.att_sens_bias_len]
        sens_bias_ic = int_cov[
            idx + adj : idx + self.att_sens_bias_len + adj,
            idx + adj : idx + self.att_sens_bias_len + adj,
        ]

        idx += self.att_sens_bias_len
        dist_param = full_state[idx : idx + self.dist_param_len]
        dist_param_ic = int_cov[
            idx + adj : idx + self.dist_param_len + adj,
            idx + adj : idx + self.dist_param_len + adj,
        ]

        # --- Update satellite dynamic state ---
        self.update_RWhs(full_state[: self.state_len])

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
            act.set_bias(act_bias[ind : ind + l])
            act.bias_std_rate = np.sqrt(act_bias_ic[ind : ind + l, ind : ind + l])
            ind += l

        # --- Update sensor biases ---
        ind = 0
        for j in self.att_sens_bias_inds:
            sens = self.attitude_sensors[j]
            l = sens.output_length
            sens.bias = sens_bias[ind : ind + l]
            sens.bias_std_rate = np.sqrt(sens_bias_ic[ind : ind + l, ind : ind + l])
            ind += l

        # --- Update disturbance parameters ---
        ind = 0
        for j in self.dist_param_inds:
            dist = self.disturbances[j]
            if dist.active:  # Only update active ones
                l = dist.main_param.size
                dist.main_param = dist_param[ind : ind + l]
                dist.std = np.sqrt(dist_param_ic[ind : ind + l, ind : ind + l])
                ind += l

    def dist_torques_jacobian(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobian of the total disturbance torque with respect to both
        the spacecraft state and disturbance model parameters.

        This method accumulates the first-order derivatives contributed by each
        disturbance source in :attr:`self.disturbances`. The quaternion components
        of the state (indices 3--6) determine the attitude-dependent part of the
        disturbance torque, so only these affect the state Jacobian.

        .. math::

            \\frac{\\partial \\boldsymbol{\\tau}_d}{\\partial \\mathbf{x}},
            \\qquad
            \\frac{\\partial \\boldsymbol{\\tau}_d}{\\partial \\boldsymbol{\\theta}_d}

        where :math:`\\boldsymbol{\\tau}_d` is the total disturbance torque,
        :math:`\\mathbf{x}` is the spacecraft state vector, and
        :math:`\\boldsymbol{\\theta}_d` represents disturbance model parameters.

        Parameters
        ----------
        x : numpy.ndarray
            Current spacecraft state vector of length ``state_len``.
        vecs : Dict[str, numpy.ndarray]
            Dictionary of environmental vectors (e.g., magnetic field, Sun vector,
            aerodynamic flow) passed to each disturbance model.

        Returns
        -------
        ddist_torq__dx : numpy.ndarray, shape (state_len, 3)
            Partial derivative of disturbance torque with respect to the state vector.
            Non-zero contributions occur only in the quaternion block (indices 3–6).
        ddist_torq__ddmp : numpy.ndarray, shape (dist_param_len, 3)
            Partial derivative of disturbance torque with respect to disturbance
            parameters. Empty if no disturbance parameters are defined.

        Notes
        -----
        - Each disturbance model ``j`` must implement:
        :func:`torque_qjac(self, vecs)` for attitude derivatives and
        :func:`torque_valjac(self, vecs)` for parameter derivatives.
        - The total Jacobian is computed by summing the contributions from all
        registered disturbances.
        """
        ddist_torq__dx = np.zeros((self.state_len,3))
        ddist_torq__dx[3:7,:] = sum([j.torque_qjac(self,vecs) for j in self.disturbances],np.zeros((4,3)))
        ddist_torq__ddmp = np.zeros((0,3))
        if self.dist_param_len>0:
            ddist_torq__ddmp = np.vstack([self.disturbances[j].torque_valjac(self,vecs) for j in self.dist_param_inds])
        return ddist_torq__dx,ddist_torq__ddmp

    def dist_torque_hess(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Hessian tensors of the total disturbance torque with respect to
        both the spacecraft state and the disturbance model parameters.

        This routine aggregates the second-order derivatives of all disturbance
        models, yielding three Hessian tensors corresponding to pure state,
        mixed state–parameter, and pure parameter terms.

        .. math::

            \\frac{\\partial^2 \\boldsymbol{\\tau}_d}{\\partial \\mathbf{x}^2}, \\qquad
            \\frac{\\partial^2 \\boldsymbol{\\tau}_d}{\\partial \\mathbf{x}\\,\\partial \\boldsymbol{\\theta}_d}, \\qquad
            \\frac{\\partial^2 \\boldsymbol{\\tau}_d}{\\partial \\boldsymbol{\\theta}_d^2}

        Parameters
        ----------
        x : numpy.ndarray
            Current spacecraft state vector of length ``state_len``.
        vecs : Dict[str, numpy.ndarray]
            Dictionary of environmental vectors required by the disturbance models.

        Returns
        -------
        dddist_torq__dxdx : numpy.ndarray, shape (state_len, state_len, 3)
            Second derivative of the disturbance torque with respect to the state.
            Only the quaternion block (indices 3–6) contains non-zero entries.
        dddist_torq__dxddmp : numpy.ndarray, shape (state_len, dist_param_len, 3)
            Cross-derivative tensor coupling state and parameter effects.
        dddist_torq__ddmpddmp : numpy.ndarray, shape (dist_param_len, dist_param_len, 3)
            Second derivative of the disturbance torque with respect to disturbance
            model parameters.

        Notes
        -----
        - Each disturbance model ``j`` must implement the following methods:

        * :func:`torque_qqhess(self, vecs)` — second derivative w.r.t. quaternion.
        * :func:`torque_qvalhess(self, vecs)` — mixed derivative (quaternion–parameter).
        * :func:`torque_valvalhess(self, vecs)` — second derivative w.r.t. parameters.

        - Parameter block sizes are managed according to
        :attr:`self.dist_param_inds` and the parameter dimension of each disturbance
        model.
        """
        dddist_torq__dxdx = np.zeros((self.state_len,self.state_len,3))
        dddist_torq__dxdx[3:7,3:7,:] = sum([j.torque_qqhess(self,vecs) for j in self.disturbances],np.zeros((4,4,3)))
        dddist_torq__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,3))
        dddist_torq__dxddmp = np.zeros((self.state_len,self.dist_param_len,3))
        ind = 0
        for j in self.dist_param_inds:
            l = self.disturbances[j].main_param.size
            dddist_torq__ddmpddmp[ind:ind+l,ind:ind+l,:] = self.disturbances[j].torque_valvalhess(self,vecs)
            dddist_torq__dxddmp[3:7,ind:ind+l,:] = self.disturbances[j].torque_qvalhess(self,vecs)
            ind += l
        return dddist_torq__dxdx,dddist_torq__dxddmp,dddist_torq__ddmpddmp
    
    def dynJacCore(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State) -> Union[np.ndarray, np.ndarray]:
        R = orbital_state.R # Position in ECI [km]
        V = orbital_state.V # Velocity in body frame [km/s]
        B = orbital_state.B # Magnetic field in ECI [T]
        S = orbital_state.S # Sun Vector in ECI [km]
        rho = orbital_state.rho # Atmospheric density [kg/m^3]

        w = x[0:3]
        q = x[4:7]
        RWhs = x[7:]
        J = self.J_0
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
        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])

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
            dact_torq__dh = np.vstack([self.actuators[j].dtorq__dh(u[j],self,x,vecs) for j in range(len(self.actuators))])
            RWjs = np.array([rw.J for rw in self.rw_actuators])
            RWaxes = np.vstack([rw.axis for rw in self.rw_actuators])
            mRWjs = np.diagflat(RWjs)
            dxdot__dx[0:3,0:3] += -skewsym(RWhs@RWaxes)@invJ_noRW
            dxdot__dx[7:,0:3] += (dact_torq__dh+np.cross(RWaxes,w))@invJ_noRW
            dxdot__du[:,7:] = block_diag(*[self.actuators[j].dstor_torq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
            dxdot__du[:,7:] -= dxdot__du[:,0:3]@RWaxes.T@mRWjs
            dxdot__dx[0:7,7:] = np.hstack([act.dstor_torq__dbasestate(u[j],self,x,vecs) for act in self.actuators])
            dxdot__dx[7:,7:] = np.diagflat([rw.dstor_torq__dh(u[j],self,x,vecs) for rw in self.rw_actuators])
            dxdot__dx[:,7:] -= dxdot__dx[:,0:3]@RWaxes.T@mRWjs
        dxdot__dab = np.zeros((self.act_bias_len,self.state_len))
        dxdot__dsb = np.zeros((self.att_sens_bias_len,self.state_len))
        dxdot__ddmp = np.zeros((self.dist_param_len,self.state_len))
        if self.act_bias_len>0:
            dact_torq__dab = np.vstack([self.actuators[j].dtorq__dbias(u[j],self,x,vecs) for j in self.act_bias_inds])
        else:
            dact_torq__dab = np.zeros((0,3))

        dxdot__dab[:,0:3] = dact_torq__dab@invJ_noRW

        dxdot__ddmp[:,0:3] = ddist_torq__ddmp@invJ_noRW

        if self.number_RW>0:
            dxdot__dab[:,7:] = block_diag(*[self.actuators[j].dstor_torq__dbias(u[j],self,x,vecs).T for j in self.act_bias_inds]).T
            dxdot__dab[:,7:] -= dxdot__dab[:,0:3]@RWaxes.T@mRWjs
            dxdot__ddmp[:,7:] -= dxdot__ddmp[:,0:3]@RWaxes.T@mRWjs

        return [dxdot__dx,dxdot__du,dxdot__dab,dxdot__dsb,dxdot__ddmp]
    

    def dynamics_Hessians(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State) -> List[List[np.ndarray]]:
        r"""
        Compute the **second-order derivatives (Hessians)** of the spacecraft attitude dynamics
        with respect to all relevant variables — including state, control inputs, actuator biases,
        sensor biases, and disturbance parameters.

        This function provides a full analytic **second-order linearization** of the rotational
        dynamics model, extending beyond the first-order Jacobians to capture curvature effects
        in the equations of motion. These Hessians are used in second-order control, estimation,
        and sensitivity analysis algorithms such as Differential Dynamic Programming (DDP),
        Iterative LQR (iLQR), or Extended Second-Order Kalman Filters.

        ---
        **Mathematical Definition**

        The nonlinear attitude dynamics are expressed as:

        .. math::

            \begin{aligned}
            \dot{\boldsymbol{\omega}} &= J^{-1}
            \Big[
                -\boldsymbol{\omega} \times (J \boldsymbol{\omega} + A_{RW}^T \mathbf{h}_{RW})
                + \boldsymbol{\tau}_{\text{act}}(\mathbf{x}, \mathbf{u})
                + \boldsymbol{\tau}_{\text{dist}}(\mathbf{x})
            \Big] \\
            \dot{\mathbf{q}} &= \tfrac{1}{2} W(\mathbf{q})^T \boldsymbol{\omega} \\
            \dot{\mathbf{h}}_{RW} &= \mathbf{u}_{RW}
                - \mathrm{diag}(J_{RW}) A_{RW}^T \dot{\boldsymbol{\omega}}
            \end{aligned}

        where:
        - :math:`J` is the spacecraft body inertia matrix,
        - :math:`A_{RW}` contains reaction wheel spin axes,
        - :math:`\mathbf{h}_{RW}` are reaction wheel momenta,
        - :math:`\boldsymbol{\tau}_{\text{act}}` and :math:`\boldsymbol{\tau}_{\text{dist}}`
        represent actuator and disturbance torques respectively.

        This function computes the full set of **second-order derivatives** of the dynamics function:

        .. math::
            f(\mathbf{x}, \mathbf{u}) = \dot{\mathbf{x}},

        i.e.

        .. math::
            \frac{\partial^2 f_i}{\partial z_j \, \partial z_k},
            \quad
            \mathbf{z} = [\mathbf{x}, \mathbf{u}, \boldsymbol{\theta}_{act}, \boldsymbol{\theta}_{sens}, \boldsymbol{\theta}_{dist}],
        
        where each tensor element represents a curvature term coupling any two of the state,
        control, or parameter dimensions.

        ---
        **Inputs**

        :param x:
            Full spacecraft state vector:
            .. math::

                \mathbf{x} =
                \begin{bmatrix}
                    \boldsymbol{\omega} \\
                    \mathbf{q} \\
                    \mathbf{h}_{RW}
                \end{bmatrix}

            - :math:`\boldsymbol{\omega}` — angular velocity in body frame [rad/s], shape ``(3,)``  
            - :math:`\mathbf{q}` — quaternion (Hamilton form, body→ECI), shape ``(4,)``  
            - :math:`\mathbf{h}_{RW}` — reaction wheel momenta, shape ``(n_{RW},)``  

        :type x: numpy.ndarray

        :param u:
            Control input vector (actuator torque or wheel speed command signals), shape ``(n_u,)``.

        :type u: numpy.ndarray

        :param orbital_state:
            Object containing orbital and environmental quantities:
            - ``R``: ECI position vector [m]  
            - ``V``: ECI velocity vector [m/s]  
            - ``B``: Magnetic field vector [T]  
            - ``S``: Sun direction vector [-]  
            - ``rho``: Atmospheric density [kg/m³]  

            These vectors are rotated into the body frame using :func:`rot_mat(q).T`, and both
            their first and second quaternion derivatives (:func:`drotmatTvecdq`,
            :func:`ddrotmatTvecdqdq`) are computed for use in torque partials.

        :type orbital_state: Orbital_State

        ---
        **Outputs**

        :return:
            Nested list of Hessian tensors. The exact structure depends on whether
            estimation-related parameters (biases, disturbances) are enabled.

            **If extended estimation is active:**

            .. code-block:: text

                [
                [ddxdot__dxdx, ddxdot__dxdu, ddxdot__dxdab, ddxdot__dxdsb, ddxdot__dxddmp],
                [ddxdot__dxdu.T, ddxdot__dudu, ddxdot__dudab, ddxdot__dudsb, ddxdot__duddmp],
                [0, 0, ddxdot__dabdab, ddxdot__dabdsb, ddxdot__dabddmp],
                [0, 0, 0, ddxdot__dsbdsb, ddxdot__dsbddmp],
                [0, 0, 0, 0, ddxdot__ddmpddmp]
                ]

            Each entry is a 3D array of the form ``(dim₁, dim₂, n_state)``, where each slice
            corresponds to the curvature of one state derivative component.

            +-------------------+--------------------------------------+--------------------------+
            | Symbol            | Description                          | Shape                    |
            +===================+======================================+==========================+
            | ``ddxdot__dxdx``  | ∂²ẋ / ∂x²  (state Hessian)           | (nₓ, nₓ, nₓ)            |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dxdu``  | ∂²ẋ / ∂x∂u  (state–control coupling) | (nₓ, nᵤ, nₓ)            |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dudu``  | ∂²ẋ / ∂u²  (control Hessian)         | (nᵤ, nᵤ, nₓ)            |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dxdab`` | ∂²ẋ / ∂x∂(actuator bias)             | (nₓ, n_{ab}, nₓ)        |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dxdsb`` | ∂²ẋ / ∂x∂(sensor bias)               | (nₓ, n_{sb}, nₓ)        |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dxddmp``| ∂²ẋ / ∂x∂(disturbance param)         | (nₓ, n_{dp}, nₓ)        |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dabdab``| ∂²ẋ / ∂(actuator bias)²              | (n_{ab}, n_{ab}, nₓ)    |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__dsbdsb``| ∂²ẋ / ∂(sensor bias)²                | (n_{sb}, n_{sb}, nₓ)    |
            +-------------------+--------------------------------------+--------------------------+
            | ``ddxdot__ddmpddmp``| ∂²ẋ / ∂(disturbance param)²        | (n_{dp}, n_{dp}, nₓ)    |
            +-------------------+--------------------------------------+--------------------------+

            **If estimation is disabled:**

            .. code-block:: text

                [
                [ddxdot__dxdx, ddxdot__dxdu],
                [ddxdot__dxdu.T, ddxdot__dudu]
                ]

        :rtype: List[List[np.ndarray]]

        ---
        **Computation Summary**

        1. **Quaternion and environment transformation**
        
        Environmental vectors (``R``, ``V``, ``B``, ``S``) are rotated into the body frame and
        their first- and second-order quaternion derivatives are computed to capture
        attitude-dependent effects in environmental torques.

        2. **Actuator torque Hessians**

        For each actuator, compute:
        - :math:`\frac{\partial^2 \boldsymbol{\tau}_{act}}{\partial \mathbf{x}^2}`,
        - :math:`\frac{\partial^2 \boldsymbol{\tau}_{act}}{\partial \mathbf{u}^2}`,
        - :math:`\frac{\partial^2 \boldsymbol{\tau}_{act}}{\partial \mathbf{x} \partial \mathbf{u}}`,
        - and mixed terms w.r.t. wheel momentum or bias parameters.

        3. **Disturbance torque Hessians**

        Second derivatives of environmental torque models
        :math:`\boldsymbol{\tau}_{dist}(\mathbf{x})`
        are obtained from :func:`dist_torque_hess`.

        4. **Reaction wheel coupling**

        If reaction wheels are active (:attr:`self.number_RW > 0`),
        adds higher-order coupling terms due to wheel inertia, axes, and stored momentum:

        .. math::
            \frac{\partial^2 \dot{\omega}}{\partial h_{RW}\, \partial x},
            \quad
            \frac{\partial^2 \dot{h}_{RW}}{\partial x^2}

        5. **Bias and disturbance parameter extension**

        If biases and disturbances are modeled, additional Hessians
        are computed for cross-derivatives w.r.t. actuator bias,
        sensor bias, and disturbance parameters.

        6. **Inertia mapping**

        All torque Hessians are post-multiplied by :math:`J^{-1}` to yield angular acceleration
        Hessians in body coordinates.

        ---
        **Notes**

        - Quaternion convention follows **Hamilton form**, representing a body→ECI rotation.
        - All returned Hessian tensors are stored as 3D arrays; index 3 corresponds to the affected
        component of :math:`\dot{\mathbf{x}}`.
        - Symmetry of second derivatives is preserved wherever applicable.
        - These tensors can be contracted with perturbation vectors to compute second-order
        variations in the dynamics, e.g.:

        .. math::
            \delta^2 \dot{\mathbf{x}} \approx
            \tfrac{1}{2} \sum_{i,j} \frac{\partial^2 f}{\partial z_i \partial z_j}
            \delta z_i \delta z_j

        ---
        **Example**

        .. code-block:: python

            ddH = sat.dynamics_Hessians(x, u, orb)
            ddx_dx, dx_du = ddH[0]
            print(ddx_dx.shape)   # (n_x, n_x, n_x)
            print(dx_du.shape)    # (n_x, n_u, n_x)

        ---
        **References**

        - Wie, B. *Space Vehicle Dynamics and Control*, 2nd Ed. AIAA, 2008.  
        - Crassidis, J.L., Junkins, J.L. *Optimal Estimation of Dynamic Systems*, 2nd Ed. CRC Press, 2011.  
        - Tassa, Y. *et al.*, “Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization,” *IROS*, 2012.  
        - Diebel, J., “Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors,” Stanford University, 2006.
        """
        w = x[0:3]#.reshape((3,1))
        q = x[3:7]#normalize(x[3:7,:])
        RWhs = x[7:]
        invJ_noRW = self.invJ_noRW
        J = self.J

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

        dact_torq__dbase = sum([self.actuators[j].dtorq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,3)))
        ddact_torq__dbasedbase = sum([self.actuators[j].ddtorq__dbasestatedbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))],np.zeros((7,7,3)))
        dact_torq__du = np.vstack([self.actuators[j].dtorq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
        ddact_torq__dudu = np.zeros((self.control_len,self.control_len,3))
        ddact_torq__dudbase = np.zeros((self.control_len,7,3))
        for j in range(len(self.actuators)):
            ddact_torq__dudu[j,j,:] = self.actuators[j].ddtorq__dudu(u[j],self,x,vecs)
            ddact_torq__dudbase[j,:,:] = self.actuators[j].ddtorq__dudbasestate(u[j],self,x,vecs)


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
                ddact_torq__dudh[j,ind,:] = self.actuators[j].ddtorq__dudh(u[j],self,x,vecs)
                ddact_torq__dhdh[ind,ind,:] = self.actuators[j].ddtorq__dhdh(u[j],self,x,vecs)
                ddact_torq__dbasedh[:,ind,:] = np.squeeze(self.actuators[j].ddtorq__dbasestatedh(u[j],self,x,vecs))

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
                ddxdot__dxdu[0:7,j,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dudbasestate(u[j],self,x,vecs),(1,0,2)))
                ddxdot__dxdu[7+ind,j,7+ind] += np.transpose(self.actuators[j].ddstor_torq__dudh(u[j],self,x,vecs),(1,0,2))
                ddxdot__dudu[j,j,7+ind] = self.actuators[j].ddstor_torq__dudu(u[j],self,x,vecs)
                ddxdot__dxdx[0:7,0:7,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedbasestate(u[j],self,x,vecs))
                ddxdot__dxdx[7+ind,0:7,7+ind] += np.squeeze(np.transpose(self.actuators[j].ddstor_torq__dbasestatedh(u[j],self,x,vecs),(1,0,2)))
                ddxdot__dxdx[0:7,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dbasestatedh(u[j],self,x,vecs))
                ddxdot__dxdx[7+ind,7+ind,7+ind] += np.squeeze(self.actuators[j].ddstor_torq__dhdh(u[j],self,x,vecs))

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
        """
        Return the [start:stop] slice in the *full* estimator state vector
        corresponding to the bias of attitude_sensors[att_sensor_index].

        Full estimator state is ordered as:
            [ base_state (state_len),
              actuator_bias (act_bias_len),
              attitude_sensor_bias (att_sens_bias_len),
              disturbance_params (dist_param_len) ]

        If the sensor has no associated bias state, returns None.
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
    
    def control_cov(self) -> np.ndarray:
        """
        Block-diagonal covariance matrix for all actuator noises.
        """
        blocks = [actuator.noise.cov() for actuator in self.actuators]
        return np.array(block_diag(*blocks))

    def control_srcov(self) -> np.ndarray:
        """
        Block-diagonal square-root covariance matrix for all actuator noises.
        """
        blocks = [actuator.noise.srcov() for actuator in self.actuators]
        return np.array(block_diag(*blocks))

    def sensor_cov(self) -> np.ndarray:
        """
        Block-diagonal covariance matrix for all attitude sensor noises.
        """
        blocks = []
        blocks.extend([attitude_sensor.noise.cov() for attitude_sensor in self.attitude_sensors])
        blocks.extend([np.atleast_2d(rw_sensor.h_meas_noise.cov()) for rw_sensor in self.rw_actuators])
        return np.array(block_diag(*blocks))

    def sensor_srcov(self) -> np.ndarray:
        """
        Block-diagonal square-root covariance matrix for all attitude sensor noises.
        """
        blocks = []
        blocks.extend([attitude_sensor.noise.srcov() for attitude_sensor in self.attitude_sensors])
        blocks.extend([np.atleast_2d(rw_sensor.h_meas_noise.srcov()) for rw_sensor in self.rw_actuators])
        return np.array(block_diag(*blocks))