__all__ = ["Satellite"]

import numpy as np

from typing import List, Dict, Union, Tuple, Any
from scipy.linalg import block_diag

import ADCS.orbits.universal_constants as uc
from ADCS.helpers.math_helpers import *
from ADCS.satellite_hardware.disturbances import Disturbance, SRP_Disturbance, General_Disturbance, Prop_Disturbance
from ADCS.satellite_hardware.sensors import Sensor, GPS
from ADCS.satellite_hardware.actuators import Actuator, RW
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.logging.logger import ADCSLogger

class Satellite:
    """
    Represents a rigid satellite with inertia, actuators, and sensors.

    This class defines the physical parameters of a spacecraft,
    including its mass, center of mass (COM), inertia matrix, and
    associated subsystems. It also provides utilities to compute
    derived inertial properties such as the inertia at the center
    of mass and bus-only inertia (excluding reaction wheels).

    Parameters
    ----------
    mass : float, optional
        Total satellite mass [kg]. Defaults to 1.0.
    COM : np.ndarray, optional
        Center of mass vector (shape (3,)). Defaults to [0, 0, 0].
    J_0 : np.ndarray, optional
        Inertia tensor about the reference origin (shape (3,3)).
        Defaults to identity.
    disturbances : List[Disturbance], optional
        List of environmental disturbance models (e.g., drag, SRP).
    sensors : List[Sensor], optional
        List of sensor models (e.g., gyros, star trackers, GPS).
    actuators : List[Actuator], optional
        List of actuator models (e.g., reaction wheels, thrusters).

    Attributes
    ----------
    mass : float
        Satellite total mass [kg].
    COM : np.ndarray
        Center of mass vector (3,).
    J_0 : np.ndarray
        Inertia tensor about the reference origin (3,3).
    J_COM : np.ndarray
        Inertia tensor about the center of mass (3,3).
    J_noRW : np.ndarray
        Inertia tensor without reaction wheel contributions (3,3).
    invJ_COM : np.ndarray
        Inverse of the inertia tensor at the center of mass.
    state_len : int
        Dimension of the satellite's full state vector.

    Raises
    ------
    ValueError
        If COM or J_0 are not of the correct shape.
    """
    def __init__(self, mass: float = 1.0, COM: np.ndarray = None, J_0: np.ndarray = None, disturbances: List[Disturbance] = [], sensors: List[Sensor] = [], actuators: List[Actuator] = [], logger: ADCSLogger = None) -> None:
        # Assign variables
        self.mass = mass # Includes angular momentum storage
        self.COM = np.asarray(COM, dtype=float) # Includes angular momentum storage
        if COM is None:
            self.COM = np.zeros(3)
        else:
            self.COM = np.asarray(COM, dtype=float)
            if self.COM.shape != (3,):
                raise ValueError(f"COM must be a numpy array of shape (3,), got {self.COM.shape}")

        if J_0 is None:
            self.J_0 = np.eye(3)
        else:
            self.J_0 = np.asarray(J_0, dtype=float)
            if self.J_0.shape != (3, 3):
                raise ValueError(f"J must be a numpy array of shape (3, 3), got {self.J_0.shape}")
        self.disturbances = disturbances
        self.sensors = sensors
        self.actuators = actuators
        self.logger = logger

        # Filter sensors
        self.attitude_sensors_ind = [j for j in sensors if j.attitude_sensor]
        self.other_sensors_ind = [j for j in sensors if not j.attitude_sensor and not isinstance(j,GPS)]
        self.orbit_sensors_ind = [j for j in sensors if isinstance(j,GPS)]

        # Filter actuators
        self.momentum_actuators_ind = [j for j in range(len(self.actuators)) if self.actuators[j].has_momentum]
        self.num_RW = sum([1 for j in self.actuators if isinstance(j,RW)])

        # Initialize state
        self.state_len = 7 + self.num_RW

    def update_J(self, J: np.ndarray = None, COM: np.ndarray = None) -> None:
        """
        Update the satellite's inertia matrices.

        Applies physical validation, symmetry checks, and the
        parallel axis theorem to shift the inertia tensor from
        the reference origin to the center of mass (COM).

        Parameters
        ----------
        J : np.ndarray, optional
            Inertia tensor about the reference origin (3,3). If None,
            uses the stored ``self.J_0``.
        COM : np.ndarray, optional
            Center of mass offset (3,). If None, uses the stored ``self.COM``.

        Raises
        ------
        ValueError
            If J cannot be reshaped to (3,3), contains non-real values,
            is non-symmetric, or not positive definite.

        Notes
        -----
        This method also computes:
            - ``self.J_COM`` : inertia about the COM,
            - ``self.J_noRW`` : inertia without reaction wheels,
            - Inverse matrices for each of these for efficient dynamics.
        """
        if J is None: J = self.J_0
        if COM is None: COM = self.COM

        try:
            J = np.array(J, dtype=float).reshape((3,3))
        except:
            raise ValueError("J must be convertible to a (3,3) array")
        
        # Physical validity checks
        if not np.all(np.isreal(J)):
            raise ValueError("J contains non-real values")
        if not np.allclose(J, J.T, rtol=1e-5, atol=1e-8):
            raise ValueError("J must be symmetric")
        J = 0.5 * (J + J.T)  # enforce symmetry

        eigvals = np.linalg.eigvals(J)
        if not np.all(eigvals > 0):
            print("Inertia eigenvalues:", eigvals)
            raise ValueError("J must be positive definite")
        
        self.J_0 = J
        self.invJ_0 = np.linalg.inv(J)

        # Apply the parallel axis theorem to move to COM
        self.J_COM = J - self.mass * (np.eye(3) * np.dot(COM, COM) - np.outer(COM, COM))
        self.invJ_COM = np.linalg.inv(self.J_COM)

        # Subtract reaction wheel contributions (if applicable)
        self.J_noRW = self.J_COM - np.sum(
            np.array([
                self.actuators[j].J * np.outer(self.actuators[j].axis, self.actuators[j].axis)
                for j in getattr(self, "momentum_inds", [])
            ]),
            axis=0
        ) if getattr(self, "momentum_inds", None) else self.J_COM

        self.invJ_noRW = np.linalg.inv(self.J_noRW)

    def _toggle_disturbances(self, dist_class: Disturbance, on: bool, ind: int | None = None) -> None:
        if ind is not None:
            d = self.disturbances[ind]
            if not isinstance(d, dist_class):
                raise ValueError(
                    f"Disturbance at index {ind} is not of type {dist_class.__name__}"
                )
            getattr(d, "turn_on" if on else "turn_off")()
            return

        # Otherwise apply to all disturbances of that type
        for d in self.disturbances:
            if isinstance(d, dist_class):
                getattr(d, "turn_on" if on else "turn_off")()

    def srp_dist_on(self):
        """Turn on all Solar Radiation Pressure disturbances."""
        self._toggle_disturbance(SRP_Disturbance, on=True)

    def srp_dist_off(self):
        """Turn off all Solar Radiation Pressure disturbances."""
        self._toggle_disturbance(SRP_Disturbance, on=False)

    def gen_dist_on(self, ind: int | None = None):
        """Turn on general disturbances."""
        self._toggle_disturbance(General_Disturbance, on=True, ind=ind)

    def gen_dist_off(self, ind: int | None = None):
        """Turn off general disturbances."""
        self._toggle_disturbance(General_Disturbance, on=False, ind=ind)

    def prop_dist_on(self, ind: int | None = None):
        """Turn on propulsion disturbances."""
        self._toggle_disturbance(Prop_Disturbance, on=True, ind=ind)

    def prop_dist_off(self, ind: int | None = None):
        """Turn off propulsion disturbances."""
        self._toggle_disturbance(Prop_Disturbance, on=False, ind=ind)

    def specific_dist_on(self, ind: int):
        """Turn on a specific disturbance by index."""
        self.disturbances[ind].turn_on()

    def specific_dist_off(self, ind: int):
        """Turn off a specific disturbance by index."""
        self.disturbances[ind].turn_off()

    def RWhs(self):
        return np.array([self.actuators[j].momentum for j in self.momentum_inds])

    def update_RWhs(self,state_or_RWhs):
        if np.size(state_or_RWhs) == self.state_len:
            RWhs = self.RWhs_from_state(state_or_RWhs)
        else:
            RWhs = state_or_RWhs
        if np.size(RWhs) != self.number_RW:
            raise ValueError("wrong number of RWhs to update")
        [self.actuators[self.momentum_inds[i]].update_momentum(RWhs[i]) for i in range(len(self.momentum_inds))]

    def RWhs_from_state(self,state):
        return state[7:]
    
    def dynamics_core(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State, verbose: bool = False, log: bool = False) -> np.ndarray:
        r"""
        Compute the full spacecraft rotational dynamics including attitude kinematics,
        external disturbances, and actuator torques, with optional reaction wheel coupling.

        This method forms the **core of the spacecraft attitude dynamics model**, computing
        the time derivative of the state vector:

        .. math::

            \dot{\mathbf{x}} =
            \begin{bmatrix}
            \dot{\boldsymbol{\omega}} \\
            \dot{\mathbf{q}} \\
            \dot{\mathbf{h}}_{\text{RW}}
            \end{bmatrix}

        where:
        
        - :math:`\boldsymbol{\omega}` is the body angular velocity vector (rad/s),
        - :math:`\mathbf{q}` is the attitude quaternion,
        - :math:`\mathbf{h}_{\text{RW}}` is the stored angular momentum in reaction wheels.

        The model includes:
        - Coordinate transformation from ECI to body frame,
        - Disturbance torques (aerodynamic, magnetic, solar, etc.),
        - Actuator torques (reaction wheels, magnetorquers, etc.),
        - Coupled rigid-body dynamics with or without reaction wheels.

        ---
        **1. Environmental Vector Transformation**

        The orbital state provides environmental quantities in the Earth-Centered Inertial (ECI) frame:
        position :math:`\mathbf{R}`, velocity :math:`\mathbf{V}`, magnetic field :math:`\mathbf{B}`,
        Sun vector :math:`\mathbf{S}`, and atmospheric density :math:`\rho`.

        These are rotated into the body frame using the quaternion-based rotation matrix:

        .. math::
            \mathbf{v}_B = \mathbf{R}_{\text{ECI}\rightarrow B}\, \mathbf{v}_{\text{ECI}}, \quad
            \mathbf{R}_{\text{ECI}\rightarrow B} = \text{rot\_mat}(\mathbf{q})^\top

        yielding:
        :math:`\mathbf{R}_B, \mathbf{V}_B, \mathbf{B}_B, \mathbf{S}_B`.

        ---
        **2. Quaternion Kinematics**

        The quaternion derivative is computed from angular velocity:

        .. math::
            \dot{\mathbf{q}} = \tfrac{1}{2}\,\mathbf{W}(\mathbf{q})^\top \boldsymbol{\omega}

        where :math:`\mathbf{W}(\mathbf{q})` is the standard quaternion kinematic matrix.

        ---
        **3. Rigid-Body Rotational Dynamics**

        The core Euler rotational dynamics are:

        .. math::
            \dot{\boldsymbol{\omega}} =
            \mathbf{J}^{-1}\left(
            \boldsymbol{\tau}_{\text{tot}} -
            \boldsymbol{\omega} \times (\mathbf{J}\boldsymbol{\omega})
            \right)

        where:
        - :math:`\mathbf{J}` is the spacecraft inertia tensor,
        - :math:`\boldsymbol{\tau}_{\text{tot}}` is the total torque acting on the body:
        \(\boldsymbol{\tau}_{\text{tot}} = \boldsymbol{\tau}_{\text{act}} + \boldsymbol{\tau}_{\text{dist}}\).

        ---
        **4. Reaction Wheel Coupling (if present)**

        When the spacecraft has reaction wheels, the body dynamics are coupled to wheel angular momentum :math:`\mathbf{h}_{\text{RW}}`:

        .. math::
            \dot{\boldsymbol{\omega}} =
            \mathbf{J}_b^{-1}\left(
            \boldsymbol{\tau}_{\text{tot}} -
            \boldsymbol{\omega} \times
            (\mathbf{J}_b\boldsymbol{\omega} + \mathbf{A}^\top \mathbf{h}_{\text{RW}})
            \right)

        where:
        - :math:`\mathbf{J}_b` is the bus (no-RW) inertia tensor,
        - :math:`\mathbf{A}` is the matrix of wheel spin axes (3×N),
        - :math:`\mathbf{h}_{\text{RW}} = \text{diag}(J_{\text{RW}})\, \boldsymbol{\omega}_{\text{RW}}`.

        The time derivative of stored wheel momentum is given by:

        .. math::
            \dot{\mathbf{h}}_{\text{RW}} =
            \mathbf{u}_{\text{RW}} -
            \text{diag}(J_{\text{RW}})\,\mathbf{A}^\top \dot{\boldsymbol{\omega}}

        where :math:`\mathbf{u}_{\text{RW}}` is the motor torque command vector.

        ---
        **5. State Derivative Assembly**

        The total derivative vector is returned as:

        .. math::
            \dot{\mathbf{x}} =
            \begin{cases}
                [\dot{\boldsymbol{\omega}}, \dot{\mathbf{q}}]^\top, & N_{\text{RW}} = 0 \\
                [\dot{\boldsymbol{\omega}}, \dot{\mathbf{q}}, \dot{\mathbf{h}}_{\text{RW}}]^\top, & N_{\text{RW}} > 0
            \end{cases}

        ---

        Parameters
        ----------
        x : numpy.ndarray
            Full spacecraft state vector:
            :math:`\mathbf{x} = [\boldsymbol{\omega}, \mathbf{q}, \mathbf{h}_{\text{RW}}]`.
        u : numpy.ndarray
            Control input vector (actuator commands).
        orbital_state : Orbital_State
            Object containing orbital and environmental parameters:
            position, velocity, magnetic field, Sun vector, and density.
        verbose : bool, optional
            If ``True``, print diagnostic information (default: ``False``).
        log : bool, optional
            If ``True``, log intermediate data for debugging or analysis.

        Returns
        -------
        numpy.ndarray
            Time derivative of the spacecraft state vector :math:`\dot{\mathbf{x}}`.

        Notes
        -----
        - All vectors are represented in the body reference frame.
        - Reaction wheel torques are internally computed by each actuator using
        :func:`Actuator.storage_torque()`.
        - The function supports both reaction-wheel and wheel-less configurations.
        """

        R = orbital_state.R # Position in ECI [km]
        V = orbital_state.V # Velocity in body frame [km/s]
        B = orbital_state.B # Magnetic field in ECI [T]
        S = orbital_state.S # Sun Vector in ECI [km]
        rho = orbital_state.rho # Atmospheric density [kg/m^3]

        w = x[0:3]
        q = x[4:7]
        h = x[7:]
        J = self.J_0
        invJ_noRW = self.invJ_noRW

        rmat_ECI2B = rot_mat(q).T
        R_B = rmat_ECI2B@R
        B_B = rmat_ECI2B@B
        S_B = rmat_ECI2B@S
        V_B = rmat_ECI2B@V

        vecs: Dict[str, np.ndarray] = {"b":B_B,"r":R_B,"s":S_B,"v":V_B,"rho":rho,"os":orbital_state}

        disturbance_torque: np.ndarray = self.dist_torques(x, vecs, log)
        actuator_torque: np.ndarray = self.act_torque(x, u, vecs)

        # Dynamics
        qdot = 0.5*Wmat(q).T
        total_torque = disturbance_torque + actuator_torque

        # Reaction wheels
        if self.number_RW==0:
            wdot = (-np.cross(w,w@J) + total_torque)@invJ_noRW
            return np.concatenate([wdot,qdot])
        else:
            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])
            # H_from_RW = sum([j.body_momentum() for j in self.actuators if j.has_momentum],np.zeros(3))
            u_RW = np.concatenate([self.actuators[j].storage_torque(u[j],self,x,vecs) for j in self.momentum_inds])
            wdot = (-np.cross(w,w@J + h@RWaxes) + total_torque)@invJ_noRW
            RW_hdot = u_RW-wdot@RWaxes.T@np.diagflat(RWjs) #u_RW-wdot@RWaxes.T@np.diagflat(RWjs)
            if verbose:
                print('wdot',wdot)
                print('hdot',RW_hdot)
                print('comp1',u_RW)
                print('comp2',-wdot@RWaxes.T@np.diagflat(RWjs))
            return np.concatenate([wdot,qdot,RW_hdot])


    def dist_torques(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        dist_list = [j.torque(self.vecs) for j in self.disturbances]
        return sum(dist_list,np.zeros(3))
    
    def act_torque(self, x: np.ndarray, u: np.ndarray, vecs: Dict[str, np.ndarray]) -> np.ndarray:
        act_list = [self.actuators[j].torque(u[j], x, vecs) for j in range(len(self.actuators))]
        return sum(act_list, np.zeros(3))
    
    def dist_torques_jacobian(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> Union[np.ndarray, np.ndarray]:
        r"""
        Compute the Jacobian of the total disturbance torque with respect to the
        system state and disturbance model parameters.

        The function aggregates the first-order derivatives of all registered
        disturbance models and returns a pair of Jacobian matrices. The state
        derivative includes only the quaternion components (indices 3--6),
        as these determine the attitude-dependent disturbance torque.

        .. math::

            \frac{\partial \boldsymbol{\tau}_d}{\partial \mathbf{x}}, \qquad
            \frac{\partial \boldsymbol{\tau}_d}{\partial \boldsymbol{\theta}_d}

        where :math:`\boldsymbol{\tau}_d` is the total disturbance torque,
        :math:`\mathbf{x}` is the state vector, and
        :math:`\boldsymbol{\theta}_d` are disturbance model parameters.

        Parameters
        ----------
        x : numpy.ndarray
            Current system state vector of length ``state_len``.
        vecs : Dict[str, numpy.ndarray]
            Dictionary of environment vectors (e.g., magnetic field, sun vector,
            aerodynamic flow) required by each disturbance model.

        Returns
        -------
        ddist_torq__dx : numpy.ndarray, shape (state_len, 3)
            Jacobian of disturbance torque with respect to the state vector.
        ddist_torq__ddmp : numpy.ndarray, shape (dist_param_len, 3)
            Jacobian of disturbance torque with respect to disturbance model
            parameters. Empty if no parameters are estimated.

        Notes
        -----
        - Only the quaternion part of the state contributes to the torque derivative.
        - Each disturbance model ``j`` must implement ``torque_qjac(self, vecs)``.
        """
        ddist_torq__dx = np.zeros((self.state_len,3))
        ddist_torq__dx[3:7,:] = sum([j.torque_qjac(self,vecs) for j in self.disturbances],np.zeros((4,3)))
        ddist_torq__ddmp = np.zeros((0,3))
        return ddist_torq__dx,ddist_torq__ddmp

    def dist_torque_hess(self, x: np.ndarray, vecs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Compute the Hessian tensors of the total disturbance torque with respect
        to the state vector and disturbance parameters.

        The function aggregates second-order derivatives of all registered
        disturbance models. The output consists of three 3D tensors representing
        second-order sensitivities with respect to state, parameter, and
        mixed (state--parameter) terms.

        .. math::

            \frac{\partial^2 \boldsymbol{\tau}_d}{\partial \mathbf{x}^2}, \qquad
            \frac{\partial^2 \boldsymbol{\tau}_d}{\partial \mathbf{x}\,\partial \boldsymbol{\theta}_d}, \qquad
            \frac{\partial^2 \boldsymbol{\tau}_d}{\partial \boldsymbol{\theta}_d^2}

        Parameters
        ----------
        x : numpy.ndarray
            Current system state vector of length ``state_len``.
        vecs : Dict[str, numpy.ndarray]
            Dictionary of environmental vectors required by the disturbance models.

        Returns
        -------
        dddist_torq__dxdx : numpy.ndarray, shape (state_len, state_len, 3)
            Second derivative of disturbance torque with respect to the state vector.
        dddist_torq__dxddmp : numpy.ndarray, shape (state_len, dist_param_len, 3)
            Mixed derivative tensor (state–parameter coupling).
        dddist_torq__ddmpddmp : numpy.ndarray, shape (dist_param_len, dist_param_len, 3)
            Second derivative of disturbance torque with respect to disturbance
            model parameters.

        Notes
        -----
        - Only the quaternion portion of the state contributes to second-order terms.
        - Each disturbance model ``j`` must implement:
        ``torque_qqhess(self, vecs)`` for quaternion–quaternion derivatives.
        """
        dddist_torq__dxdx = np.zeros((self.state_len,self.state_len,3))
        dddist_torq__dxdx[3:7,3:7,:] = sum([j.torque_qqhess(self,vecs) for j in self.disturbances],np.zeros((4,4,3)))
        dddist_torq__ddmpddmp = np.zeros((self.dist_param_len,self.dist_param_len,3))
        dddist_torq__dxddmp = np.zeros((self.state_len,self.dist_param_len,3))
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
            RWjs = np.array([self.actuators[j].J for j in self.momentum_inds])
            RWaxes = np.vstack([self.actuators[j].axis for j in self.momentum_inds])
            mRWjs = np.diagflat(RWjs)
            dxdot__dx[0:3,0:3] += -skewsym(RWhs@RWaxes)@invJ_noRW
            dxdot__dx[7:,0:3] += (dact_torq__dh+np.cross(RWaxes,w))@invJ_noRW
            dxdot__du[:,7:] = block_diag(*[self.actuators[j].dstor_torq__du(u[j],self,x,vecs) for j in range(len(self.actuators))])
            dxdot__du[:,7:] -= dxdot__du[:,0:3]@RWaxes.T@mRWjs
            dxdot__dx[0:7,7:] = np.hstack([self.actuators[j].dstor_torq__dbasestate(u[j],self,x,vecs) for j in range(len(self.actuators))])
            dxdot__dx[7:,7:] = np.diagflat([self.actuators[j].dstor_torq__dh(u[j],self,x,vecs) for j in self.momentum_inds])
            dxdot__dx[:,7:] -= dxdot__dx[:,0:3]@RWaxes.T@mRWjs
        return [dxdot__dx,dxdot__du]
    
    def dynamics_Hessians(self, x: np.ndarray, u: np.ndarray, orbital_state: Orbital_State) -> List[List[np.ndarray]]:
        r"""
        Compute the **second-order partial derivatives (Hessians)** of the spacecraft attitude dynamics
        with respect to the system state and control inputs.

        This function analytically evaluates the **Hessian tensors** of the spacecraft’s nonlinear
        rotational dynamics model:

        .. math::

            \dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, \mathbf{p})

        where the state vector :math:`\mathbf{x}` includes angular velocity, quaternion, and reaction
        wheel angular momenta, and :math:`\mathbf{u}` represents actuator torques or control inputs.

        The Hessians quantify the **second-order curvature** of the dynamics — i.e., how the Jacobians
        (first derivatives) themselves change with respect to the state and control. These tensors are
        essential for second-order estimation or control algorithms such as Differential Dynamic
        Programming (DDP), iterative LQR (iLQR), and nonlinear uncertainty propagation.

        **Mathematical Formulation**

        The rotational dynamics are defined as:

        .. math::

            \begin{aligned}
            \dot{\boldsymbol{\omega}} &= J^{-1} \left[
                -\boldsymbol{\omega} \times \left(J \boldsymbol{\omega} + A_{RW}^T \mathbf{h}_{RW}\right)
                + \boldsymbol{\tau}_{act}(\mathbf{x}, \mathbf{u})
                + \boldsymbol{\tau}_{dist}(\mathbf{x})
            \right] \\
            \dot{\mathbf{q}} &= \tfrac{1}{2} W(\mathbf{q})^T \boldsymbol{\omega} \\
            \dot{\mathbf{h}}_{RW} &= \mathbf{u}_{RW}
                - \mathrm{diag}(J_{RW}) A_{RW}^T \dot{\boldsymbol{\omega}}
            \end{aligned}

        The second derivatives are computed for each component of :math:`f(\mathbf{x}, \mathbf{u})`:

        .. math::

            \frac{\partial^2 f_i}{\partial z_j \, \partial z_k},
            \quad
            \mathbf{z} = [\mathbf{x}, \mathbf{u}]

        where each element of the returned Hessian tensors represents a mixed second derivative of the
        system dynamics with respect to state and/or control variables.

        ---
        **Inputs**

        :param x: 
            Current spacecraft state vector.
            
            .. math::

                \mathbf{x} =
                \begin{bmatrix}
                    \boldsymbol{\omega} \\[3pt]
                    \mathbf{q} \\[3pt]
                    \mathbf{h}_{RW}
                \end{bmatrix}
                \in \mathbb{R}^{n_x}

            - :math:`\boldsymbol{\omega}` — body angular velocity (rad/s), shape ``(3,)``  
            - :math:`\mathbf{q}` — attitude quaternion (Hamilton convention, body→ECI), shape ``(4,)``  
            - :math:`\mathbf{h}_{RW}` — reaction wheel momenta, shape ``(n_{RW},)``  

        :type x: np.ndarray

        :param u:
            Control input vector representing actuator torque commands, wheel speed commands,
            or other control variables. Shape ``(n_u,)``.

        :type u: np.ndarray

        :param orbital_state:
            Object containing current orbital and environmental parameters:
            - ``R`` — position vector in ECI [m]  
            - ``V`` — velocity vector in ECI [m/s]  
            - ``B`` — magnetic field vector in ECI [T]  
            - ``S`` — Sun direction vector in ECI  
            - ``rho`` — atmospheric density [kg/m³]

            These are internally transformed into the body frame and their first and second
            derivatives with respect to the quaternion are computed.

        :type orbital_state: Orbital_State

        ---
        **Outputs**

        :return:
            Nested list of Hessian tensors for each derivative block.  
            When estimation extensions are disabled (`self.estimated == False`):

            .. code-block:: text

                [
                [ddxdot__dxdx, ddxdot__dxdu],
                [ddxdot__dxdu.T, ddxdot__dudu]
                ]

            where each element is a 3D array:

            +-------------------+----------------------------+--------------------------+
            | Symbol            | Definition                 | Shape                    |
            +===================+============================+==========================+
            | ``ddxdot__dxdx``  | ∂²ẋ / ∂x² (state Hessian)  | (nₓ, nₓ, nₓ)            |
            +-------------------+----------------------------+--------------------------+
            | ``ddxdot__dxdu``  | ∂²ẋ / ∂x∂u (cross term)    | (nₓ, nᵤ, nₓ)            |
            +-------------------+----------------------------+--------------------------+
            | ``ddxdot__dudu``  | ∂²ẋ / ∂u² (input Hessian)  | (nᵤ, nᵤ, nₓ)            |
            +-------------------+----------------------------+--------------------------+

            Each 3D tensor’s third index corresponds to a state derivative component.

        :rtype: List[List[np.ndarray]]

        ---
        **Computation Steps**

        1. **Frame transformation**

        Environmental vectors are rotated from ECI to body frame using:

        .. math::

            \mathbf{v}_B = R_{ECI\to B} \, \mathbf{v}_{ECI}

        and both their first and second quaternion derivatives
        (:func:`drotmatTvecdq`, :func:`ddrotmatTvecdqdq`) are evaluated.

        2. **Torque derivatives**

        For each actuator and disturbance source:
        - Compute first- and second-order derivatives of the generated torque:
            :math:`\frac{\partial \boldsymbol{\tau}}{\partial \mathbf{x}}`,
            :math:`\frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{x}^2}`,
            :math:`\frac{\partial^2 \boldsymbol{\tau}}{\partial \mathbf{u}^2}`,
            and mixed derivatives.

        3. **Hessian assembly**

        Populate the combined second derivative tensors:
        - ``ddxdot__dxdx`` — curvature of dynamics wrt state
        - ``ddxdot__dxdu`` — curvature wrt state–control interaction
        - ``ddxdot__dudu`` — curvature wrt control inputs

        4. **Reaction wheel coupling**

        If reaction wheels are present (:attr:`self.number_RW > 0`),
        additional second-order coupling terms are added to capture
        body–wheel cross-dynamics via the wheel inertia, spin axes,
        and stored momentum.

        5. **Final tensor contraction**

        All torques are pre-multiplied by :math:`J^{-1}` (the inverse of the bus inertia
        tensor excluding reaction wheels) to yield accelerations in body coordinates.

        ---
        **Notes**

        - The quaternion formulation follows the **Hamilton convention**, where
        :math:`\mathbf{q}` represents the body-to-ECI rotation.
        - Returned tensors are symmetric along appropriate axes for physical consistency.
        - Designed for use in second-order dynamic linearization and trajectory optimization.

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
        - Diebel, J. “Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors.” Stanford University, 2006.  
        - Tassa, Y. *et al.* “Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization.” IROS, 2012.  
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

        return [[ddxdot__dxdx,ddxdot__dxdu],[ddxdot__dxdu.T,ddxdot__dudu]]


    def rk4(self, x: np.ndarray, u: np.ndarray, dt: float, orbital_state0: Orbital_State, orbital_state1: Orbital_State, verbose: bool=False,mid_orbital_state: Orbital_State = None, quat_as_vec: bool = True, give_err_est = False) -> np.ndarray:
        r"""
        Integrate the spacecraft rotational dynamics forward in time using a **fourth-order Runge–Kutta (RK4)**
        method or an optional **commutator-free fifth-order (CG5)** scheme for quaternion propagation.

        This method advances the state vector :math:`\mathbf{x}` over one time step :math:`\Delta t`
        according to the nonlinear dynamics model

        .. math::

            \dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}, t),

        where :math:`\mathbf{x}` contains the spacecraft angular velocity, quaternion attitude, and (optionally)
        reaction wheel states. The quaternion is renormalized at each substep to prevent numerical drift.

        **Integration Formulas**

        *For standard RK4:*

        .. math::
            \begin{aligned}
            k_1 &= f(\mathbf{x}_n, \mathbf{u}, t_n) \\
            k_2 &= f(\mathbf{x}_n + \tfrac{1}{2} \Delta t\, k_1, \mathbf{u}, t_n + \tfrac{1}{2}\Delta t) \\
            k_3 &= f(\mathbf{x}_n + \tfrac{1}{2} \Delta t\, k_2, \mathbf{u}, t_n + \tfrac{1}{2}\Delta t) \\
            k_4 &= f(\mathbf{x}_n + \Delta t\, k_3, \mathbf{u}, t_n + \Delta t) \\
            \mathbf{x}_{n+1} &= \mathbf{x}_n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
            \end{aligned}

        *For quaternion integration*, each intermediate state’s quaternion is renormalized and the final quaternion
        is normalized again after the update.

        *For CG5 (Lie group) variant:*  
        Quaternion increments are represented using rotational exponentials :math:`\exp(\mathbf{F}_i)` to ensure
        attitude integration on the manifold :math:`\mathrm{SO}(3)` via the commutator-free Runge–Kutta method.

        ---
        
        Parameters
        
        :param x:
            Current state vector :math:`\mathbf{x}` containing:
            - :math:`\boldsymbol{\omega}` — angular velocity in body frame [rad/s]
            - :math:`\mathbf{q}` — attitude quaternion (Hamilton form, body→ECI)
            - (optional) reaction wheel states

            Shape: ``(n_x,)``

        :type x: numpy.ndarray

        :param u:
            Control input vector :math:`\mathbf{u}` (e.g., actuator or wheel torques).  
            Shape: ``(n_u,)``

        :type u: numpy.ndarray

        :param dt:
            Integration time step [s]

        :type dt: float

        :param orbital_state0:
            Orbital/environmental state at the beginning of the step.
            Provides quantities such as :math:`\mathbf{R}, \mathbf{V}, \mathbf{B}, \mathbf{S}, \rho`
            used in environmental torque models.

        :type orbital_state0: Orbital_State

        :param orbital_state1:
            Orbital/environmental state at the end of the step.

        :type orbital_state1: Orbital_State

        :param verbose:
            If ``True``, prints intermediate derivative stages :math:`k_1, k_2, k_3, k_4`.

        :type verbose: bool, optional

        :param mid_orbital_state:
            Precomputed midpoint orbital state.  
            If ``None``, computed as ``orbital_state0.average(orbital_state1)``.

        :type mid_orbital_state: Optional[Orbital_State], optional

        :param quat_as_vec:
            If ``True``, uses standard RK4 with quaternion normalization.  
            If ``False``, uses commutator-free fifth-order (CG5) method with exponential maps.

        :type quat_as_vec: bool, optional

        :type save_info: bool, optional

        :param give_err_est:
            If ``True``, performs a third-order embedded RK method for **error estimation**, returning an additional
            error vector :math:`\hat{e}` representing the difference between the 4th- and 3rd-order solutions.

        :type give_err_est: bool, optional

        ---
        **Returns**

        :return:
            - If ``give_err_est = False`` → Updated state vector :math:`\mathbf{x}_{n+1}` (shape ``(n_x,)``).  
            - If ``give_err_est = True`` → Tuple ``(x_next, err_est)``:
            
            * :math:`x_{\text{next}}` — next-step state vector  
            * :math:`\hat{e}` — elementwise integration error estimate

        :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

        ---
        **Computation Notes**

        1. Quaternion components are renormalized after each substep to prevent drift.  
        2. The error estimate (when enabled) uses an embedded RK3 formula:

        .. math::
            \mathbf{x}_{n+1}^{(3)} = \mathbf{x}_n + \tfrac{\Delta t}{6}(k_1 + 4k_2 + k_{33})

        The elementwise difference :math:`|\mathbf{x}_{n+1}^{(4)} - \mathbf{x}_{n+1}^{(3)}|` provides
        a per-state local truncation error estimate.
        3. The CG5 variant propagates quaternions directly on the Lie group using exponential maps
        :math:`\exp(\mathbf{F}_i)`, avoiding small-angle linearizations.

        ---
        **Example**

        .. code-block:: python

            x_next = sat.rk4(x, u, 0.1, orb0, orb1)
            # or with embedded error estimate
            x_next, err = sat.rk4(x, u, 0.1, orb0, orb1, give_err_est=True)

        ---
        **References**

        - Hairer, E., Lubich, C., Wanner, G., *Geometric Numerical Integration*, 2nd Ed., Springer, 2006.  
        - Wie, B., *Space Vehicle Dynamics and Control*, AIAA, 2008.  
        - Celledoni, E., et al., “Commutator-Free Lie Group Methods,” *J. Comput. Phys.*, 2003.  
        """
        x[3:7] = normalize(x[3:7])
        if quat_as_vec:
            if mid_orbital_state is None:
                mid_orbital_state = orbital_state0.average(orbital_state1)

            k1 = self.dynamics(x, u, orbital_state0, verbose=verbose)
            k2_in = x + 0.5 * dt * k1
            k2_in[3:7] = normalize(k2_in[3:7])
            k2 = self.dynamics(k2_in, u, mid_orbital_state, verbose=verbose)

            k3_in = x + 0.5 * dt * k2
            k3_in[3:7] = normalize(k3_in[3:7])
            k3 = self.dynamics(k3_in, u, mid_orbital_state, verbose=verbose)

            k4_in = x + dt * k3
            k4_in[3:7] = normalize(k4_in[3:7])
            k4 = self.dynamics(k4_in, u, orbital_state1, verbose=verbose,)

            out = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out[3:7] = normalize(out[3:7])

            if verbose:
                print("k1:", k1)
                print("k2:", k2)
                print("k3:", k3)
                print("k4:", k4)

            if give_err_est:
                k33_in = x + dt * (2 * k2 - k1)
                k33_in[3:7] = normalize(k33_in[3:7])
                k33 = self.dynamics(k33_in, u, orbital_state1, verbose=verbose)

                out3 = x + (dt / 6.0) * (k1 + 4 * k2 + k33)
                out3[3:7] = normalize(out3[3:7])

                est_err = np.zeros(out.size - 1)
                est_err[:3] = np.abs(out[:3] - out3[:3])
                est_err[6:] = np.abs(out[7:] - out3[7:])
                est_err[3:6] = np.abs(quat_to_vec3(quat_mult(quat_inv(out[3:7]), out3[3:7]), 0))
                return out, est_err

            return out

        else:
            if give_err_est:
                raise ValueError("Error estimation not implemented for CG5 method.")

            ki = [np.zeros_like(x) for _ in range(5)]
            F = [np.zeros(3) for _ in range(5)]

            if mid_orbital_state is None:
                mid_orbital_state = [
                    orbital_state0.average(orbital_state1, CG5_c[i]) for i in range(5)
                ]

            for j in range(5):
                midstate = x + dt * sum([uc.CG5_a[j, i] * ki[i] for i in range(j)], np.zeros_like(x))
                if j > 0:
                    midstate[3:7] = normalize(
                        quat_mult(x[3:7], *[rot_exp(uc.CG5_a[j, i] * F[i]) for i in range(j)])
                    )
                ki[j] = self.dynamics(midstate, u, mid_orbital_state[j], verbose=verbose)
                F[j] = dt * midstate[0:3]

            out = x + dt * sum([uc.CG5_b[i] * ki[i] for i in range(5)], np.zeros_like(x))
            out[3:7] = normalize(
                quat_mult(x[3:7], *[rot_exp(uc.CG5_b[i] * F[i]) for i in range(5)])
            )

            return out