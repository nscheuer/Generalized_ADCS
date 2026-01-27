__all__ = ["Controller"]

import numpy as np
from typing import List, Tuple, Type, Optional

from ADCS.CONOPS.goals import Goal
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.sensors import Sensor
from ADCS.satellite_hardware.actuators import Actuator

class Controller():
    r"""
    Base abstract controller for all ADCS control law implementations.

    This class defines the core interface required in every ADCS controller.
    Controllers are expected to compute actuator input commands that achieve
    a desired torque or attitude regulation objective.

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        Estimated satellite model containing inertia, actuator layout,
        and sensor configuration.
    include_disturbances : bool, optional
        If True, the controller will query the estimated satellite's disturbance
        model and add feedforward compensation to the control torque. Default False.
    **kwargs : dict
        Optional keyword parameters forwarded to derived controllers.

    Notes
    -----
    Controllers that inherit from this base must override
    :meth:`~ADCS.controller.Controller.find_u`.

    """
    def __init__(self, est_sat: EstimatedSatellite, include_disturbances: bool = False, **kwargs) -> None:
        r"""
        Initializes the base controller class.

        Parameters
        ----------
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite model.
        include_disturbances : bool, optional
            Enable disturbance feedforward compensation. Default False.
        **kwargs : dict
            Additional optional arguments needed by subclasses.

        """
        self.include_disturbances = include_disturbances
        self._last_dist_torque = np.zeros(3)  # Store for debugging/logging


    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal: Goal | None, **kwargs) -> np.ndarray:
        r"""
        Computes actuator inputs required to satisfy the control objective.

        This function must be overridden by subclasses. Implementations
        typically compute a torque objective in the body frame and then
        allocate actuator commands accordingly.

        Parameters
        ----------
        x_hat : :class:`numpy.ndarray`
            Estimated spacecraft state vector. Conventionally includes angular
            velocity and quaternion attitude estimates.
        sens : :class:`numpy.ndarray`
            Flattened sensor measurement vector from onboard hardware such as
            magnetometers, sun sensors, star trackers, etc.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite object providing inertia, actuator geometries,
            and possible reaction wheel momentum estimates.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Estimated orbital state (e.g. ECI position, velocity).
        goal_vector_eci : :class:`numpy.ndarray` or None
            Desired pointing direction expressed in the ECI frame.
        w_ref : :class:`numpy.ndarray` or None
            Desired angular rate reference in the body frame.

        Returns
        -------
        :class:`numpy.ndarray`
            Actuator command vector. The internal indexing convention must
            match that of the satellite model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the derived controller.

        """
        raise NotImplementedError(
            "find_u() must be implemented in child controller classes."
        )

    def get_disturbance_torque(self, x_hat: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State) -> np.ndarray:
        r"""
        Computes the estimated disturbance torque from the satellite's disturbance models.

        This is used for feedforward compensation when `include_disturbances=True`.

        Parameters
        ----------
        x_hat : :class:`numpy.ndarray`
            Estimated spacecraft state vector.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite with disturbance models.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Estimated orbital state.

        Returns
        -------
        :class:`numpy.ndarray`
            Estimated disturbance torque in body frame [N·m], shape (3,).
        """
        if not self.include_disturbances:
            return np.zeros(3)
        
        try:
            # Use the satellite's dist_torques method
            dist_torque = est_sat.dist_torques(x=x_hat, os=os_hat)
            self._last_dist_torque = dist_torque.copy()
            return dist_torque
        except Exception:
            # If satellite doesn't have disturbance models or method fails
            return np.zeros(3)

    def compensate_disturbance(self, tau_des: np.ndarray, x_hat: np.ndarray, 
                                est_sat: EstimatedSatellite, os_hat: Orbital_State) -> np.ndarray:
        r"""
        Adds disturbance feedforward compensation to a desired torque.

        If `include_disturbances=True`, subtracts the estimated disturbance torque
        from the desired control torque so the controller counteracts it.

        .. math::

            \tau_{compensated} = \tau_{des} - \tau_{dist}

        Parameters
        ----------
        tau_des : :class:`numpy.ndarray`
            Desired control torque before disturbance compensation [N·m].
        x_hat : :class:`numpy.ndarray`
            Estimated spacecraft state vector.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite with disturbance models.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Estimated orbital state.

        Returns
        -------
        :class:`numpy.ndarray`
            Compensated control torque [N·m], shape (3,).

        Notes
        -----
        Subclasses should call this method after computing their baseline
        desired torque to add disturbance feedforward:

        .. code-block:: python

            tau_des = self.compute_pd_torque(...)  # Controller-specific
            tau_des = self.compensate_disturbance(tau_des, x_hat, est_sat, os_hat)

        """
        dist_torque = self.get_disturbance_torque(x_hat, est_sat, os_hat)
        return tau_des - dist_torque

    def compute_gyroscopic_torque(self, x_hat: np.ndarray, est_sat: EstimatedSatellite) -> np.ndarray:
        r"""
        Computes the gyroscopic/coupling torque for feedforward compensation.

        The gyroscopic torque arises from the cross-coupling of angular velocity
        with the total angular momentum (body + reaction wheels):

        .. math::

            \tau_{gyro} = \omega \times (J\omega + h_{rw})

        This term appears on the RHS of Euler's equation and must be counteracted
        for precise attitude control.

        Parameters
        ----------
        x_hat : :class:`numpy.ndarray`
            Estimated spacecraft state vector [ω, q, ...].
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite with inertia and RW states.

        Returns
        -------
        :class:`numpy.ndarray`
            Gyroscopic torque in body frame [N·m], shape (3,).
        """
        from ADCS.satellite_hardware.actuators import RW
        
        w = x_hat[0:3]
        J = est_sat.J_0
        
        # Get RW angular momentum
        rws = [a for a in est_sat.actuators if isinstance(a, RW)]
        n_rw = len(rws)
        
        if n_rw > 0 and len(x_hat) >= 7 + n_rw:
            h_rw_states = x_hat[7:7 + n_rw]
        else:
            h_rw_states = np.array([rw.h for rw in rws]) if rws else np.array([])
        
        # Sum RW momentum in body frame
        h_rw_body = np.zeros(3)
        for i, rw in enumerate(rws):
            if i < len(h_rw_states):
                h_rw_body += np.asarray(rw.axis).flatten() * h_rw_states[i]
        
        # Gyroscopic coupling: ω × (Jω + h_rw)
        tau_gyro = np.cross(w, J @ w + h_rw_body)
        
        return tau_gyro

    def apply_feedforward_compensation(self, tau_baseline: np.ndarray, x_hat: np.ndarray,
                                        est_sat: EstimatedSatellite, os_hat: Orbital_State,
                                        include_gyroscopic: bool = True) -> np.ndarray:
        r"""
        Applies both gyroscopic and disturbance feedforward compensation to a baseline torque.

        This is a convenience method that combines:
        1. Gyroscopic compensation (always applied if `include_gyroscopic=True`)
        2. Disturbance compensation (applied if `self.include_disturbances=True`)

        .. math::

            \tau_{des} = \tau_{baseline} + \tau_{gyro} - \tau_{dist}

        Parameters
        ----------
        tau_baseline : :class:`numpy.ndarray`
            Baseline control torque (e.g., PD feedback) [N·m].
        x_hat : :class:`numpy.ndarray`
            Estimated spacecraft state vector.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite model.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Estimated orbital state.
        include_gyroscopic : bool, optional
            Whether to add gyroscopic compensation. Default True.

        Returns
        -------
        :class:`numpy.ndarray`
            Fully compensated desired torque [N·m], shape (3,).

        Notes
        -----
        This method is intended for use in LP/QP controllers where the feedforward
        terms should be added BEFORE the allocation/optimization step.

        Example usage in a controller:

        .. code-block:: python

            # Compute baseline PD torque
            tau_pd = -self.p_gain * q_err - self.d_gain * w_err
            
            # Add feedforward compensation
            tau_des = self.apply_feedforward_compensation(tau_pd, x_hat, est_sat, os_hat)
            
            # Now pass tau_des to LP/QP allocator
            u = self.allocate(tau_des, ...)

        """
        tau_des = tau_baseline.copy()
        
        # Add gyroscopic compensation
        if include_gyroscopic:
            tau_gyro = self.compute_gyroscopic_torque(x_hat, est_sat)
            tau_des = tau_des + tau_gyro
        
        # Add disturbance compensation
        tau_des = self.compensate_disturbance(tau_des, x_hat, est_sat, os_hat)
        
        return tau_des


    def build_sensor_matrix_pinv(self, sensors: List[Sensor], sensor_type: Type[Sensor]) -> Tuple[np.ndarray, List[int]]:
        r"""
        Constructs a measurement reconstruction matrix for a specific
        class of sensors by computing a Moore–Penrose pseudoinverse
        based on their sensing axes.

        Given a stacked sensor measurement vector :math:`y`, this matrix
        maps sensor outputs to a physical 3-vector (e.g. the estimated
        magnetic field in body coordinates).

        .. math::

            \mathbf{v}_{body} \approx M_{\mathrm{sens}} \, y

        Parameters
        ----------
        sensors : list[:class:`~ADCS.satellite_hardware.sensors.Sensor`]
            List of all satellite sensors.
        sensor_type : type
            Sensor type to extract. Must inherit from
            :class:`~ADCS.satellite_hardware.sensors.Sensor`.

        Returns
        -------
        M_sens : :class:`numpy.ndarray`
            Pseudoinverse reconstruction matrix of shape (3, N), mapping
            measurements to a 3-vector physical estimate.
        indices : list[int]
            Flattened measurement indices belonging to the selected sensor type.

        Raises
        ------
        TypeError
            If `sensor_type` does not subclass Sensor.
        ValueError
            If no sensors of the given type are found.

        """
        if not issubclass(sensor_type, Sensor):
            raise TypeError(f"sensor_type must be a subclass of Sensor, got {sensor_type}")
        
        # 1. Collect axes for the target sensor type and track indices
        active_cols = []
        active_indices = []
        
        current_idx = 0
        
        for sens in sensors:
            # Get the sensing axis (or axes) for this sensor
            # shape is (3, N_outputs), usually (3,1)
            axis = np.asarray(sens.axis, dtype=float).reshape(3, -1) 
            n_outputs = axis.shape[1]
            
            # If this is the sensor we want, record its axis and indices
            if isinstance(sens, sensor_type):
                # Append each column of the axis matrix individually
                for i in range(n_outputs):
                    active_cols.append(axis[:, i])
                    active_indices.append(current_idx + i)
            
            # Always increment the index counter so the full 'y' vector alignment is preserved
            current_idx += n_outputs

        if not active_cols:
            raise ValueError(f"No sensors of type {sensor_type.__name__} found.")

        # 2. Compute the Pinv for ONLY the active sensors
        # A_sub shape: (3, N_active)
        A_sub = np.column_stack(active_cols)
        
        # M_sub shape: (3, N_active) -- usually (3,3) for MTMs
        M_sub = np.linalg.pinv(A_sub)

        # 3. Create the full matrix (3, N_total) filled with zeros
        # current_idx now holds the total length of 'y'
        M_full = np.zeros((3, current_idx))

        # 4. Slot the computed inverse into the correct columns
        # This ensures M_full @ y only "sees" the relevant values
        M_full[:, active_indices] = M_sub

        return M_full, active_indices
    

    def build_torque_to_u_matrix_pinv(self, actuators: List[Actuator], actuator_type: Type[Actuator]) -> Tuple[np.ndarray, List[int]]:
        r"""
        Builds the actuator allocation matrix that maps desired body torques
        into actuator command space via the pseudoinverse of the actuator
        direction matrix.

        The columns of the internal matrix correspond to the unit axis
        directions along which the selected actuators can impart torque.
        Its Moore–Penrose pseudoinverse yields the minimum-norm command
        vector that best matches a desired torque.

        .. math::

            \mathbf{u} \approx A^{\dagger} \, \boldsymbol{\tau}

        Parameters
        ----------
        actuators : list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]
            List of all satellite actuators.
        actuator_type : type
            Target actuator type (e.g. :class:`~ADCS.satellite_hardware.actuators.RW`
            or :class:`~ADCS.satellite_hardware.actuators.MTQ`).

        Returns
        -------
        M_act : :class:`numpy.ndarray`
            Allocation pseudoinverse of size (N, 3), mapping desired torque to commands.
        indices : list[int]
            Indices of the actuator channels belonging to the selected type.

        Raises
        ------
        TypeError
            If `actuator_type` does not subclass Actuator.

        """
        if not issubclass(actuator_type, Actuator):
            raise TypeError(f"actuator_type must be a subclass of Actuator, got {actuator_type}")
        
        active_cols = []
        active_indices = []
        curr_global_idx = 0

        # 1. Scan all actuators to build total dimension and find active ones
        for act in actuators:
            # Flatten axis to handle potential multi-axis components if any
            axis = np.asarray(act.axis, dtype=float).reshape(3, -1)
            num_inputs = axis.shape[1]

            if isinstance(act, actuator_type):
                for i in range(num_inputs):
                    active_cols.append(axis[:, i])
                    active_indices.append(curr_global_idx + i)

            curr_global_idx += num_inputs

        # 2. Initialize the full output matrix with zeros (N_total_inputs x 3)
        M_act = np.zeros((curr_global_idx, 3))

        # 3. Only compute pinv if we actually found matching actuators
        if active_cols:
            # A_sub shape: (3, N_active)
            A_sub = np.column_stack(active_cols)
            
            # M_sub shape: (N_active, 3)
            M_sub = np.linalg.pinv(A_sub)

            # Map the active sub-matrix into the global matrix
            M_act[active_indices, :] = M_sub
        
        # If active_cols was empty, M_act remains all zeros, which is correct (no torque capability)
        return M_act, active_indices
    

    def build_u_to_torque_matrix_pinv(self, actuators: List[Actuator], actuator_type: Type[Actuator]) -> np.ndarray:
        r"""
        Builds the forward mapping matrix that converts actuator inputs directly
        into the physical torque contribution direction in the body frame.

        Unlike :meth:`~ADCS.controller.Controller.build_torque_to_u_matrix_pinv`,
        which gives the inverse allocation mapping from torque to commands,
        this matrix is used in the forward physical model:

        .. math::

            \boldsymbol{\tau} = A \, \mathbf{u}

        Examples
        --------
        - For reaction wheels:
          each column of :math:`A` is the torque axis direction.
        - For magnetorquers:
          each column represents the direction of the magnetic dipole.
          Body torque is then computed by

        .. math::

            \boldsymbol{\tau} = \left[\mathbf{B}\right]_{\times} A \mathbf{u}

        Parameters
        ----------
        actuators : list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]
            List of all satellite actuators.
        actuator_type : type
            Target actuator type.

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of stacked axis vectors, shape (3, N). If no actuators
            of the requested type are found, returns an empty (3, 0) matrix.

        """
        if not issubclass(actuator_type, Actuator):
            raise TypeError(f"actuator_type must be a subclass of Actuator, got {actuator_type}")

        cols = []
        for act in actuators:
            axis = np.asarray(act.axis, dtype=float).reshape(3, -1)
            num_inputs = axis.shape[1]

            if isinstance(act, actuator_type):
                # Target actuator: Append the actual torque axis
                for i in range(num_inputs):
                    cols.append(axis[:, i])
            else:
                # Non-target actuator: Append zeros to maintain alignment with the full 'u' vector
                for i in range(num_inputs):
                    cols.append(np.zeros(3))
        
        if not cols:
            return np.zeros((3, 0))
        
        return np.column_stack(cols)
    

    def find_max_torque(self, actuators: List[Actuator], actuator_type: Optional[Type[Actuator]] = None) -> np.ndarray:
        r"""
        Extracts the maximum actuator command magnitude for the given actuator type.

        The returned vector matches the number of command channels for that type.
        This value is typically used to clip, saturate, or normalize actuator
        control inputs.

        Parameters
        ----------
        actuators : list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]
            List of all actuators.
        actuator_type : type
            Target actuator type.

        Returns
        -------
        :class:`numpy.ndarray`
            Vector of maximum allowable actuator input values.

        Raises
        ------
        ValueError
            If no actuators of the requested type are present.

        """
        if actuator_type is None:
            # Return limits for all actuators in order
            max_u_limits = np.array([act.u_max for act in actuators])
            if len(max_u_limits) == 0:
                # Only raise an error if the list of actuators itself is empty
                raise ValueError("The actuator list is empty.")
            return max_u_limits
        else:
            # Original logic: return limits only for the specified type
            max_torque = np.array([act.u_max for act in actuators if isinstance(act, actuator_type)])
            if len(max_torque) == 0:
                raise ValueError(f"No actuators of type {actuator_type.__name__} found to determine max input limit (u_max).")
            return max_torque