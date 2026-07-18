__all__ = ["Controller"]

import numpy as np

from ADCS.state import EstimatedState, State
from typing import List, Tuple, Type, Optional

from ADCS.CONOPS.goals import Goal
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.sensors import Sensor
from ADCS.satellite_hardware.actuators import Actuator

class Controller():
    r"""
    Base abstract controller class for all ADCS control law implementations.

    This class defines the common interface and shared utility methods required
    by all attitude determination and control system controllers. A controller
    is responsible for computing actuator command inputs that achieve a desired
    attitude, angular rate, or torque objective.

    Controllers derived from this base class typically implement a specific
    control law, such as magnetic detumbling, reaction-wheel stabilization, or
    pointing control. The base class additionally provides helper methods for
    sensor reconstruction and actuator allocation using Moore–Penrose
    pseudoinverses.

    Any concrete controller must override
    :meth:`~ADCS.controller.controller.Controller.find_u`.

    """
    def __init__(self, est_sat: EstimatedSatellite, **kwargs) -> None:
        r"""
        Initializes the base controller.

        This constructor provides a common initialization entry point for all
        controller implementations. Derived controllers may extract and store
        satellite parameters, actuator layouts, or sensor configurations from
        the estimated satellite model.

        :param est_sat: Estimated satellite model containing inertia, actuators,
                        and sensors
        :type est_sat: ~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite
        :param kwargs: Optional keyword arguments passed to derived controllers
        :type kwargs: dict
        :return: None
        :rtype: None

        """
        pass


    def find_u(self, x_hat: EstimatedState, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal: Goal | None, **kwargs) -> np.ndarray:
        r"""
        Computes actuator command inputs that satisfy the control objective.

        This method defines the primary controller interface and must be
        implemented by all derived controller classes. Implementations typically
        compute a desired control torque in the body frame and then allocate
        actuator commands accordingly.

        In abstract form, the control problem can be written as

        .. math::

            \boldsymbol{\tau}_{\mathrm{des}} = f(x, y, g)

        where the desired torque is a function of the estimated state, sensor
        measurements, and mission goal. Actuator commands are then computed as

        .. math::

            \mathbf{u} = A^{\dagger} \boldsymbol{\tau}_{\mathrm{des}}

        where :math:`A^{\dagger}` denotes an actuator allocation pseudoinverse.

        :param x_hat: Estimated spacecraft state vector
        :type x_hat: ADCS.state.EstimatedState
        :param sens: Flattened sensor measurement vector
        :type sens: numpy.ndarray
        :param est_sat: Estimated satellite model providing hardware properties
        :type est_sat: ~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite
        :param os_hat: Estimated orbital state
        :type os_hat: ~ADCS.orbits.orbital_state.Orbital_State
        :param goal: Optional mission goal definition
        :type goal: ~ADCS.CONOPS.goals.Goal or None
        :param kwargs: Optional controller-specific parameters
        :type kwargs: dict
        :return: Actuator command vector
        :rtype: numpy.ndarray

        """
        raise NotImplementedError(
            "find_u() must be implemented in child controller classes."
        )


    def build_sensor_matrix_pinv(self, sensors: List[Sensor], sensor_type: Type[Sensor]) -> Tuple[np.ndarray, List[int]]:
        r"""
        Constructs a sensor reconstruction matrix using a Moore–Penrose
        pseudoinverse.

        This method builds a linear mapping that reconstructs a three-dimensional
        physical vector from a stacked sensor measurement vector. Only sensors of
        the specified type are used in the reconstruction.

        Let :math:`y` denote the flattened measurement vector and
        :math:`A \in \mathbb{R}^{3 \times N}` the stacked sensing axis matrix of
        the selected sensors. The reconstructed vector is

        .. math::

            \mathbf{v}_{\mathrm{body}} = A^{\dagger} \, y

        where :math:`A^{\dagger}` is the Moore–Penrose pseudoinverse.

        :param sensors: List of all satellite sensors
        :type sensors: list[~ADCS.satellite_hardware.sensors.Sensor]
        :param sensor_type: Sensor type to include in the reconstruction
        :type sensor_type: type
        :return: Reconstruction matrix and corresponding measurement indices
        :rtype: tuple[numpy.ndarray, list[int]]

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
        Builds an actuator allocation matrix mapping desired body torque to
        actuator command space.

        This method constructs the pseudoinverse of the actuator direction
        matrix formed from the torque axes of the selected actuators.

        Let :math:`A \in \mathbb{R}^{3 \times N}` be the stacked actuator axis
        matrix. The minimum-norm actuator command vector satisfying a desired
        torque :math:`\boldsymbol{\tau}` is

        .. math::

            \mathbf{u} = A^{\dagger} \boldsymbol{\tau}

        :param actuators: List of all satellite actuators
        :type actuators: list[~ADCS.satellite_hardware.actuators.Actuator]
        :param actuator_type: Target actuator type
        :type actuator_type: type
        :return: Allocation matrix and actuator command indices
        :rtype: tuple[numpy.ndarray, list[int]]

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
        Builds the forward mapping matrix from actuator commands to body-frame
        torque directions.

        This matrix represents the physical contribution of each actuator input
        to the generated body torque:

        .. math::

            \boldsymbol{\tau} = A \, \mathbf{u}

        where each column of :math:`A` corresponds to an actuator torque axis.

        For magnetorquers, the resulting torque is computed as

        .. math::

            \boldsymbol{\tau} = [\mathbf{B}]_{\times} A \mathbf{u}

        where :math:`[\mathbf{B}]_{\times}` denotes the skew-symmetric matrix of
        the geomagnetic field.

        :param actuators: List of all satellite actuators
        :type actuators: list[~ADCS.satellite_hardware.actuators.Actuator]
        :param actuator_type: Target actuator type
        :type actuator_type: type
        :return: Forward torque mapping matrix
        :rtype: numpy.ndarray

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
        Extracts maximum allowable actuator command magnitudes.

        This method returns a vector of actuator saturation limits. These values
        are typically used to clip or normalize control inputs prior to command
        execution.

        If no actuator type is specified, limits for all actuators are returned
        in command vector order.

        :param actuators: List of all satellite actuators
        :type actuators: list[~ADCS.satellite_hardware.actuators.Actuator]
        :param actuator_type: Optional actuator type filter
        :type actuator_type: type or None
        :return: Vector of maximum actuator command values
        :rtype: numpy.ndarray

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