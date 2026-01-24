__all__ = ["Orbit_GPS"]

import numpy as np
from typing import List, Optional
from scipy.linalg import block_diag

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedOrbital_State
from ADCS.estimators.orbit_estimators import Orbit_Estimator

class Orbit_GPS(Orbit_Estimator):
    r"""
    Pass-through GPS-based orbit estimator.

    This estimator directly converts raw GPS measurements into an inertial
    orbital state without performing any temporal propagation or filtering.
    No dynamic model, prediction step, or recursive estimation is applied.
    Instead, the most recent GPS measurement is assumed to be the best
    available estimate of the satellite state.

    Given a GPS measurement expressed in the Earth-Centered Earth-Fixed (ECEF)
    frame, the estimator constructs the inertial estimate via a deterministic
    coordinate transformation:

    .. math::

        \hat{\mathbf{x}}_{ECI}(t)
        =
        \begin{bmatrix}
        \mathbf{r}_{ECI} \\
        \mathbf{v}_{ECI}
        \end{bmatrix}
        =
        T_{ECEF \rightarrow ECI}(t)
        \begin{bmatrix}
        \mathbf{r}_{ECEF} \\
        \mathbf{v}_{ECEF}
        \end{bmatrix}

    where :math:`T_{ECEF \rightarrow ECI}(t)` is the time-dependent rotation
    defined by the Earth orientation and ephemeris models.

    This estimator is primarily intended for:

    * Initializing more advanced orbit estimators
    * Scenarios where GPS data is trusted without filtering
    * Debugging and validation of coordinate-frame transformations

    The estimated covariance matrix is constructed directly from the GPS
    sensor noise parameters provided by the satellite hardware model.

    :inherits:
        :class:`~ADCS.estimators.orbit_estimators.Orbit_Estimator`

    """
    def __init__(
        self, 
        est_sat: EstimatedSatellite, 
        J2000: float,
        os_template: Orbital_State
    ) -> None:
        r"""
        Initialize the GPS pass-through orbit estimator.

        This constructor configures the estimator with a satellite hardware
        model and a template orbital state. The template state is used solely
        to access shared environment models (e.g., ephemeris and density)
        required for coordinate frame conversions.

        No state estimate is produced during initialization; instead, the
        estimator is configured via :meth:`reset`.

        :param est_sat:
            Satellite hardware model containing GPS sensor definitions and
            noise characteristics.
        :type est_sat:
            :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`

        :param J2000:
            Current epoch expressed in seconds since J2000. This parameter is
            not used internally during initialization but is retained for
            interface consistency.
        :type J2000:
            float

        :param os_template:
            Template orbital state used to access ephemeris and environmental
            models required for ECEF–ECI transformations.
        :type os_template:
            :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :return:
            None
        :rtype:
            None

        """
        super().__init__(est_sat=est_sat, dt=0.0)
        self.reset(est_sat=est_sat, os_template=os_template)

    def reset(
        self, 
        est_sat: EstimatedSatellite, 
        os_template: Orbital_State
    ) -> None:
        r"""
        Reset the estimator configuration and noise parameters.

        This method reinitializes the estimator by extracting GPS sensor noise
        characteristics from the provided satellite hardware model. The
        standard deviation of the first available GPS sensor is stored and
        later used to construct the state covariance matrix.

        No state estimate is created during reset; the estimator will produce
        an estimate upon the next call to :meth:`update`.

        :param est_sat:
            Satellite hardware model containing GPS sensor definitions.
        :type est_sat:
            :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`

        :param os_template:
            Template orbital state providing access to ephemeris and
            environmental models.
        :type os_template:
            :class:`~ADCS.orbits.orbital_state.Orbital_State`

        :raises ValueError:
            If the satellite hardware model does not contain any GPS sensors.

        :return:
            None
        :rtype:
            None

        """
        self.est_sat = est_sat
        self.os_template = os_template
        
        gps_sensors = self.est_sat.GPS_sensors
        if not gps_sensors:
            raise ValueError("Orbit_GPS requires at least one GPS sensor.")
            
        blocks = []
        for gps in gps_sensors:
            std = gps.noise.std_noise

            self.gps_std = std

    def update(
        self,
        GPS_measurements: List[np.ndarray],
        J2000: float
    ) -> EstimatedOrbital_State:
        r"""
        Update the orbital state estimate using GPS measurements.

        This method converts the most recent GPS measurement into an inertial
        orbital state and constructs a corresponding covariance matrix based
        on sensor noise specifications.

        The update procedure is as follows:

        1. A temporary
           :class:`~ADCS.orbits.orbital_state.Orbital_State` is created at the
           current epoch to enable frame transformations.
        2. The GPS measurement is transformed from ECEF to ECI coordinates.
        3. A covariance matrix :math:`\mathbf{P}` is constructed assuming
           independent measurement noise:

           .. math::

               \mathbf{P}
               =
               \mathrm{diag}
               \left(
               \sigma_{r}^2,
               \sigma_{r}^2,
               \sigma_{r}^2,
               \sigma_{v}^2,
               \sigma_{v}^2,
               \sigma_{v}^2
               \right)

        If only position is measured, the velocity is either inherited from the
        previous estimate or set to zero, with a deliberately large velocity
        uncertainty to reflect the lack of direct measurement.

        :param GPS_measurements:
            List of GPS measurement vectors. Each measurement must have length
            3 (position only) or 6 (position and velocity).
        :type GPS_measurements:
            list[numpy.ndarray]

        :param J2000:
            Current epoch expressed in seconds since J2000.
        :type J2000:
            float

        :return:
            Estimated orbital state containing the inertial state vector
            :math:`\hat{\mathbf{x}}` and covariance matrix :math:`\mathbf{P}`.
        :rtype:
            :class:`~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedOrbital_State`

        :raises ValueError:
            If the GPS measurement vector does not have length 3 or 6.

        """
        
        if not GPS_measurements:
            return self.os_hat


        m = np.asarray(GPS_measurements[0]).reshape(-1)

        # 1. Create a temporary state at current J2000 to handle frame conversion.
        #    We inherit the physics models (Ephemeris) from the template.
        temp_os = Orbital_State(
            ephem=self.os_template.ephem,
            J2000=J2000,
            R=np.zeros(3), # Dummy values
            V=np.zeros(3),
            density_model=self.os_template.density_model
        )

        # 2. Convert ECEF Measurement -> ECI State
        if m.size == 3:
            # Position only
            r_ecef = m
            r_eci = temp_os.ecef_to_eci(r_ecef)

            if self.os_hat:
                v_eci = self.os_hat.os.V
            else:
                v_eci = np.zeros(3)
                
            std_pos = self.gps_std
            std_vel = 1000.0 # High uncertainty for unmeasured velocity
            P = np.diag([std_pos]*3 + [std_vel]*3)**2

        elif m.size == 6:
            # Position + Velocity
            r_ecef = m[0:3]
            v_ecef = m[3:6]

            r_eci = temp_os.ecef_to_eci(r_ecef)
            v_eci = temp_os.ecef_to_eci(v_ecef)
            
            std = self.gps_std
            P = np.diag([std]*6)**2
            
        else:
            raise ValueError(f"Unknown GPS measurement size: {m.size}")

        # 3. Create the new State
        new_os = Orbital_State(
            ephem=self.os_template.ephem,
            J2000=J2000,
            R=r_eci,
            V=v_eci,
            density_model=self.os_template.density_model
        )

        # 4. Return Estimated State
        self.os_hat = EstimatedOrbital_State(
            os=new_os,
            P=P,
            Q=np.zeros((6,6)) # No process noise in a static pass-through
        )

        return self.os_hat