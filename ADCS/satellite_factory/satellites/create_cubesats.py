__all__ = [
    'create_beavercube1_cubesat',
    'create_beavercube2_cubesat',
    'create_3_3_beavercube2_cubesat',
    'create_estcube1_cubesat',
    'create_rax1_cubesat',
    'create_rax2_cubesat',
]

import numpy as np
from typing import List

from ADCS.satellite_factory.actuators import create_cubewheel_smallplus_rw, create_estcube1_magnetorquers, create_isis_magnetorquer_board
from ADCS.satellite_factory.sensors import (
    create_Clydespace_3U_array,
    create_ICM20948_IMU,
    create_adis16405_gyros,
    create_adis16405_magnetometers,
    create_hamamatsu_s3931_sun_sensors,
    create_hmc5883l_magnetometers,
    create_itg3200_gyros,
    create_isis_magnetometer,
    create_micromag3_magnetometers,
    create_osram_sfh2430_sun_sensors,
)
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import GeometryFace, GeometryConfig, Drag_Disturbance, GG_Disturbance, SRP_Disturbance
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_constants import MathConstants

def create_beavercube1_cubesat(estimated: bool = False):
    mass = 4
    COM = np.zeros(3)
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
    
    # Actuators
    mtqs: List[MTQ] = create_isis_magnetorquer_board(estimate_bias=estimated)
    
    # Sensors
    mtms: List[MTM] = create_isis_magnetometer(estimate_bias=estimated)
    gyros: List[Gyro] = create_ICM20948_IMU(estimate_bias=estimated)
    solar_panel_1 = create_Clydespace_3U_array(axis=np.array([1, 0, 0]), estimate_bias=estimated)
    solar_panel_2 = create_Clydespace_3U_array(axis=np.array([0, 1, 0]), estimate_bias=estimated)
    suns: List[SunPair] = solar_panel_1+solar_panel_2

    # Disturbances
    geometry_faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2)]
    config = GeometryConfig(geometry_faces)
    gg_dist = [GG_Disturbance()]
    drag_dist = [Drag_Disturbance(config)]
    srp_dist = [SRP_Disturbance(config)]

    boresight = np.array([0, 1, 0])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, sensors=mtms+gyros+suns, actuators=mtqs, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, sensors=mtms+gyros+suns, actuators=mtqs, boresight=boresight)


def create_estcube1_cubesat(estimated: bool = False):
    r"""
    Create an ESTCube-1 magnetic-control CubeSat model.

    The preset captures the best-documented ESTCube-1 flight-fitted inertia and
    published ADCS component configuration: two Honeywell HMC5883L three-axis
    magnetometers, four InvenSense ITG-3200 three-axis gyros, twelve Hamamatsu
    S3931 one-dimensional Sun sensor channels arranged as six two-channel Sun
    sensor assemblies, and three nominal electromagnetic coils with no reaction
    wheels.

    Sources:

    * `ESTCube-1 attitude determination flight results
      <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
    * `ESTCube-1 magnetic actuator flight results
      <https://www.sciencedirect.com/science/article/pii/S0094576515302216>`__
    """
    mass = 1.048
    COM = np.zeros(3)
    J = 1e-3 * np.array([
        [1.813, 0.024, 0.042],
        [0.024, 1.963, 0.029],
        [0.042, 0.029, 1.796],
    ])

    mtqs: List[MTQ] = create_estcube1_magnetorquers(estimate_bias=estimated)
    mtms: List[MTM] = create_hmc5883l_magnetometers(estimate_bias=estimated)
    gyros: List[Gyro] = create_itg3200_gyros(estimate_bias=estimated)
    suns = create_hamamatsu_s3931_sun_sensors(estimate_bias=estimated)

    geometry_faces: List[GeometryFace] = [
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.05, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.05, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
    ]
    config = GeometryConfig(geometry_faces)
    disturbances = [GG_Disturbance(), Drag_Disturbance(config), SRP_Disturbance(config)]

    boresight = np.array([0, 0, 1])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=mtqs, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=mtqs, boresight=boresight)


def _rax_inertia() -> np.ndarray:
    return 1e-2 * np.diag([2.91058, 2.91058, 0.59261])


def _rax_sun_sensor_layout(variant: int) -> np.ndarray:
    if variant == 1:
        return np.array([
            [0, 0], [180, 0], [90, 0], [270, 0], [0, 90],
            [0, 90], [0, 90], [0, -90], [0, -90],
        ])
    if variant == 2:
        return np.array([
            [17, -10], [0, 20], [-17, -10],
            [-162, -10], [180, 20], [162, -10],
            [72, 10], [107, 10], [90, -20],
            [-107, 10], [-72, 10], [-90, -20],
            [0, 90], [0, 90], [0, 90],
            [0, -90], [0, -90],
        ])
    raise ValueError("RAX variant must be 1 or 2.")


def _create_rax_cubesat(mass: float, variant: int, estimated: bool = False):
    COM = np.zeros(3)
    J = _rax_inertia()

    gyros: List[Gyro] = create_adis16405_gyros(estimate_bias=estimated)
    mtms: List[MTM] = (
        create_adis16405_magnetometers(estimate_bias=estimated)
        + create_micromag3_magnetometers(estimate_bias=estimated)
    )
    suns = create_osram_sfh2430_sun_sensors(
        az_el_deg=_rax_sun_sensor_layout(variant),
        estimate_bias=estimated,
    )

    geometry_faces: List[GeometryFace] = [
        GeometryFace(area=0.1*0.34, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.34, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.34, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.34, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.17, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.17, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
    ]
    config = GeometryConfig(geometry_faces)
    disturbances = [GG_Disturbance(), Drag_Disturbance(config), SRP_Disturbance(config)]

    boresight = np.array([0, 0, 1])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=[], boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=[], boresight=boresight)


def create_rax1_cubesat(estimated: bool = False):
    r"""
    Create a RAX-1 spacecraft model.

    RAX-1 was a 3U CubeSat using passive magnetic attitude stabilization:
    four permanent magnets aligned with the long body-Z axis and two HyMu80
    hysteresis strips along transverse axes. This package does not yet have
    passive magnetic stabilization actuator classes, so those devices are
    documented here but are not returned as commandable actuators.

    The inertia tensor follows the published RAX dynamics model,
    ``diag(2.91058, 2.91058, 0.59261) * 1e-2 kg m^2``, reflecting the body-Z
    principal symmetry axis.

    Source: `RAX attitude determination system design
    <https://doi.org/10.1016/j.actaastro.2012.02.001>`__
    """
    return _create_rax_cubesat(mass=2.8, variant=1, estimated=estimated)


def create_rax2_cubesat(estimated: bool = False):
    r"""
    Create a RAX-2 spacecraft model.

    RAX-2 used the same core RAX attitude-determination hardware as RAX-1, with
    the revised 17-photodiode OSRAM SFH2430 Sun-sensor layout. Passive magnetic
    stabilization hardware is documented but not represented as commandable
    actuators because this package does not yet include permanent-magnet or
    hysteresis-strip actuator models.

    The inertia tensor follows the published RAX dynamics model,
    ``diag(2.91058, 2.91058, 0.59261) * 1e-2 kg m^2``, reflecting the body-Z
    principal symmetry axis.

    Source: `RAX-1/RAX-2 flight attitude-determination results
    <https://doi.org/10.1016/j.actaastro.2014.02.026>`__
    """
    return _create_rax_cubesat(mass=2.9, variant=2, estimated=estimated)


def create_beavercube2_cubesat(estimated: bool = False):
    mass = 4
    COM = np.zeros(3)
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
    
    # Actuators
    mtqs: List[MTQ] = create_isis_magnetorquer_board(estimate_bias=estimated)
    rws: List[RW] = [create_cubewheel_smallplus_rw(axis=np.array([0, 0, 1]), estimate_bias=estimated)]
    
    # Sensors
    mtms: List[MTM] = create_isis_magnetometer(estimate_bias=estimated)
    gyros: List[Gyro] = create_ICM20948_IMU(estimate_bias=estimated)
    solar_panel_1 = create_Clydespace_3U_array(axis=np.array([1, 0, 0]), estimate_bias=estimated)
    solar_panel_2 = create_Clydespace_3U_array(axis=np.array([0, 1, 0]), estimate_bias=estimated)
    suns: List[SunPair] = solar_panel_1+solar_panel_2

    # Disturbances
    geometry_faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2)]
    config = GeometryConfig(geometry_faces)
    gg_dist = [GG_Disturbance()]
    drag_dist = [Drag_Disturbance(config)]
    srp_dist = [SRP_Disturbance(config)]

    boresight = np.array([0, 1, 0])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws, boresight=boresight)


def create_3_3_beavercube2_cubesat(estimated: bool = False):
    mass = 4
    COM = np.zeros(3)
    J =  np.array([[0.03136490806, 5.88304e-05, -0.00671361357],
                [5.88304e-05, 0.03409127827, -0.00012334756],
                [-0.00671361357, -0.00012334756, 0.01004091997]])
    
    # Actuators
    mtqs: List[MTQ] = create_isis_magnetorquer_board(estimate_bias=estimated)
    rws: List[RW] = [create_cubewheel_smallplus_rw(axis=np.array([1, 0, 0]), estimate_bias=estimated),
                     create_cubewheel_smallplus_rw(axis=np.array([0, 1, 0]), estimate_bias=estimated),
                     create_cubewheel_smallplus_rw(axis=np.array([0, 0, 1]), estimate_bias=estimated)]
    
    # Sensors
    mtms: List[MTM] = create_isis_magnetometer(estimate_bias=estimated)
    gyros: List[Gyro] = create_ICM20948_IMU(estimate_bias=estimated)
    solar_panel_1 = create_Clydespace_3U_array(axis=np.array([1, 0, 0]), estimate_bias=estimated)
    solar_panel_2 = create_Clydespace_3U_array(axis=np.array([0, 1, 0]), estimate_bias=estimated)
    suns: List[SunPair] = solar_panel_1+solar_panel_2

    # Disturbances
    geometry_faces: List[GeometryFace] = [GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
                                 GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2)]
    config = GeometryConfig(geometry_faces)
    gg_dist = [GG_Disturbance()]
    drag_dist = [Drag_Disturbance(config)]
    srp_dist = [SRP_Disturbance(config)]

    boresight = np.array([0, 1, 0])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws, boresight=boresight)
