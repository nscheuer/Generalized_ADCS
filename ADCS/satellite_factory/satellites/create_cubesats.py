__all__ = [
    'create_beavercube1_cubesat',
    'create_beavercube2_cubesat',
    'create_3_3_beavercube2_cubesat',
    'create_brite_austria',
    'create_estcube1_cubesat',
    'create_lightsail2',
    'create_moveii_cubesat',
    'create_rax1_cubesat',
    'create_rax2_cubesat',
]

import numpy as np
from typing import List

from ADCS.satellite_factory.actuators import (
    create_cubewheel_smallplus_rw,
    create_estcube1_magnetorquers,
    create_gnb_air_core_magnetorquers,
    create_isis_magnetorquer_board,
    create_moveii_pcb_magnetorquers,
    create_sfl_reaction_wheels,
    create_sinclair_interplanetary_momentum_wheel,
    create_stras_space_torque_rods,
)
from ADCS.satellite_factory.sensors import (
    create_aeroastro_mst,
    create_analog_devices_pib_gyros,
    create_Clydespace_3U_array,
    create_elmos_sun_sensors,
    create_gnb_magnetometer,
    create_gnb_rate_sensors,
    create_gnb_sun_sensors,
    create_ICM20948_IMU,
    create_adis16405_gyros,
    create_adis16405_magnetometers,
    create_bmx055_gyros,
    create_bmx055_magnetometers,
    create_hamamatsu_s3931_sun_sensors,
    create_hmc5883l_magnetometers,
    create_honeywell_lightsail2_magnetometers,
    create_itg3200_gyros,
    create_isis_magnetometer,
    create_intrepid_mainboard_gyros,
    create_micromag3_magnetometers,
    create_nano_iss60_sun_sensors,
    create_osram_sfh2430_sun_sensors,
)
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, GeometryFace, GeometryConfig, Drag_Disturbance, GG_Disturbance, SRP_Disturbance
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.helpers.math_constants import MathConstants


def _box_geometry_faces(dimensions: np.ndarray) -> List[GeometryFace]:
    """Create a six-face rectangular-prism geometry centered on the body origin."""
    x_len, y_len, z_len = np.asarray(dimensions, dtype=float)
    half = np.array([x_len, y_len, z_len]) / 2.0
    eta_s = 0.5
    eta_d = 0.2
    eta_a = 0.3
    CD = 2.2
    return [
        GeometryFace(area=y_len*z_len, centroid=half[0]*MathConstants.unitvecs[0], normal=MathConstants.unitvecs[0], eta_s=eta_s, eta_d=eta_d, eta_a=eta_a, CD=CD),
        GeometryFace(area=y_len*z_len, centroid=-half[0]*MathConstants.unitvecs[0], normal=-MathConstants.unitvecs[0], eta_s=eta_s, eta_d=eta_d, eta_a=eta_a, CD=CD),
        GeometryFace(area=x_len*z_len, centroid=half[1]*MathConstants.unitvecs[1], normal=MathConstants.unitvecs[1], eta_s=eta_s, eta_d=eta_d, eta_a=eta_a, CD=CD),
        GeometryFace(area=x_len*z_len, centroid=-half[1]*MathConstants.unitvecs[1], normal=-MathConstants.unitvecs[1], eta_s=eta_s, eta_d=eta_d, eta_a=eta_a, CD=CD),
        GeometryFace(area=x_len*y_len, centroid=half[2]*MathConstants.unitvecs[2], normal=MathConstants.unitvecs[2], eta_s=eta_s, eta_d=eta_d, eta_a=eta_a, CD=CD),
        GeometryFace(area=x_len*y_len, centroid=-half[2]*MathConstants.unitvecs[2], normal=-MathConstants.unitvecs[2], eta_s=eta_s, eta_d=eta_d, eta_a=eta_a, CD=CD),
    ]


def _environment_disturbances(geometry_faces: List[GeometryFace]) -> List:
    config = GeometryConfig(geometry_faces)
    return [GG_Disturbance(), Drag_Disturbance(config), SRP_Disturbance(config)]


def _box_environment_disturbances(dimensions: np.ndarray) -> List:
    return _environment_disturbances(_box_geometry_faces(dimensions))


def _lightsail2_geometry_faces() -> List[GeometryFace]:
    r"""
    Approximate LightSail 2 as a 3U core plus a deployed two-sided sail.

    Public mission summaries give a 3U CubeSat bus and a deployed sail area of
    32 m^2. The exact panel centroid coordinates are not encoded in the public
    factory sources, so the sail is represented as four coplanar triangular
    quadrants on each side of a square sail with total area 32 m^2.
    """
    faces = _box_geometry_faces(np.array([0.1, 0.1, 0.3]))
    total_sail_area = 32.0
    half_side = np.sqrt(total_sail_area) / 2.0
    centroid_offset = 2.0 * half_side / 3.0
    sail_centroids = [
        np.array([centroid_offset, 0.0, 0.0]),
        np.array([0.0, centroid_offset, 0.0]),
        np.array([-centroid_offset, 0.0, 0.0]),
        np.array([0.0, -centroid_offset, 0.0]),
    ]
    sail_area = total_sail_area / 4.0
    for normal in (MathConstants.unitvecs[2], -MathConstants.unitvecs[2]):
        faces.extend(
            GeometryFace(area=sail_area, centroid=centroid, normal=normal, eta_s=0.8, eta_d=0.1, eta_a=0.1, CD=2.2)
            for centroid in sail_centroids
        )
    return faces


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

    disturbances = _box_environment_disturbances(np.array([0.1, 0.1, 0.3]))

    boresight = np.array([0, 1, 0])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=mtqs, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=mtqs, boresight=boresight)


def create_brite_austria(estimated: bool = False):
    r"""
    Create a BRITE-Austria/TUGSAT-1 spacecraft model.

    BRITE-Austria used the Generic Nanosatellite Bus with three reaction
    wheels, three magnetorquers, three rate sensors, one three-axis
    magnetometer, six dedicated Sun sensors, and one AeroAstro Miniature Star
    Tracker. The representative BRITE/GNB inertia tensor is used here. No
    defensible numerical TUGSAT-1 COM vector was found, so ``COM = 0``.
    Disturbances use a 20 cm cube geometry from the public BRITE/GNB bus
    description with generic optical and aerodynamic coefficients.
    """
    mass = 6.9
    COM = np.zeros(3)
    J = np.array([
        [0.0465, -0.0007, 0.0004],
        [-0.0007, 0.0486, -0.0021],
        [0.0004, -0.0021, 0.0482],
    ])

    rws: List[RW] = create_sfl_reaction_wheels(estimate_bias=estimated)
    mtqs: List[MTQ] = create_gnb_air_core_magnetorquers(estimate_bias=estimated)
    gyros: List[Gyro] = create_gnb_rate_sensors(estimate_bias=estimated)
    mtms: List[MTM] = create_gnb_magnetometer(estimate_bias=estimated)
    suns = create_gnb_sun_sensors(estimate_bias=estimated)
    star_trackers = [create_aeroastro_mst(estimate_bias=estimated)]
    disturbances = _box_environment_disturbances(np.array([0.2, 0.2, 0.2]))
    boresight = np.array([0, 0, 1])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=gyros+mtms+suns+star_trackers, actuators=rws+mtqs, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=gyros+mtms+suns+star_trackers, actuators=rws+mtqs, boresight=boresight)


def create_lightsail2(estimated: bool = False):
    r"""
    Create a LightSail 2 deployed-sail spacecraft model.

    LightSail 2 used one +Y Sinclair Interplanetary momentum wheel, three
    Stras Space magnetic torque rods, two three-axis Honeywell magnetometers,
    three primary PIB gyros, three secondary mainboard gyros, and five coarse
    Elmos Sun sensors. The inertia tensor is the published deployed-sail tensor
    and the COM is taken from the LightSail-B deployed corner-cube drawing.
    Disturbances use a 3U bus plus a two-sided 32 m^2 deployed-sail geometry
    with generic aerodynamic and approximate aluminized-Mylar optical
    coefficients.
    """
    mass = 4.93
    COM = np.array([0.00046, -0.00003, 0.13746])
    J = np.array([
        [3.79, -1.90e-4, -8.18e-4],
        [-1.90e-4, 3.79, 1.47e-3],
        [-8.18e-4, 1.47e-3, 7.33],
    ])

    rw: List[RW] = [create_sinclair_interplanetary_momentum_wheel(estimate_bias=estimated)]
    mtqs: List[MTQ] = create_stras_space_torque_rods(estimate_bias=estimated)
    mtms: List[MTM] = create_honeywell_lightsail2_magnetometers(estimate_bias=estimated)
    gyros: List[Gyro] = (
        create_analog_devices_pib_gyros(estimate_bias=estimated)
        + create_intrepid_mainboard_gyros(estimate_bias=estimated)
    )
    suns = create_elmos_sun_sensors(estimate_bias=estimated)
    disturbances = _environment_disturbances(_lightsail2_geometry_faces())
    boresight = np.array([0, 0, 1])

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=rw+mtqs, boresight=boresight)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=disturbances, sensors=mtms+gyros+suns, actuators=rw+mtqs, boresight=boresight)


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
    mtms: List[MTM] = [
        mtm
        for _ in range(2)
        for mtm in create_hmc5883l_magnetometers(estimate_bias=estimated)
    ]
    gyros: List[Gyro] = [
        gyro
        for _ in range(4)
        for gyro in create_itg3200_gyros(estimate_bias=estimated)
    ]
    sun_axes = np.repeat(np.vstack((np.eye(3), -np.eye(3))), 2, axis=0)
    suns = [
        sun
        for axis in sun_axes
        for sun in create_hamamatsu_s3931_sun_sensors(axes=np.array([axis]), estimate_bias=estimated)
    ]

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


def create_moveii_cubesat(estimated: bool = False):
    r"""
    Create a MOVE-II CubeSat model.

    MOVE-II is a 1U CubeSat with a distributed magnetorquer-based ADCS: six
    ADCS PCBs carry Bosch Sensortec BMX055 gyros and magnetometers, the five
    outer panels carry Solar MEMS NANO-ISS60 Sun sensors, and the ADCS panels
    include custom PCB magnetorquer coils.

    The inertia tensor follows the diagonal values printed in the MOVE-II
    magnetic-control simulation paper,
    ``diag(0.00297, 0.00330, 0.00320) kg m^2``. No public center-of-mass vector
    was found in the attached source summary, so the factory uses ``COM = 0``.

    Source:
        `Hardware-In-The-Loop and Software-In-The-Loop Testing of the MOVE-II
        CubeSat <https://doi.org/10.3390/aerospace6120130>`__
    """
    mass = 1.2
    COM = np.zeros(3)
    J = np.diag([0.00297, 0.00330, 0.00320])

    mtqs: List[MTQ] = create_moveii_pcb_magnetorquers(estimate_bias=estimated)
    mtms: List[MTM] = [
        mtm
        for _ in range(6)
        for mtm in create_bmx055_magnetometers(estimate_bias=estimated)
    ]
    gyros: List[Gyro] = [
        gyro
        for _ in range(6)
        for gyro in create_bmx055_gyros(estimate_bias=estimated)
    ]
    sun_axes = np.vstack((np.eye(3), -np.eye(2, 3)))
    suns = [
        sun
        for axis in sun_axes
        for sun in create_nano_iss60_sun_sensors(axes=np.array([axis]), estimate_bias=estimated)
    ]

    geometry_faces: List[GeometryFace] = [
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[0]*0.05, normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[0]*0.05, normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[1]*0.05, normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[1]*0.05, normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.05, normal=MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.05, normal=-MathConstants.unitvecs[2], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
    ]
    config = GeometryConfig(geometry_faces)
    residual_dipole = Dipole_Disturbance(dipole_torque=np.array([-0.001, 0.012, -0.045]))
    disturbances = [GG_Disturbance(), Drag_Disturbance(config), SRP_Disturbance(config), residual_dipole]

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
