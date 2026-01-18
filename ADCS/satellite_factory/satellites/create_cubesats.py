import numpy as np
from typing import List

from ADCS.satellite_factory.actuators import create_cubewheel_smallplus_rw, create_isis_magnetorquer_board
from ADCS.satellite_factory.sensors import create_Clydespace_3U_array, create_ICM20948_IMU, create_isis_magnetometer
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

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, sensors=mtms+gyros+suns, actuators=mtqs)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, sensors=mtms+gyros+suns, actuators=mtqs)


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

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws)


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

    if estimated:
        return EstimatedSatellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws)
    else:
        return Satellite(mass=mass, COM=COM, J_0=J, disturbances=gg_dist+drag_dist+srp_dist, sensors=mtms+gyros+suns, actuators=mtqs+rws)
