import numpy as np
from typing import List

from ADCS.satellite_factory.actuators import create_cubewheel_smallplus_rw, create_isis_magnetorquer_board
from ADCS.satellite_factory.sensors import create_Clydespace_3U_array, create_ICM20948_IMU, create_isis_magnetometer
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM, Gyro, SunPair
from ADCS.satellite_hardware.disturbances import (
    GeometryFace, GeometryConfig, Drag_Disturbance, GG_Disturbance, 
    SRP_Disturbance, Dipole_Disturbance, General_Disturbance
)
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

def create_wisniewski_test_satellite(estimated: bool = False, 
                                      use_disturbances: bool = True,
                                      disturbance_mode: str = 'full'):
    """Create satellite matching Wisniewski sliding mode test parameters from thesis.
    
    Parameters from thesis Table 'Wisniewski Comparison Details':
    - Inertia: J = diag([3.428, 2.904, 1.275]) kg·m²
    - MTQ max: 20 Am²
    - Mass: ~50 kg (estimated)
    
    Parameters
    ----------
    estimated : bool
        If True, return EstimatedSatellite; else Satellite
    use_disturbances : bool
        If True, include disturbance models
    disturbance_mode : str
        - 'full': Include GG, drag, SRP, dipole (for truth or disturbance-aware)
        - 'general': Include GG + general disturbance estimator (for all-in-one)
        - 'gg_only': Include only GG (for Clean case per thesis)
        - 'none': No disturbances
    """
    mass = 50.0
    COM = np.zeros(3)
    J = np.diag([3.428, 2.904, 1.275])  # kg·m² from thesis
    
    # Actuators - three-axis MTQs with 20 Am² max
    mtqs: List[MTQ] = [
        MTQ(axis=MathConstants.unitvecs[0], max_torque=20.0),
        MTQ(axis=MathConstants.unitvecs[1], max_torque=20.0),
        MTQ(axis=MathConstants.unitvecs[2], max_torque=20.0),
    ]
    
    # Sensors - simple 3-axis MTM
    mtms: List[MTM] = [
        MTM(axis=MathConstants.unitvecs[0]),
        MTM(axis=MathConstants.unitvecs[1]),
        MTM(axis=MathConstants.unitvecs[2]),
    ]
    
    # Geometry for drag/SRP (from thesis common_sats.py)
    geometry_faces: List[GeometryFace] = [
        GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[0]*0.05, 
                     normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[0]*0.05, 
                     normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=MathConstants.unitvecs[1]*0.05, 
                     normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.1*0.3, centroid=-MathConstants.unitvecs[1]*0.05, 
                     normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.3, eta_a=0.2, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=MathConstants.unitvecs[2]*0.15, 
                     normal=MathConstants.unitvecs[2], eta_s=0.3, eta_d=0.5, eta_a=0.2, CD=2.2),
        GeometryFace(area=0.1*0.1, centroid=-MathConstants.unitvecs[2]*0.15, 
                     normal=-MathConstants.unitvecs[2], eta_s=0.25, eta_d=0.6, eta_a=0.15, CD=2.2),
    ]
    config = GeometryConfig(geometry_faces)
    
    # Disturbances based on mode
    disturbances = []
    if use_disturbances and disturbance_mode != 'none':
        if disturbance_mode == 'gg_only':
            # Clean case: GG only (per thesis Table)
            disturbances.append(GG_Disturbance())
        elif disturbance_mode == 'full':
            # Full model: GG, drag, SRP, dipole (MTQ bias equivalent)
            disturbances.append(GG_Disturbance())
            disturbances.append(Drag_Disturbance(config))
            disturbances.append(SRP_Disturbance(config))
            disturbances.append(Dipole_Disturbance(
                dipole_moment=np.array([0.05, 0.0001, 0.2]),
                estimate_dist=estimated
            ))
        elif disturbance_mode == 'general':
            # All-in-one: analytical GG + general disturbance estimator
            disturbances.append(GG_Disturbance())
            disturbances.append(General_Disturbance(
                torque_init=np.zeros(3),
                std=5e-6,
                mag_max=1e-3,
                estimate_dist=True
            ))
    
    boresight = np.array([0, 0, 1])
    
    if estimated:
        return EstimatedSatellite(
            mass=mass, COM=COM, J_0=J, 
            disturbances=disturbances,
            sensors=mtms, actuators=mtqs, 
            boresight=boresight
        )
    else:
        return Satellite(
            mass=mass, COM=COM, J_0=J, 
            disturbances=disturbances,
            sensors=mtms, actuators=mtqs, 
            boresight=boresight
        )


def create_lovera_test_satellite(estimated: bool = False, 
                                  use_disturbances: bool = True,
                                  disturbance_mode: str = 'full'):
    """Create satellite matching Lovera PD test parameters from thesis.
    
    Parameters from thesis Table 'Lovera Comparison Details':
    - Inertia: J = diag([27, 17, 25]) kg·m²
    - MTQ max: 50 Am² (based on original Lovera paper)
    - Mass: ~100 kg (estimated)
    
    Same disturbance options as Wisniewski test.
    """
    mass = 100.0
    COM = np.zeros(3)
    J = np.diag([27.0, 17.0, 25.0])  # kg·m² from thesis (Lovera paper values)
    
    # Actuators - three-axis MTQs with 50 Am² max
    mtqs: List[MTQ] = [
        MTQ(axis=MathConstants.unitvecs[0], max_torque=50.0),
        MTQ(axis=MathConstants.unitvecs[1], max_torque=50.0),
        MTQ(axis=MathConstants.unitvecs[2], max_torque=50.0),
    ]
    
    # Sensors - simple 3-axis MTM
    mtms: List[MTM] = [
        MTM(axis=MathConstants.unitvecs[0]),
        MTM(axis=MathConstants.unitvecs[1]),
        MTM(axis=MathConstants.unitvecs[2]),
    ]
    
    # Geometry for drag/SRP (scaled up from Wisniewski case)
    geometry_faces: List[GeometryFace] = [
        GeometryFace(area=0.2*0.4, centroid=MathConstants.unitvecs[0]*0.1, 
                     normal=MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.2*0.4, centroid=-MathConstants.unitvecs[0]*0.1, 
                     normal=-MathConstants.unitvecs[0], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.2*0.4, centroid=MathConstants.unitvecs[1]*0.1, 
                     normal=MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.2, eta_a=0.3, CD=2.2),
        GeometryFace(area=0.2*0.4, centroid=-MathConstants.unitvecs[1]*0.1, 
                     normal=-MathConstants.unitvecs[1], eta_s=0.5, eta_d=0.3, eta_a=0.2, CD=2.2),
        GeometryFace(area=0.2*0.2, centroid=MathConstants.unitvecs[2]*0.2, 
                     normal=MathConstants.unitvecs[2], eta_s=0.3, eta_d=0.5, eta_a=0.2, CD=2.2),
        GeometryFace(area=0.2*0.2, centroid=-MathConstants.unitvecs[2]*0.2, 
                     normal=-MathConstants.unitvecs[2], eta_s=0.25, eta_d=0.6, eta_a=0.15, CD=2.2),
    ]
    config = GeometryConfig(geometry_faces)
    
    # Disturbances based on mode (same as Wisniewski)
    disturbances = []
    if use_disturbances and disturbance_mode != 'none':
        if disturbance_mode == 'full':
            disturbances.append(GG_Disturbance())
            disturbances.append(Drag_Disturbance(config))
            disturbances.append(SRP_Disturbance(config))
            disturbances.append(Dipole_Disturbance(
                dipole_moment=np.array([0.05, 0.0001, 0.2]),
                estimate_dist=estimated
            ))
        elif disturbance_mode == 'general':
            disturbances.append(GG_Disturbance())
            disturbances.append(General_Disturbance(
                torque_init=np.zeros(3),
                std=5e-6,
                mag_max=1e-3,
                estimate_dist=True
            ))
    
    boresight = np.array([0, 0, 1])
    
    if estimated:
        return EstimatedSatellite(
            mass=mass, COM=COM, J_0=J, 
            disturbances=disturbances,
            sensors=mtms, actuators=mtqs, 
            boresight=boresight
        )
    else:
        return Satellite(
            mass=mass, COM=COM, J_0=J, 
            disturbances=disturbances,
            sensors=mtms, actuators=mtqs, 
            boresight=boresight
        )
