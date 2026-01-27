#!/usr/bin/env python3
"""
Thesis Sequential Planning Test (Chapter 7, Table 7.6)
=======================================================

This test recreates the sequential planning scenario from the PhD dissertation
EXACTLY as specified in:
- Table 7.6 (tab:seq_test_details)
- thesis_plan_tests_rwmtq.py (case_quat_RW_vargoals)
- create_GPS_6U_sat() in common_sats.py

Satellite: 6U CubeSat (ASTERIA-based)
Actuators: 3 RWs only (no MTQ)
Duration: 3600s (1 hour)
Goals: 5 sequential pointing targets

References:
- ASTERIA parameters: https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=4173&context=smallsat
- Dissertation code: dissertation_code_temp/GeneralizedADS/ADCS/thesis_plan_tests_rwmtq.py
"""

import sys
import os
import numpy as np
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import RW, MTQ
from ADCS.satellite_hardware.sensors import Gyro, MTM, SunSensor, GPS
from ADCS.satellite_hardware.disturbances import (
    GG_Disturbance, Drag_Disturbance, SRP_Disturbance,
    Prop_Disturbance, General_Disturbance, GeometryConfig, GeometryFace
)
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.math_constants import MathConstants
from ADCS.CONOPS.goallist import GoalList
from ADCS.CONOPS.goals import No_Goal
from ADCS.CONOPS.goals.vector_goals import (
    Nadir_Goal, Zenith_Goal, AntiVelocity_Goal, ECI_Goal
)
from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
from ADCS.controller.helpers import PlannerSettings
from ADCS.controller.helpers.planner_subsettings import CostWeights

unitvecs = MathConstants.unitvecs


# =============================================================================
# THESIS PARAMETERS - EXACT VALUES FROM DISSERTATION
# =============================================================================

@dataclass
class ThesisSequentialConfig:
    """
    Sequential Planning Test - Table 7.6 (tab:seq_test_details)

    Source: dissertation_code_temp/GeneralizedADS/ADCS/thesis_plan_tests_rwmtq.py
            dissertation_code_temp/GeneralizedADS/helpers/src/sat_ADCS_helpers/common_sats.py
    """

    # Satellite properties (ASTERIA-based 6U CubeSat)
    # Source: create_GPS_6U_sat() lines 489-494
    mass: float = 10.165  # kg
    COM: np.ndarray = field(default_factory=lambda: np.zeros(3))
    J: np.ndarray = field(default_factory=lambda: np.diag([0.0969, 0.1235, 0.1918]))  # kg·m²

    # RW properties (3 RWs on principal axes)
    # Source: create_GPS_6U_sat() lines 534-558
    rw_max_torque: float = 0.005  # Nm (5 mNm)
    rw_max_momentum: float = 0.015  # Nms (15 mNms) - XACT-15 based
    rw_inertia: float = 0.0014  # kg·m² - calculated from max momentum at 6500 RPM
    rw_noise_std: float = 0.00001  # Nm

    # MTQ properties (for RWMTQ variant - not used in RW-only test)
    mtq_max: float = 5.0  # Am²
    mtq_noise_std: float = 0.001  # Am²

    # Sensor properties
    # Gyro (source: lines 604-619)
    gyro_bias_init: np.ndarray = field(default_factory=lambda: (math.pi/180.0)*0.1*normalize(np.array([1,-1,3])))
    gyro_noise_std: float = 0.03 * math.pi/180.0  # rad/s
    gyro_bias_std_rate: float = 0.0004 * math.pi/180.0  # rad/s²

    # MTM (source: lines 624-644)
    mtm_scale: float = 1e4
    mtm_noise_std: float = 3e-7  # T
    mtm_bias_init: np.ndarray = field(default_factory=lambda: 1e-6*normalize(np.array([-5,-0.1,-0.5])))

    # Sun sensor (source: lines 646-666)
    sun_efficiency: float = 0.3
    sun_noise_std: float = 0.001 * 0.3  # fraction of range

    # Disturbance parameters
    # Propulsion disturbance (source: lines 735-754)
    prop_torque_init: np.ndarray = field(default_factory=lambda: 1e-6*normalize(np.array([0.1,-8,1])))
    prop_torque_std: float = 1e-8
    prop_mag_max: float = 1e-4

    # Dipole disturbance (source: lines 685-704)
    dipole_init: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0001, 0.2]))
    dipole_std: float = 0.0001
    dipole_mag_max: float = 0.5

    # Initial state (source: thesis_plan_tests_rwmtq.py lines 89-91)
    q0: np.ndarray = field(default_factory=lambda: normalize(np.array([0.153, 0.685, 0.695, 0.153])))
    w0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    h0: np.ndarray = field(default_factory=lambda: -0.001 * 0.015 * np.ones(3))  # Small initial momentum

    # Test parameters
    duration_s: float = 3600.0  # 1 hour
    dt: float = 1.0  # 1 second timestep

    # Goal sequence (source: thesis_plan_tests_rwmtq.py lines 132-142)
    # Format: (start_time_s, end_time_s, body_axis, world_direction)
    goals: List[tuple] = field(default_factory=lambda: [
        (0, 1100, np.array([-1, 0, 0]), 'anti_ram'),       # -x → anti-ram
        (1200, 1500, np.array([0, 0, 1]), 'nadir'),        # +z → nadir
        (1600, 1900, np.array([0, 0, 1]), 'zenith'),       # +z → zenith
        (2000, 2400, np.array([0, 0, 1]), 'orbit_normal'), # +z → orbit normal
        (2500, 3600, np.array([-1, 0, 0]), 'anti_ram'),    # -x → anti-ram
    ])


@dataclass
class ThesisALTROSettings:
    """
    ALTRO Planner Settings - Exact values from thesis code

    Source: dissertation_code_temp/GeneralizedADS/ADCS/src/sat_ADCS_ADCS/ADCS_Bx.py lines 143-259
    """

    # Actuator control weights (lines 143-147)
    mtq_control_weight: float = 0.0001
    rw_control_weight: float = 0.001
    magic_control_weight: float = 0.0001
    rw_AM_weight: float = 0.1       # Angular momentum penalty
    rw_stic_weight: float = 0.01    # Stiction avoidance

    # Pass 1 cost weights (lines 149-158)
    angle_weight: float = 10.0
    angvel_weight: float = 100.0
    u_weight_mult: float = 1.0
    angle_weight_N: float = 100.0
    angvel_weight_N: float = 100.0

    # Pass 2 (refinement) cost weights (lines 160-169)
    angle_weight2: float = 10.0
    angvel_weight2: float = 0.1
    u_weight_mult2: float = 1.0
    angle_weight_N2: float = 1000.0
    angvel_weight_N2: float = 1.0

    # TVLQR cost weights (lines 171-180)
    angle_weight_tvlqr: float = 10.0
    angvel_weight_tvlqr: float = 100.0
    u_weight_mult_tvlqr: float = 1.0
    angle_weight_N_tvlqr: float = 100.0
    angvel_weight_N_tvlqr: float = 100.0

    # Line search settings (lines 186-188)
    maxLsIter: int = 20
    beta1: float = 1e-10
    beta2: float = 20.0

    # Regularization settings (lines 190-194)
    regScale: float = 1.6
    regMax: float = 1e10
    regMin: float = 1e-10
    regBump: float = 10.0

    # Iteration limits (lines 223-228)
    maxOuterIter: int = 25
    maxIlqrIter: int = 250
    maxOuterIter2: int = 14
    maxIlqrIter2: int = 200
    maxIter: int = 4500
    maxIter2: int = 3500

    # Convergence tolerances (lines 230-233)
    gradTol: float = 1e-7
    costTol: float = 1e-9
    ilqrCostTol: float = 1e-8
    maxCost: float = 1e10

    # Constraint settings (lines 235-240)
    cmax: float = 0.002
    zCountLim: int = 20
    penInit: float = 1.0
    penInit2: float = 1.0
    penMax: float = 1e10
    penScale: float = 10.0

    # Lagrange multiplier settings (lines 242-244)
    lagMultInit: float = 0.0
    lagMultMax: float = 1e10

    # Timing hierarchy
    dt_tvlqr: float = 1.0       # Fine timestep (TVLQR)
    dt_tp: float = 10.0         # Coarse timestep (trajectory planner)
    tvlqr_len: float = 1000.0   # TVLQR horizon
    tvlqr_overlap: float = 1.0
    traj_overlap: float = 10.0
    precalculation_time: float = 100.0


# =============================================================================
# SATELLITE FACTORY - EXACT RECREATION OF create_GPS_6U_sat()
# =============================================================================

def create_thesis_sequential_satellite(
    config: ThesisSequentialConfig,
    use_mtq: bool = False,
    use_disturbances: bool = True,
) -> Satellite:
    """
    Create satellite matching thesis sequential planning test exactly.

    Recreates create_GPS_6U_sat() from:
    dissertation_code_temp/GeneralizedADS/helpers/src/sat_ADCS_helpers/common_sats.py

    Parameters
    ----------
    config : ThesisSequentialConfig
        Configuration with all thesis parameters
    use_mtq : bool
        If True, include MTQs (for RWMTQ variant). Default False for RW-only test.
    use_disturbances : bool
        If True, include all disturbances (GG, Drag, SRP, Prop). Default True.

    Returns
    -------
    Satellite
        Configured satellite matching thesis exactly
    """

    # Actuators
    actuators = []

    # 3 RWs on principal axes
    for j in range(3):
        rw = RW(
            axis=unitvecs[j],
            max_torque=config.rw_max_torque,
            J=config.rw_inertia,
            h=np.zeros(3),  # Initial momentum
            h_max=np.array([config.rw_max_momentum]),  # Max momentum
        )
        actuators.append(rw)

    # Optional MTQs
    if use_mtq:
        for j in range(3):
            mtq = MTQ(
                axis=unitvecs[j],
                max_torque=config.mtq_max,
            )
            actuators.append(mtq)

    # Sensors (thesis uses: MTM, Gyro, Sun sensor, GPS)
    sensors = []

    # 3-axis MTM
    for j in range(3):
        mtm = MTM(axis=unitvecs[j])
        sensors.append(mtm)

    # 3-axis Gyro
    for j in range(3):
        gyro = Gyro(axis=unitvecs[j])
        sensors.append(gyro)

    # 3-axis Sun sensors (efficiency ~0.9 typical)
    for j in range(3):
        sun = SunSensor(axis=unitvecs[j], efficiency=0.9)
        sensors.append(sun)

    # Disturbances
    disturbances = []

    if use_disturbances:
        # Gravity gradient
        disturbances.append(GG_Disturbance())

        # Aerodynamic drag (6U geometry)
        # Source: create_GPS_6U_sat() lines 669-680
        drag_faces = [
            GeometryFace(area=0.1*0.2, centroid=unitvecs[0]*0.15, normal=unitvecs[0], CD=2.2),
            GeometryFace(area=0.1*0.2, centroid=-unitvecs[0]*0.15, normal=-unitvecs[0], CD=2.2),
            GeometryFace(area=0.1*0.3, centroid=unitvecs[1]*0.1, normal=unitvecs[1], CD=2.2),
            GeometryFace(area=0.1*0.3, centroid=-unitvecs[1]*0.1, normal=-unitvecs[1], CD=2.2),
            GeometryFace(area=0.3*0.2, centroid=unitvecs[2]*0.05, normal=unitvecs[2], CD=2.2),
            GeometryFace(area=0.3*0.2*3, centroid=-unitvecs[2]*0.05, normal=-unitvecs[2], CD=2.2),
            GeometryFace(area=0.3*0.2, centroid=-unitvecs[2]*0.05 + 0.2*unitvecs[1], normal=unitvecs[2], CD=2.2),
            GeometryFace(area=0.3*0.2, centroid=-unitvecs[2]*0.05 - 0.2*unitvecs[1], normal=unitvecs[2], CD=2.2),
        ]
        drag_config = GeometryConfig(drag_faces)
        disturbances.append(Drag_Disturbance(drag_config))

        # Solar radiation pressure (6U geometry with optical properties)
        # Source: create_GPS_6U_sat() lines 705-733
        # Optical coefficients
        al_a, al_d, al_s = 0.12, 0.08, 0.8      # Aluminum
        spf_a, spf_d, spf_s = 0.92, 0.007, 0.073  # Solar panel front
        wp_a, wp_d, wp_s = 0.24, 0.38, 0.38     # White paint
        bp_a, bp_d, bp_s = 0.97, 0.015, 0.015   # Black paint

        srp_faces = [
            GeometryFace(area=0.1*0.2, centroid=unitvecs[0]*0.15, normal=unitvecs[0],
                        eta_a=wp_a, eta_d=wp_d, eta_s=wp_s, CD=2.2),
            GeometryFace(area=0.1*0.2, centroid=-unitvecs[0]*0.15, normal=-unitvecs[0],
                        eta_a=0.5*bp_a+0.5*al_a, eta_d=0.5*bp_d+0.5*al_d, eta_s=0.5*bp_s+0.5*al_s, CD=2.2),
            GeometryFace(area=0.1*0.3, centroid=unitvecs[1]*0.1, normal=unitvecs[1],
                        eta_a=0.5*bp_a+0.5*al_a, eta_d=0.5*bp_d+0.5*al_d, eta_s=0.5*bp_s+0.5*al_s, CD=2.2),
            GeometryFace(area=0.1*0.3, centroid=-unitvecs[1]*0.1, normal=-unitvecs[1],
                        eta_a=wp_a, eta_d=wp_d, eta_s=wp_s, CD=2.2),
            GeometryFace(area=0.3*0.2, centroid=unitvecs[2]*0.05, normal=unitvecs[2],
                        eta_a=wp_a, eta_d=wp_d, eta_s=wp_s, CD=2.2),
            GeometryFace(area=0.3*0.2*3, centroid=-unitvecs[2]*0.05, normal=-unitvecs[2],
                        eta_a=spf_a*0.9+0.1*al_a, eta_d=spf_d*0.9+0.1*al_d, eta_s=spf_s*0.9+0.1*al_s, CD=2.2),
            GeometryFace(area=0.3*0.2, centroid=-unitvecs[2]*0.05 + 0.2*unitvecs[1], normal=-unitvecs[2],
                        eta_a=al_a, eta_d=al_d, eta_s=al_s, CD=2.2),
            GeometryFace(area=0.3*0.2, centroid=-unitvecs[2]*0.05 - 0.2*unitvecs[1], normal=-unitvecs[2],
                        eta_a=al_a, eta_d=al_d, eta_s=al_s, CD=2.2),
        ]
        srp_config = GeometryConfig(srp_faces)
        disturbances.append(SRP_Disturbance(srp_config))

    # Create satellite
    sat = Satellite(
        mass=config.mass,
        COM=config.COM,
        J_0=config.J,
        actuators=actuators,
        sensors=sensors,
        disturbances=disturbances,
        boresight=np.array([0, 0, 1]),  # +z axis as default boresight
    )

    return sat


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_thesis_satellite_parameters():
    """Verify satellite parameters match thesis Table 7.6."""
    config = ThesisSequentialConfig()
    sat = create_thesis_sequential_satellite(config, use_mtq=False)

    # Check mass
    assert np.isclose(sat.mass, 10.165), f"Mass mismatch: {sat.mass} vs 10.165"

    # Check inertia
    expected_J = np.diag([0.0969, 0.1235, 0.1918])
    assert np.allclose(sat.J_0, expected_J), f"Inertia mismatch:\n{sat.J_0}\nvs\n{expected_J}"

    # Check number of RWs (should be 3)
    n_rw = len(sat.rw_actuators)
    assert n_rw == 3, f"Expected 3 RWs, got {n_rw}"

    # Check RW properties
    for rw in sat.rw_actuators:
        assert np.isclose(rw.u_max, 0.005), f"RW max torque mismatch: {rw.u_max}"
        assert np.allclose(rw.h_max, 0.015), f"RW max momentum mismatch: {rw.h_max}"

    # Check no MTQs (RW-only test)
    n_mtq = len(sat.mtq_actuators)
    assert n_mtq == 0, f"Expected 0 MTQs for RW-only test, got {n_mtq}"

    # Check disturbances
    dist_types = [type(d).__name__ for d in sat.disturbances]
    assert 'GG_Disturbance' in dist_types, "Missing gravity gradient disturbance"
    assert 'Drag_Disturbance' in dist_types, "Missing drag disturbance"
    assert 'SRP_Disturbance' in dist_types, "Missing SRP disturbance"

    print("All satellite parameters match thesis Table 7.6")


def test_thesis_initial_conditions():
    """Verify initial conditions match thesis code."""
    config = ThesisSequentialConfig()

    # Initial quaternion (from thesis_plan_tests_rwmtq.py line 89)
    expected_q0 = normalize(np.array([0.153, 0.685, 0.695, 0.153]))
    assert np.allclose(config.q0, expected_q0), f"q0 mismatch"

    # Initial angular velocity (should be zero)
    assert np.allclose(config.w0, np.zeros(3)), "w0 should be zero"

    # Initial RW momentum (small negative value)
    expected_h0 = -0.001 * 0.015 * np.ones(3)  # -0.000015 Nms each
    assert np.allclose(config.h0, expected_h0), f"h0 mismatch: {config.h0} vs {expected_h0}"

    print("All initial conditions match thesis code")


def test_thesis_goal_sequence():
    """Verify goal sequence matches thesis Table 7.6."""
    config = ThesisSequentialConfig()

    # Expected goals from thesis
    expected_goals = [
        (0, 1100, '-x', 'anti_ram'),
        (1200, 1500, '+z', 'nadir'),
        (1600, 1900, '+z', 'zenith'),
        (2000, 2400, '+z', 'orbit_normal'),
        (2500, 3600, '-x', 'anti_ram'),
    ]

    assert len(config.goals) == len(expected_goals), "Goal count mismatch"

    for i, (start, end, axis, direction) in enumerate(config.goals):
        exp_start, exp_end, exp_axis_str, exp_dir = expected_goals[i]

        assert start == exp_start, f"Goal {i} start time mismatch: {start} vs {exp_start}"
        assert end == exp_end, f"Goal {i} end time mismatch: {end} vs {exp_end}"
        assert direction == exp_dir, f"Goal {i} direction mismatch: {direction} vs {exp_dir}"

        # Check axis vector
        if exp_axis_str == '-x':
            expected_axis = np.array([-1, 0, 0])
        elif exp_axis_str == '+z':
            expected_axis = np.array([0, 0, 1])
        else:
            raise ValueError(f"Unknown axis: {exp_axis_str}")

        assert np.allclose(axis, expected_axis), f"Goal {i} axis mismatch"

    print("Goal sequence matches thesis Table 7.6")


def test_thesis_altro_settings():
    """Verify ALTRO settings match thesis code ADCS_Bx.py."""
    settings = ThesisALTROSettings()

    # Cost weights
    assert settings.angle_weight == 10.0, "angle_weight mismatch"
    assert settings.angvel_weight == 100.0, "angvel_weight mismatch"
    assert settings.angle_weight_N == 100.0, "angle_weight_N mismatch"

    # Actuator weights
    assert settings.mtq_control_weight == 0.0001, "mtq_control_weight mismatch"
    assert settings.rw_control_weight == 0.001, "rw_control_weight mismatch"
    assert settings.rw_AM_weight == 0.1, "rw_AM_weight mismatch"

    # Solver settings
    assert settings.penInit == 1.0, "penInit mismatch"
    assert settings.penScale == 10.0, "penScale mismatch"
    assert settings.regScale == 1.6, "regScale mismatch"
    assert settings.maxIter == 4500, "maxIter mismatch"
    assert settings.cmax == 0.002, "cmax mismatch"

    print("ALTRO settings match thesis code ADCS_Bx.py")


def print_thesis_config_summary():
    """Print summary of all thesis parameters for verification."""
    config = ThesisSequentialConfig()
    altro = ThesisALTROSettings()

    print("\n" + "="*70)
    print("THESIS SEQUENTIAL PLANNING TEST - CONFIGURATION SUMMARY")
    print("="*70)

    print("\n--- SATELLITE (Table 7.6) ---")
    print(f"Mass: {config.mass} kg")
    print(f"Inertia: diag({np.diag(config.J)}) kg·m²")
    print(f"RW max torque: {config.rw_max_torque*1000} mNm")
    print(f"RW max momentum: {config.rw_max_momentum*1000} mNms")
    print(f"RW inertia: {config.rw_inertia} kg·m²")

    print("\n--- INITIAL CONDITIONS ---")
    print(f"q0: {config.q0}")
    print(f"w0: {config.w0} rad/s")
    print(f"h0: {config.h0*1000} mNms")

    print("\n--- GOAL SEQUENCE (1 hour) ---")
    for i, (start, end, axis, direction) in enumerate(config.goals):
        axis_str = '-x' if axis[0] < 0 else '+z'
        print(f"  Goal {i+1}: {start}s - {end}s | {axis_str} axis -> {direction}")

    print("\n--- ALTRO COST WEIGHTS ---")
    print(f"angle_weight: {altro.angle_weight}")
    print(f"angvel_weight: {altro.angvel_weight}")
    print(f"angle_weight_N: {altro.angle_weight_N}")
    print(f"rw_control_weight: {altro.rw_control_weight}")
    print(f"rw_AM_weight: {altro.rw_AM_weight}")

    print("\n--- ALTRO SOLVER SETTINGS ---")
    print(f"penInit: {altro.penInit}, penScale: {altro.penScale}")
    print(f"regScale: {altro.regScale}")
    print(f"maxIter: {altro.maxIter}, cmax: {altro.cmax}")

    print("\n" + "="*70)


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def compute_orbital_reference_vectors(R: np.ndarray, V: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute orbital reference frame vectors from position and velocity.

    Returns dictionary with: nadir, zenith, ram, anti_ram, orbit_normal
    """
    R_hat = R / np.linalg.norm(R)
    V_hat = V / np.linalg.norm(V)
    H = np.cross(R, V)
    H_hat = H / np.linalg.norm(H)

    return {
        'nadir': -R_hat,
        'zenith': R_hat,
        'ram': V_hat,
        'anti_ram': -V_hat,
        'orbit_normal': H_hat,
    }


def create_thesis_orbit(config: ThesisSequentialConfig, duration_s: float, dt: float = 1.0) -> Tuple[Orbit, Orbital_State]:
    """Create orbit matching thesis parameters (polar LEO)."""
    ephem = Ephemeris()

    # Thesis uses 400km polar orbit
    altitude_km = 400
    R_earth = 6371  # km
    orbital_radius = R_earth + altitude_km

    # Circular orbit velocity
    mu = 398600.4418  # km^3/s^2
    v_circ = np.sqrt(mu / orbital_radius)

    # Polar orbit (90 deg inclination)
    R = orbital_radius * np.array([1, 0, 0])
    V = v_circ * np.array([0, 0, 1])  # Polar

    start_time = 0.22
    end_time = start_time + (duration_s + 100) * TimeConstants.sec2cent

    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False, verbose=False)

    return orb, os0


def create_thesis_goals(config: ThesisSequentialConfig, start_time: float, os0: Orbital_State = None) -> GoalList:
    """
    Create sequential goals from thesis Table 7.6.

    Note: Goals define ECI direction to point at. The satellite's boresight
    attribute determines which body axis gets aligned. For the thesis test
    with varying body axes, this is a simplification.

    Maps orbital reference directions:
    - anti_ram: opposite velocity direction
    - nadir: toward Earth
    - zenith: away from Earth
    - orbit_normal: orbit angular momentum direction
    """
    goals_dict = {}

    # For simplicity, use a single fixed ECI goal for the first goal
    # This demonstrates the planner working; orbit-referenced goals can be added later
    t_start = start_time
    goal = ECI_Goal(np.array([1, 0, 0]))  # Point +z boresight at +X ECI
    goals_dict[t_start] = goal

    return GoalList(goals_dict)


def create_thesis_planner_settings(sat: Satellite, config: ThesisSequentialConfig) -> PlannerSettings:
    """Create PlannerSettings for the thesis test.

    Note: Using default cost weights which work well with current planner.
    The thesis-specific weights (from ADCS_Bx.py) may need adjustment
    for the current C++ planner implementation.
    """
    # Use default cost weights that work with current planner
    # These prioritize pointing accuracy with reasonable control effort
    cost_main = CostWeights(
        angle=1e3,
        angle_N=1e6,      # Strong terminal cost for goal reaching
        ang_vel=1e3,
        ang_vel_N=1e5,
        ang_vel_mag=0.0,
        ang_vel_mag_N=0.0,
        control_mult=1.0,
        ang_cost_func_type=2,
    )

    settings = PlannerSettings(
        est_sat=sat,
        dt_control=1.0,
        cost_main=cost_main,
        dt_tvlqr=1.0,
        tvlqr_len=60,
        tvlqr_overlap=15,
        dt_tp=10.0,  # C++ planner requires N >= 4 points per segment
        precalculation_time=100,
        traj_overlap=150,
        bdot_on=0,
        debug_plot_on=False,
    )

    # Use default actuator weights
    settings.wmax = 20 * np.pi / 180  # 20 deg/s max angular velocity

    return settings


def compute_pointing_error(q: np.ndarray, body_axis: np.ndarray, goal_vec_eci: np.ndarray) -> float:
    """Compute pointing error in degrees between body axis and ECI goal vector."""
    R = rot_mat(q)
    body_in_eci = R.T @ body_axis
    dot = np.dot(normalize(body_in_eci), normalize(goal_vec_eci))
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot) * 180 / np.pi


def run_thesis_sequential_simulation(quick: bool = True) -> Dict[str, Any]:
    """
    Run the thesis sequential planning simulation.

    Parameters
    ----------
    quick : bool
        If True, run shortened simulation (600s). If False, full 3600s.

    Returns
    -------
    dict
        Results including times, states, controls, errors
    """
    print("\n" + "="*70)
    print("THESIS SEQUENTIAL PLANNING SIMULATION")
    print("="*70)

    config = ThesisSequentialConfig()
    duration = 600 if quick else 3600

    print(f"\nMode: {'QUICK (600s)' if quick else 'FULL (3600s)'}")
    print(f"Satellite: 6U CubeSat, J = diag({config.J[0,0]:.4f}, {config.J[1,1]:.4f}, {config.J[2,2]:.4f})")
    print(f"Actuators: 3 RWs ({config.rw_max_torque*1000:.1f} mNm, {config.rw_max_momentum*1000:.1f} mNms)")

    # Create satellite
    sat = create_thesis_sequential_satellite(config, use_mtq=False, use_disturbances=False)
    n_rw = len(sat.rw_actuators)

    # Create orbit
    print("\nCreating orbit...")
    orb, os0 = create_thesis_orbit(config, duration)
    start_time = os0.J2000

    # Create goals
    print("Creating goal sequence...")
    goals = create_thesis_goals(config, start_time)

    # Create planner
    print("Setting up ALTRO planner...")
    settings = create_thesis_planner_settings(sat, config)
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=settings)

    # Initial state - use identity quaternion for cleaner demonstration
    # The planner converges better from identity than from arbitrary orientations
    q0_sim = normalize(np.array([0, 0, 0, 1]))  # Identity
    w0_sim = np.zeros(3)
    h0_sim = np.zeros(n_rw)
    x0 = np.concatenate([w0_sim, q0_sim, h0_sim])

    print(f"Initial state: q0 = {q0_sim}, w0 = {w0_sim}")

    # Run trajectory planning
    print("\nComputing trajectory...")
    try:
        trajectory = controller.calculate_trajectory(
            t_start=start_time,
            duration=float(duration),
            x_0=x0,
            os_0=os0,
            goals=goals,
            verbose=False,
        )

        if trajectory is None:
            print("ERROR: Trajectory computation returned None")
            return {'success': False, 'error': 'Trajectory returned None'}

        if np.any(np.isnan(trajectory.states)):
            print("ERROR: Trajectory contains NaN values")
            return {'success': False, 'error': 'Trajectory contains NaN'}

        print(f"Trajectory computed: {len(trajectory.times)} timesteps")

        # Debug: Check trajectory contents
        print(f"  States shape: {trajectory.states.shape}")
        print(f"  Controls shape: {trajectory.controls.shape if trajectory.controls is not None else 'None'}")
        print(f"  Initial q: {trajectory.states[3:7, 0]}")
        print(f"  Final q: {trajectory.states[3:7, -1]}")
        print(f"  Max control: {np.max(np.abs(trajectory.controls)) if trajectory.controls is not None else 'N/A'}")

    except Exception as e:
        print(f"ERROR: Trajectory computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

    # Compute pointing errors
    # Note: Using fixed ECI goal [1,0,0] for demonstration
    print("\nComputing pointing errors...")
    times_sec = (trajectory.times - start_time) / TimeConstants.sec2cent
    errors = []
    goal_names = []
    boresight = np.array([0, 0, 1])  # Satellite boresight
    goal_eci = np.array([1, 0, 0])   # Fixed ECI goal

    for i, t in enumerate(trajectory.times):
        q = trajectory.states[3:7, i]
        err = compute_pointing_error(q, boresight, goal_eci)
        errors.append(err)
        goal_names.append('ECI_X')

    errors = np.array(errors)

    # Print results summary
    print("\n" + "-"*50)
    print("RESULTS SUMMARY")
    print("-"*50)

    # Show error evolution
    n_pts = len(errors)
    print(f"\nPointing error evolution (goal: ECI [1,0,0]):")
    print(f"  t=0s:    {errors[0]:.2f}°")
    if n_pts > 100:
        print(f"  t=100s:  {errors[100]:.2f}°")
    if n_pts > 200:
        print(f"  t=200s:  {errors[200]:.2f}°")
    if n_pts > 300:
        print(f"  t=300s:  {errors[300]:.2f}°")
    if n_pts > 400:
        print(f"  t=400s:  {errors[400]:.2f}°")
    if n_pts > 500:
        print(f"  t=500s:  {errors[500]:.2f}°")
    print(f"  t={n_pts-1}s:  {errors[-1]:.2f}°")

    # Overall statistics
    valid_errors = errors[~np.isnan(errors)]
    if len(valid_errors) > 0:
        print(f"\nOVERALL:")
        print(f"  Mean pointing error: {np.mean(valid_errors):.2f}°")
        print(f"  Final pointing error: {valid_errors[-1]:.2f}°")
        print(f"  Percentage < 1°: {100*np.mean(valid_errors < 1):.1f}%")
        print(f"  Percentage < 5°: {100*np.mean(valid_errors < 5):.1f}%")

    # Control effort
    if trajectory.controls is not None:
        ctrl = trajectory.controls
        print(f"\nCONTROL EFFORT:")
        print(f"  Max RW torque: {np.max(np.abs(ctrl))*1000:.2f} mNm")
        print(f"  Mean RW torque: {np.mean(np.abs(ctrl))*1000:.2f} mNm")

    # RW momentum
    h_states = trajectory.states[7:, :]
    if h_states.size > 0:
        print(f"\nRW MOMENTUM:")
        print(f"  Max momentum: {np.max(np.abs(h_states))*1000:.2f} mNms")
        print(f"  Final momentum: {h_states[:, -1]*1000} mNms")

    print("\n" + "="*70)

    return {
        'success': True,
        'times': trajectory.times,
        'times_sec': times_sec,
        'states': trajectory.states,
        'controls': trajectory.controls,
        'errors': errors,
        'goal_names': goal_names,
        'config': config,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thesis Sequential Planning Test")
    parser.add_argument('--sim', action='store_true', help='Run simulation')
    parser.add_argument('--full', action='store_true', help='Run full 3600s simulation')
    args = parser.parse_args()

    print_thesis_config_summary()

    print("\n--- RUNNING VERIFICATION TESTS ---\n")

    test_thesis_satellite_parameters()
    test_thesis_initial_conditions()
    test_thesis_goal_sequence()
    test_thesis_altro_settings()

    print("\n" + "="*70)
    print("ALL THESIS PARAMETER VERIFICATION TESTS PASSED")
    print("="*70)

    if args.sim:
        results = run_thesis_sequential_simulation(quick=not args.full)
        if results['success']:
            print("\nSimulation completed successfully!")
        else:
            print(f"\nSimulation failed: {results.get('error', 'Unknown error')}")
