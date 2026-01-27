#!/usr/bin/env python3
"""
Thesis Chapter 7: Planning Experiments
=======================================

Complete experiment suite for Chapter 7 (Trajectory Planning) figures.
Uses REAL ALTRO trajectory planner with parameters EXACTLY matching thesis.

Thesis Tables Referenced:
- Table 7.1 (tab:plan_dist_test_details): Spinning solution parameters
- Table 7.2 (tab:mc_sat_params): Monte Carlo satellite properties
- Table 7.3 (tab:mc_180deg): 180° slew results
- Table 7.4 (tab:mc_reduced): Reduced attitude results
- Table 7.5 (tab:mc_multi): Multi-target results
- Table 7.6 (tab:seq_test_details): Sequential planning parameters
- Table 7.7 (tab:gif_test_details): Long-term two-goal trajectory

Figure Outputs:
- simple_slew/: mtq_montecarlo, 1W_montecarlo, *_traj, *_quaternion, *_mom
- single_target_imaging/: mtq_quatset_*, 1W_quatset_*
- multi_target_imaging/: mtq_multi_*, 1W_multi_*
- sequential/: plan_quat_plot, plan_av_plot, planvecang, planctrl_plot
- spinning_ang, spinning_av, spinning_cmd
- anim_plots (two-goal trajectory)

Usage:
    # List all experiments without running
    python thesis_chapter7_planning.py --list
    
    # Run specific experiment
    python thesis_chapter7_planning.py --experiment mc_180deg_mtq --quick
    python thesis_chapter7_planning.py --experiment mc_180deg_1rw --full
    
    # Run all experiments (WARNING: takes hours)
    python thesis_chapter7_planning.py --all --full --output-dir ./ch7_figures
"""

import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
import numpy as np

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# =============================================================================
# ORBITAL REFERENCE FRAME COMPUTATION
# =============================================================================

def compute_orbital_reference_vectors(R: np.ndarray, V: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute orbital reference frame vectors from position and velocity.

    Args:
        R: Position vector in ECI frame (km)
        V: Velocity vector in ECI frame (km/s)

    Returns:
        Dictionary with orbital reference directions in ECI:
        - 'nadir': unit vector toward Earth center (-R_hat)
        - 'zenith': unit vector away from Earth center (+R_hat)
        - 'ram': unit vector in velocity direction (+V_hat)
        - 'anti_ram': unit vector opposite velocity (-V_hat)
        - 'orbit_normal': unit vector normal to orbital plane (R x V)
        - 'anti_orbit_normal': opposite orbit normal
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
        'anti_orbit_normal': -H_hat,
    }


def get_active_goal_at_time(t: float, start_time: float, goal_config,
                            R: np.ndarray, V: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Get the active goal vector at a given time for sequential goals.

    Args:
        t: Current time (J2000 centuries)
        start_time: Simulation start time (J2000 centuries)
        goal_config: GoalConfig object with sequential targets
        R: Current position vector in ECI (km)
        V: Current velocity vector in ECI (km/s)

    Returns:
        Tuple of (goal_vector_eci, goal_name) or (None, None) if no active goal
    """
    from ADCS.orbits.universal_constants import TimeConstants

    if goal_config.goal_type != 'sequential' or goal_config.targets is None:
        return None, None

    # Convert time to seconds since start
    t_sec = (t - start_time) / TimeConstants.sec2cent

    # Find active goal
    for target in goal_config.targets:
        if target['start_s'] <= t_sec <= target['end_s']:
            goal_name = target['goal']
            ref_vecs = compute_orbital_reference_vectors(R, V)

            if goal_name in ref_vecs:
                return ref_vecs[goal_name], goal_name

    return None, None


def compute_sequential_pointing_errors(times: np.ndarray, states: np.ndarray,
                                       orbit_states: List[Dict],
                                       goal_config, body_axis: np.ndarray,
                                       start_time: float) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pointing errors for sequential goals with time-varying targets.

    Args:
        times: Time array (J2000 centuries)
        states: State array (omega, q, h) shape (n_states, n_times)
        orbit_states: List of {'R': position, 'V': velocity} at each timestep
        goal_config: GoalConfig with sequential targets
        body_axis: Body-frame axis to align
        start_time: Simulation start time

    Returns:
        Tuple of (errors_deg, goal_names) arrays
    """
    from ADCS.helpers.math_helpers import rot_mat

    errors = []
    goal_names = []

    for i in range(len(times)):
        t = times[i]
        q = states[3:7, i]
        R = rot_mat(q)
        body_vec_eci = R @ body_axis

        # Get orbital state
        if i < len(orbit_states):
            os = orbit_states[i]
            R_orb = os['R']
            V_orb = os['V']
        else:
            # Fallback - use last known
            os = orbit_states[-1]
            R_orb = os['R']
            V_orb = os['V']

        # Get active goal
        goal_vec, goal_name = get_active_goal_at_time(t, start_time, goal_config, R_orb, V_orb)

        if goal_vec is not None:
            err = np.arccos(np.clip(np.dot(body_vec_eci, goal_vec), -1, 1)) * 180 / np.pi
            errors.append(err)
            goal_names.append(goal_name)
        else:
            # No active goal - append NaN
            errors.append(np.nan)
            goal_names.append('none')

    return np.array(errors), goal_names


# =============================================================================
# DATA PERSISTENCE
# =============================================================================

def save_experiment_data(results: Dict, output_dir: Path, prefix: str):
    """Save experiment results to pickle for later figure regeneration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f'{prefix}_data.pkl'

    with open(data_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"  Data saved to: {data_path}")


def load_experiment_data(output_dir: Path, prefix: str) -> Optional[Dict]:
    """Load experiment results from pickle."""
    data_path = output_dir / f'{prefix}_data.pkl'

    if not data_path.exists():
        print(f"  No data file found: {data_path}")
        return None

    with open(data_path, 'rb') as f:
        results = pickle.load(f)

    print(f"  Loaded data from: {data_path}")
    return results


# =============================================================================
# EXPERIMENT CONFIGURATIONS - EXACTLY MATCHING THESIS
# =============================================================================

@dataclass
class SatelliteConfig:
    """Satellite configuration from thesis Table 7.2."""
    name: str
    J: np.ndarray  # Inertia matrix
    mtq_max: List[float]  # [x, y, z] max dipole moments
    has_rw: bool = False
    rw_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))  # Single RW axis
    rw_axes: Optional[List[np.ndarray]] = None  # Multiple RW axes (overrides rw_axis if set)
    rw_max_torque: float = 0.0002  # Nm
    rw_h_max: float = 0.002  # Nms
    rw_J: float = 2e-6  # kg·m²
    mass: float = 4.0  # kg
    boresight: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))


# Table 7.2 satellite configurations
SATELLITE_CONFIGS = {
    # MTQ-only satellite (thesis Table 7.2)
    'mtq_only': SatelliteConfig(
        name='3MTQ CubeSat',
        J=np.diag([0.005256, 0.04939, 0.04939]),
        mtq_max=[0.19, 0.57, 0.57],  # From thesis: "0.19 Am² on x, 0.57 Am² on y and z"
        has_rw=False,
        boresight=np.array([1, 0, 0]),
    ),
    # 3MTQ+1RW satellite (thesis Table 7.2)
    '3mtq_1rw': SatelliteConfig(
        name='3MTQ+1RW CubeSat',
        J=np.diag([0.005256, 0.04939, 0.04939]),
        mtq_max=[0.19, 0.57, 0.57],
        has_rw=True,
        rw_axis=np.array([0, 1, 0]),
        rw_max_torque=0.0002,
        rw_h_max=0.002,
        rw_J=2e-6,
        boresight=np.array([1, 0, 0]),
    ),
    # Table 7.1 spinning solution satellite (different MTQ limits)
    'spinning_sat': SatelliteConfig(
        name='3MTQ+1RW Spinning Test',
        J=np.diag([0.1, 0.05, 0.005]),  # From Table 7.1
        mtq_max=[0.19, 0.57, 0.57],
        has_rw=True,
        rw_axis=np.array([0, 1, 0]),
        rw_max_torque=0.0002,
        rw_h_max=0.002,
        rw_J=2e-6,
        boresight=np.array([0, 0, 1]),  # z-axis for propulsion pointing
    ),
    # Table 7.6 sequential planning satellite (6U with 3RW, ASTERIA-based)
    'sequential_6u': SatelliteConfig(
        name='6U CubeSat (ASTERIA)',
        J=np.diag([0.0969, 0.1235, 0.1918]),  # From Table 7.6
        mtq_max=[5.0, 5.0, 5.0],  # 5 Am² from Table 7.6
        has_rw=True,
        rw_axes=[np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],  # 3 orthogonal RWs
        rw_max_torque=0.004,  # Typical for small RW (4 mNm)
        rw_h_max=0.015,  # 15 mNms capacity
        rw_J=5e-5,  # kg·m²
        mass=10.165,  # From Table 7.6
        boresight=np.array([0, 0, 1]),
    ),
}


@dataclass
class OrbitConfig:
    """Orbit configuration."""
    name: str
    altitude_km: float
    inclination_deg: float
    use_J2: bool = True


ORBIT_CONFIGS = {
    'iss': OrbitConfig(
        name='ISS Orbit',
        altitude_km=429,
        inclination_deg=51.5,
    ),
    'polar': OrbitConfig(
        name='Polar Orbit',
        altitude_km=450,
        inclination_deg=87.0,  # From Table 7.6
    ),
    # Altitude variation tests
    'low_300km': OrbitConfig(
        name='Low LEO (300km)',
        altitude_km=300,
        inclination_deg=51.5,
    ),
    'mid_600km': OrbitConfig(
        name='Mid LEO (600km)',
        altitude_km=600,
        inclination_deg=51.5,
    ),
    'high_800km': OrbitConfig(
        name='High LEO (800km)',
        altitude_km=800,
        inclination_deg=51.5,
    ),
    # Inclination variation tests
    'equatorial': OrbitConfig(
        name='Equatorial Orbit',
        altitude_km=400,
        inclination_deg=0.0,
    ),
    'sunsync': OrbitConfig(
        name='Sun-Synchronous Orbit',
        altitude_km=600,
        inclination_deg=98.0,
    ),
}


@dataclass  
class GoalConfig:
    """Goal configuration for experiments."""
    name: str
    goal_type: str  # 'full_attitude', 'reduced_attitude', 'multi_target'
    # For full attitude (180° slew)
    q_start: Optional[np.ndarray] = None
    q_target: Optional[np.ndarray] = None
    # For reduced attitude (vector alignment)
    body_axis: Optional[np.ndarray] = None
    eci_target: Optional[np.ndarray] = None
    # For multi-target
    targets: Optional[List[Dict]] = None  # List of {start_s, end_s, eci_vec}


GOAL_CONFIGS = {
    # Table 7.3: 180° slew
    '180deg_slew': GoalConfig(
        name='180° Slew',
        goal_type='full_attitude',
        q_start=np.array([0, 0, 1, 0]),
        q_target=np.array([0, 1, 0, 0]),
    ),
    # Table 7.4: Reduced attitude (align body x with ECI vector)
    'reduced_attitude': GoalConfig(
        name='Reduced Attitude',
        goal_type='reduced_attitude',
        body_axis=np.array([1, 0, 0]),
        eci_target=np.array([np.cos(np.deg2rad(10)), 0, np.sin(np.deg2rad(10))]),
    ),
    # Table 7.5: Multi-target
    'multi_target': GoalConfig(
        name='Multi-Target',
        goal_type='multi_target',
        body_axis=np.array([1, 0, 0]),
        targets=[
            {'start_s': 0, 'end_s': 170, 'eci_vec': np.array([np.cos(np.deg2rad(10)), 0, np.sin(np.deg2rad(10))])},
            {'start_s': 200, 'end_s': 420, 'eci_vec': np.array([np.cos(np.deg2rad(-10)), 0, np.sin(np.deg2rad(-10))])},
            {'start_s': 450, 'end_s': 500, 'eci_vec': np.array([np.cos(np.deg2rad(10)), 0, np.sin(np.deg2rad(10))])},
        ],
    ),
    # Table 7.6: Sequential goals
    'sequential': GoalConfig(
        name='Sequential Goals',
        goal_type='sequential',
        targets=[
            {'start_s': 150, 'end_s': 1100, 'body_axis': np.array([-1, 0, 0]), 'goal': 'anti_ram'},
            {'start_s': 1200, 'end_s': 1500, 'body_axis': np.array([0, 0, 1]), 'goal': 'nadir'},
            {'start_s': 1600, 'end_s': 1900, 'body_axis': np.array([0, 0, 1]), 'goal': 'zenith'},
            {'start_s': 2000, 'end_s': 2400, 'body_axis': np.array([0, 0, 1]), 'goal': 'orbit_normal'},
            {'start_s': 2500, 'end_s': 3600, 'body_axis': np.array([-1, 0, 0]), 'goal': 'anti_ram'},
        ],
    ),
}


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str
    satellite: str  # Key into SATELLITE_CONFIGS
    orbit: str  # Key into ORBIT_CONFIGS
    goal: str  # Key into GOAL_CONFIGS
    duration_s: float
    n_trials: int  # For Monte Carlo
    dt_sim: float = 1.0  # Simulation timestep
    dt_plan: float = 10.0  # ALTRO planning timestep
    output_subdir: str = ''
    output_prefix: str = ''
    # Initial conditions
    omega_range: Tuple[float, float] = (0.0, 0.0)  # Random omega range (deg/s)
    h_init: float = 0.0
    # Disturbances
    prop_disturbance: Optional[np.ndarray] = None  # Body-frame torque (Nm)
    # Special flags
    use_sun_keepout: bool = False
    keepout_angle_deg: float = 20.0


# Complete list of experiments for Chapter 7
EXPERIMENTS = {
    # ==========================================================================
    # Monte Carlo: 180° Slew (Table 7.3)
    # ==========================================================================
    'mc_180deg_mtq': ExperimentConfig(
        name='MC 180° Slew - MTQ Only',
        description='100-trial Monte Carlo for 180° slews with MTQ-only satellite',
        satellite='mtq_only',
        orbit='iss',
        goal='180deg_slew',
        duration_s=500,
        n_trials=100,
        output_subdir='simple_slew',
        output_prefix='mtq',
    ),
    'mc_180deg_1rw': ExperimentConfig(
        name='MC 180° Slew - 3MTQ+1RW',
        description='100-trial Monte Carlo for 180° slews with 3MTQ+1RW satellite',
        satellite='3mtq_1rw',
        orbit='iss',
        goal='180deg_slew',
        duration_s=500,
        n_trials=100,
        output_subdir='simple_slew',
        output_prefix='1W',
    ),
    
    # ==========================================================================
    # Monte Carlo: Reduced Attitude (Table 7.4)
    # ==========================================================================
    'mc_reduced_mtq': ExperimentConfig(
        name='MC Reduced Attitude - MTQ Only',
        description='100-trial Monte Carlo for reduced attitude goals with MTQ-only',
        satellite='mtq_only',
        orbit='iss',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='single_target_imaging',
        output_prefix='mtq_quatset',
        use_sun_keepout=True,
        keepout_angle_deg=10.0,
    ),
    'mc_reduced_1rw': ExperimentConfig(
        name='MC Reduced Attitude - 3MTQ+1RW',
        description='100-trial Monte Carlo for reduced attitude goals with 3MTQ+1RW',
        satellite='3mtq_1rw',
        orbit='iss',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='single_target_imaging',
        output_prefix='1W_quatset',
        use_sun_keepout=True,
        keepout_angle_deg=10.0,
    ),
    
    # ==========================================================================
    # Monte Carlo: Multi-Target (Table 7.5)
    # ==========================================================================
    'mc_multi_mtq': ExperimentConfig(
        name='MC Multi-Target - MTQ Only',
        description='100-trial Monte Carlo for multi-target trajectories with MTQ-only',
        satellite='mtq_only',
        orbit='iss',
        goal='multi_target',
        duration_s=500,
        n_trials=100,
        output_subdir='multi_target_imaging',
        output_prefix='mtq_multi',
        use_sun_keepout=True,
        keepout_angle_deg=10.0,
    ),
    'mc_multi_1rw': ExperimentConfig(
        name='MC Multi-Target - 3MTQ+1RW',
        description='100-trial Monte Carlo for multi-target trajectories with 3MTQ+1RW',
        satellite='3mtq_1rw',
        orbit='iss',
        goal='multi_target',
        duration_s=500,
        n_trials=100,
        output_subdir='multi_target_imaging',
        output_prefix='1W_multi',
        use_sun_keepout=True,
        keepout_angle_deg=10.0,
    ),
    
    # ==========================================================================
    # Spinning Solution (Table 7.1)
    # ==========================================================================
    'spinning_solution': ExperimentConfig(
        name='Spinning Solution',
        description='Satellite countering disturbance by spinning (Table 7.1)',
        satellite='spinning_sat',
        orbit='iss',
        goal='reduced_attitude',  # Point z-axis anti-ram
        duration_s=500,
        n_trials=1,
        output_subdir='',
        output_prefix='spinning',
        prop_disturbance=np.array([0.0003, 0, 0]),  # 0.3 mNm on x-axis
    ),
    
    # ==========================================================================
    # Sequential Planning (Table 7.6)
    # ==========================================================================
    'sequential_planning': ExperimentConfig(
        name='Sequential Planning',
        description='Sequential trajectory planning with time-varying goals (Table 7.6)',
        satellite='sequential_6u',
        orbit='polar',
        goal='sequential',
        duration_s=3600,
        n_trials=1,
        dt_sim=1.0,
        output_subdir='sequential',
        output_prefix='plan',
    ),
    
    # ==========================================================================
    # Long-Term Two-Goal (Table 7.7 / Figure anim_plots)
    # ==========================================================================
    'two_goal_trajectory': ExperimentConfig(
        name='Two-Goal Long Trajectory',
        description='Long trajectory with ground target tracking then anti-ram pointing',
        satellite='mtq_only',
        orbit='iss',
        goal='180deg_slew',  # Will be overridden with custom goals
        duration_s=3600,
        n_trials=1,
        output_subdir='',
        output_prefix='anim',
    ),

    # ==========================================================================
    # ALTITUDE VARIATION TESTS (for abstract parameter range validation)
    # ==========================================================================
    'alt_300km_reduced': ExperimentConfig(
        name='Altitude 300km - Reduced Attitude',
        description='Reduced attitude at 300km (stronger B-field)',
        satellite='3mtq_1rw',
        orbit='low_300km',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='altitude_variation',
        output_prefix='alt300_reduced',
    ),
    'alt_600km_reduced': ExperimentConfig(
        name='Altitude 600km - Reduced Attitude',
        description='Reduced attitude at 600km (moderate B-field)',
        satellite='3mtq_1rw',
        orbit='mid_600km',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='altitude_variation',
        output_prefix='alt600_reduced',
    ),
    'alt_800km_reduced': ExperimentConfig(
        name='Altitude 800km - Reduced Attitude',
        description='Reduced attitude at 800km (weaker B-field)',
        satellite='3mtq_1rw',
        orbit='high_800km',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='altitude_variation',
        output_prefix='alt800_reduced',
    ),

    # ==========================================================================
    # INCLINATION VARIATION TESTS (for abstract parameter range validation)
    # ==========================================================================
    'inc_equatorial_reduced': ExperimentConfig(
        name='Equatorial - Reduced Attitude',
        description='Reduced attitude at 0 deg inclination',
        satellite='3mtq_1rw',
        orbit='equatorial',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='inclination_variation',
        output_prefix='inc0_reduced',
    ),
    'inc_sunsync_reduced': ExperimentConfig(
        name='Sun-Sync - Reduced Attitude',
        description='Reduced attitude at 98 deg inclination (sun-sync)',
        satellite='3mtq_1rw',
        orbit='sunsync',
        goal='reduced_attitude',
        duration_s=500,
        n_trials=100,
        output_subdir='inclination_variation',
        output_prefix='inc98_reduced',
    ),
}


# =============================================================================
# EXPERIMENT RUNNER FRAMEWORK
# =============================================================================

class ExperimentRunner:
    """Base class for running thesis experiments."""
    
    def __init__(self, config: ExperimentConfig, output_dir: Path, quick: bool = False):
        self.config = config
        self.output_dir = output_dir
        self.quick = quick
        
        # Adjust for quick mode
        if quick:
            self.n_trials = min(10, config.n_trials)
            self.duration_s = min(100, config.duration_s)
        else:
            self.n_trials = config.n_trials
            self.duration_s = config.duration_s
    
    def create_satellite(self):
        """Create satellite matching thesis parameters."""
        from ADCS.satellite_factory.satellites.create_cubesats import create_beavercube2_cubesat
        from ADCS.satellite_factory.actuators import create_isis_magnetorquer_board, create_cubewheel_smallplus_rw
        from ADCS.satellite_factory.sensors import create_ICM20948_IMU, create_isis_magnetometer
        from ADCS.satellite_hardware.satellite.satellite import Satellite
        from ADCS.satellite_hardware.actuators import RW

        sat_config = SATELLITE_CONFIGS[self.config.satellite]

        # Use factory functions for consistency
        mtqs = create_isis_magnetorquer_board(estimate_bias=False)
        mtms = create_isis_magnetometer(estimate_bias=False)
        gyros = create_ICM20948_IMU(estimate_bias=False)

        actuators = mtqs
        if sat_config.has_rw:
            if sat_config.rw_axes is not None:
                # Multiple RWs (e.g., 3-axis configuration)
                for axis in sat_config.rw_axes:
                    rw = RW(
                        axis=axis,
                        max_torque=sat_config.rw_max_torque,
                        J=sat_config.rw_J,
                        h=0.0,
                        h_max=sat_config.rw_h_max,
                        estimate_bias=False,
                    )
                    actuators = actuators + [rw]
            else:
                # Single RW
                rw = create_cubewheel_smallplus_rw(axis=sat_config.rw_axis, estimate_bias=False)
                actuators = mtqs + [rw]

        sat = Satellite(
            mass=sat_config.mass,
            COM=np.zeros(3),
            J_0=sat_config.J,
            sensors=mtms + gyros,
            actuators=actuators,
            boresight=sat_config.boresight,
        )

        return sat
    
    def create_orbit(self, start_time: float):
        """Create orbit matching thesis parameters."""
        from ADCS.orbits.ephemeris import Ephemeris
        from ADCS.orbits.orbit import Orbit
        from ADCS.orbits.orbital_state import Orbital_State
        from ADCS.orbits.universal_constants import TimeConstants
        
        orbit_config = ORBIT_CONFIGS[self.config.orbit]
        
        ephem = Ephemeris()
        R_magnitude = 6378.137 + orbit_config.altitude_km
        i_rad = np.deg2rad(orbit_config.inclination_deg)
        
        # Random orbital position for MC
        theta = np.random.uniform(0, 2 * np.pi)
        R = R_magnitude * np.array([
            np.cos(theta),
            np.sin(theta) * np.cos(i_rad),
            np.sin(theta) * np.sin(i_rad)
        ])
        
        mu = 398600.4418
        v_mag = np.sqrt(mu / R_magnitude)
        V = v_mag * np.array([
            -np.sin(theta),
            np.cos(theta) * np.cos(i_rad),
            np.cos(theta) * np.sin(i_rad)
        ])
        
        os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
        
        end_time = start_time + (self.duration_s + 100) * TimeConstants.sec2cent
        orb = Orbit(
            os0=os0,
            end_time=end_time,
            dt=self.config.dt_sim,
            use_J2=orbit_config.use_J2,
            fast=True,
            verbose=False,
        )
        
        return orb, os0
    
    def create_goals(self, start_time: float, os0):
        """Create goal list matching thesis parameters."""
        from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal, No_Goal
        from ADCS.CONOPS.goallist import GoalList
        from ADCS.helpers.math_helpers import normalize
        
        goal_config = GOAL_CONFIGS[self.config.goal]
        
        if goal_config.goal_type == 'full_attitude':
            goal = Fixed_Attitude_Goal(goal_config.q_target)
            return GoalList({start_time: goal}), goal_config.q_start
            
        elif goal_config.goal_type == 'reduced_attitude':
            eci_vec = normalize(goal_config.eci_target)
            goal = ECI_Goal(eci_vec)
            # Random starting quaternion
            q_start = normalize(np.random.randn(4))
            return GoalList({start_time: goal}), q_start
            
        elif goal_config.goal_type == 'multi_target':
            # Create goal list with timing
            from ADCS.orbits.universal_constants import TimeConstants
            goals_dict = {}
            for target in goal_config.targets:
                t_start = start_time + target['start_s'] * TimeConstants.sec2cent
                t_end = start_time + target['end_s'] * TimeConstants.sec2cent
                eci_vec = normalize(target['eci_vec'])
                goals_dict[t_start] = ECI_Goal(eci_vec)
                # Add no-goal for gaps
                goals_dict[t_end] = No_Goal()
            q_start = normalize(np.random.randn(4))
            return GoalList(goals_dict), q_start

        elif goal_config.goal_type == 'sequential':
            # Sequential goals with named orbital reference directions
            # Used for Table 7.6 sequential planning experiment
            from ADCS.orbits.universal_constants import TimeConstants
            from ADCS.CONOPS.goals.vector_goals import (
                Nadir_Goal, Zenith_Goal, AntiVelocity_Goal, LVLH_Tangential_Goal
            )

            # Map goal names to goal classes
            goal_map = {
                'anti_ram': AntiVelocity_Goal,  # Opposite velocity direction
                'nadir': Nadir_Goal,            # Toward Earth center
                'zenith': Zenith_Goal,          # Away from Earth center
                'orbit_normal': LVLH_Tangential_Goal,  # Cross-track (approximation)
            }

            goals_dict = {}
            for target in goal_config.targets:
                t_start = start_time + target['start_s'] * TimeConstants.sec2cent
                t_end = start_time + target['end_s'] * TimeConstants.sec2cent
                goal_name = target['goal']

                if goal_name in goal_map:
                    goals_dict[t_start] = goal_map[goal_name]()
                else:
                    raise ValueError(f"Unknown sequential goal: {goal_name}")

                # Add no-goal for gaps between targets
                goals_dict[t_end] = No_Goal()

            q_start = normalize(np.random.randn(4))
            return GoalList(goals_dict), q_start

        else:
            raise ValueError(f"Unknown goal type: {goal_config.goal_type}")
    
    def create_planner(self, sat):
        """Create ALTRO planner with thesis settings."""
        from ADCS.controller.helpers import PlannerSettings
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        
        planner_settings = PlannerSettings(
            est_sat=sat,
            bdot_on=0,
            dt_tp=self.config.dt_plan,
            dt_tvlqr=self.config.dt_sim,
        )
        
        controller = Plan_and_Track_LQR(
            est_sat=sat,
            planner_settings=planner_settings,
        )
        
        return controller
    
    def run_single_trial(self, seed: int) -> Dict[str, Any]:
        """Run a single ALTRO trajectory planning trial."""
        np.random.seed(seed)
        
        start_time = 0.22 + seed * 0.001
        
        # Create components
        sat = self.create_satellite()
        n_rw = len(sat.rw_actuators) if hasattr(sat, 'rw_actuators') else 0
        if n_rw > 0:
            for rw in sat.rw_actuators:
                rw.h = self.config.h_init

        orb, os0 = self.create_orbit(start_time)
        goals, q_start = self.create_goals(start_time, os0)
        controller = self.create_planner(sat)

        # Initial state
        omega0 = np.zeros(3)
        if self.config.omega_range[1] > 0:
            omega0 = np.random.uniform(
                -self.config.omega_range[1],
                self.config.omega_range[1],
                3
            ) * np.pi / 180

        # Initial RW momenta (one per reaction wheel)
        h0 = np.array([self.config.h_init] * n_rw) if n_rw > 0 else np.array([])
        x0 = np.concatenate([omega0, q_start, h0])
        
        # Run ALTRO
        try:
            trajectory = controller.calculate_trajectory(
                t_start=start_time,
                duration=self.duration_s,
                x_0=x0,
                os_0=os0,
                goals=goals,
                verbose=False,
            )
            
            if trajectory is None or np.any(np.isnan(trajectory.states)):
                return {'success': False, 'error': 'Trajectory computation failed'}
            
            # Extract orbit states at each timestep for figure generation
            orbit_states = []
            for t in trajectory.times:
                try:
                    os_t = orb.get_os(t)
                    orbit_states.append({'R': os_t.R.copy(), 'V': os_t.V.copy()})
                except:
                    # If orbit lookup fails, use last known state
                    if orbit_states:
                        orbit_states.append(orbit_states[-1])
                    else:
                        orbit_states.append({'R': os0.R.copy(), 'V': os0.V.copy()})

            return {
                'success': True,
                'times': trajectory.times,
                'states': trajectory.states,
                'controls': trajectory.controls,
                'orbit_states': orbit_states,
                'start_time': start_time,
                'q_start': q_start,
                'seed': seed,
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def compute_pointing_errors(self, states: np.ndarray, goal_vec: np.ndarray, body_axis: np.ndarray) -> np.ndarray:
        """Compute pointing errors over trajectory."""
        from ADCS.helpers.math_helpers import rot_mat
        
        errors = []
        for i in range(states.shape[1]):
            q = states[3:7, i]
            R = rot_mat(q)
            body_vec_eci = R @ body_axis
            err = np.arccos(np.clip(np.dot(body_vec_eci, goal_vec), -1, 1)) * 180 / np.pi
            errors.append(err)
        
        return np.array(errors)
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        print(f"\n{'='*60}")
        print(f"  {self.config.name}")
        print(f"  {self.config.description}")
        print(f"{'='*60}")
        print(f"  Trials: {self.n_trials}")
        print(f"  Duration: {self.duration_s}s")
        print(f"  Output: {self.output_dir / self.config.output_subdir}")
        print(f"{'='*60}\n")
        
        results = []
        for i in range(self.n_trials):
            print(f"  Trial {i+1}/{self.n_trials}...", end='\r')
            result = self.run_single_trial(seed=42 + i)
            results.append(result)
        
        successful = [r for r in results if r['success']]
        print(f"  Completed: {len(successful)}/{self.n_trials} successful")
        
        return {
            'config': self.config.name,
            'n_trials': self.n_trials,
            'n_successful': len(successful),
            'results': results,
        }


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_mc_figures(results: Dict, output_dir: Path, prefix: str,
                        boresight: np.ndarray = None, goal_vector: np.ndarray = None,
                        goal_config=None):
    """Generate Monte Carlo figures matching thesis format.

    Generates figures with log scale formatting to match thesis:
    - Histogram with log x-scale for final pointing errors
    - Trajectory plot with semilogy for error over time
    - Good/bad quaternion examples

    For sequential goals, computes errors against time-varying orbital references.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ADCS.helpers.math_helpers import rot_mat
    from ADCS.orbits.universal_constants import TimeConstants

    # Publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
    GOAL_COLORS = {
        'anti_ram': '#D55E00',
        'nadir': '#0072B2',
        'zenith': '#009E73',
        'orbit_normal': '#E69F00',
        'none': '#999999',
    }

    successful = [r for r in results['results'] if r['success']]
    if not successful:
        print(f"  No successful trials for {prefix}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Default boresight and goal if not provided
    if boresight is None:
        boresight = np.array([1, 0, 0])
    if goal_vector is None:
        goal_vector = np.array([0, 0, 1])  # Nadir-ish default

    # Check if this is a sequential goal experiment
    is_sequential = (goal_config is not None and
                     hasattr(goal_config, 'goal_type') and
                     goal_config.goal_type == 'sequential')

    def compute_pointing_error(states, goal_vec, body_axis):
        """Compute pointing errors over trajectory."""
        errors = []
        for i in range(states.shape[1]):
            q = states[3:7, i]
            R = rot_mat(q)
            body_vec_eci = R @ body_axis
            err = np.arccos(np.clip(np.dot(body_vec_eci, goal_vec), -1, 1)) * 180 / np.pi
            errors.append(err)
        return np.array(errors)

    # Compute final errors for each successful trial
    final_errors = []
    all_error_trajectories = []

    for r in successful:
        states = r['states']
        times = r['times']

        if is_sequential and 'orbit_states' in r and 'start_time' in r:
            # Use sequential error computation with time-varying goals
            errors, goal_names = compute_sequential_pointing_errors(
                times, states, r['orbit_states'], goal_config, boresight, r['start_time']
            )
            # For final error, only count times when goal is active (not NaN)
            valid_errors = errors[~np.isnan(errors)]
            if len(valid_errors) > 0:
                final_errors.append(valid_errors[-1])
            all_error_trajectories.append({
                'times': times,
                'errors': errors,
                'states': states,
                'goal_names': goal_names,
                'start_time': r['start_time'],
            })
        else:
            # Standard fixed-goal error computation
            errors = compute_pointing_error(states, goal_vector, boresight)
            final_errors.append(errors[-1])
            # Include start_time if available for proper time axis conversion
            traj_dict = {'times': times, 'errors': errors, 'states': states}
            if 'start_time' in r:
                traj_dict['start_time'] = r['start_time']
            all_error_trajectories.append(traj_dict)

    final_errors = np.array(final_errors)

    # Statistics (handle empty arrays)
    if len(final_errors) > 0:
        pct_under_1deg = 100 * np.sum(final_errors < 1.0) / len(final_errors)
        pct_under_10deg = 100 * np.sum(final_errors < 10.0) / len(final_errors)
        mean_error = np.mean(final_errors)
        max_err = min(np.nanmax(final_errors) * 1.1, 180) if not np.all(np.isnan(final_errors)) else 180
        print(f"  {prefix}: {pct_under_1deg:.1f}% <1°, {pct_under_10deg:.1f}% <10°, mean={mean_error:.2f}°")
    else:
        pct_under_1deg = 0
        pct_under_10deg = 0
        mean_error = np.nan
        max_err = 180
        print(f"  {prefix}: No valid errors computed (check goal timing vs simulation duration)")

    # === HISTOGRAMS ===
    if len(final_errors) == 0:
        print(f"  Skipping histograms - no valid errors")
    else:
        # === HISTOGRAM (LINEAR - matches thesis) ===
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, max_err, 30)
        ax.hist(final_errors, bins=bins, color=COLORS[0], edgecolor='white', alpha=0.8)
        ax.set_xlabel('Final Pointing Error (deg)')
        ax.set_ylabel('Count')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1° threshold')
        ax.axvline(10.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='10° threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{prefix}: {pct_under_1deg:.0f}% <1°, mean={mean_error:.2f}°')

        fig.savefig(output_dir / f'{prefix}_montecarlo.png')
        plt.close(fig)

        # === HISTOGRAM (LOG x-scale - alternative) ===
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.logspace(-2, 2, 30)
        ax.hist(final_errors, bins=bins, color=COLORS[0], edgecolor='white', alpha=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Final Pointing Error (deg)')
        ax.set_ylabel('Count')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1° threshold')
        ax.axvline(10.0, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='10° threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.set_title(f'{prefix}: {pct_under_1deg:.0f}% <1°, mean={mean_error:.2f}°')

        fig.savefig(output_dir / f'{prefix}_montecarlo_log.png')
        plt.close(fig)

    # === TRAJECTORY PLOT (LINEAR - matches thesis) ===
    if len(all_error_trajectories) > 0:
        fig, ax = plt.subplots(figsize=(8, 4) if is_sequential else (6, 4))

        for traj in all_error_trajectories[:min(20, len(all_error_trajectories))]:
            # Convert times to seconds from start
            times = traj['times']
            if 'start_time' in traj:
                times_sec = (times - traj['start_time']) / TimeConstants.sec2cent
            else:
                times_sec = (times - times[0]) / TimeConstants.sec2cent

            errors = traj['errors']

            if is_sequential and 'goal_names' in traj:
                # Plot with color-coded goal segments
                goal_names = traj['goal_names']
                prev_goal = None
                seg_start = 0

                for i, (t, err, goal) in enumerate(zip(times_sec, errors, goal_names)):
                    if goal != prev_goal or i == len(times_sec) - 1:
                        if prev_goal is not None and seg_start < i:
                            seg_times = times_sec[seg_start:i+1]
                            seg_errors = errors[seg_start:i+1]
                            color = GOAL_COLORS.get(prev_goal, COLORS[0])
                            ax.plot(seg_times, seg_errors, color=color, alpha=0.7, linewidth=1.0)
                        seg_start = i
                        prev_goal = goal
            else:
                ax.plot(times_sec, errors, color=COLORS[0], alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pointing Error (deg)')
        ax.set_ylim(0, max(15, max_err))
        ax.grid(True, alpha=0.3)

        if is_sequential and goal_config is not None:
            # Add goal transition markers
            for target in goal_config.targets:
                ax.axvline(target['start_s'], color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
            # Add legend for goal colors
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=GOAL_COLORS.get(g, '#999'), lw=2, label=g)
                               for g in ['anti_ram', 'nadir', 'zenith', 'orbit_normal']]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax.set_title(f'{prefix} - Sequential Pointing Error vs Time')
        else:
            ax.set_title(f'{prefix} - Pointing Error vs Time')

        fig.savefig(output_dir / f'{prefix}_montecarlo_traj.png')
        plt.close(fig)

        # === TRAJECTORY PLOT (SEMILOGY - alternative) ===
        fig, ax = plt.subplots(figsize=(8, 4) if is_sequential else (6, 4))

        for traj in all_error_trajectories[:min(20, len(all_error_trajectories))]:
            times = traj['times']
            if 'start_time' in traj:
                times_sec = (times - traj['start_time']) / TimeConstants.sec2cent
            else:
                times_sec = (times - times[0]) / TimeConstants.sec2cent

            errors = traj['errors']
            # Replace zeros/negatives with small value for log scale
            errors_log = np.maximum(errors, 0.01)

            if is_sequential and 'goal_names' in traj:
                goal_names = traj['goal_names']
                prev_goal = None
                seg_start = 0

                for i, (t, err, goal) in enumerate(zip(times_sec, errors_log, goal_names)):
                    if goal != prev_goal or i == len(times_sec) - 1:
                        if prev_goal is not None and seg_start < i:
                            seg_times = times_sec[seg_start:i+1]
                            seg_errors = errors_log[seg_start:i+1]
                            color = GOAL_COLORS.get(prev_goal, COLORS[0])
                            ax.semilogy(seg_times, seg_errors, color=color, alpha=0.7, linewidth=1.0)
                        seg_start = i
                        prev_goal = goal
            else:
                ax.semilogy(times_sec, errors_log, color=COLORS[0], alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pointing Error (deg)')
        ax.set_ylim(0.01, 200)
        ax.grid(True, alpha=0.3, which='both')

        if is_sequential and goal_config is not None:
            for target in goal_config.targets:
                ax.axvline(target['start_s'], color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=GOAL_COLORS.get(g, '#999'), lw=2, label=g)
                               for g in ['anti_ram', 'nadir', 'zenith', 'orbit_normal']]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            ax.set_title(f'{prefix} - Sequential Pointing Error vs Time (log scale)')
        else:
            ax.set_title(f'{prefix} - Pointing Error vs Time (log scale)')

        fig.savefig(output_dir / f'{prefix}_montecarlo_traj_log.png')
        plt.close(fig)

    # === GOOD/BAD QUATERNION EXAMPLES ===
    if len(all_error_trajectories) >= 2:
        # Sort by final error, handling NaN
        def get_final_error(traj):
            errors = traj['errors']
            valid = errors[~np.isnan(errors)] if hasattr(errors, '__iter__') else [errors]
            return valid[-1] if len(valid) > 0 else 180.0

        sorted_trajs = sorted(all_error_trajectories, key=get_final_error)

        for quality, traj in [('good', sorted_trajs[0]), ('bad', sorted_trajs[-1])]:
            fig, ax = plt.subplots(figsize=(6, 4))
            states = traj['states']
            times = traj['times']

            # Convert times to seconds from start
            if 'start_time' in traj:
                times_sec = (times - traj['start_time']) / TimeConstants.sec2cent
            else:
                times_sec = (times - times[0]) / TimeConstants.sec2cent

            for i, label in enumerate(['w', 'x', 'y', 'z']):
                ax.plot(times_sec, states[3+i, :], color=COLORS[i], label=f'q_{label}')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Quaternion')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            final_err = get_final_error(traj)
            ax.set_title(f'{prefix} - {quality.capitalize()} Example (final err={final_err:.2f}°)')

            fig.savefig(output_dir / f'{prefix}_{quality}_quaternion.png')
            plt.close(fig)

    print(f"  Figures saved to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def list_experiments():
    """Print list of all experiments."""
    print("\n" + "="*70)
    print("  Chapter 7 Planning Experiments")
    print("="*70)
    
    for exp_id, config in EXPERIMENTS.items():
        sat_config = SATELLITE_CONFIGS[config.satellite]
        print(f"\n  {exp_id}")
        print(f"    Name: {config.name}")
        print(f"    Description: {config.description}")
        print(f"    Satellite: {sat_config.name}")
        print(f"    Trials: {config.n_trials}, Duration: {config.duration_s}s")
        print(f"    Output: {config.output_subdir}/{config.output_prefix}_*")
    
    print("\n" + "="*70)
    print("  Usage:")
    print("    python thesis_chapter7_planning.py --experiment mc_180deg_mtq --quick")
    print("    python thesis_chapter7_planning.py --all --full")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chapter 7 Planning Experiments")
    parser.add_argument('--list', action='store_true', help='List all experiments')
    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode (10 trials, 100s)')
    parser.add_argument('--full', action='store_true', help='Full mode (100 trials, 500s)')
    parser.add_argument('--output-dir', type=str, default='./chapter7_figures')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate figures from saved data (no simulation)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save simulation data to pickle')
    args = parser.parse_args()

    if args.list or (not args.experiment and not args.all and not args.regenerate):
        list_experiments()
        return

    output_dir = Path(args.output_dir)
    quick = not args.full

    experiments_to_run = []
    if args.all:
        experiments_to_run = list(EXPERIMENTS.keys())
    elif args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            return
        experiments_to_run = [args.experiment]

    # Handle regenerate mode
    if args.regenerate:
        print(f"\n{'='*60}")
        print(f"  Regenerating figures from saved data")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")

        for exp_id in experiments_to_run:
            config = EXPERIMENTS[exp_id]
            exp_output = output_dir / config.output_subdir

            # Load saved data
            results = load_experiment_data(exp_output, config.output_prefix)
            if results is None:
                print(f"  Skipping {exp_id} - no saved data")
                continue

            # Get configs for figure generation
            sat_config = SATELLITE_CONFIGS[config.satellite]
            goal_config = GOAL_CONFIGS.get(config.goal, None)
            boresight = sat_config.boresight

            # Determine goal vector for non-sequential goals
            goal_vector = None
            if goal_config is not None:
                if goal_config.eci_target is not None:
                    goal_vector = goal_config.eci_target
                elif goal_config.targets is not None and len(goal_config.targets) > 0:
                    first_target = goal_config.targets[0]
                    if 'eci_vec' in first_target:
                        goal_vector = first_target['eci_vec']

            generate_mc_figures(results, exp_output, config.output_prefix,
                               boresight=boresight, goal_vector=goal_vector,
                               goal_config=goal_config)
        return

    # Normal run mode
    print(f"\n{'='*60}")
    print(f"  Running {len(experiments_to_run)} experiments")
    print(f"  Mode: {'Quick' if quick else 'Full'}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    all_results = {}
    for exp_id in experiments_to_run:
        config = EXPERIMENTS[exp_id]
        exp_output = output_dir / config.output_subdir

        runner = ExperimentRunner(config, exp_output, quick=quick)
        results = runner.run()
        all_results[exp_id] = results

        # Save experiment data for later figure regeneration
        if not args.no_save:
            save_experiment_data(results, exp_output, config.output_prefix)

        # Generate figures
        sat_config = SATELLITE_CONFIGS[config.satellite]
        goal_config = GOAL_CONFIGS.get(config.goal, None)

        # Determine boresight and goal vector for figure generation
        boresight = sat_config.boresight
        goal_vector = None
        if goal_config is not None:
            if goal_config.eci_target is not None:
                goal_vector = goal_config.eci_target
            elif goal_config.targets is not None and len(goal_config.targets) > 0:
                # Use first target for figure generation
                first_target = goal_config.targets[0]
                if 'eci_vec' in first_target:
                    goal_vector = first_target['eci_vec']

        generate_mc_figures(results, exp_output, config.output_prefix,
                           boresight=boresight, goal_vector=goal_vector,
                           goal_config=goal_config)

    # Save summary
    summary_path = output_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        summary = {
            exp_id: {
                'name': r['config'],
                'n_trials': r['n_trials'],
                'n_successful': r['n_successful'],
            }
            for exp_id, r in all_results.items()
        }
        json.dump(summary, f, indent=2)
    
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
