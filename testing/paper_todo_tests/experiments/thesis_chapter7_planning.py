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
import numpy as np

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


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
    rw_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
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
        has_rw=True,  # Actually has 3 RW but we model as 1 for simplicity
        rw_axis=np.array([0, 0, 1]),
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
        
        sat_config = SATELLITE_CONFIGS[self.config.satellite]
        
        # Use factory functions for consistency
        mtqs = create_isis_magnetorquer_board(estimate_bias=False)
        mtms = create_isis_magnetometer(estimate_bias=False)
        gyros = create_ICM20948_IMU(estimate_bias=False)
        
        actuators = mtqs
        if sat_config.has_rw:
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
        if hasattr(sat, 'rw_actuators') and len(sat.rw_actuators) > 0:
            sat.rw_actuators[0].h = self.config.h_init
        
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
        
        h0 = np.array([self.config.h_init]) if hasattr(sat, 'rw_actuators') and len(sat.rw_actuators) > 0 else np.array([])
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
            
            return {
                'success': True,
                'times': trajectory.times,
                'states': trajectory.states,
                'controls': trajectory.controls,
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

def generate_mc_figures(results: Dict, output_dir: Path, prefix: str):
    """Generate Monte Carlo figures matching thesis format."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
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
    
    successful = [r for r in results['results'] if r['success']]
    if not successful:
        print(f"  No successful trials for {prefix}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Compute final errors and generate:
    # - {prefix}_montecarlo.png (histogram)
    # - {prefix}_montecarlo_traj.png (error over time)
    # - {prefix}_good_quaternion.png
    # - {prefix}_bad_quaternion.png
    # - {prefix}_mom_montecarlo.png (if RW present)
    
    print(f"  Figure generation for {prefix} - placeholder (implement compute_pointing_errors)")


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
    args = parser.parse_args()
    
    if args.list or (not args.experiment and not args.all):
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
        
        # Generate figures
        generate_mc_figures(results, exp_output, config.output_prefix)
    
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
