#!/usr/bin/env python3
"""
Thesis Figure Configurations
============================

This file contains the EXACT parameter values from the thesis for each test case.
Use these to ensure reproducibility of thesis figures.

Chapter 6 - Disturbance Control:
- Table 5.1: Wie PD controller (Space Shuttle-like)
- Table 5.2: Lovera MTQ-PD controller  
- Section 5.2.3: Wisniewski sliding mode

Chapter 7 - Planning:
- Table 7.1 (tab:plan_dist_test_details): Spinning solution
- Table 7.2 (tab:mc_sat_params): Monte Carlo satellite
- Table 7.6 (tab:seq_test_details): Sequential planning

ALTRO Planner Settings (from dissertation_code_temp/GeneralizedADS/ADCS/src/sat_ADCS_ADCS/ADCS_Bx.py):
- Cost weights are validated against the thesis code defaults
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# =============================================================================
# CHAPTER 6: DISTURBANCE CONTROL CONFIGURATIONS
# =============================================================================

@dataclass
class Ch6_Wie_Config:
    """
    Wie PD Controller - Table 5.1 (Space Shuttle-like)
    Section 6.3 in thesis
    """
    name: str = "Wie PD Controller"
    
    # Satellite properties
    J: np.ndarray = field(default_factory=lambda: np.diag([10000.0, 9000.0, 12000.0]))  # kg·m²
    mass: float = 10000.0  # kg
    
    # Controller gains (from thesis)
    Kp: float = 5.0       # Proportional gain (scalar, multiplied by I₃)
    Kd: float = 200.0     # Derivative gain
    
    # Actuators: "magic" thrusters (represented as high-authority RW)
    max_torque: float = 20.0  # Nm
    
    # Test parameters
    duration_s: float = 7200.0  # 2 hours
    dt: float = 1.0
    
    # Initial conditions
    omega_init: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.001]))  # rad/s
    q_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))


@dataclass  
class Ch6_Lovera_Config:
    """
    Lovera MTQ-PD Controller - Table 5.2
    Section 6.4 in thesis
    """
    name: str = "Lovera MTQ-PD Controller"
    
    # Satellite properties (from thesis Table 5.2)
    J: np.ndarray = field(default_factory=lambda: np.diag([27.0, 17.0, 25.0]))  # kg·m²
    mass: float = 100.0  # kg
    
    # MTQ properties
    mtq_max: float = 50.0  # Am² (from thesis)
    
    # Controller gains (from thesis)
    eps: float = 0.01    # ε parameter
    kp: float = 50.0     # Proportional gain
    kv: float = 50.0     # Velocity (derivative) gain
    
    # Test parameters  
    duration_s: float = 36000.0  # 10 hours (from thesis)
    dt: float = 1.0
    
    # Initial conditions (from thesis)
    omega_init: np.ndarray = field(default_factory=lambda: np.array([1, 1, -1]) * np.pi/180)  # 1 deg/s each
    q_init: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.2, 0.3, np.sqrt(1-0.01-0.04-0.09)]))


@dataclass
class Ch6_Wisniewski_Config:
    """
    Wisniewski Sliding Mode Controller - Section 5.2.3
    Parameters from thesis Table 'Wisniewski Comparison Details'
    
    Reference: Wisniewski (1998) test case recreation
    """
    name: str = "Wisniewski Sliding Mode"
    
    # Satellite properties from thesis Table
    J: np.ndarray = field(default_factory=lambda: np.diag([3.428, 2.904, 1.275]))  # kg·m²
    mass: float = 50.0  # kg (approximate, not critical for MTQ-only)
    
    # MTQ properties from thesis
    mtq_max: float = 20.0  # Am² (thesis: "Maximum Magnetic Moment: 20 Am²")
    
    # Controller gains from thesis
    lambda_q: float = 0.002   # Quaternion error gain (scalar for λ_q·I₃)
    lambda_s: float = 0.003   # Sliding surface gain (scalar for λ_s·I₃)
    
    # CubeSat variant gains (for comparison)
    lambda_q_cubesat: float = 0.0001
    lambda_s_cubesat: float = 0.0003
    
    # Test parameters from thesis
    duration_s: float = 36000.0  # 10 hours
    dt: float = 1.0  # "Update Period: 1 second"
    
    # Initial conditions from thesis:
    # "Body-frame Angular Velocity: [-0.002, 0.002, 0.002] rad/s"
    omega_init: np.ndarray = field(default_factory=lambda: np.array([-0.002, 0.002, 0.002]))
    
    # "Orientation: (roll, pitch, yaw) = (60°, 100°, -100°) from desired attitude"
    # Convert RPY to quaternion - this gives ~175° error from identity
    q_init: np.ndarray = field(default_factory=lambda: _rpy_to_quat(60, 100, -100))


def _rpy_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert roll-pitch-yaw (degrees) to quaternion [x, y, z, w]."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


# =============================================================================
# CHAPTER 7: PLANNING CONFIGURATIONS
# =============================================================================

@dataclass
class Ch7_Spinning_Config:
    """
    Planning Around Disturbance - Table 7.1 (tab:plan_dist_test_details)
    Section 7.5.2 (sec:spinning_satellite)
    
    3U CubeSat countering body-fixed disturbance by spinning
    """
    name: str = "Spinning Solution"
    
    # Satellite inertia (from thesis Table 7.1)
    J: np.ndarray = field(default_factory=lambda: np.array([
        [0.1, 0, 0.00013],
        [0, 0.05, -0.00021],
        [0.00013, -0.00021, 0.005]
    ]))
    mass: float = 4.0  # kg
    
    # MTQ properties (from thesis)
    mtq_max_x: float = 0.19   # Am² on x-axis
    mtq_max_yz: float = 0.57  # Am² on y and z axes
    
    # RW properties (single RW on y-axis)
    rw_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    rw_max_torque: float = 0.0002   # 0.2 mNm
    rw_max_momentum: float = 0.002  # 2 mNms
    rw_inertia: float = 2e-6        # kg·m²
    
    # Propulsion disturbance (body-fixed)
    disturbance_torque: np.ndarray = field(default_factory=lambda: np.array([0.0003, 0, 0]))  # 0.3 mNm on x
    
    # Orbit
    orbital_radius: float = 6800.0  # km
    inclination: float = 51.5       # degrees
    
    # Goal: point z-axis (propulsion) anti-ram
    goal_body_axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))  # +z
    goal_direction: str = "anti-ram"
    
    # Initial state (from thesis)
    omega_init: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # deg/s
    q_init: np.ndarray = field(default_factory=lambda: np.array([-0.232, -0.664, -0.234, -0.671]))
    h_init: float = 0.0  # Stored momentum
    
    # Test parameters
    duration_s: float = 500.0
    dt: float = 1.0
    dt_coarse: float = 10.0  # Coarse timestep for initial pass


@dataclass
class Ch7_MonteCarlo_Config:
    """
    Monte Carlo Satellite Properties - Table 7.2 (tab:mc_sat_params)
    Section 7.5.3 (sec:mc_planning)
    """
    name: str = "Monte Carlo Tests"
    
    # Satellite inertia (from thesis Table 7.2)
    J_xx: float = 0.005256   # kg·m²
    J_yy: float = 0.04939    # kg·m²
    J_zz: float = 0.04939    # kg·m²
    
    @property
    def J(self) -> np.ndarray:
        return np.diag([self.J_xx, self.J_yy, self.J_zz])
    
    mass: float = 4.0  # kg
    
    # MTQ properties
    mtq_max_x: float = 0.19  # Am²
    
    # RW properties (only for 3MTQ+1RW variant)
    rw_max_torque: float = 0.0002   # Nm
    rw_max_momentum: float = 0.002  # Nms
    rw_inertia: float = 2e-6        # kg·m²
    rw_axis: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    
    # Test case: 180° slew (from thesis)
    q_start_180: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1, 0]))  # [x,y,z,w]
    q_target_180: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0, 0]))
    
    # Test parameters
    n_trials_quick: int = 10
    n_trials_full: int = 100
    duration_quick: float = 100.0
    duration_full: float = 500.0


@dataclass
class Ch7_Sequential_Config:
    """
    Sequential Planning - Table 7.6 (tab:seq_test_details)
    Section 7.5.4 (subsec:sequential_trajectories)
    
    6U CubeSat with 3RW (ASTERIA-based)
    """
    name: str = "Sequential Planning"
    
    # Satellite properties (ASTERIA-based 6U)
    J: np.ndarray = field(default_factory=lambda: np.diag([0.0969, 0.1235, 0.1918]))  # kg·m²
    mass: float = 10.165  # kg
    
    # RW properties (3 RWs)
    rw_max_torque: float = 0.001   # Nm
    rw_max_momentum: float = 0.01  # Nms
    
    # Goal sequence from thesis Table 7.6:
    # -x anti-ram (150s-1100s)
    # z nadir (1200s-1500s)  
    # z zenith (1600s-1900s)
    # z orbit normal (2000s-2400s)
    # -x anti-ram (2500s on)
    goals: List[Tuple[float, float, str, np.ndarray, str]] = field(default_factory=lambda: [
        (150, 1100, '-x', np.array([-1, 0, 0]), 'anti-ram'),
        (1200, 1500, '+z', np.array([0, 0, 1]), 'nadir'),
        (1600, 1900, '+z', np.array([0, 0, 1]), 'zenith'),
        (2000, 2400, '+z', np.array([0, 0, 1]), 'orbit_normal'),
        (2500, 3600, '-x', np.array([-1, 0, 0]), 'anti-ram'),
    ])
    
    # Test parameters
    duration_s: float = 3600.0  # 1 hour
    dt: float = 1.0
    trajectory_overlap: float = 150.0  # seconds


# =============================================================================
# QUICK REFERENCE FUNCTIONS
# =============================================================================

def get_ch6_configs():
    """Get all Chapter 6 test configurations."""
    return {
        'wie': Ch6_Wie_Config(),
        'lovera': Ch6_Lovera_Config(),
        'wisniewski': Ch6_Wisniewski_Config(),
    }


def get_ch7_configs():
    """Get all Chapter 7 test configurations."""
    return {
        'spinning': Ch7_Spinning_Config(),
        'monte_carlo': Ch7_MonteCarlo_Config(),
        'sequential': Ch7_Sequential_Config(),
    }


# =============================================================================
# ALTRO PLANNER SETTINGS - EXACT VALUES FROM THESIS CODE
# =============================================================================

@dataclass
class ALTROCostWeights:
    """
    ALTRO cost function weights from thesis code (ADCS_Bx.py lines 149-180).
    
    Cost function: ℓ = angle_weight·ℓ_q + angvel_weight·(½ω'ω) + u_weight·(½u'Ru)
    Terminal cost: ℓ_N = angle_weight_N·ℓ_q + angvel_weight_N·(½ω'ω)
    
    Pass 1 (main optimization):
        angle_weight = 10, angvel_weight = 100
        angle_weight_N = 100, angvel_weight_N = 100
        
    Pass 2 (refinement):
        angle_weight2 = 10, angvel_weight2 = 0.1
        angle_weight_N2 = 1000, angvel_weight_N2 = 1.0
        
    TVLQR (trajectory following):
        angle_weight_tvlqr = 10, angvel_weight_tvlqr = 100
        angle_weight_N_tvlqr = 100, angvel_weight_N_tvlqr = 100
    """
    # Pass 1 - main optimization  
    angle_weight: float = 10.0
    angvel_weight: float = 100.0
    u_weight_mult: float = 1.0
    angle_weight_N: float = 100.0
    angvel_weight_N: float = 100.0
    
    # Pass 2 - refinement
    angle_weight2: float = 10.0
    angvel_weight2: float = 0.1
    u_weight_mult2: float = 1.0
    angle_weight_N2: float = 1000.0
    angvel_weight_N2: float = 1.0
    
    # TVLQR - trajectory following
    angle_weight_tvlqr: float = 10.0
    angvel_weight_tvlqr: float = 100.0
    u_weight_mult_tvlqr: float = 1.0
    angle_weight_N_tvlqr: float = 100.0
    angvel_weight_N_tvlqr: float = 100.0


@dataclass
class ALTROSolverSettings:
    """
    ALTRO solver settings from thesis code (ADCS_Bx.py lines 186-230).
    
    These control the augmented Lagrangian / iLQR convergence.
    """
    # Line search settings
    maxLsIter: int = 20
    beta1: float = 1e-10
    beta2: float = 500.0
    
    # Augmented Lagrangian settings
    lagMultInit: float = 0.0
    lagMultMax: float = 1e20
    penInit: float = 1.0        # Initial penalty
    penMax: float = 1e16
    penScale: float = 10.0      # Penalty scaling factor
    
    # Regularization settings  
    regInit: float = 0.01       # Initial regularization
    regMin: float = 1e-8
    regMax: float = 1e30
    regScale: float = 1.6       # Regularization scaling
    regBump: float = 10.0
    
    # Convergence criteria
    maxOuterIter: int = 30      # Max AL iterations
    maxIlqrIter: int = 250      # Max iLQR iterations per AL step
    maxIter: int = 4500         # Absolute max iterations
    gradTol: float = 1e-4       # Gradient tolerance
    ilqrCostTol: float = 0.01   # iLQR cost change tolerance
    costTol: float = 1e-4       # Overall cost tolerance
    zCountLim: int = 10
    cmax: float = 0.002         # Max constraint violation


@dataclass
class ALTROActuatorWeights:
    """
    Actuator-specific weights from thesis code (ADCS_Bx.py lines 143-147).
    """
    mtq_control_weight: float = 0.0001
    rw_control_weight: float = 0.001
    magic_control_weight: float = 0.0001
    rw_AM_weight: float = 0.1       # Penalize RW momentum buildup
    rw_stic_weight: float = 0.01    # Penalize RW stiction region


@dataclass
class ALTROTimingSettings:
    """
    ALTRO timing settings from thesis code.
    """
    dt_tvlqr: float = 1.0       # TVLQR timestep (fine)
    dt_tp: float = 10.0         # Trajectory planner timestep (coarse)
    tvlqr_len: float = 1000.0   # TVLQR horizon length (s)
    tvlqr_overlap: float = 1.0
    traj_overlap: float = 10.0
    precalculation_time: float = 100.0


def get_thesis_planner_settings() -> Dict:
    """Get all ALTRO planner settings matching thesis code exactly."""
    return {
        'cost_weights': ALTROCostWeights(),
        'solver': ALTROSolverSettings(),
        'actuator_weights': ALTROActuatorWeights(),
        'timing': ALTROTimingSettings(),
    }


def print_config_summary():
    """Print summary of all thesis configurations."""
    print("\n" + "="*70)
    print("THESIS FIGURE CONFIGURATIONS")
    print("="*70)
    
    print("\nCHAPTER 6 - DISTURBANCE CONTROL")
    print("-"*40)
    ch6 = get_ch6_configs()
    for name, cfg in ch6.items():
        print(f"\n{cfg.name}:")
        print(f"  Inertia: {np.diag(cfg.J)}")
        print(f"  Duration: {cfg.duration_s}s ({cfg.duration_s/3600:.1f} hours)")
    
    print("\n\nCHAPTER 7 - PLANNING")
    print("-"*40)
    ch7 = get_ch7_configs()
    
    spin = ch7['spinning']
    print(f"\n{spin.name}:")
    print(f"  Disturbance: {spin.disturbance_torque*1000} mNm")
    print(f"  Duration: {spin.duration_s}s")
    
    mc = ch7['monte_carlo']
    print(f"\n{mc.name}:")
    print(f"  Inertia: [{mc.J_xx}, {mc.J_yy}, {mc.J_zz}]")
    print(f"  180° slew: q=[0,0,1,0] → [0,1,0,0]")
    
    seq = ch7['sequential']
    print(f"\n{seq.name}:")
    print(f"  Mass: {seq.mass} kg")
    print(f"  Goals: {len(seq.goals)} sequential targets")
    for start, end, axis, _, direction in seq.goals:
        print(f"    {start}s-{end}s: {axis} → {direction}")
    
    print("\n\nALTRO PLANNER SETTINGS")
    print("-"*40)
    planner = get_thesis_planner_settings()
    cw = planner['cost_weights']
    print(f"\nCost Weights (Pass 1):")
    print(f"  angle_weight = {cw.angle_weight}, angvel_weight = {cw.angvel_weight}")
    print(f"  angle_weight_N = {cw.angle_weight_N}, angvel_weight_N = {cw.angvel_weight_N}")
    
    sv = planner['solver']
    print(f"\nSolver Settings:")
    print(f"  penInit = {sv.penInit}, penScale = {sv.penScale}")
    print(f"  regInit = {sv.regInit}, regScale = {sv.regScale}")
    print(f"  maxIter = {sv.maxIter}, cmax = {sv.cmax}")


if __name__ == "__main__":
    print_config_summary()
