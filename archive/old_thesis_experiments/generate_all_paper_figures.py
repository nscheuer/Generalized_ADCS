#!/usr/bin/env python3
"""
Generate ALL Paper Figures
==========================

Complete figure generation for all 4 papers. Includes:
- Monte Carlo simulations
- Direction preservation analysis (LP vs QP)
- Torque polytope visualization
- Momentum management comparison
- Actuator failure simulation
- ALTRO trajectory planning
- Multi-target sequences

Usage:
    # Run everything (takes 8-16 hours for full)
    python generate_all_paper_figures.py --all --full
    
    # Quick test
    python generate_all_paper_figures.py --all --quick
    
    # Specific paper
    python generate_all_paper_figures.py --paper 3p1 --full
    
    # Specific figure
    python generate_all_paper_figures.py --figure direction_preservation --quick
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull
from tqdm import tqdm

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../..")))

# ADCS imports
from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal, No_Goal, Nadir_Goal
from ADCS.CONOPS.goallist import GoalList
from ADCS.controller import MTQ_w_RW_LP, MTQ_w_RW_QP
from ADCS.controller.mtq_lovera import MTQ_Lovera
from ADCS.controller.mtq_wisniewski import MTQ_Wisniewski
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.helpers.orbit_factory import create_random_circular_orbit
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_factory.satellites.create_cubesats import (
    create_beavercube1_cubesat,
    create_beavercube2_cubesat,
    create_3_3_beavercube2_cubesat,
)
from ADCS.helpers.math_helpers import normalize, rot_mat
from ADCS.helpers.math_constants import MathConstants

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Experiment configuration."""
    n_trials: int = 100
    duration_s: float = 1000
    dt: float = 2.0
    altitude_km: float = 400
    inclination_deg: float = 51.6
    body_boresight: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    use_reduced_attitude: bool = True
    
    # Controller gains (from working test files)
    mtq_p_gain: float = 0.001
    mtq_d_gain: float = 0.005
    rw_p_gain: float = 1.0  # Higher gain for LP/QP controllers
    rw_d_gain: float = 0.5
    rw_c_gain: float = 0.01  # Momentum management
    
    @classmethod
    def quick(cls):
        return cls(n_trials=10, duration_s=200)
    
    @classmethod
    def full(cls):
        return cls(n_trials=100, duration_s=1000)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_orbit(altitude_km: float, inclination_deg: float, dt: float, duration_s: float,
                 seed: int = 0, precompute_env: bool = True) -> Tuple[Orbit, float]:
    """Create orbit with random starting position.

    Args:
        altitude_km: Orbit altitude
        inclination_deg: Orbit inclination
        dt: Time step
        duration_s: Duration
        seed: Random seed for reproducibility
        precompute_env: If True, batch-compute B and S vectors (~100x faster)
    """
    np.random.seed(seed)
    ephem = Ephemeris()

    R_mag = 6378.137 + altitude_km
    inc = np.deg2rad(inclination_deg)
    theta = np.random.uniform(0, 2 * np.pi)

    R = R_mag * np.array([np.cos(theta), np.sin(theta) * np.cos(inc), np.sin(theta) * np.sin(inc)])
    V_mag = np.sqrt(398600.4418 / R_mag)
    V = V_mag * np.array([-np.sin(theta), np.cos(theta) * np.cos(inc), np.cos(theta) * np.sin(inc)])

    # Use fixed start time in 2024 (within ephemeris range) with small offset for variety
    # J2000 = 0.24 corresponds to ~2024
    start_time = 0.24 + (seed % 1000) * 0.00001  # Keep within valid ephemeris range
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V, fast=True)
    end_time = start_time + (duration_s + 100) * TimeConstants.sec2cent

    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=True, verbose=False)

    # Batch-compute environment vectors (~100x faster than per-step computation)
    if precompute_env:
        orb.populate_environment(compute_B=True, compute_S=True, verbose=False)

    return orb, start_time


def compute_pointing_error(q: np.ndarray, goal, boresight: np.ndarray, os) -> float:
    """Compute pointing error in degrees."""
    R = rot_mat(q)
    boresight_eci = R @ boresight
    
    if isinstance(goal, ECI_Goal):
        goal_vec, _ = goal.to_ref(os)
        goal_vec = goal_vec / np.linalg.norm(goal_vec)
    elif isinstance(goal, Fixed_Attitude_Goal):
        q_goal = goal.q_ref
        q_err = np.array([
            q[0]*q_goal[0] + q[1]*q_goal[1] + q[2]*q_goal[2] + q[3]*q_goal[3],
            -q[0]*q_goal[1] + q[1]*q_goal[0] - q[2]*q_goal[3] + q[3]*q_goal[2],
            -q[0]*q_goal[2] + q[1]*q_goal[3] + q[2]*q_goal[0] - q[3]*q_goal[1],
            -q[0]*q_goal[3] - q[1]*q_goal[2] + q[2]*q_goal[1] + q[3]*q_goal[0],
        ])
        return 2 * np.rad2deg(np.arccos(np.clip(abs(q_err[0]), 0, 1)))
    else:
        goal_vec = np.array([0, 0, 1])
    
    dot = np.clip(np.dot(boresight_eci, goal_vec), -1, 1)
    return np.rad2deg(np.arccos(dot))


def compute_direction_error(tau_des: np.ndarray, tau_ach: np.ndarray) -> float:
    """Compute angle between desired and achieved torque vectors."""
    norm_des = np.linalg.norm(tau_des)
    norm_ach = np.linalg.norm(tau_ach)
    if norm_des < 1e-12 or norm_ach < 1e-12:
        return 0.0
    dot = np.clip(np.dot(tau_des, tau_ach) / (norm_des * norm_ach), -1, 1)
    return np.rad2deg(np.arccos(dot))


# =============================================================================
# SATELLITE FACTORIES
# =============================================================================

# Use THESIS parameters by default for paper figures
# BeaverCube parameters differ from thesis (different J, MTQ limits, RW torque)
USE_THESIS_PARAMS = True  # Set False to use BeaverCube hardware params

def create_sat_3p0():
    """Create MTQ-only satellite (3+0 configuration)."""
    if USE_THESIS_PARAMS:
        return create_sat_thesis_mtq()
    return create_beavercube1_cubesat(estimated=False)

def create_sat_3p1():
    """Create hybrid satellite (3MTQ + 1RW configuration)."""
    if USE_THESIS_PARAMS:
        return create_sat_thesis_3p1()
    return create_beavercube2_cubesat(estimated=False)

def create_sat_3p3():
    """Create fully-actuated satellite (3MTQ + 3RW configuration)."""
    if USE_THESIS_PARAMS:
        return create_sat_thesis_3p3()
    return create_3_3_beavercube2_cubesat(estimated=False)

def create_sat_thesis_mtq():
    """Create MTQ-only satellite matching thesis Table 7.2.

    Thesis parameters:
    - J = diag([0.005256, 0.04939, 0.04939]) kg·m²
    - MTQ max dipole: 0.19 Am² (x), 0.57 Am² (y,z)
    - Boresight: x-axis (payload camera)
    """
    from ADCS.satellite_hardware.sensors import MTM, Gyro
    J = np.diag([0.005256, 0.04939, 0.04939])
    # NOTE: MTQ class uses 'max_torque' but stores max magnetic dipole [Am²]
    mtqs = [
        MTQ(axis=np.array([1, 0, 0]), max_torque=0.19),
        MTQ(axis=np.array([0, 1, 0]), max_torque=0.57),
        MTQ(axis=np.array([0, 0, 1]), max_torque=0.57),
    ]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    gyros = [Gyro(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=J, actuators=mtqs, sensors=mtms+gyros, boresight=np.array([1,0,0]))

def create_sat_thesis_3p1():
    """Create 3+1 satellite matching thesis Table 7.2.

    Thesis parameters:
    - J = diag([0.005256, 0.04939, 0.04939]) kg·m²
    - MTQ max dipole: 0.19 Am² (x), 0.57 Am² (y,z)
    - RW: y-axis, max_torque=0.0002 Nm, h_max=0.002 Nms
    """
    from ADCS.satellite_hardware.sensors import MTM, Gyro
    J = np.diag([0.005256, 0.04939, 0.04939])
    mtqs = [
        MTQ(axis=np.array([1, 0, 0]), max_torque=0.19),
        MTQ(axis=np.array([0, 1, 0]), max_torque=0.57),
        MTQ(axis=np.array([0, 0, 1]), max_torque=0.57),
    ]
    rw = RW(axis=np.array([0, 1, 0]), max_torque=0.0002, J=2e-6, h=0.0, h_max=0.002)
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    gyros = [Gyro(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=J, actuators=mtqs+[rw], sensors=mtms+gyros, boresight=np.array([1,0,0]))

def create_sat_thesis_3p3():
    """Create 3+3 satellite (fully actuated) matching thesis parameters.

    Thesis parameters:
    - J = diag([0.005256, 0.04939, 0.04939]) kg·m²
    - MTQ max dipole: 0.19 Am² (x), 0.57 Am² (y,z)
    - RWs: 3 orthogonal, max_torque=0.0002 Nm each, h_max=0.002 Nms
    """
    from ADCS.satellite_hardware.sensors import MTM, Gyro
    J = np.diag([0.005256, 0.04939, 0.04939])
    mtqs = [
        MTQ(axis=np.array([1, 0, 0]), max_torque=0.19),
        MTQ(axis=np.array([0, 1, 0]), max_torque=0.57),
        MTQ(axis=np.array([0, 0, 1]), max_torque=0.57),
    ]
    # 3 orthogonal RWs for full 3-axis control
    rws = [
        RW(axis=np.array([1, 0, 0]), max_torque=0.0002, J=2e-6, h=0.0, h_max=0.002),
        RW(axis=np.array([0, 1, 0]), max_torque=0.0002, J=2e-6, h=0.0, h_max=0.002),
        RW(axis=np.array([0, 0, 1]), max_torque=0.0002, J=2e-6, h=0.0, h_max=0.002),
    ]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    gyros = [Gyro(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=J, actuators=mtqs+rws, sensors=mtms+gyros, boresight=np.array([1,0,0]))


# =============================================================================
# CONTROLLER FACTORIES
# =============================================================================

def create_ctrl_lovera(sat, cfg):
    return MTQ_Lovera(est_sat=sat, p_gain=cfg.mtq_p_gain, d_gain=cfg.mtq_d_gain, eps=1.0)

def create_ctrl_wisniewski(sat, cfg):
    return MTQ_Wisniewski(est_sat=sat, p_gain=cfg.mtq_p_gain, d_gain=cfg.mtq_d_gain, eps=0.1)

def create_ctrl_lp(sat, cfg):
    # h_target is always 3D (body-frame momentum target)
    h_target = np.zeros(3)
    return MTQ_w_RW_LP(est_sat=sat, p_gain=cfg.rw_p_gain, d_gain=cfg.rw_d_gain, 
                       c_gain=cfg.rw_c_gain, h_target=h_target)

def create_ctrl_qp(sat, cfg):
    # h_target is always 3D (body-frame momentum target)
    h_target = np.zeros(3)
    return MTQ_w_RW_QP(est_sat=sat, p_gain=cfg.rw_p_gain, d_gain=cfg.rw_d_gain,
                       c_gain=cfg.rw_c_gain, h_target=h_target)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(
    sat: Satellite,
    controller,
    goal,
    orb: Orbit,
    start_time: float,
    cfg: Config,
    x0: np.ndarray,
    record_torques: bool = False,
    disable_actuator_at: Optional[Tuple[float, int]] = None,  # (time_s, actuator_idx)
) -> Dict[str, Any]:
    """Run a single simulation with optional torque recording and actuator failure."""
    N = int(cfg.duration_s / cfg.dt)
    n_act = len(sat.actuators)
    
    x = x0.copy()
    for i, rw in enumerate(sat.rw_actuators):
        if len(x) > 7 + i:
            rw.h = x[7 + i]
    
    # History arrays
    time_hist = np.zeros(N)
    state_hist = np.zeros((N, len(x)))
    u_hist = np.zeros((N, n_act))
    error_hist = np.zeros(N)
    
    if record_torques:
        tau_des_hist = np.zeros((N, 3))
        tau_ach_hist = np.zeros((N, 3))
        dir_error_hist = np.zeros(N)
    
    t = 0
    original_max_torques = [a.max_torque if hasattr(a, 'max_torque') else (a.max_m if hasattr(a, 'max_m') else 1.0) 
                           for a in sat.actuators]
    
    for i in range(N):
        # Check for actuator failure
        if disable_actuator_at and t >= disable_actuator_at[0]:
            act_idx = disable_actuator_at[1]
            if hasattr(sat.actuators[act_idx], 'max_torque'):
                sat.actuators[act_idx].max_torque = 0.0
            if hasattr(sat.actuators[act_idx], 'max_m'):
                sat.actuators[act_idx].max_m = 0.0
        
        J2000 = start_time + t * TimeConstants.sec2cent
        os = orb.get_os(J2000)
        sens = sat.sensor_readings(x=x, os=os)
        
        u = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os, goal=goal)
        
        error_deg = compute_pointing_error(x[3:7], goal, cfg.body_boresight, os)
        
        if record_torques:
            # Compute desired torque from PD law: tau = -kp*q_err - kd*omega
            q = x[3:7]
            omega = x[:3]
            
            # Get goal quaternion
            if isinstance(goal, ECI_Goal):
                goal_vec, _ = goal.to_ref(os)
                goal_vec = goal_vec / np.linalg.norm(goal_vec)
                # For reduced attitude, compute quaternion that aligns boresight with goal
                R = rot_mat(q)
                boresight_eci = R @ cfg.body_boresight
                # Simple approximation: q_err ~ cross(boresight, goal)
                q_err_vec = np.cross(boresight_eci, goal_vec)
            else:
                q_err_vec = q[1:4]  # Use vector part of quaternion as error
            
            # PD torque (simplified)
            tau_des = -cfg.rw_p_gain * q_err_vec * 1000 - cfg.rw_d_gain * omega * 100
            
            # Achieved torque
            tau_ach = sat.act_torque(u=u, os=os, x=x) if hasattr(sat, 'act_torque') else np.zeros(3)
            
            tau_des_hist[i] = tau_des
            tau_ach_hist[i] = tau_ach
            dir_error_hist[i] = compute_direction_error(tau_des, tau_ach)
        
        time_hist[i] = t
        state_hist[i] = x
        u_hist[i] = u
        error_hist[i] = error_deg
        
        # Propagate
        t_next = t + cfg.dt
        os_next = orb.get_os(start_time + t_next * TimeConstants.sec2cent)
        
        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, cfg.dt),
            y0=x,
            method='RK45',
            args=(u, os, os_next),
            rtol=1e-6, atol=1e-6,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t = t_next
    
    # Restore actuators
    for idx, max_t in enumerate(original_max_torques):
        if hasattr(sat.actuators[idx], 'max_torque'):
            sat.actuators[idx].max_torque = max_t
        if hasattr(sat.actuators[idx], 'max_m'):
            sat.actuators[idx].max_m = max_t
    
    final_idx = int(0.8 * N)
    result = {
        'time': time_hist,
        'state': state_hist,
        'u': u_hist,
        'error_deg': error_hist,
        'final_error_deg': float(np.mean(error_hist[final_idx:])),
        'min_error_deg': float(np.min(error_hist[final_idx:])),
    }
    
    if record_torques:
        result['tau_des'] = tau_des_hist
        result['tau_ach'] = tau_ach_hist
        result['dir_error_deg'] = dir_error_hist
        result['mean_dir_error'] = float(np.mean(dir_error_hist[dir_error_hist > 0]))
    
    return result


def run_monte_carlo(sat_factory, ctrl_factory, cfg: Config, name: str, 
                    record_torques: bool = False) -> Dict[str, Any]:
    """Run Monte Carlo campaign."""
    all_errors = []
    all_trajectories = []
    all_dir_errors = [] if record_torques else None
    
    for trial in tqdm(range(cfg.n_trials), desc=f"  {name}", leave=False):
        seed = trial * 1000
        np.random.seed(seed)
        
        orb, start_time = create_orbit(cfg.altitude_km, cfg.inclination_deg, cfg.dt, cfg.duration_s, seed)
        sat = sat_factory()
        ctrl = ctrl_factory(sat, cfg)
        
        w0 = normalize(np.random.randn(3)) * np.random.uniform(0.001, 0.01)
        q0 = normalize(np.random.randn(4))
        n_rw = len(sat.rw_actuators)
        h0 = np.random.uniform(-0.0001, 0.0001, n_rw) if n_rw > 0 else np.array([])
        x0 = np.concatenate([w0, q0, h0])
        
        if cfg.use_reduced_attitude:
            goal = ECI_Goal(normalize(np.random.randn(3)))
        else:
            goal = Fixed_Attitude_Goal(normalize(np.random.randn(4)))
        
        result = run_simulation(sat, ctrl, goal, orb, start_time, cfg, x0, record_torques=record_torques)
        
        all_errors.append(result['final_error_deg'])
        if trial < 10:
            all_trajectories.append(result['error_deg'].tolist())
        if record_torques and 'mean_dir_error' in result:
            all_dir_errors.append(result['mean_dir_error'])
    
    errors = np.array(all_errors)
    output = {
        'name': name,
        'n_trials': cfg.n_trials,
        'duration_s': cfg.duration_s,
        'errors': errors.tolist(),
        'trajectories': all_trajectories,
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'median': float(np.median(errors)),
        'pct_1deg': float(100 * np.sum(errors < 1) / len(errors)),
        'pct_5deg': float(100 * np.sum(errors < 5) / len(errors)),
        'pct_10deg': float(100 * np.sum(errors < 10) / len(errors)),
    }
    
    if record_torques and all_dir_errors:
        output['mean_dir_error'] = float(np.mean(all_dir_errors))
        output['dir_errors'] = all_dir_errors
    
    return output


# =============================================================================
# FIGURE GENERATORS
# =============================================================================

def save_fig(fig, output_dir: Path, name: str):
    """Save figure in both PNG and PDF."""
    fig.savefig(output_dir / f'{name}.png')
    fig.savefig(output_dir / f'{name}.pdf')
    plt.close(fig)
    print(f"    Saved: {name}")


# -----------------------------------------------------------------------------
# 3+1 PAPER FIGURES
# -----------------------------------------------------------------------------

def gen_3p1_architecture_comparison(cfg: Config, output_dir: Path):
    """Generate architecture comparison figures (3+0, 3+1, 3+3)."""
    print("\n  [3+1] Architecture Comparison...")
    
    configs = {
        '3+0': (create_sat_3p0, create_ctrl_lovera),
        '3+1': (create_sat_3p1, create_ctrl_lp),
        '3+3': (create_sat_3p3, create_ctrl_lp),
    }
    
    results = {}
    for name, (sat_f, ctrl_f) in configs.items():
        results[name] = run_monte_carlo(sat_f, ctrl_f, cfg, name)
    
    # Error trajectories
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.arange(0, cfg.duration_s, cfg.dt)
    for i, (name, res) in enumerate(results.items()):
        color = COLORS[i]
        for traj in res['trajectories']:
            ax.plot(times[:len(traj)], traj, color=color, alpha=0.15, linewidth=0.5)
        mean_traj = np.mean(res['trajectories'], axis=0)
        ax.plot(times[:len(mean_traj)], mean_traj, color=color, linewidth=2, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    save_fig(fig, output_dir, 'fig_3p1_error_trajectories')
    
    # CDF
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, res) in enumerate(results.items()):
        errors = np.sort(res['errors'])
        cdf = np.arange(1, len(errors)+1) / len(errors) * 100
        ax.plot(errors, cdf, color=COLORS[i], linewidth=1.5, label=name)
    ax.set_xscale('log')
    ax.set_xlabel('Final Pointing Error (deg)')
    ax.set_ylabel('Cumulative %')
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(10.0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlim(0.01, 200)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_dir, 'fig_3p1_cdf')
    
    # Success bars
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(results))
    width = 0.25
    names = list(results.keys())
    ax.bar(x - width, [results[n]['pct_1deg'] for n in names], width, label='<1°', color=COLORS[2])
    ax.bar(x, [results[n]['pct_5deg'] for n in names], width, label='<5°', color=COLORS[3])
    ax.bar(x + width, [results[n]['pct_10deg'] for n in names], width, label='<10°', color=COLORS[1])
    ax.set_ylabel('Success Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.legend()
    save_fig(fig, output_dir, 'fig_3p1_success_rates')
    
    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(-2, 2, 40)
    for i, (name, res) in enumerate(results.items()):
        ax.hist(res['errors'], bins=bins, alpha=0.6, color=COLORS[i], label=name, edgecolor='white')
    ax.set_xscale('log')
    ax.set_xlabel('Final Pointing Error (deg)')
    ax.set_ylabel('Count')
    ax.legend()
    save_fig(fig, output_dir, 'fig_3p1_histogram')
    
    # Save data
    with open(output_dir / 'data_3p1_architecture.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'} 
                   for k, v in results.items()}, f, indent=2)
    
    # LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Architecture Comparison Monte Carlo Results}
\begin{tabular}{lccccc}
\toprule
Config & Mean & Std & $<1°$ & $<5°$ & $<10°$ \\
\midrule
"""
    for name, res in results.items():
        latex += f"{name} & {res['mean']:.2f}$^\\circ$ & {res['std']:.2f}$^\\circ$ & "
        latex += f"{res['pct_1deg']:.0f}\\% & {res['pct_5deg']:.0f}\\% & {res['pct_10deg']:.0f}\\% \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(output_dir / 'table_3p1_architecture.tex', 'w') as f:
        f.write(latex)
    
    return results


def gen_3p1_torque_envelope(cfg: Config, output_dir: Path):
    """Generate torque envelope comparison for different configs."""
    print("\n  [3+1] Torque Envelope Comparison...")
    
    def compute_torque_envelope(sat, B_body, n_samples=1000):
        """Compute achievable torque set given B-field."""
        torques = []
        
        # Sample MTQ commands
        for _ in range(n_samples):
            m = np.zeros(3)
            for mtq in sat.mtq_actuators:
                # u_max is the max dipole moment
                max_m = mtq.u_max if hasattr(mtq, 'u_max') else 0.2
                m += mtq.axis * np.random.uniform(-max_m, max_m)
            tau_mtq = np.cross(m, B_body)
            
            tau_rw = np.zeros(3)
            for rw in sat.rw_actuators:
                max_torque = rw.max_torque if hasattr(rw, 'max_torque') else 0.001
                tau_rw += rw.axis * np.random.uniform(-max_torque, max_torque)
            
            torques.append(tau_mtq + tau_rw)
        
        return np.array(torques)
    
    # Different B-field orientations
    B_fields = [
        np.array([1, 0, 0]) * 30e-6,
        np.array([0, 1, 0]) * 30e-6,
        np.array([1, 1, 1]) / np.sqrt(3) * 30e-6,
    ]
    B_names = ['B along X', 'B along Y', 'B diagonal']
    
    configs = {
        '3+0': create_sat_3p0(),
        '3+1': create_sat_3p1(),
        '3+3': create_sat_3p3(),
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax, B, B_name in zip(axes, B_fields, B_names):
        for i, (name, sat) in enumerate(configs.items()):
            torques = compute_torque_envelope(sat, B)
            # Project to 2D (x-y plane)
            ax.scatter(torques[:, 0] * 1e6, torques[:, 1] * 1e6, 
                      alpha=0.3, s=1, color=COLORS[i], label=name)
        
        ax.set_xlabel(r'$\tau_x$ ($\mu$Nm)')
        ax.set_ylabel(r'$\tau_y$ ($\mu$Nm)')
        ax.set_title(B_name)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(markerscale=5)
    
    fig.suptitle('Achievable Torque Envelopes (X-Y projection)', y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_3p1_torque_envelope')


def gen_3p1_momentum_management(cfg: Config, output_dir: Path):
    """Generate momentum management comparison (continuous vs scheduled)."""
    print("\n  [3+1] Momentum Management...")
    
    # Use longer duration for momentum study
    cfg_mom = Config(n_trials=1, duration_s=5400, dt=2.0)  # 90 min = 1 orbit
    
    sat = create_sat_3p1()
    orb, start_time = create_orbit(cfg_mom.altitude_km, cfg_mom.inclination_deg, 
                                    cfg_mom.dt, cfg_mom.duration_s, seed=42)
    
    q0 = normalize(np.array([0.1, 0.2, 0.3, np.sqrt(1-0.01-0.04-0.09)]))
    w0 = np.array([0.001, -0.001, 0.002])
    h0 = np.array([0.0005])  # Start with some momentum
    x0 = np.concatenate([w0, q0, h0])
    
    goal = ECI_Goal(np.array([0, 0, 1]))  # Nadir
    
    # Continuous desaturation (normal LP controller)
    ctrl_cont = create_ctrl_lp(sat, cfg_mom)
    result_cont = run_simulation(sat, ctrl_cont, goal, orb, start_time, cfg_mom, x0)
    
    # For scheduled, we'd need a modified controller - simulate with lower c_gain
    cfg_sched = Config(n_trials=1, duration_s=5400, dt=2.0)
    cfg_sched.rw_c_gain = 0.0001  # Lower momentum management
    ctrl_sched = create_ctrl_lp(sat, cfg_sched)
    result_sched = run_simulation(sat, ctrl_sched, goal, orb, start_time, cfg_sched, x0)
    
    # Extract wheel momentum from state
    h_cont = result_cont['state'][:, 7] if result_cont['state'].shape[1] > 7 else np.zeros(len(result_cont['time']))
    h_sched = result_sched['state'][:, 7] if result_sched['state'].shape[1] > 7 else np.zeros(len(result_sched['time']))
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    t_min = result_cont['time'] / 60
    
    axes[0].plot(t_min, h_cont * 1000, label='Continuous desat', color=COLORS[0])
    axes[0].plot(t_min, h_sched * 1000, label='Reduced desat', color=COLORS[1])
    axes[0].axhline(2, color='red', linestyle='--', alpha=0.5, label='h_max')
    axes[0].axhline(-2, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Wheel Momentum (mNms)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_min, result_cont['error_deg'], label='Continuous', color=COLORS[0], alpha=0.8)
    axes[1].plot(t_min, result_sched['error_deg'], label='Reduced', color=COLORS[1], alpha=0.8)
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Pointing Error (deg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Momentum Management Comparison (1 Orbit)', y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_3p1_momentum_management')


def gen_3p1_graceful_degradation(cfg: Config, output_dir: Path):
    """Generate wheel failure graceful degradation figure."""
    print("\n  [3+1] Graceful Degradation (Wheel Failure)...")
    
    cfg_fail = Config(n_trials=1, duration_s=1200, dt=2.0)
    
    sat = create_sat_3p1()
    orb, start_time = create_orbit(cfg_fail.altitude_km, cfg_fail.inclination_deg,
                                    cfg_fail.dt, cfg_fail.duration_s, seed=42)
    
    q0 = normalize(np.array([0.1, 0.2, 0.3, np.sqrt(1-0.01-0.04-0.09)]))
    w0 = np.array([0.001, -0.001, 0.002])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    goal = ECI_Goal(np.array([0, 0, 1]))
    ctrl = create_ctrl_lp(sat, cfg_fail)
    
    # Run with wheel failure at t=500s
    # Find RW index
    rw_idx = None
    for i, act in enumerate(sat.actuators):
        if isinstance(act, RW):
            rw_idx = i
            break
    
    result = run_simulation(sat, ctrl, goal, orb, start_time, cfg_fail, x0,
                           disable_actuator_at=(500, rw_idx) if rw_idx else None)
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    t = result['time']
    
    axes[0].plot(t, result['error_deg'], color=COLORS[0], linewidth=1.5)
    axes[0].axvline(500, color='red', linestyle='--', alpha=0.7, label='RW Failure')
    axes[0].set_ylabel('Pointing Error (deg)')
    axes[0].legend()
    axes[0].annotate('3+1 Mode', (200, 5), fontsize=9)
    axes[0].annotate('3+0 Mode\n(Degraded)', (800, 15), fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # RW command (will be 0 after failure)
    if rw_idx:
        rw_cmd = result['u'][:, rw_idx]
        axes[1].plot(t, rw_cmd * 1e6, color=COLORS[1], linewidth=1.5)
    axes[1].axvline(500, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RW Torque Command (μNm)')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('Graceful Degradation After Wheel Failure', y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_3p1_graceful_degradation')


# -----------------------------------------------------------------------------
# GENERALIZED CONTROL PAPER FIGURES
# -----------------------------------------------------------------------------

def gen_lp_vs_qp_comparison(cfg: Config, output_dir: Path):
    """Generate LP vs QP allocation comparison with direction analysis."""
    print("\n  [Generalized] LP vs QP Comparison...")
    
    results = {}
    for name, ctrl_f in [('LP', create_ctrl_lp), ('QP', create_ctrl_qp)]:
        results[name] = run_monte_carlo(create_sat_3p1, ctrl_f, cfg, name, record_torques=True)
    
    # Error trajectories
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.arange(0, cfg.duration_s, cfg.dt)
    for i, (name, res) in enumerate(results.items()):
        for traj in res['trajectories']:
            ax.plot(times[:len(traj)], traj, color=COLORS[i], alpha=0.15, linewidth=0.5)
        mean_traj = np.mean(res['trajectories'], axis=0)
        ax.plot(times[:len(mean_traj)], mean_traj, color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    save_fig(fig, output_dir, 'fig_lp_vs_qp_trajectories')
    
    # CDF
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, res) in enumerate(results.items()):
        errors = np.sort(res['errors'])
        cdf = np.arange(1, len(errors)+1) / len(errors) * 100
        ax.plot(errors, cdf, color=COLORS[i], linewidth=1.5, label=name)
    ax.set_xscale('log')
    ax.set_xlabel('Final Pointing Error (deg)')
    ax.set_ylabel('Cumulative %')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, output_dir, 'fig_lp_vs_qp_cdf')
    
    # Save data
    with open(output_dir / 'data_lp_vs_qp.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'}
                   for k, v in results.items()}, f, indent=2)
    
    return results


def gen_direction_preservation(cfg: Config, output_dir: Path):
    """Generate direction preservation figure showing LP vs QP torque alignment.
    
    This directly tests the allocators on achievable torques (perpendicular to B)
    to show LP preserves direction perfectly while QP minimizes 2-norm error.
    """
    print("\n  [Generalized] Direction Preservation Analysis...")
    
    # Create satellite and controllers for direct allocation testing
    sat_lp = create_sat_3p1()
    sat_qp = create_sat_3p1()
    
    cfg_test = Config(n_trials=1, duration_s=100, dt=1.0)
    ctrl_lp = create_ctrl_lp(sat_lp, cfg_test)
    ctrl_qp = create_ctrl_qp(sat_qp, cfg_test)
    
    # Test n random achievable torques (perpendicular to B)
    n_test = 100
    np.random.seed(42)
    
    dir_errors_lp = []
    dir_errors_qp = []
    tau_des_list = []
    tau_lp_list = []
    tau_qp_list = []
    
    for _ in range(n_test):
        # Fixed B-field in Z direction for cleaner 2D visualization
        b_body = np.array([0, 0, 40e-6])
        b_hat = np.array([0, 0, 1])
        
        # Desired torque in X-Y plane (perpendicular to B, so achievable by MTQs)
        theta = np.random.uniform(0, 2 * np.pi)
        tau_mag = np.random.uniform(1e-6, 20e-6)  # 1-20 μNm
        tau_des = np.array([np.cos(theta), np.sin(theta), 0]) * tau_mag
        
        # Get allocations from LP and QP
        try:
            u_rw_lp, u_mtq_lp, alpha_lp = ctrl_lp.allocate_max_torque_in_direction(tau_des, b_body, sat_lp)
        except:
            continue
            
        try:
            u_rw_qp, u_mtq_qp, alpha_qp = ctrl_qp.allocate_max_torque_in_direction(tau_des, b_body, sat_qp)
        except:
            continue
        
        # Compute achieved torques
        # MTQ torque: τ = m × B (m is the dipole moment vector = u_mtq)
        tau_mtq_lp = np.cross(u_mtq_lp, b_body)
        tau_mtq_qp = np.cross(u_mtq_qp, b_body)
        
        # RW torque: τ = -u_rw * axis (reaction torque)
        rw_axis = sat_lp.rw_actuators[0].axis if len(sat_lp.rw_actuators) > 0 else np.array([0, 0, 1])
        tau_rw_lp = -u_rw_lp[0] * rw_axis if len(u_rw_lp) > 0 else np.zeros(3)
        tau_rw_qp = -u_rw_qp[0] * rw_axis if len(u_rw_qp) > 0 else np.zeros(3)
        
        tau_ach_lp = tau_mtq_lp + tau_rw_lp
        tau_ach_qp = tau_mtq_qp + tau_rw_qp
        
        # Compute direction errors
        dir_err_lp = compute_direction_error(tau_des, tau_ach_lp)
        dir_err_qp = compute_direction_error(tau_des, tau_ach_qp)
        
        dir_errors_lp.append(dir_err_lp)
        dir_errors_qp.append(dir_err_qp)
        tau_des_list.append(tau_des)
        tau_lp_list.append(tau_ach_lp)
        tau_qp_list.append(tau_ach_qp)
    
    dir_errors_lp = np.array(dir_errors_lp)
    dir_errors_qp = np.array(dir_errors_qp)
    
    mean_lp = np.nanmean(dir_errors_lp)
    mean_qp = np.nanmean(dir_errors_qp)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scale = 1e6  # Convert to μNm
    
    # LP: τ_des vs τ_ach 
    ax = axes[0]
    for i in range(min(50, len(tau_des_list))):
        tau_des = np.array(tau_des_list[i])
        tau_ach = np.array(tau_lp_list[i])
        if np.linalg.norm(tau_des) > 1e-12:
            ax.arrow(0, 0, tau_des[0]*scale, tau_des[1]*scale,
                    head_width=0.3, color=COLORS[0], alpha=0.3, length_includes_head=True)
            ax.arrow(0, 0, tau_ach[0]*scale, tau_ach[1]*scale,
                    head_width=0.3, color=COLORS[2], alpha=0.5, length_includes_head=True)
    
    ax.set_xlabel(r'$\tau_x$ ($\mu$Nm)')
    ax.set_ylabel(r'$\tau_y$ ($\mu$Nm)')
    ax.set_title(f'LP Allocation (Direction-Preserving)\nMean direction error: {mean_lp:.1f}°')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # QP
    ax = axes[1]
    for i in range(min(50, len(tau_des_list))):
        tau_des = np.array(tau_des_list[i])
        tau_ach = np.array(tau_qp_list[i])
        if np.linalg.norm(tau_des) > 1e-12:
            ax.arrow(0, 0, tau_des[0]*scale, tau_des[1]*scale,
                    head_width=0.3, color=COLORS[0], alpha=0.3, length_includes_head=True)
            ax.arrow(0, 0, tau_ach[0]*scale, tau_ach[1]*scale,
                    head_width=0.3, color=COLORS[1], alpha=0.5, length_includes_head=True)
    
    ax.set_xlabel(r'$\tau_x$ ($\mu$Nm)')
    ax.set_ylabel(r'$\tau_y$ ($\mu$Nm)')
    ax.set_title(f'QP Allocation (Norm-Minimizing)\nMean direction error: {mean_qp:.1f}°')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[0], alpha=0.3, label='Desired'),
        Patch(facecolor=COLORS[2], alpha=0.5, label='LP Achieved'),
        Patch(facecolor=COLORS[1], alpha=0.5, label='QP Achieved'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_direction_preservation')
    
    # Direction error histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    
    bins = np.linspace(0, 90, 30)
    ax.hist(dir_errors_lp[~np.isnan(dir_errors_lp)], bins=bins, alpha=0.6, color=COLORS[2], label=f'LP (μ={mean_lp:.1f}°)', edgecolor='white')
    ax.hist(dir_errors_qp[~np.isnan(dir_errors_qp)], bins=bins, alpha=0.6, color=COLORS[1], label=f'QP (μ={mean_qp:.1f}°)', edgecolor='white')
    ax.set_xlabel('Direction Error (deg)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Torque Direction Error Distribution (100 random tests)')
    save_fig(fig, output_dir, 'fig_direction_error_histogram')


def gen_framework_versatility(cfg: Config, output_dir: Path):
    """Generate framework versatility demo - same control law, different actuators."""
    print("\n  [Generalized] Framework Versatility Demo...")
    
    configs = {
        '3+0 (Lovera)': (create_sat_3p0, create_ctrl_lovera),
        '3+1 (LP)': (create_sat_3p1, create_ctrl_lp),
        '3+3 (LP)': (create_sat_3p3, create_ctrl_lp),
    }
    
    results = {}
    for name, (sat_f, ctrl_f) in configs.items():
        results[name] = run_monte_carlo(sat_f, ctrl_f, cfg, name)
    
    # Single figure showing all configs
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.arange(0, cfg.duration_s, cfg.dt)
    for i, (name, res) in enumerate(results.items()):
        mean_traj = np.mean(res['trajectories'], axis=0)
        ax.plot(times[:len(mean_traj)], mean_traj, color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('Same Goal, Different Actuator Configurations')
    save_fig(fig, output_dir, 'fig_framework_versatility')
    
    # Save data
    with open(output_dir / 'data_framework_versatility.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'}
                   for k, v in results.items()}, f, indent=2)


def gen_controllability_analysis(cfg: Config, output_dir: Path):
    """Generate controllability vs inclination figure."""
    print("\n  [Generalized] Controllability Analysis...")

    inclinations = [0, 15, 30, 45, 60, 75, 90]
    configs = {
        '3+0': create_sat_3p0,
        '3+1': create_sat_3p1,
        '3+3': create_sat_3p3,
    }

    # For each config and inclination, compute "effective controllability"
    # by running short simulation and measuring convergence
    results = {name: [] for name in configs}

    cfg_short = Config(n_trials=5, duration_s=300, dt=2.0)

    for inc in tqdm(inclinations, desc="  Inclinations"):
        cfg_short.inclination_deg = inc
        for name, sat_f in configs.items():
            if name == '3+0':
                ctrl_f = create_ctrl_lovera
            else:
                ctrl_f = create_ctrl_lp

            res = run_monte_carlo(sat_f, ctrl_f, cfg_short, f"{name}@{inc}°")
            results[name].append(res['pct_10deg'] / 100)  # Normalize to 0-1

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, ctrl_vals) in enumerate(results.items()):
        ax.plot(inclinations, ctrl_vals, 'o-', color=COLORS[i], label=name, markersize=6)

    ax.set_xlabel('Orbit Inclination (deg)')
    ax.set_ylabel('Controllability Index\n(fraction < 10°)')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Effective Controllability vs Orbit Inclination')
    save_fig(fig, output_dir, 'fig_controllability_vs_inclination')


def gen_torque_polytope_evolution(cfg: Config, output_dir: Path):
    """Generate 3D torque polytope evolution over one orbit for Generalized Control paper."""
    print("\n  [Generalized] Torque Polytope Evolution...")

    # Parameters
    sat = create_sat_3p1()  # BC2: 3MTQ + 1RW
    orbit_period_s = 5400   # ~90 min ISS orbit
    n_samples = 1500        # Points to sample polytope

    # Create orbit
    orb, start_time = create_orbit(400, 51.6, 1.0, orbit_period_s + 100, seed=42)

    # Time points: 0, 15, 30, 45, 60, 75 minutes
    t_snapshots = [0, 900, 1800, 2700, 3600, 4500]

    fig = plt.figure(figsize=(14, 9))

    for idx, t in enumerate(t_snapshots):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')

        # Get B-field at this time
        J2000 = start_time + t * TimeConstants.sec2cent
        os = orb.get_os(J2000)
        B_eci = os.B  # Tesla

        # Use a fixed body attitude for visualization
        B_body = B_eci  # Simplified (assume identity attitude)

        # Sample achievable torques
        torques = []
        for _ in range(n_samples):
            # Random MTQ dipole within limits
            m = np.zeros(3)
            for mtq in sat.mtq_actuators:
                max_m = mtq.u_max if hasattr(mtq, 'u_max') else getattr(mtq, 'max_m', 0.5)
                m += mtq.axis * np.random.uniform(-max_m, max_m)
            tau_mtq = np.cross(m, B_body)

            # Random RW torque within limits
            tau_rw = np.zeros(3)
            for rw in sat.rw_actuators:
                max_t = rw.u_max if hasattr(rw, 'u_max') else getattr(rw, 'max_torque', 0.0002)
                tau_rw += rw.axis * np.random.uniform(-max_t, max_t)

            torques.append(tau_mtq + tau_rw)

        torques = np.array(torques) * 1e6  # Convert to μNm

        # Plot scatter (convex hull can be tricky in 3D)
        ax.scatter(torques[:, 0], torques[:, 1], torques[:, 2],
                   s=1, alpha=0.4, c=COLORS[idx % len(COLORS)])

        # Try to show convex hull faces
        try:
            hull = ConvexHull(torques)
            for simplex in hull.simplices:
                pts = torques[simplex]
                tri = Poly3DCollection([pts], alpha=0.15,
                                        facecolor=COLORS[idx % len(COLORS)],
                                        edgecolor='gray', linewidth=0.2)
                ax.add_collection3d(tri)
        except Exception:
            pass  # Hull may fail for degenerate cases

        ax.set_xlabel(r'$\tau_x$ ($\mu$Nm)', fontsize=8)
        ax.set_ylabel(r'$\tau_y$ ($\mu$Nm)', fontsize=8)
        ax.set_zlabel(r'$\tau_z$ ($\mu$Nm)', fontsize=8)
        ax.set_title(f't = {t//60} min\n|B| = {np.linalg.norm(B_body)*1e6:.1f} μT', fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle('Achievable Torque Polytope Evolution Over One Orbit (3MTQ+1RW)', fontsize=12)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_torque_polytope_evolution')


def gen_desaturation_scheduling(cfg: Config, output_dir: Path):
    """Show momentum evolution with different desaturation strategies."""
    print("\n  [Generalized] Desaturation Scheduling...")

    # Parameters - 3 orbits
    duration_s = 16200  # 3 orbits = 3 * 90 min
    dt = 2.0

    sat = create_sat_3p1()
    orb, start_time = create_orbit(400, 51.6, dt, duration_s + 100, seed=42)

    # Initial state with some momentum buildup
    q0 = normalize(np.array([0.1, 0.2, 0.3, np.sqrt(1-0.14)]))
    w0 = np.array([0.001, -0.001, 0.002])
    h0 = np.array([0.0008])  # Start with 0.8 mNms (40% of capacity)
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(np.array([0, 0, 1]))  # Nadir pointing

    # Run TWO simulations with different c_gain
    results = {}
    for name, c_gain in [('Continuous Desat\n(c=0.001)', 0.001),
                         ('Pointing Priority\n(c=0.0001)', 0.0001)]:
        cfg_run = Config(n_trials=1, duration_s=duration_s, dt=dt)
        cfg_run.rw_c_gain = c_gain

        # Create fresh satellite and controller
        sat_run = create_sat_3p1()
        ctrl = create_ctrl_lp(sat_run, cfg_run)

        result = run_simulation(sat_run, ctrl, goal, orb, start_time, cfg_run, x0.copy())
        results[name] = result

    # Compute B-field magnitude over time
    B_mag = []
    for t in results[list(results.keys())[0]]['time']:
        J2000 = start_time + t * TimeConstants.sec2cent
        os = orb.get_os(J2000)
        B_mag.append(np.linalg.norm(os.B) * 1e6)  # μT
    B_mag = np.array(B_mag)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_min = results[list(results.keys())[0]]['time'] / 60

    # Panel 1: Wheel momentum
    ax = axes[0]
    for i, (name, res) in enumerate(results.items()):
        if res['state'].shape[1] > 7:
            h = res['state'][:, 7] * 1000  # mNms
            ax.plot(t_min, h, label=name, linewidth=1.5, color=COLORS[i])
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='h_max')
    ax.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Wheel Momentum\n(mNms)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Pointing error
    ax = axes[1]
    for i, (name, res) in enumerate(results.items()):
        ax.plot(t_min, res['error_deg'], label=name, linewidth=1.5, color=COLORS[i], alpha=0.8)
    ax.set_ylabel('Pointing Error\n(deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 200)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 3: B-field magnitude (shows orbital variation)
    ax = axes[2]
    ax.plot(t_min, B_mag, color='purple', linewidth=1.5)
    ax.set_ylabel('|B| (μT)')
    ax.set_xlabel('Time (min)')
    ax.grid(True, alpha=0.3)

    # Mark high B-field regions (good for desaturation)
    B_thresh = np.percentile(B_mag, 70)
    in_window = B_mag > B_thresh
    for axi in axes:
        start_idx = None
        for i, is_high in enumerate(in_window):
            if is_high and start_idx is None:
                start_idx = i
            elif not is_high and start_idx is not None:
                axi.axvspan(t_min[start_idx], t_min[i-1], alpha=0.1, color='green')
                start_idx = None
        if start_idx is not None:
            axi.axvspan(t_min[start_idx], t_min[-1], alpha=0.1, color='green')

    # Add orbit markers
    for orbit_num in range(1, 4):
        t_orbit = orbit_num * 90
        for axi in axes:
            axi.axvline(t_orbit, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Desaturation Scheduling: Trade-off Between Pointing and Momentum Management', fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_desaturation_scheduling')


def gen_wcdta_sphere(cfg: Config, output_dir: Path):
    """Visualize Worst-Case Directional Torque Authority for different configs."""
    print("\n  [Generalized] WCDTA Sphere Visualization...")

    configs = {
        '3+0 (MTQ only)': create_sat_3p0(),
        '3+1 (Hybrid)': create_sat_3p1(),
        '3+3 (Full RW)': create_sat_3p3(),
    }

    # Sample B-field (typical ISS value)
    B_body = np.array([20e-6, 15e-6, 25e-6])  # Tesla

    # Sample directions on unit sphere using Fibonacci lattice for uniform coverage
    n_dirs = 200
    indices = np.arange(0, n_dirs, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_dirs)
    theta = np.pi * (1 + 5**0.5) * indices
    directions = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ]).T

    fig = plt.figure(figsize=(14, 5))

    for ax_idx, (name, sat) in enumerate(configs.items()):
        ax = fig.add_subplot(1, 3, ax_idx+1, projection='3d')

        # For each direction, find max achievable torque magnitude
        max_torques = []
        for d in directions:
            # Sample random actuator commands, project onto d
            samples = []
            for _ in range(300):
                m = np.zeros(3)
                for mtq in sat.mtq_actuators:
                    max_m = mtq.u_max if hasattr(mtq, 'u_max') else getattr(mtq, 'max_m', 0.5)
                    m += mtq.axis * np.random.uniform(-max_m, max_m)
                tau_mtq = np.cross(m, B_body)

                tau_rw = np.zeros(3)
                for rw in sat.rw_actuators:
                    max_t = rw.u_max if hasattr(rw, 'u_max') else getattr(rw, 'max_torque', 0.0002)
                    tau_rw += rw.axis * np.random.uniform(-max_t, max_t)

                tau = tau_mtq + tau_rw
                samples.append(np.dot(tau, d))
            max_torques.append(max(samples))

        max_torques = np.array(max_torques) * 1e6  # μNm

        # Normalize for visualization
        max_val = max_torques.max()
        min_val = max_torques.min()

        # Plot WCDTA "sphere" (distorted by actuation limits)
        colors = plt.cm.viridis((max_torques - min_val) / (max_val - min_val + 1e-10))

        for i, d in enumerate(directions):
            r = max_torques[i] / max_val  # Normalize radius
            ax.scatter(d[0]*r, d[1]*r, d[2]*r, c=[colors[i]], s=20, alpha=0.8)

        # Add unit sphere wireframe for reference
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color='gray', alpha=0.1, linewidth=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name}\nMin: {min_val:.1f} μNm, Max: {max_val:.1f} μNm', fontsize=10)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)

    fig.suptitle('Worst-Case Directional Torque Authority (WCDTA)\nRadius = max torque in that direction', fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_wcdta_sphere')


def gen_actuator_failure_generalized(cfg: Config, output_dir: Path):
    """Show system response before/during/after actuator failure (for Generalized paper)."""
    print("\n  [Generalized] Actuator Failure Response...")

    duration_s = 2000
    failure_time_s = 700

    sat = create_sat_3p1()
    orb, start_time = create_orbit(400, 51.6, 2.0, duration_s + 100, seed=42)

    q0 = normalize(np.array([0.5, 0.3, 0.1, np.sqrt(1-0.35)]))
    w0 = np.array([0.005, -0.003, 0.004])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(np.array([0, 0, 1]))

    cfg_run = Config(n_trials=1, duration_s=duration_s, dt=2.0)
    ctrl = create_ctrl_lp(sat, cfg_run)

    # Find RW index
    rw_idx = None
    for i, act in enumerate(sat.actuators):
        if isinstance(act, RW):
            rw_idx = i
            break

    # Run with failure
    result = run_simulation(sat, ctrl, goal, orb, start_time, cfg_run, x0,
                           disable_actuator_at=(failure_time_s, rw_idx) if rw_idx else None)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    t = result['time']

    # Panel 1: Pointing Error
    ax = axes[0]
    ax.plot(t, result['error_deg'], 'b-', linewidth=1.5)
    ax.axvline(failure_time_s, color='red', linestyle='--', linewidth=2, label='RW Failure')
    ax.set_ylabel('Pointing Error (deg)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Annotate phases
    ax.annotate('3+1 Mode\n(Full capability)', (200, 20), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.annotate('3+0 Mode\n(Graceful degradation)', (1300, 40), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Panel 2: RW torque command
    ax = axes[1]
    if rw_idx is not None:
        ax.plot(t, result['u'][:, rw_idx] * 1e6, 'g-', linewidth=1.5)
    ax.axvline(failure_time_s, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('RW Command\n(μNm)')
    ax.grid(True, alpha=0.3)
    ax.annotate('RW active', (300, 50), fontsize=9)
    ax.annotate('RW disabled', (1200, 0), fontsize=9)

    # Panel 3: MTQ activity (sum of absolute dipole moments)
    ax = axes[2]
    n_mtq = len([a for a in sat.actuators if not isinstance(a, RW)])
    mtq_sum = np.sum(np.abs(result['u'][:, :n_mtq]), axis=1)
    ax.plot(t, mtq_sum, 'm-', linewidth=1.5)
    ax.axvline(failure_time_s, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('MTQ Activity\n(Am²)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Graceful Degradation: Automatic Fallback After RW Failure', fontsize=11)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_actuator_failure_response')


# -----------------------------------------------------------------------------
# PLANNER PAPER FIGURES
# -----------------------------------------------------------------------------

def gen_planner_pd_baseline(cfg: Config, output_dir: Path):
    """Generate PD baseline results for planner comparison."""
    print("\n  [Planner] PD Baseline...")
    
    configs = {
        '3+0 PD': (create_sat_3p0, create_ctrl_lovera),
        '3+1 PD': (create_sat_3p1, create_ctrl_lp),
    }
    
    results = {}
    for name, (sat_f, ctrl_f) in configs.items():
        results[name] = run_monte_carlo(sat_f, ctrl_f, cfg, name)
    
    # Trajectories
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.arange(0, cfg.duration_s, cfg.dt)
    for i, (name, res) in enumerate(results.items()):
        for traj in res['trajectories']:
            ax.plot(times[:len(traj)], traj, color=COLORS[i], alpha=0.15, linewidth=0.5)
        mean_traj = np.mean(res['trajectories'], axis=0)
        ax.plot(times[:len(mean_traj)], mean_traj, color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 200)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('PD Control Baseline (for Planner Comparison)')
    save_fig(fig, output_dir, 'fig_pd_baseline_trajectories')
    
    # Save data
    with open(output_dir / 'data_pd_baseline.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'}
                   for k, v in results.items()}, f, indent=2)
    
    return results


def gen_planner_altro_comparison(cfg: Config, output_dir: Path):
    """Generate ALTRO vs PD comparison using real planner."""
    print("\n  [Planner] ALTRO vs PD Comparison...")
    
    ALTRO_AVAILABLE = False
    try:
        # Use same import path as debug_planner.py to avoid pybind11 conflict
        import trajectory_planner.build.tplaunch as tplaunch
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        ALTRO_AVAILABLE = True
    except ImportError as e:
        print(f"    WARNING: ALTRO planner not available ({e}), generating placeholder")
    except Exception as e:
        print(f"    WARNING: ALTRO error ({type(e).__name__}: {e}), generating placeholder")
    
    if not ALTRO_AVAILABLE:
        # Generate placeholder with expected results from thesis
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        t = np.linspace(0, 500, 250)
        
        # MTQ-only
        ax = axes[0]
        np.random.seed(42)
        for i in range(5):
            pd_err = 90 * np.exp(-t / 400) + 15 + 10 * np.random.randn(len(t)) * np.exp(-t/200)
            plan_err = 90 * np.exp(-t / 100) + 3 + 5 * np.random.randn(len(t)) * np.exp(-t/100)
            ax.plot(t, np.clip(pd_err, 0.1, 200), color=COLORS[0], alpha=0.2)
            ax.plot(t, np.clip(plan_err, 0.1, 200), color=COLORS[1], alpha=0.2)
        ax.plot([], [], color=COLORS[0], label='PD')
        ax.plot([], [], color=COLORS[1], label='ALTRO')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pointing Error (deg)')
        ax.set_yscale('log')
        ax.set_ylim(0.1, 200)
        ax.legend()
        ax.set_title('MTQ-Only (3+0) - PLACEHOLDER')
        ax.grid(True, alpha=0.3, which='both')
        
        # 3+1
        ax = axes[1]
        for i in range(5):
            pd_err = 90 * np.exp(-t / 150) + 2 + 3 * np.random.randn(len(t)) * np.exp(-t/100)
            plan_err = 90 * np.exp(-t / 50) + 0.1 + 0.3 * np.random.randn(len(t)) * np.exp(-t/50)
            ax.plot(t, np.clip(pd_err, 0.01, 200), color=COLORS[0], alpha=0.2)
            ax.plot(t, np.clip(plan_err, 0.01, 200), color=COLORS[1], alpha=0.2)
        ax.plot([], [], color=COLORS[0], label='PD')
        ax.plot([], [], color=COLORS[1], label='ALTRO')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pointing Error (deg)')
        ax.set_yscale('log')
        ax.set_ylim(0.01, 200)
        ax.legend()
        ax.set_title('Hybrid (3+1) - PLACEHOLDER')
        ax.grid(True, alpha=0.3, which='both')
        
        fig.suptitle('ALTRO vs PD (Thesis Expected Results)', y=1.02)
        fig.tight_layout()
        save_fig(fig, output_dir, 'fig_altro_vs_pd')
        return
    
    # Real ALTRO implementation
    cfg_altro = Config(n_trials=min(cfg.n_trials, 10), duration_s=500, dt=2.0)
    
    results_pd = {}
    results_altro = {}
    
    for sat_name, sat_f in [('3+0', create_sat_thesis_mtq), ('3+1', create_sat_thesis_3p1)]:
        # PD baseline
        if sat_name == '3+0':
            ctrl_f = create_ctrl_lovera
        else:
            ctrl_f = create_ctrl_lp
        results_pd[sat_name] = run_monte_carlo(sat_f, ctrl_f, cfg_altro, f'{sat_name} PD')
        
        # ALTRO
        altro_errors = []
        altro_trajectories = []
        
        for trial in tqdm(range(cfg_altro.n_trials), desc=f"  ALTRO {sat_name}"):
            seed = trial * 1000
            np.random.seed(seed)
            
            try:
                sat = sat_f()
                orb, start_time = create_orbit(cfg_altro.altitude_km, cfg_altro.inclination_deg,
                                                cfg_altro.dt, cfg_altro.duration_s, seed)
                
                planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10, dt_tvlqr=1)
                controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)
                
                w0 = normalize(np.random.randn(3)) * np.random.uniform(0.001, 0.01)
                q0 = normalize(np.random.randn(4))
                n_rw = len(sat.rw_actuators)
                h0 = np.zeros(n_rw)
                x0 = np.concatenate([w0, q0, h0])
                
                goal = ECI_Goal(normalize(np.random.randn(3)))
                os0 = orb.get_os(start_time)
                goals = GoalList({start_time: goal})
                
                # Calculate trajectory
                trajectory = controller.calculate_trajectory(
                    t_start=start_time,
                    duration=cfg_altro.duration_s,
                    x_0=x0,
                    os_0=os0,
                    goals=goals,
                    verbose=False,
                )
                
                if trajectory is not None and not np.any(np.isnan(trajectory.states)):
                    # Compute errors from trajectory
                    errors = []
                    for i in range(trajectory.states.shape[1]):
                        q = trajectory.states[3:7, i]
                        err = compute_pointing_error(q, goal, cfg_altro.body_boresight, os0)
                        errors.append(err)
                    
                    altro_errors.append(np.mean(errors[-50:]))
                    if trial < 10:
                        altro_trajectories.append(errors)
            except Exception as e:
                print(f"    Trial {trial} failed: {e}")
                continue
        
        if altro_errors:
            results_altro[sat_name] = {
                'errors': altro_errors,
                'trajectories': altro_trajectories,
                'mean': np.mean(altro_errors),
                'pct_10deg': 100 * np.sum(np.array(altro_errors) < 10) / len(altro_errors),
            }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    times = np.arange(0, cfg_altro.duration_s, cfg_altro.dt)
    
    for ax_idx, sat_name in enumerate(['3+0', '3+1']):
        ax = axes[ax_idx]
        
        # PD
        if sat_name in results_pd:
            for traj in results_pd[sat_name]['trajectories']:
                ax.plot(times[:len(traj)], traj, color=COLORS[0], alpha=0.15, linewidth=0.5)
            mean_pd = np.mean(results_pd[sat_name]['trajectories'], axis=0)
            ax.plot(times[:len(mean_pd)], mean_pd, color=COLORS[0], linewidth=2, label='PD')
        
        # ALTRO
        if sat_name in results_altro and results_altro[sat_name]['trajectories']:
            for traj in results_altro[sat_name]['trajectories']:
                ax.plot(range(len(traj)), traj, color=COLORS[1], alpha=0.15, linewidth=0.5)
            mean_altro = np.mean(results_altro[sat_name]['trajectories'], axis=0)
            ax.plot(range(len(mean_altro)), mean_altro, color=COLORS[1], linewidth=2, label='ALTRO')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pointing Error (deg)')
        ax.set_yscale('log')
        ax.set_ylim(0.01 if sat_name == '3+1' else 0.1, 200)
        ax.legend()
        ax.set_title(f'{sat_name}')
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('ALTRO vs PD Control', y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_altro_vs_pd')


def gen_multi_target_sequence(cfg: Config, output_dir: Path):
    """Generate multi-target sequence figure."""
    print("\n  [Planner] Multi-Target Sequence...")
    
    # Use regular controller with goal switching
    cfg_multi = Config(n_trials=1, duration_s=500, dt=1.0)
    
    sat = create_sat_3p1()
    orb, start_time = create_orbit(cfg_multi.altitude_km, cfg_multi.inclination_deg,
                                    cfg_multi.dt, cfg_multi.duration_s, seed=42)
    
    q0 = normalize(np.array([0.5, 0.3, 0.1, np.sqrt(1-0.25-0.09-0.01)]))
    w0 = np.array([0.005, -0.003, 0.004])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    # Three target windows
    targets = [
        (0, 150, np.array([0, 0, 1])),      # Target 1: nadir
        (180, 380, np.array([1, 0, 0])),    # Target 2: ram
        (420, 500, np.array([0, 1, 0])),    # Target 3: cross-track
    ]
    
    ctrl = create_ctrl_lp(sat, cfg_multi)
    
    N = int(cfg_multi.duration_s / cfg_multi.dt)
    time_hist = np.zeros(N)
    error_hist = np.zeros(N)
    target_hist = np.zeros(N)
    
    x = x0.copy()
    t = 0
    
    for i in range(N):
        # Find current goal
        current_goal = None
        current_target = 0
        for tgt_idx, (t_start, t_end, goal_vec) in enumerate(targets):
            if t_start <= t <= t_end:
                current_goal = ECI_Goal(goal_vec)
                current_target = tgt_idx + 1
                break
        
        if current_goal is None:
            current_goal = ECI_Goal(np.array([0, 0, 1]))  # Default
        
        J2000 = start_time + t * TimeConstants.sec2cent
        os = orb.get_os(J2000)
        sens = sat.sensor_readings(x=x, os=os)
        
        u = ctrl.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os, goal=current_goal)
        error = compute_pointing_error(x[3:7], current_goal, cfg_multi.body_boresight, os)
        
        time_hist[i] = t
        error_hist[i] = error
        target_hist[i] = current_target
        
        # Propagate
        t_next = t + cfg_multi.dt
        os_next = orb.get_os(start_time + t_next * TimeConstants.sec2cent)
        
        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0, cfg_multi.dt),
            y0=x,
            method='RK45',
            args=(u, os, os_next),
            rtol=1e-6, atol=1e-6,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t = t_next
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_hist, error_hist, color=COLORS[0], linewidth=1.5)
    
    # Shade target windows
    colors_targets = [COLORS[2], COLORS[3], COLORS[4]]
    for tgt_idx, (t_start, t_end, _) in enumerate(targets):
        ax.axvspan(t_start, t_end, alpha=0.2, color=colors_targets[tgt_idx])
        ax.annotate(f'Target {tgt_idx+1}', ((t_start + t_end)/2, 1), ha='center', fontsize=9)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 100)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='1° threshold')
    ax.legend(loc='upper right')
    ax.set_title('Multi-Target Sequence (3 Targets, 500s)')
    ax.grid(True, alpha=0.3, which='both')
    save_fig(fig, output_dir, 'fig_multi_target_sequence')


def gen_altro_timing_benchmark(cfg: Config, output_dir: Path):
    """Generate ALTRO solve time benchmark figure."""
    print("\n  [Planner] ALTRO Timing Benchmark...")

    try:
        import tplaunch
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        ALTRO_AVAILABLE = True
    except ImportError:
        print("    WARNING: ALTRO planner not available, generating placeholder")
        ALTRO_AVAILABLE = False

    if not ALTRO_AVAILABLE:
        # Generate placeholder with expected results
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Panel 1: Solve time histogram (placeholder)
        ax = axes[0]
        np.random.seed(42)
        # Simulated solve times based on thesis expectations
        solve_times = np.random.lognormal(mean=2.5, sigma=0.5, size=100)  # ~10-30s typical
        ax.hist(solve_times, bins=20, color=COLORS[0], edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(solve_times), color='red', linestyle='--', label=f'Mean: {np.mean(solve_times):.1f}s')
        ax.axvline(np.percentile(solve_times, 95), color='orange', linestyle=':', label=f'95th pct: {np.percentile(solve_times, 95):.1f}s')
        ax.set_xlabel('Solve Time (s)')
        ax.set_ylabel('Count')
        ax.set_title('ALTRO Solve Time Distribution\n(PLACEHOLDER - run with ALTRO built)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Horizon scaling (placeholder)
        ax = axes[1]
        horizons = np.array([100, 250, 500, 750, 1000])
        # Simulated scaling: roughly quadratic in horizon length
        times_mean = 2 + (horizons / 100) ** 1.5 * 3
        times_std = times_mean * 0.2
        ax.errorbar(horizons, times_mean, yerr=times_std, marker='o', capsize=5,
                    color=COLORS[0], linewidth=1.5, markersize=8)
        ax.set_xlabel('Trajectory Horizon (s)')
        ax.set_ylabel('Solve Time (s)')
        ax.set_title('Solve Time vs Horizon\n(PLACEHOLDER)')
        ax.grid(True, alpha=0.3)

        fig.suptitle('ALTRO Timing Benchmarks (Placeholder - ALTRO not built)', y=1.02)
        fig.tight_layout()
        save_fig(fig, output_dir, 'fig_altro_timing')

        # Save placeholder data
        placeholder_data = {
            'note': 'PLACEHOLDER - ALTRO not available',
            'expected_mean_solve_time_s': 15.0,
            'expected_95pct_solve_time_s': 40.0,
        }
        with open(output_dir / 'data_altro_timing_placeholder.json', 'w') as f:
            json.dump(placeholder_data, f, indent=2)
        return

    # Real ALTRO timing benchmark
    n_trials = min(cfg.n_trials, 20)  # Limit for timing tests
    horizons = [100, 250, 500]
    dt_tp = 10  # Planning timestep

    solve_times_by_horizon = {h: [] for h in horizons}

    for horizon in tqdm(horizons, desc="  Horizons"):
        for trial in range(n_trials):
            seed = trial * 1000
            np.random.seed(seed)

            try:
                sat = create_sat_thesis_3p1()
                orb, start_time = create_orbit(400, 51.6, 1.0, horizon + 100, seed)

                planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=dt_tp, dt_tvlqr=1)
                controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)

                w0 = normalize(np.random.randn(3)) * 0.005
                q0 = normalize(np.random.randn(4))
                h0 = np.zeros(len(sat.rw_actuators))
                x0 = np.concatenate([w0, q0, h0])

                goal = ECI_Goal(normalize(np.random.randn(3)))
                os0 = orb.get_os(start_time)
                goals = GoalList({start_time: goal})

                import time as time_module
                t_start_solve = time_module.time()

                trajectory = controller.calculate_trajectory(
                    t_start=start_time,
                    duration=horizon,
                    x_0=x0,
                    os_0=os0,
                    goals=goals,
                    verbose=False,
                )

                solve_time = time_module.time() - t_start_solve

                if trajectory is not None and not np.any(np.isnan(trajectory.states)):
                    solve_times_by_horizon[horizon].append(solve_time)

            except Exception as e:
                print(f"    Horizon {horizon}, trial {trial} failed: {e}")
                continue

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Histogram for longest horizon
    ax = axes[0]
    if solve_times_by_horizon[500]:
        times = solve_times_by_horizon[500]
        ax.hist(times, bins=15, color=COLORS[0], edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(times):.1f}s')
        ax.axvline(np.percentile(times, 95), color='orange', linestyle=':',
                   label=f'95th pct: {np.percentile(times, 95):.1f}s')
        ax.legend()
    ax.set_xlabel('Solve Time (s)')
    ax.set_ylabel('Count')
    ax.set_title('ALTRO Solve Time Distribution (500s horizon)')
    ax.grid(True, alpha=0.3)

    # Panel 2: Horizon scaling
    ax = axes[1]
    means = [np.mean(solve_times_by_horizon[h]) if solve_times_by_horizon[h] else 0 for h in horizons]
    stds = [np.std(solve_times_by_horizon[h]) if solve_times_by_horizon[h] else 0 for h in horizons]
    ax.errorbar(horizons, means, yerr=stds, marker='o', capsize=5,
                color=COLORS[0], linewidth=1.5, markersize=8)
    ax.set_xlabel('Trajectory Horizon (s)')
    ax.set_ylabel('Solve Time (s)')
    ax.set_title('Solve Time vs Horizon')
    ax.grid(True, alpha=0.3)

    fig.suptitle('ALTRO Timing Benchmarks', y=1.02)
    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_altro_timing')

    # Save data
    timing_data = {
        'horizons': horizons,
        'solve_times': {str(h): solve_times_by_horizon[h] for h in horizons},
        'means': {str(h): np.mean(solve_times_by_horizon[h]) if solve_times_by_horizon[h] else None for h in horizons},
        'stds': {str(h): np.std(solve_times_by_horizon[h]) if solve_times_by_horizon[h] else None for h in horizons},
    }
    with open(output_dir / 'data_altro_timing.json', 'w') as f:
        json.dump(timing_data, f, indent=2)


def gen_tvlqr_tracking(cfg: Config, output_dir: Path):
    """Generate TVLQR tracking visualization showing planned vs actual trajectory."""
    print("\n  [Planner] TVLQR Tracking Visualization...")

    try:
        import tplaunch
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        ALTRO_AVAILABLE = True
    except ImportError:
        print("    WARNING: ALTRO not available, generating placeholder")
        ALTRO_AVAILABLE = False

    if not ALTRO_AVAILABLE:
        # Placeholder showing expected behavior
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        t = np.linspace(0, 500, 500)

        # Panel 1: Planned vs actual quaternion component
        ax = axes[0]
        np.random.seed(42)
        q_planned = 0.7 + 0.3 * np.exp(-t / 100)
        q_actual = q_planned + 0.02 * np.random.randn(len(t)) * np.exp(-t / 200)
        ax.plot(t, q_planned, 'b--', linewidth=1.5, label='Planned')
        ax.plot(t, q_actual, 'r-', linewidth=1, alpha=0.8, label='Actual (TVLQR)')
        ax.set_ylabel('Quaternion q₀')
        ax.legend()
        ax.set_title('TVLQR Tracking: Planned vs Actual Trajectory\n(PLACEHOLDER)')
        ax.grid(True, alpha=0.3)

        # Panel 2: Tracking error
        ax = axes[1]
        tracking_error = np.abs(q_actual - q_planned) * 180 / np.pi
        ax.plot(t, tracking_error, 'g-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Tracking Error (deg)')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_fig(fig, output_dir, 'fig_tvlqr_tracking')
        return

    # Real TVLQR tracking visualization
    sat = create_sat_thesis_3p1()
    orb, start_time = create_orbit(400, 51.6, 1.0, 600, seed=42)

    planner_settings = PlannerSettings(est_sat=sat, bdot_on=0, dt_tp=10, dt_tvlqr=1)
    controller = Plan_and_Track_LQR(est_sat=sat, planner_settings=planner_settings)

    q0 = normalize(np.array([0.5, 0.3, 0.1, np.sqrt(1-0.35)]))
    w0 = np.array([0.005, -0.003, 0.004])
    h0 = np.zeros(len(sat.rw_actuators))
    x0 = np.concatenate([w0, q0, h0])

    goal = ECI_Goal(np.array([0, 0, 1]))
    os0 = orb.get_os(start_time)
    goals = GoalList({start_time: goal})

    # Calculate trajectory
    trajectory = controller.calculate_trajectory(
        t_start=start_time,
        duration=500,
        x_0=x0,
        os_0=os0,
        goals=goals,
        verbose=False,
    )

    if trajectory is None:
        print("    ALTRO trajectory planning failed")
        return

    # Now simulate with TVLQR tracking
    cfg_sim = Config(n_trials=1, duration_s=500, dt=1.0)
    result = run_simulation(sat, controller, goal, orb, start_time, cfg_sim, x0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Panel 1: Planned vs actual pointing error
    ax = axes[0]
    t_plan = np.arange(trajectory.states.shape[1])
    plan_errors = []
    for i in range(trajectory.states.shape[1]):
        q = trajectory.states[3:7, i]
        err = compute_pointing_error(q, goal, cfg_sim.body_boresight, os0)
        plan_errors.append(err)

    ax.plot(t_plan, plan_errors, 'b--', linewidth=1.5, label='Planned')
    ax.plot(result['time'], result['error_deg'], 'r-', linewidth=1, alpha=0.8, label='Actual (TVLQR)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('TVLQR Tracking: Planned vs Actual')
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Tracking error
    ax = axes[1]
    # Interpolate planned to match actual times
    plan_interp = np.interp(result['time'], t_plan, plan_errors)
    tracking_error = np.abs(result['error_deg'] - plan_interp)
    ax.plot(result['time'], tracking_error, 'g-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (deg)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, output_dir, 'fig_tvlqr_tracking')


# -----------------------------------------------------------------------------
# PACKAGE PAPER FIGURES
# -----------------------------------------------------------------------------

def gen_controller_comparison(cfg: Config, output_dir: Path):
    """Generate controller comparison (Lovera vs Wisniewski)."""
    print("\n  [Package] Controller Comparison...")
    
    configs = {
        'Lovera': (create_sat_3p0, create_ctrl_lovera),
        'Wisniewski': (create_sat_3p0, create_ctrl_wisniewski),
    }
    
    results = {}
    for name, (sat_f, ctrl_f) in configs.items():
        results[name] = run_monte_carlo(sat_f, ctrl_f, cfg, name)
    
    # Trajectories
    fig, ax = plt.subplots(figsize=(6, 4))
    times = np.arange(0, cfg.duration_s, cfg.dt)
    for i, (name, res) in enumerate(results.items()):
        for traj in res['trajectories']:
            ax.plot(times[:len(traj)], traj, color=COLORS[i], alpha=0.15, linewidth=0.5)
        mean_traj = np.mean(res['trajectories'], axis=0)
        ax.plot(times[:len(mean_traj)], mean_traj, color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 200)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_title('MTQ Controller Comparison')
    save_fig(fig, output_dir, 'fig_controller_comparison')
    
    # Save data
    with open(output_dir / 'data_controller_comparison.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'trajectories'}
                   for k, v in results.items()}, f, indent=2)
    
    return results


def gen_quickstart_demo(output_dir: Path):
    """Generate quickstart demo figure showing 3+3 architecture converging quickly."""
    print("\n  [Package] Quickstart Demo...")
    
    # Use 3+3 (4 RW pyramid) for demo since it converges quickly
    cfg = Config(n_trials=1, duration_s=300, dt=1.0)
    sat = create_sat_3p3()  # 3MTQ + 3RW (pyramid) - fully actuated
    orb, start_time = create_orbit(cfg.altitude_km, cfg.inclination_deg, cfg.dt, cfg.duration_s, seed=42)
    
    # Start with significant attitude error (~60 deg from target)
    q0 = normalize(np.array([0.5, 0.3, 0.1, np.sqrt(1-0.25-0.09-0.01)]))
    w0 = np.array([0.005, -0.003, 0.004])  # Moderate initial rate
    # 3+3 has 3 RWs  
    h0 = np.zeros(len(sat.rw_actuators))
    x0 = np.concatenate([w0, q0, h0])
    
    # Use ECI goal for pointing
    goal_vec = normalize(np.array([0, 0, 1]))
    goal = ECI_Goal(goal_vec)
    ctrl = create_ctrl_lp(sat, cfg)
    
    result = run_simulation(sat, ctrl, goal, orb, start_time, cfg, x0)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result['time'], result['error_deg'], color=COLORS[0], linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_title('Quickstart Example: Inertial Pointing (3MTQ+4RW)')
    ax.set_ylim(0, max(100, np.max(result['error_deg']) * 1.1))
    ax.grid(True, alpha=0.3)
    
    # Add code snippet
    textstr = 'sat = create_3_3_beavercube2()\nctrl = MTQ_w_RW_LP(sat)\ngoal = ECI_Goal([0,0,1])'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Add final error annotation
    final_err = result['final_error_deg']
    ax.annotate(f'Final: {final_err:.1f}°', xy=(result['time'][-1], result['error_deg'][-1]),
                xytext=(-50, 20), textcoords='offset points', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    save_fig(fig, output_dir, 'fig_quickstart_demo')


# =============================================================================
# MAIN RUNNER
# =============================================================================

FIGURES = {
    '3p1': {
        'architecture': ('Architecture Comparison', gen_3p1_architecture_comparison),
        'torque_envelope': ('Torque Envelope', gen_3p1_torque_envelope),
        'momentum': ('Momentum Management', gen_3p1_momentum_management),
        'degradation': ('Graceful Degradation', gen_3p1_graceful_degradation),
    },
    'generalized': {
        'lp_vs_qp': ('LP vs QP Comparison', gen_lp_vs_qp_comparison),
        'direction': ('Direction Preservation', gen_direction_preservation),
        'versatility': ('Framework Versatility', gen_framework_versatility),
        'controllability': ('Controllability Analysis', gen_controllability_analysis),
        'polytope': ('Torque Polytope Evolution', gen_torque_polytope_evolution),
        'desaturation': ('Desaturation Scheduling', gen_desaturation_scheduling),
        'wcdta': ('WCDTA Sphere', gen_wcdta_sphere),
        'failure': ('Actuator Failure Response', gen_actuator_failure_generalized),
    },
    'planner': {
        'pd_baseline': ('PD Baseline', gen_planner_pd_baseline),
        'altro': ('ALTRO vs PD', gen_planner_altro_comparison),
        'multi_target': ('Multi-Target Sequence', gen_multi_target_sequence),
        'timing': ('ALTRO Timing Benchmark', gen_altro_timing_benchmark),
        'tvlqr': ('TVLQR Tracking', gen_tvlqr_tracking),
    },
    'package': {
        'controllers': ('Controller Comparison', gen_controller_comparison),
        'quickstart': ('Quickstart Demo', lambda cfg, out: gen_quickstart_demo(out)),
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate All Paper Figures")
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--paper', type=str, choices=['3p1', 'generalized', 'planner', 'package'])
    parser.add_argument('--figure', type=str, help='Specific figure to generate')
    parser.add_argument('--quick', action='store_true', help='Quick mode (10 trials, 200s)')
    parser.add_argument('--full', action='store_true', help='Full mode (100 trials, 1000s)')
    parser.add_argument('--output-dir', type=str, default='./paper_figures')
    parser.add_argument('--list', action='store_true', help='List available figures')
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Figures:")
        for paper, figs in FIGURES.items():
            print(f"\n  {paper}:")
            for fig_id, (name, _) in figs.items():
                print(f"    {fig_id}: {name}")
        return
    
    cfg = Config.quick() if args.quick else (Config.full() if args.full else Config())
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*60)
    print("  Generate All Paper Figures")
    print(f"  Mode: {'Quick' if args.quick else ('Full' if args.full else 'Default')}")
    print(f"  Trials: {cfg.n_trials}, Duration: {cfg.duration_s}s")
    print(f"  Output: {output_dir}")
    print("="*60)
    
    start_time = time.time()
    
    # Determine which figures to generate
    if args.figure:
        # Find the figure across all papers
        for paper, figs in FIGURES.items():
            if args.figure in figs:
                paper_dir = output_dir / paper
                paper_dir.mkdir(parents=True, exist_ok=True)
                name, func = figs[args.figure]
                func(cfg, paper_dir)
                break
        else:
            print(f"Unknown figure: {args.figure}")
            return
    elif args.paper:
        paper_dir = output_dir / args.paper
        paper_dir.mkdir(parents=True, exist_ok=True)
        for fig_id, (name, func) in FIGURES[args.paper].items():
            func(cfg, paper_dir)
    elif args.all:
        for paper, figs in FIGURES.items():
            paper_dir = output_dir / paper
            paper_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"  Paper: {paper}")
            print(f"{'='*60}")
            for fig_id, (name, func) in figs.items():
                func(cfg, paper_dir)
    else:
        print("Specify --all, --paper, or --figure")
        return
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"  Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
