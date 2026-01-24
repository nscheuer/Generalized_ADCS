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
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../../trajectory_planner/build")))

# ADCS imports
from ADCS.CONOPS.goals import ECI_Goal, Fixed_Attitude_Goal, No_Goal
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
    
    # Controller gains
    mtq_p_gain: float = 0.001
    mtq_d_gain: float = 0.005
    rw_p_gain: float = 0.0001
    rw_d_gain: float = 0.001
    rw_c_gain: float = 0.001
    
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
                 seed: int = 0) -> Tuple[Orbit, float]:
    """Create orbit with random starting position."""
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
    os0 = Orbital_State(ephem=ephem, J2000=start_time, R=R, V=V)
    end_time = start_time + (duration_s + 100) * TimeConstants.sec2cent
    
    orb = Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=True, verbose=False)
    return orb, start_time


def compute_pointing_error(q: np.ndarray, goal, boresight: np.ndarray, os) -> float:
    """Compute pointing error in degrees."""
    R = rot_mat(q)
    boresight_eci = R @ boresight
    
    if isinstance(goal, ECI_Goal):
        goal_vec, _ = goal.to_ref(os)
        goal_vec = goal_vec / np.linalg.norm(goal_vec)
    elif isinstance(goal, Fixed_Attitude_Goal):
        q_goal = goal.q_goal
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

def create_sat_3p0():
    return create_beavercube1_cubesat(estimated=False)

def create_sat_3p1():
    return create_beavercube2_cubesat(estimated=False)

def create_sat_3p3():
    return create_3_3_beavercube2_cubesat(estimated=False)

def create_sat_thesis_mtq():
    """Create MTQ-only satellite matching thesis Table 7.2."""
    from ADCS.satellite_hardware.sensors import MTM, Gyro
    J = np.diag([0.005256, 0.04939, 0.04939])
    mtqs = [
        MTQ(axis=np.array([1, 0, 0]), max_m=0.19),
        MTQ(axis=np.array([0, 1, 0]), max_m=0.57),
        MTQ(axis=np.array([0, 0, 1]), max_m=0.57),
    ]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    gyros = [Gyro(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=J, actuators=mtqs, sensors=mtms+gyros, boresight=np.array([1,0,0]))

def create_sat_thesis_3p1():
    """Create 3+1 satellite matching thesis Table 7.2."""
    from ADCS.satellite_hardware.sensors import MTM, Gyro
    J = np.diag([0.005256, 0.04939, 0.04939])
    mtqs = [
        MTQ(axis=np.array([1, 0, 0]), max_m=0.19),
        MTQ(axis=np.array([0, 1, 0]), max_m=0.57),
        MTQ(axis=np.array([0, 0, 1]), max_m=0.57),
    ]
    rw = RW(axis=np.array([0, 1, 0]), max_torque=0.0002, J=2e-6, h=0.0, h_max=0.002)
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    gyros = [Gyro(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=J, actuators=mtqs+[rw], sensors=mtms+gyros, boresight=np.array([1,0,0]))


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
        
        if record_torques and hasattr(controller, 'tau_ref'):
            tau_des = controller.tau_ref if controller.tau_ref is not None else np.zeros(3)
            tau_ach = sat.tau_from_u(u, os) if hasattr(sat, 'tau_from_u') else np.zeros(3)
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
        n_mtq = len(sat.mtq_actuators)
        n_rw = len(sat.rw_actuators)
        
        # Sample MTQ commands
        for _ in range(n_samples):
            m = np.zeros(3)
            for j, mtq in enumerate(sat.mtq_actuators):
                m += mtq.axis * np.random.uniform(-mtq.max_m, mtq.max_m)
            tau_mtq = np.cross(m, B_body)
            
            tau_rw = np.zeros(3)
            for rw in sat.rw_actuators:
                tau_rw += rw.axis * np.random.uniform(-rw.max_torque, rw.max_torque)
            
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
    """Generate direction preservation figure showing LP vs QP torque alignment."""
    print("\n  [Generalized] Direction Preservation Analysis...")
    
    # Run single detailed simulation for each allocator
    cfg_single = Config(n_trials=1, duration_s=500, dt=1.0)
    
    sat_lp = create_sat_3p1()
    sat_qp = create_sat_3p1()
    orb, start_time = create_orbit(cfg_single.altitude_km, cfg_single.inclination_deg,
                                    cfg_single.dt, cfg_single.duration_s, seed=42)
    
    q0 = normalize(np.array([0.5, 0.3, 0.1, np.sqrt(1-0.25-0.09-0.01)]))
    w0 = np.array([0.005, -0.003, 0.004])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    goal = ECI_Goal(np.array([0, 0, 1]))
    
    ctrl_lp = create_ctrl_lp(sat_lp, cfg_single)
    ctrl_qp = create_ctrl_qp(sat_qp, cfg_single)
    
    res_lp = run_simulation(sat_lp, ctrl_lp, goal, orb, start_time, cfg_single, x0, record_torques=True)
    res_qp = run_simulation(sat_qp, ctrl_qp, goal, orb, start_time, cfg_single, x0.copy(), record_torques=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # LP: τ_des vs τ_ach scatter
    ax = axes[0]
    tau_des_lp = res_lp.get('tau_des', np.zeros((len(res_lp['time']), 3)))
    tau_ach_lp = res_lp.get('tau_ach', np.zeros((len(res_lp['time']), 3)))
    
    # Sample points for clarity
    idx = np.arange(0, len(tau_des_lp), 5)
    scale = 1e6  # Convert to μNm
    
    for i in idx[:50]:
        if np.linalg.norm(tau_des_lp[i]) > 1e-12:
            ax.arrow(0, 0, tau_des_lp[i, 0]*scale, tau_des_lp[i, 1]*scale,
                    head_width=0.5, color=COLORS[0], alpha=0.3, length_includes_head=True)
            ax.arrow(0, 0, tau_ach_lp[i, 0]*scale, tau_ach_lp[i, 1]*scale,
                    head_width=0.5, color=COLORS[2], alpha=0.5, length_includes_head=True)
    
    ax.set_xlabel(r'$\tau_x$ ($\mu$Nm)')
    ax.set_ylabel(r'$\tau_y$ ($\mu$Nm)')
    ax.set_title(f'LP Allocation\nMean direction error: {res_lp.get("mean_dir_error", 0):.2f}°')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # QP
    ax = axes[1]
    tau_des_qp = res_qp.get('tau_des', np.zeros((len(res_qp['time']), 3)))
    tau_ach_qp = res_qp.get('tau_ach', np.zeros((len(res_qp['time']), 3)))
    
    for i in idx[:50]:
        if np.linalg.norm(tau_des_qp[i]) > 1e-12:
            ax.arrow(0, 0, tau_des_qp[i, 0]*scale, tau_des_qp[i, 1]*scale,
                    head_width=0.5, color=COLORS[0], alpha=0.3, length_includes_head=True)
            ax.arrow(0, 0, tau_ach_qp[i, 0]*scale, tau_ach_qp[i, 1]*scale,
                    head_width=0.5, color=COLORS[1], alpha=0.5, length_includes_head=True)
    
    ax.set_xlabel(r'$\tau_x$ ($\mu$Nm)')
    ax.set_ylabel(r'$\tau_y$ ($\mu$Nm)')
    ax.set_title(f'QP Allocation\nMean direction error: {res_qp.get("mean_dir_error", 0):.2f}°')
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
    dir_err_lp = res_lp.get('dir_error_deg', np.zeros(len(res_lp['time'])))
    dir_err_qp = res_qp.get('dir_error_deg', np.zeros(len(res_qp['time'])))
    
    bins = np.linspace(0, 90, 50)
    ax.hist(dir_err_lp[dir_err_lp > 0], bins=bins, alpha=0.6, color=COLORS[0], label='LP', edgecolor='white')
    ax.hist(dir_err_qp[dir_err_qp > 0], bins=bins, alpha=0.6, color=COLORS[1], label='QP', edgecolor='white')
    ax.set_xlabel('Direction Error (deg)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Torque Direction Error Distribution')
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
    
    try:
        import tplaunch
        from ADCS.controller.plan_and_track_lqr import Plan_and_Track_LQR
        from ADCS.controller.helpers import PlannerSettings
        ALTRO_AVAILABLE = True
    except ImportError:
        print("    WARNING: ALTRO planner not available, generating placeholder")
        ALTRO_AVAILABLE = False
    
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
    """Generate quickstart demo figure."""
    print("\n  [Package] Quickstart Demo...")
    
    cfg = Config(n_trials=1, duration_s=300, dt=1.0)
    sat = create_sat_3p1()
    orb, start_time = create_orbit(cfg.altitude_km, cfg.inclination_deg, cfg.dt, cfg.duration_s, seed=42)
    
    q0 = normalize(np.array([0.5, 0.3, 0.1, np.sqrt(1-0.25-0.09-0.01)]))
    w0 = np.array([0.01, -0.005, 0.008])
    h0 = np.array([0.0])
    x0 = np.concatenate([w0, q0, h0])
    
    goal = ECI_Goal(np.array([0, 0, 1]))  # Nadir
    ctrl = create_ctrl_lp(sat, cfg)
    
    result = run_simulation(sat, ctrl, goal, orb, start_time, cfg, x0)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result['time'], result['error_deg'], color=COLORS[0], linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_title('Quickstart Example: Nadir Pointing (3+1 Hybrid)')
    ax.grid(True, alpha=0.3)
    
    # Add code snippet
    textstr = 'sat = create_beavercube2_cubesat()\nctrl = MTQ_w_RW_LP(sat)\ngoal = ECI_Goal([0,0,1])'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, family='monospace')
    
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
    },
    'planner': {
        'pd_baseline': ('PD Baseline', gen_planner_pd_baseline),
        'altro': ('ALTRO vs PD', gen_planner_altro_comparison),
        'multi_target': ('Multi-Target Sequence', gen_multi_target_sequence),
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
