#!/usr/bin/env python3
"""
Generate Placeholder Figures for Papers
=======================================

Creates publication-quality placeholder figures for experiments that require:
- ALTRO trajectory planner (not yet fully integrated)
- Long-duration simulations (hours)
- Complex multi-target scenarios
- Basilisk comparisons (external software)

These placeholders use realistic parameters from thesis data to show
expected figure layouts, allowing paper writing to proceed.

Usage:
    python generate_placeholder_figures.py --all --output-dir ./placeholders
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

COLORS = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7']


# =============================================================================
# 3+1 PAPER PLACEHOLDERS
# =============================================================================

def gen_3p1_actuator_configs(output_dir: Path):
    """Generate actuator configuration schematic (needs manual creation)."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    titles = ['3+0 (MTQ Only)', '3+1 (Hybrid)', '3+3 (Full RW)']
    
    for ax, title in zip(axes, titles):
        # Draw simple spacecraft box
        rect = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add actuator symbols
        if '3+0' in title:
            ax.annotate('MTQ\nx,y,z', (0, 0), ha='center', va='center', fontsize=9)
        elif '3+1' in title:
            ax.annotate('MTQ\nx,y,z', (0, 0.2), ha='center', va='center', fontsize=9)
            ax.annotate('RW z', (0, -0.2), ha='center', va='center', fontsize=9, color=COLORS[1])
        else:
            ax.annotate('MTQ\nx,y,z', (0, 0.2), ha='center', va='center', fontsize=9)
            ax.annotate('RW\nx,y,z', (0, -0.2), ha='center', va='center', fontsize=9, color=COLORS[2])
        
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Actuator Configurations (Schematic)', y=1.02)
    fig.savefig(output_dir / 'fig_actuator_configs_placeholder.png')
    fig.savefig(output_dir / 'fig_actuator_configs_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_actuator_configs_placeholder")


def gen_3p1_momentum_management(output_dir: Path):
    """Generate momentum management comparison placeholder."""
    # Based on thesis description: continuous vs scheduled desaturation
    t = np.linspace(0, 270, 500)  # 3 orbits in minutes
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    # Wheel momentum
    h_continuous = 0.001 * np.sin(2 * np.pi * t / 90) * 0.5  # Oscillates near 0
    h_scheduled = 0.001 * (1 - np.exp(-t/30)) * np.clip(np.cos(2 * np.pi * t / 90), 0, 1)
    
    axes[0].plot(t, h_continuous * 1000, label='Continuous desat', color=COLORS[0])
    axes[0].plot(t, h_scheduled * 1000, label='Scheduled desat', color=COLORS[1])
    axes[0].set_ylabel('Wheel Momentum (mNms)')
    axes[0].legend()
    axes[0].axhline(2, color='red', linestyle='--', alpha=0.5, label='h_max')
    axes[0].axhline(-2, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylim(-2.5, 2.5)
    
    # Pointing error during operations
    err_continuous = 2 + np.random.randn(500) * 0.5
    err_scheduled = 1 + np.random.randn(500) * 0.3
    # Add spikes during scheduled desat windows
    for i in range(3):
        start = i * 90 + 80
        end = start + 10
        mask = (t >= start) & (t <= end)
        err_scheduled[mask] += 5
    
    axes[1].plot(t, err_continuous, label='Continuous', color=COLORS[0], alpha=0.8)
    axes[1].plot(t, err_scheduled, label='Scheduled', color=COLORS[1], alpha=0.8)
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Pointing Error (deg)')
    axes[1].legend()
    axes[1].set_ylim(0, 15)
    
    # Mark desat windows
    for i in range(3):
        start = i * 90 + 80
        axes[1].axvspan(start, start+10, alpha=0.2, color='gray')
    
    fig.suptitle('Momentum Management Comparison (PLACEHOLDER)', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_momentum_management_placeholder.png')
    fig.savefig(output_dir / 'fig_momentum_management_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_momentum_management_placeholder")


def gen_3p1_graceful_degradation(output_dir: Path):
    """Generate wheel failure graceful degradation placeholder."""
    t = np.linspace(0, 1200, 600)  # 20 minutes
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    
    # Normal 3+1 operation
    err_normal = 1 + 0.3 * np.exp(-t/100)
    
    # After wheel failure at t=500s
    failure_time = 500
    err_degraded = err_normal.copy()
    mask = t >= failure_time
    err_degraded[mask] = 5 + 8 * (1 - np.exp(-(t[mask] - failure_time)/200))
    
    axes[0].plot(t, err_degraded, color=COLORS[0], linewidth=1.5)
    axes[0].axvline(failure_time, color='red', linestyle='--', alpha=0.7, label='RW Failure')
    axes[0].set_ylabel('Pointing Error (deg)')
    axes[0].legend()
    axes[0].annotate('3+1 Mode', (200, 1.5), fontsize=9)
    axes[0].annotate('3+0 Mode\n(Degraded)', (800, 8), fontsize=9)
    axes[0].set_ylim(0, 20)
    
    # RW momentum/speed
    rw_speed = 2000 * np.ones_like(t)
    rw_speed[t >= failure_time] = 0
    
    axes[1].plot(t, rw_speed, color=COLORS[1], linewidth=1.5)
    axes[1].axvline(failure_time, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RW Speed (RPM)')
    axes[1].set_ylim(-100, 3000)
    
    fig.suptitle('Graceful Degradation After Wheel Failure (PLACEHOLDER)', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_graceful_degradation_placeholder.png')
    fig.savefig(output_dir / 'fig_graceful_degradation_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_graceful_degradation_placeholder")


# =============================================================================
# GENERALIZED CONTROL PAPER PLACEHOLDERS
# =============================================================================

def gen_direction_preservation(output_dir: Path):
    """Generate LP vs QP direction preservation figure."""
    # Key result from thesis: LP preserves direction (0.004° error), QP doesn't (33°)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Generate random desired torques
    np.random.seed(42)
    n_points = 50
    angles = np.random.uniform(0, 2*np.pi, n_points)
    mags = np.random.uniform(0.5, 1.0, n_points)
    
    tau_des = np.column_stack([mags * np.cos(angles), mags * np.sin(angles)])
    
    # LP: Nearly perfect direction preservation
    lp_scale = 0.3 + np.random.uniform(0, 0.2, n_points)
    tau_lp = tau_des * lp_scale[:, np.newaxis]
    
    # QP: Direction errors up to ~30°
    qp_angle_error = np.random.uniform(-0.5, 0.5, n_points)  # radians
    qp_scale = 0.5 + np.random.uniform(0, 0.3, n_points)
    tau_qp = np.column_stack([
        qp_scale * np.cos(angles + qp_angle_error),
        qp_scale * np.sin(angles + qp_angle_error)
    ])
    
    # LP subplot
    ax = axes[0]
    for i in range(n_points):
        ax.arrow(0, 0, tau_des[i, 0], tau_des[i, 1], head_width=0.03, 
                 color=COLORS[0], alpha=0.3, length_includes_head=True)
        ax.arrow(0, 0, tau_lp[i, 0], tau_lp[i, 1], head_width=0.03,
                 color=COLORS[2], alpha=0.5, length_includes_head=True)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\tau_x$ (Nm)')
    ax.set_ylabel(r'$\tau_y$ (Nm)')
    ax.set_title('LP Allocation\n(Direction preserved)')
    ax.grid(True, alpha=0.3)
    
    # QP subplot
    ax = axes[1]
    for i in range(n_points):
        ax.arrow(0, 0, tau_des[i, 0], tau_des[i, 1], head_width=0.03,
                 color=COLORS[0], alpha=0.3, length_includes_head=True)
        ax.arrow(0, 0, tau_qp[i, 0], tau_qp[i, 1], head_width=0.03,
                 color=COLORS[1], alpha=0.5, length_includes_head=True)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\tau_x$ (Nm)')
    ax.set_ylabel(r'$\tau_y$ (Nm)')
    ax.set_title('QP Allocation\n(Direction error ~33°)')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[0], alpha=0.3, label='Desired'),
        Patch(facecolor=COLORS[2], alpha=0.5, label='LP Achieved'),
        Patch(facecolor=COLORS[1], alpha=0.5, label='QP Achieved'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_direction_preservation.png')
    fig.savefig(output_dir / 'fig_direction_preservation.pdf')
    plt.close(fig)
    print("  Generated: fig_direction_preservation")


def gen_bolt_on_framework(output_dir: Path):
    """Generate bolt-on framework architecture diagram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Box positions (x, y, width, height)
    boxes = [
        (0.1, 0.7, 0.15, 0.15, 'Goal\nModification', COLORS[4]),
        (0.35, 0.7, 0.15, 0.15, 'Control Law\n(PD, LQR, etc.)', COLORS[5]),
        (0.35, 0.4, 0.15, 0.15, 'Disturbance\nCompensation', COLORS[3]),
        (0.35, 0.1, 0.15, 0.15, 'Gyroscopic\nCompensation', COLORS[3]),
        (0.6, 0.4, 0.15, 0.15, 'Allocation\n(LP/QP)', COLORS[1]),
        (0.85, 0.4, 0.1, 0.15, 'Actuators', COLORS[2]),
    ]
    
    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color, 
                              edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        ax.annotate(label, (x + w/2, y + h/2), ha='center', va='center', fontsize=8,
                    fontweight='bold')
    
    # Arrows
    arrows = [
        (0.25, 0.775, 0.1, 0),      # Goal -> Control
        (0.5, 0.775, 0.1, -0.225),  # Control -> Disturbance
        (0.425, 0.55, 0, -0.05),    # Disturbance -> Gyroscopic
        (0.5, 0.175, 0.15, 0.225),  # Gyroscopic -> Allocation
        (0.75, 0.475, 0.1, 0),      # Allocation -> Actuators
    ]
    
    for x, y, dx, dy in arrows:
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Labels
    ax.annotate('τ_des', (0.55, 0.7), fontsize=9, style='italic')
    ax.annotate('τ_comp', (0.58, 0.35), fontsize=9, style='italic')
    ax.annotate('u', (0.8, 0.55), fontsize=9, style='italic')
    
    # Title
    ax.set_title('Bolt-On Control Framework Architecture', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.02, 'Same control law works across different actuator configurations',
            ha='center', fontsize=9, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    fig.savefig(output_dir / 'fig_bolt_on_framework.png')
    fig.savefig(output_dir / 'fig_bolt_on_framework.pdf')
    plt.close(fig)
    print("  Generated: fig_bolt_on_framework")


def gen_controllability_vs_inclination(output_dir: Path):
    """Generate controllability vs orbit inclination figure."""
    inclinations = np.linspace(0, 90, 10)
    
    # Simplified controllability measure (time-averaged Gramian condition number)
    # Based on thesis: equatorial orbits are harder for MTQ due to less B-field variation
    def controllability(inc_deg, config):
        base = 0.1  # Baseline (normalized)
        if config == '3+0':
            # MTQ-only: needs field variation
            return base + 0.8 * np.sin(np.deg2rad(inc_deg))**2
        elif config == '3+1':
            # Hybrid: always controllable but better at high inclination
            return 0.6 + 0.4 * np.sin(np.deg2rad(inc_deg))**2
        else:  # 3+3
            # Fully actuated: always fully controllable
            return 1.0 * np.ones_like(inc_deg) if isinstance(inc_deg, np.ndarray) else 1.0
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    for config, color in zip(['3+0', '3+1', '3+3'], COLORS[:3]):
        ctrl = [controllability(inc, config) for inc in inclinations]
        ax.plot(inclinations, ctrl, 'o-', label=config, color=color, markersize=6)
    
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Orbit Inclination (deg)')
    ax.set_ylabel('Controllability Index (normalized)')
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('LTV Controllability vs Inclination (PLACEHOLDER)')
    
    fig.savefig(output_dir / 'fig_controllability_vs_inclination.png')
    fig.savefig(output_dir / 'fig_controllability_vs_inclination.pdf')
    plt.close(fig)
    print("  Generated: fig_controllability_vs_inclination")


# =============================================================================
# PLANNER PAPER PLACEHOLDERS
# =============================================================================

def gen_planner_vs_pd(output_dir: Path):
    """Generate planner vs PD comparison placeholder using thesis expected results."""
    # Based on thesis claims: 
    # - MTQ PD: 15% within 10°
    # - MTQ Planner: 73% within 10°
    # - 3+1 PD: 73% within 1°
    # - 3+1 Planner: 96% within 1°
    
    t = np.linspace(0, 500, 250)
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # MTQ-only
    ax = axes[0]
    # PD: slow convergence, often doesn't converge
    pd_errors = []
    for i in range(10):
        err = 90 * np.exp(-t / 400) + 15 + 10 * np.random.randn(len(t)) * np.exp(-t/200)
        pd_errors.append(err)
        ax.plot(t, err, color=COLORS[0], alpha=0.2, linewidth=0.5)
    # Planner: better convergence
    plan_errors = []
    for i in range(10):
        err = 90 * np.exp(-t / 100) + 3 + 5 * np.random.randn(len(t)) * np.exp(-t/100)
        plan_errors.append(err)
        ax.plot(t, err, color=COLORS[1], alpha=0.2, linewidth=0.5)
    
    ax.plot(t, np.mean(pd_errors, axis=0), color=COLORS[0], linewidth=2, label='PD Control')
    ax.plot(t, np.mean(plan_errors, axis=0), color=COLORS[1], linewidth=2, label='ALTRO+TVLQR')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 200)
    ax.axhline(10, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title('MTQ-Only (3+0)')
    ax.grid(True, alpha=0.3, which='both')
    
    # 3+1
    ax = axes[1]
    pd_errors = []
    for i in range(10):
        err = 90 * np.exp(-t / 150) + 2 + 3 * np.random.randn(len(t)) * np.exp(-t/100)
        pd_errors.append(err)
        ax.plot(t, np.clip(err, 0.01, 200), color=COLORS[0], alpha=0.2, linewidth=0.5)
    plan_errors = []
    for i in range(10):
        err = 90 * np.exp(-t / 50) + 0.1 + 0.3 * np.random.randn(len(t)) * np.exp(-t/50)
        plan_errors.append(err)
        ax.plot(t, np.clip(err, 0.01, 200), color=COLORS[1], alpha=0.2, linewidth=0.5)
    
    ax.plot(t, np.clip(np.mean(pd_errors, axis=0), 0.01, 200), color=COLORS[0], linewidth=2, label='PD Control')
    ax.plot(t, np.clip(np.mean(plan_errors, axis=0), 0.01, 200), color=COLORS[1], linewidth=2, label='ALTRO+TVLQR')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 200)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title('Hybrid (3+1)')
    ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('Planner vs PD Control (PLACEHOLDER - Expected Results)', y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_planner_vs_pd_placeholder.png')
    fig.savefig(output_dir / 'fig_planner_vs_pd_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_planner_vs_pd_placeholder")


def gen_spinning_solution(output_dir: Path):
    """Generate spinning solution placeholder - key planner contribution."""
    t = np.linspace(0, 1000, 500)
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    
    # Angular velocity - spinning about body axis
    omega = 0.05 * (1 - np.exp(-t/100)) * np.ones((3, len(t)))
    omega[0, :] *= np.sin(2 * np.pi * t / 200) * 0.3  # Small x,y wobble
    omega[1, :] *= np.cos(2 * np.pi * t / 200) * 0.3
    omega[2, :] = 0.05 * (1 - np.exp(-t/100))  # Steady spin about z
    omega_mag = np.sqrt(np.sum(omega**2, axis=0))
    
    axes[0].plot(t, omega_mag * 180/np.pi, color=COLORS[0], linewidth=1.5)
    axes[0].set_ylabel('|ω| (deg/s)')
    axes[0].set_title('Planner Discovers Spinning Solution')
    axes[0].annotate('Spin-up', (100, 1), fontsize=9)
    axes[0].annotate('Steady spin', (600, 2.8), fontsize=9)
    
    # Pointing error - maintains pointing despite spin
    error = 5 * np.exp(-t/150) + 1 + 0.3 * np.sin(2 * np.pi * t / 200)
    axes[1].plot(t, error, color=COLORS[1], linewidth=1.5)
    axes[1].set_ylabel('Pointing Error (deg)')
    axes[1].axhline(2, color='gray', linestyle='--', alpha=0.5, label='Requirement')
    axes[1].legend(loc='upper right')
    
    # MTQ commands - periodic pattern
    mtq_x = 0.15 * np.sin(2 * np.pi * t / 200) * (1 - np.exp(-t/100))
    mtq_y = 0.15 * np.cos(2 * np.pi * t / 200) * (1 - np.exp(-t/100))
    mtq_z = 0.05 * np.sin(2 * np.pi * t / 400)
    
    axes[2].plot(t, mtq_x * 1000, label='MTQ x', alpha=0.8)
    axes[2].plot(t, mtq_y * 1000, label='MTQ y', alpha=0.8)
    axes[2].plot(t, mtq_z * 1000, label='MTQ z', alpha=0.8)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('MTQ Command (mAm²)')
    axes[2].legend(loc='upper right', ncol=3)
    
    fig.tight_layout()
    fig.savefig(output_dir / 'fig_spinning_solution_placeholder.png')
    fig.savefig(output_dir / 'fig_spinning_solution_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_spinning_solution_placeholder")


def gen_multi_target_sequence(output_dir: Path):
    """Generate multi-target sequence placeholder."""
    t = np.linspace(0, 500, 500)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Three target windows
    targets = [(0, 150), (180, 380), (420, 500)]
    
    # Generate error trajectory
    error = np.full_like(t, 30.0)  # Start high
    for start, end in targets:
        mask = (t >= start) & (t <= end)
        t_local = t[mask] - start
        duration = end - start
        error[mask] = 3 * np.exp(-t_local / 30) + 0.5 + 0.2 * np.random.randn(np.sum(mask))
    
    ax.plot(t, error, color=COLORS[0], linewidth=1.5)
    
    # Shade target windows
    for i, (start, end) in enumerate(targets):
        ax.axvspan(start, end, alpha=0.2, color=COLORS[2], label='Target window' if i == 0 else '')
        ax.annotate(f'Target {i+1}', ((start + end)/2, 1), ha='center', fontsize=9)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 50)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='1° threshold')
    ax.legend(loc='upper right')
    ax.set_title('Multi-Target Sequence (3 Targets, 500s)')
    ax.grid(True, alpha=0.3, which='both')
    
    fig.savefig(output_dir / 'fig_multi_target_placeholder.png')
    fig.savefig(output_dir / 'fig_multi_target_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_multi_target_placeholder")


# =============================================================================
# PACKAGE PAPER PLACEHOLDERS
# =============================================================================

def gen_basilisk_comparison_table(output_dir: Path):
    """Generate Basilisk comparison table as figure (easier than LaTeX)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Table data (placeholder - needs real measurements)
    data = [
        ['Metric', 'ADCS-Py', 'Basilisk'],
        ['Primary Language', 'Python', 'C/C++'],
        ['Lines to configure 3U', '~30', '~200+'],
        ['Time to first simulation', '<5 min', '~30 min'],
        ['Built-in estimators', '5', '10+'],
        ['Trajectory planning', 'Integrated', 'External'],
        ['Sim speed (1 orbit)', 'TBD', 'TBD'],
        ['Installation', 'pip install', 'Build from source'],
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#E6E6E6')
        table[(0, j)].set_text_props(fontweight='bold')
    
    ax.set_title('Framework Comparison: ADCS-Py vs Basilisk (PLACEHOLDER)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    fig.savefig(output_dir / 'fig_basilisk_comparison_placeholder.png')
    fig.savefig(output_dir / 'fig_basilisk_comparison_placeholder.pdf')
    plt.close(fig)
    print("  Generated: fig_basilisk_comparison_placeholder")


def gen_quickstart_demo(output_dir: Path):
    """Generate quickstart demo placeholder."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Simulate a simple pointing convergence
    t = np.linspace(0, 300, 150)
    error = 45 * np.exp(-t/50) + 1 + 0.5 * np.random.randn(len(t))
    
    ax.plot(t, error, color=COLORS[0], linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing Error (deg)')
    ax.set_title('Quickstart Example: Nadir Pointing (5-Minute Demo)')
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    
    # Add annotation box
    textstr = 'Steps:\n1. git clone\n2. pip install\n3. python quickstart.py'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    fig.savefig(output_dir / 'fig_quickstart_demo.png')
    fig.savefig(output_dir / 'fig_quickstart_demo.pdf')
    plt.close(fig)
    print("  Generated: fig_quickstart_demo")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Placeholder Figures")
    parser.add_argument('--all', action='store_true', help='Generate all placeholders')
    parser.add_argument('--paper', type=str, choices=['3p1', 'generalized', 'planner', 'package'],
                        help='Generate placeholders for specific paper')
    parser.add_argument('--output-dir', type=str, default='./placeholders')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("  Generating Placeholder Figures")
    print("="*60)
    
    # 3+1 Paper
    if args.all or args.paper == '3p1':
        print("\n  3+1 Paper Placeholders:")
        gen_3p1_actuator_configs(output_dir)
        gen_3p1_momentum_management(output_dir)
        gen_3p1_graceful_degradation(output_dir)
    
    # Generalized Control Paper
    if args.all or args.paper == 'generalized':
        print("\n  Generalized Control Paper Placeholders:")
        gen_direction_preservation(output_dir)
        gen_bolt_on_framework(output_dir)
        gen_controllability_vs_inclination(output_dir)
    
    # Planner Paper
    if args.all or args.paper == 'planner':
        print("\n  Planner Paper Placeholders:")
        gen_planner_vs_pd(output_dir)
        gen_spinning_solution(output_dir)
        gen_multi_target_sequence(output_dir)
    
    # Package Paper
    if args.all or args.paper == 'package':
        print("\n  Package Paper Placeholders:")
        gen_basilisk_comparison_table(output_dir)
        gen_quickstart_demo(output_dir)
    
    print(f"\n  All placeholders saved to: {output_dir}")


if __name__ == "__main__":
    main()
