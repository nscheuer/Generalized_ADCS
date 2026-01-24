#!/usr/bin/env python3
"""
Paper Experiments
=================

Experiment configurations for the 4 conference/journal papers.

Papers:
1. 3+1 Paper: "3MTQ+1RW Attitude Control" - Compares architectures
2. Generalized Control Paper: "Generalized Attitude Control" - LP vs QP allocation
3. Planner Paper: "MTQ Trajectory Planning" - ALTRO planner results
4. Package Paper: "Generalized ADCS Python Package" - Software validation

Usage:
    python paper_experiments.py --paper 3p1 --list
    python paper_experiments.py --paper generalized --experiment lp_vs_qp --quick
    python paper_experiments.py --paper planner --all --full
"""

import sys
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import numpy as np

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# =============================================================================
# 3+1 PAPER EXPERIMENTS
# =============================================================================

PAPER_3P1_EXPERIMENTS = {
    # Architecture comparison: 3+0, 3+1, 3+2, 3+3
    'arch_3p0_mtq_only': {
        'name': '3+0 MTQ Only',
        'description': '3 MTQ, 0 RW - Underactuated baseline',
        'satellite': 'mtq_only',
        'controller': 'MTQ_Lovera',
        'controller_params': {'p_gain': 0.001, 'd_gain': 0.005, 'eps': 1.0},
        'n_trials': 100,
        'duration_s': 1000,
        'goal': 'random_eci',
        'output_prefix': 'arch_3p0',
    },
    'arch_3p1_lp': {
        'name': '3+1 with LP',
        'description': '3 MTQ + 1 RW with LP allocation',
        'satellite': '3mtq_1rw',
        'controller': 'MTQ_w_RW_LP',
        'controller_params': {'p_gain': 0.0001, 'd_gain': 0.001, 'c_gain': 0.001},
        'n_trials': 100,
        'duration_s': 1000,
        'goal': 'random_eci',
        'output_prefix': 'arch_3p1_lp',
    },
    'arch_3p1_qp': {
        'name': '3+1 with QP',
        'description': '3 MTQ + 1 RW with QP allocation',
        'satellite': '3mtq_1rw',
        'controller': 'MTQ_w_RW_QP',
        'controller_params': {'p_gain': 0.0001, 'd_gain': 0.001, 'c_gain': 0.001},
        'n_trials': 100,
        'duration_s': 1000,
        'goal': 'random_eci',
        'output_prefix': 'arch_3p1_qp',
    },
    'arch_3p2_lp': {
        'name': '3+2 with LP',
        'description': '3 MTQ + 2 RW',
        'satellite': '3mtq_2rw',
        'controller': 'MTQ_w_RW_LP',
        'controller_params': {'p_gain': 0.0001, 'd_gain': 0.001, 'c_gain': 0.001},
        'n_trials': 100,
        'duration_s': 1000,
        'goal': 'random_eci',
        'output_prefix': 'arch_3p2',
    },
    'arch_3p3_lp': {
        'name': '3+3 with LP',
        'description': '3 MTQ + 3 RW - Fully actuated',
        'satellite': '3mtq_3rw',
        'controller': 'MTQ_w_RW_LP',
        'controller_params': {'p_gain': 0.0001, 'd_gain': 0.001, 'c_gain': 0.001},
        'n_trials': 100,
        'duration_s': 1000,
        'goal': 'random_eci',
        'output_prefix': 'arch_3p3',
    },
    # Wisniewski comparison
    'wisniewski_baseline': {
        'name': 'Wisniewski Baseline',
        'description': 'Wisniewski MTQ controller as baseline',
        'satellite': 'mtq_only',
        'controller': 'MTQ_Wisniewski',
        'controller_params': {'p_gain': 0.001, 'd_gain': 0.005, 'eps': 0.1},
        'n_trials': 100,
        'duration_s': 1000,
        'goal': 'random_eci',
        'output_prefix': 'wis_baseline',
    },
}


# =============================================================================
# GENERALIZED CONTROL PAPER EXPERIMENTS
# =============================================================================

PAPER_GENERALIZED_EXPERIMENTS = {
    # LP vs QP comparison
    'lp_vs_qp_3p1': {
        'name': 'LP vs QP (3+1)',
        'description': 'Compare LP and QP allocation on 3MTQ+1RW',
        'satellite': '3mtq_1rw',
        'controllers': ['MTQ_w_RW_LP', 'MTQ_w_RW_QP'],
        'n_trials': 100,
        'duration_s': 500,
        'goal': 'random_eci',
        'output_prefix': 'lp_qp_3p1',
    },
    'lp_vs_qp_3p3': {
        'name': 'LP vs QP (3+3)',
        'description': 'Compare LP and QP allocation on 3MTQ+3RW',
        'satellite': '3mtq_3rw',
        'controllers': ['MTQ_w_RW_LP', 'MTQ_w_RW_QP'],
        'n_trials': 100,
        'duration_s': 500,
        'goal': 'random_eci',
        'output_prefix': 'lp_qp_3p3',
    },
    # Allocation methods
    'allocation_comparison': {
        'name': 'Allocation Methods',
        'description': 'Compare pseudoinverse, LP, QP, QP-weighted allocations',
        'satellite': '3mtq_3rw',
        'methods': ['pseudoinverse', 'lp', 'qp', 'qp_weighted'],
        'n_trials': 100,
        'duration_s': 500,
        'output_prefix': 'allocation',
    },
    # Goal formulation (full vs reduced attitude)
    'goal_full_attitude': {
        'name': 'Full Attitude Goal',
        'description': 'Tracking specific quaternion',
        'satellite': '3mtq_1rw',
        'controller': 'MTQ_w_RW_LP',
        'goal': 'fixed_quaternion',
        'n_trials': 100,
        'duration_s': 500,
        'output_prefix': 'goal_full',
    },
    'goal_reduced_attitude': {
        'name': 'Reduced Attitude Goal',
        'description': 'Tracking ECI vector alignment',
        'satellite': '3mtq_1rw',
        'controller': 'MTQ_w_RW_LP',
        'goal': 'random_eci',
        'n_trials': 100,
        'duration_s': 500,
        'output_prefix': 'goal_reduced',
    },
}


# =============================================================================
# PLANNER PAPER EXPERIMENTS
# =============================================================================

PAPER_PLANNER_EXPERIMENTS = {
    # Single trajectory demos
    'single_traj_mtq': {
        'name': 'Single Trajectory MTQ',
        'description': 'Single ALTRO trajectory for MTQ-only satellite',
        'satellite': 'mtq_only',
        'use_altro': True,
        'duration_s': 500,
        'goal': 'eci_vector',
        'output_prefix': 'traj_mtq',
    },
    'single_traj_1rw': {
        'name': 'Single Trajectory 3+1',
        'description': 'Single ALTRO trajectory for 3MTQ+1RW satellite',
        'satellite': '3mtq_1rw',
        'use_altro': True,
        'duration_s': 500,
        'goal': 'eci_vector',
        'output_prefix': 'traj_1rw',
    },
    # Monte Carlo (same as thesis but for paper)
    'mc_mtq_paper': {
        'name': 'MC MTQ (Paper)',
        'description': 'Monte Carlo for paper figures',
        'satellite': 'mtq_only',
        'use_altro': True,
        'n_trials': 100,
        'duration_s': 500,
        'goal': 'random_eci',
        'output_prefix': 'mc_mtq',
    },
    'mc_1rw_paper': {
        'name': 'MC 3+1 (Paper)',
        'description': 'Monte Carlo for paper figures',
        'satellite': '3mtq_1rw',
        'use_altro': True,
        'n_trials': 100,
        'duration_s': 500,
        'goal': 'random_eci',
        'output_prefix': 'mc_1rw',
    },
    # Sequential planning demo
    'sequential_demo': {
        'name': 'Sequential Planning',
        'description': 'Sequential trajectory planning demonstration',
        'satellite': '3mtq_3rw',
        'use_altro': True,
        'duration_s': 3600,
        'goals': 'sequential_4_targets',
        'output_prefix': 'sequential',
    },
}


# =============================================================================
# PACKAGE PAPER EXPERIMENTS
# =============================================================================

PAPER_PACKAGE_EXPERIMENTS = {
    # Validation tests
    'validation_dynamics': {
        'name': 'Dynamics Validation',
        'description': 'Validate dynamics against analytical solutions',
        'test_type': 'dynamics',
        'output_prefix': 'val_dyn',
    },
    'validation_estimation': {
        'name': 'Estimation Validation',
        'description': 'Validate UKF convergence',
        'test_type': 'estimation',
        'output_prefix': 'val_est',
    },
    'validation_control': {
        'name': 'Control Validation',
        'description': 'Validate controller implementations',
        'test_type': 'control',
        'output_prefix': 'val_ctrl',
    },
    # Performance benchmarks
    'benchmark_simulation': {
        'name': 'Simulation Benchmark',
        'description': 'Simulation speed benchmarks',
        'test_type': 'benchmark',
        'output_prefix': 'bench_sim',
    },
    'benchmark_planner': {
        'name': 'Planner Benchmark',
        'description': 'ALTRO planner speed benchmarks',
        'test_type': 'benchmark',
        'output_prefix': 'bench_plan',
    },
}


# =============================================================================
# SATELLITE CONFIGURATIONS (shared across papers)
# =============================================================================

SATELLITE_CONFIGS = {
    'mtq_only': {
        'name': '3MTQ CubeSat',
        'J': [0.005256, 0.04939, 0.04939],
        'mtq_max': [0.19, 0.57, 0.57],
        'rws': [],
    },
    '3mtq_1rw': {
        'name': '3MTQ+1RW CubeSat',
        'J': [0.005256, 0.04939, 0.04939],
        'mtq_max': [0.19, 0.57, 0.57],
        'rws': [{'axis': [0, 1, 0], 'max_torque': 0.0002, 'h_max': 0.002, 'J': 2e-6}],
    },
    '3mtq_2rw': {
        'name': '3MTQ+2RW CubeSat',
        'J': [0.005256, 0.04939, 0.04939],
        'mtq_max': [0.19, 0.57, 0.57],
        'rws': [
            {'axis': [1, 0, 0], 'max_torque': 0.0002, 'h_max': 0.002, 'J': 2e-6},
            {'axis': [0, 1, 0], 'max_torque': 0.0002, 'h_max': 0.002, 'J': 2e-6},
        ],
    },
    '3mtq_3rw': {
        'name': '3MTQ+3RW CubeSat',
        'J': [0.03, 0.03, 0.01],  # BeaverCube-like
        'mtq_max': [0.2, 0.2, 0.2],
        'rws': [
            {'axis': [1, 0, 0], 'max_torque': 0.0002, 'h_max': 0.002, 'J': 2e-6},
            {'axis': [0, 1, 0], 'max_torque': 0.0002, 'h_max': 0.002, 'J': 2e-6},
            {'axis': [0, 0, 1], 'max_torque': 0.0002, 'h_max': 0.002, 'J': 2e-6},
        ],
    },
}


# =============================================================================
# PAPER REGISTRY
# =============================================================================

PAPERS = {
    '3p1': {
        'name': '3+1 Paper',
        'full_name': '3MTQ+1RW Attitude Control for CubeSats',
        'experiments': PAPER_3P1_EXPERIMENTS,
        'output_dir': '3p1_paper',
    },
    'generalized': {
        'name': 'Generalized Control Paper',
        'full_name': 'Generalized Attitude Control System',
        'experiments': PAPER_GENERALIZED_EXPERIMENTS,
        'output_dir': 'generalized_paper',
    },
    'planner': {
        'name': 'Planner Paper',
        'full_name': 'MTQ Trajectory Planning with ALTRO',
        'experiments': PAPER_PLANNER_EXPERIMENTS,
        'output_dir': 'planner_paper',
    },
    'package': {
        'name': 'Package Paper',
        'full_name': 'Generalized ADCS Python Package',
        'experiments': PAPER_PACKAGE_EXPERIMENTS,
        'output_dir': 'package_paper',
    },
}


# =============================================================================
# RUNNER FRAMEWORK
# =============================================================================

def create_satellite(sat_config_name: str):
    """Create satellite from configuration."""
    from ADCS.satellite_factory.satellites.create_cubesats import (
        create_beavercube1_cubesat,
        create_beavercube2_cubesat,
        create_3_3_beavercube2_cubesat,
    )
    
    if sat_config_name == 'mtq_only':
        return create_beavercube1_cubesat(estimated=False)
    elif sat_config_name == '3mtq_1rw':
        return create_beavercube2_cubesat(estimated=False)
    elif sat_config_name == '3mtq_3rw':
        return create_3_3_beavercube2_cubesat(estimated=False)
    else:
        # Create custom satellite
        config = SATELLITE_CONFIGS[sat_config_name]
        from ADCS.satellite_hardware.satellite.satellite import Satellite
        from ADCS.satellite_hardware.actuators import MTQ, RW
        from ADCS.satellite_hardware.sensors import MTM, Gyro
        from ADCS.helpers.math_constants import MathConstants
        
        mtqs = [MTQ(axis=MathConstants.unitvecs[i], max_torque=config['mtq_max'][i]) 
                for i in range(3)]
        rws = [RW(axis=np.array(rw['axis']), max_torque=rw['max_torque'], 
                  J=rw['J'], h=0.0, h_max=rw['h_max']) 
               for rw in config['rws']]
        mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
        gyros = [Gyro(axis=j) for j in MathConstants.unitvecs]
        
        return Satellite(
            mass=4.0,
            J_0=np.diag(config['J']),
            actuators=mtqs + rws,
            sensors=mtms + gyros,
            boresight=np.array([1, 0, 0]),
        )


def run_experiment(paper: str, exp_id: str, output_dir: Path, quick: bool = False):
    """Run a single experiment."""
    paper_info = PAPERS[paper]
    exp_config = paper_info['experiments'][exp_id]
    
    print(f"\n{'='*60}")
    print(f"  {exp_config['name']}")
    print(f"  {exp_config.get('description', '')}")
    print(f"{'='*60}")
    
    # Adjust for quick mode
    n_trials = exp_config.get('n_trials', 1)
    duration = exp_config.get('duration_s', 500)
    
    if quick:
        n_trials = min(10, n_trials)
        duration = min(100, duration)
    
    print(f"  Trials: {n_trials}, Duration: {duration}s")
    print(f"  Output: {output_dir / exp_config['output_prefix']}")
    
    # Placeholder - actual implementation would run simulations
    results = {
        'experiment': exp_id,
        'name': exp_config['name'],
        'n_trials': n_trials,
        'duration_s': duration,
        'status': 'configured',
    }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def list_experiments(paper: str = None):
    """Print list of experiments."""
    papers_to_show = [paper] if paper else list(PAPERS.keys())
    
    for p in papers_to_show:
        if p not in PAPERS:
            print(f"Unknown paper: {p}")
            continue
        
        paper_info = PAPERS[p]
        print(f"\n{'='*70}")
        print(f"  {paper_info['name']}: {paper_info['full_name']}")
        print(f"{'='*70}")
        
        for exp_id, exp_config in paper_info['experiments'].items():
            print(f"\n  {exp_id}")
            print(f"    Name: {exp_config['name']}")
            if 'description' in exp_config:
                print(f"    Description: {exp_config['description']}")
            if 'n_trials' in exp_config:
                print(f"    Trials: {exp_config['n_trials']}")
            if 'duration_s' in exp_config:
                print(f"    Duration: {exp_config['duration_s']}s")


def main():
    parser = argparse.ArgumentParser(description="Paper Experiments")
    parser.add_argument('--paper', type=str, choices=list(PAPERS.keys()),
                        help='Paper to run experiments for')
    parser.add_argument('--list', action='store_true', help='List experiments')
    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--full', action='store_true', help='Full mode')
    parser.add_argument('--output-dir', type=str, default='./paper_figures')
    args = parser.parse_args()
    
    if args.list:
        list_experiments(args.paper)
        return
    
    if not args.paper:
        print("Please specify --paper (3p1, generalized, planner, package)")
        print("Use --list to see available experiments")
        return
    
    output_dir = Path(args.output_dir) / PAPERS[args.paper]['output_dir']
    quick = not args.full
    
    experiments_to_run = []
    if args.all:
        experiments_to_run = list(PAPERS[args.paper]['experiments'].keys())
    elif args.experiment:
        if args.experiment not in PAPERS[args.paper]['experiments']:
            print(f"Unknown experiment: {args.experiment}")
            return
        experiments_to_run = [args.experiment]
    else:
        list_experiments(args.paper)
        return
    
    all_results = {}
    for exp_id in experiments_to_run:
        results = run_experiment(args.paper, exp_id, output_dir, quick=quick)
        all_results[exp_id] = results
    
    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
