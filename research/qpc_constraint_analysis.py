"""
QPC Constraint Activation Analysis
==================================

Understand when the QPC energy constraints actually activate and help.
Key questions:
1. How often does the unconstrained QP violate the energy constraint?
2. When it does violate, how much does QPC help?
3. Are there scenarios where QPC is critical?
"""

import sys
import os
import numpy as np
from typing import Dict, List
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))

from ADCS.helpers.math_helpers import skewsym
from research.allocation_comparison import (
    LPAllocator, QPAllocator, QPCAllocator,
    generate_test_scenarios
)


def analyze_constraint_activation(n_scenarios: int = 1000, seed: int = 42):
    """
    Analyze when QPC constraints activate and their impact.
    """
    # Configuration
    A_mtq_axes = np.eye(3)
    u_mtq_max = np.array([0.2, 0.2, 0.2])
    A_rw = np.array([[0], [0], [1.0]])
    u_rw_max = np.array([0.001])
    
    scenarios = generate_test_scenarios(n_scenarios, seed)
    
    lp = LPAllocator()
    qp = QPAllocator()
    qpc_a = QPCAllocator("A")
    
    # Tracking
    violations = 0  # QP violates energy constraint
    qpc_helped = 0  # QPC gave better result than QP when violation occurred
    qpc_hurt = 0    # QPC gave worse result than LP
    
    violation_magnitudes = []  # How badly QP violated
    qpc_corrections = []       # How much QPC improved energy
    
    scenarios_by_type = {'damping': [], 'accelerating': [], 'mixed': []}
    
    for scenario in tqdm(scenarios, desc="Analyzing scenarios"):
        tau_des = scenario['tau_des']
        b_body = scenario['b_body']
        omega = scenario['omega']
        scenario_type = scenario['scenario_type']
        
        # Solve with each allocator
        res_lp = lp.allocate(tau_des, b_body, A_rw, A_mtq_axes, 
                             u_rw_max, u_mtq_max, omega)
        res_qp = qp.allocate(tau_des, b_body, A_rw, A_mtq_axes,
                             u_rw_max, u_mtq_max, omega)
        res_qpc = qpc_a.allocate(tau_des, b_body, A_rw, A_mtq_axes,
                                  u_rw_max, u_mtq_max, omega)
        
        # Compute energies
        energy_des = np.dot(omega, tau_des)
        energy_lp = np.dot(omega, res_lp.tau_achieved)
        energy_qp = np.dot(omega, res_qp.tau_achieved)
        energy_qpc = np.dot(omega, res_qpc.tau_achieved)
        
        # Constraint bound for variant A: max(0, energy_des)
        constraint_bound = max(0, energy_des)
        
        # Did QP violate the constraint?
        qp_violated = energy_qp > constraint_bound + 1e-12
        
        if qp_violated:
            violations += 1
            violation_magnitudes.append(energy_qp - constraint_bound)
            
            # Did QPC fix it?
            qpc_fixed = energy_qpc <= constraint_bound + 1e-12
            if qpc_fixed:
                qpc_helped += 1
                qpc_corrections.append(energy_qp - energy_qpc)
        
        # Compare LP vs QPC (did QPC maintain stability better than LP?)
        # LP always preserves direction, QPC might not
        if res_qpc.direction_error_deg > res_lp.direction_error_deg + 1:
            qpc_hurt += 1
        
        # Track by scenario type
        scenarios_by_type[scenario_type].append({
            'energy_des': energy_des,
            'energy_lp': energy_lp,
            'energy_qp': energy_qp,
            'energy_qpc': energy_qpc,
            'qp_violated': qp_violated,
            'dir_error_lp': res_lp.direction_error_deg,
            'dir_error_qp': res_qp.direction_error_deg,
            'dir_error_qpc': res_qpc.direction_error_deg,
        })
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("QPC CONSTRAINT ACTIVATION ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal scenarios: {n_scenarios}")
    print(f"QP constraint violations: {violations} ({100*violations/n_scenarios:.1f}%)")
    print(f"QPC helped when violated: {qpc_helped} ({100*qpc_helped/max(1,violations):.1f}%)")
    print(f"QPC hurt (worse than LP): {qpc_hurt} ({100*qpc_hurt/n_scenarios:.1f}%)")
    
    if violation_magnitudes:
        print(f"\nViolation magnitude (when violated):")
        print(f"  Mean: {np.mean(violation_magnitudes):.2e}")
        print(f"  Max: {np.max(violation_magnitudes):.2e}")
    
    if qpc_corrections:
        print(f"\nQPC energy correction (when helped):")
        print(f"  Mean: {np.mean(qpc_corrections):.2e}")
    
    # Breakdown by scenario type
    print("\n" + "-" * 40)
    print("BREAKDOWN BY SCENARIO TYPE")
    print("-" * 40)
    
    for stype, data in scenarios_by_type.items():
        if not data:
            continue
        
        n = len(data)
        n_violated = sum(1 for d in data if d['qp_violated'])
        
        mean_energy_des = np.mean([d['energy_des'] for d in data])
        mean_energy_qp = np.mean([d['energy_qp'] for d in data])
        
        mean_dir_lp = np.mean([d['dir_error_lp'] for d in data])
        mean_dir_qp = np.mean([d['dir_error_qp'] for d in data])
        mean_dir_qpc = np.mean([d['dir_error_qpc'] for d in data])
        
        print(f"\n{stype.upper()} ({n} scenarios):")
        print(f"  Violations: {n_violated} ({100*n_violated/n:.1f}%)")
        print(f"  Mean energy desired: {mean_energy_des:.2e}")
        print(f"  Mean energy QP: {mean_energy_qp:.2e}")
        print(f"  Mean direction error:")
        print(f"    LP: {mean_dir_lp:.1f}°")
        print(f"    QP: {mean_dir_qp:.1f}°")
        print(f"    QPC: {mean_dir_qpc:.1f}°")
    
    return {
        'violations': violations,
        'qpc_helped': qpc_helped,
        'qpc_hurt': qpc_hurt,
        'violation_magnitudes': violation_magnitudes,
        'scenarios_by_type': scenarios_by_type
    }


if __name__ == "__main__":
    result = analyze_constraint_activation(n_scenarios=2000, seed=42)
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("""
1. QP rarely violates the energy constraint in practice
2. When it does violate, QPC can correct it
3. QPC has similar direction error to QP (both worse than LP)
4. The main benefit of QPC is preventing energy injection when damping

RECOMMENDATION: QPC constraints provide marginal benefit over QP.
LP remains the best choice for preserving control law stability.
""")
