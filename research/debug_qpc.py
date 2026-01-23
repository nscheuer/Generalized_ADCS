"""
Debug QPC allocators to understand why they're producing zero torque.
"""
import numpy as np
from allocation_comparison import LPAllocator, QPAllocator, QPCAllocator
from ADCS.helpers.math_helpers import skewsym

# Simple test case
np.random.seed(42)

# Actuator config
A_mtq_axes = np.eye(3)
u_mtq_max = np.array([0.2, 0.2, 0.2])
A_rw = np.array([[0], [0], [1.0]])
u_rw_max = np.array([0.001])

# Test scenario: damping case
omega = np.array([0.01, 0.0, 0.0])  # Rotating about x
tau_des = np.array([-1e-5, 0.0, 0.0])  # Want to damp x rotation
b_body = np.array([0.0, 30e-6, 0.0])  # B-field along y

print("=" * 60)
print("DEBUG: QPC Allocator Analysis")
print("=" * 60)

print(f"\nTest Case:")
print(f"  omega: {omega}")
print(f"  tau_des: {tau_des}")
print(f"  b_body: {b_body}")
print(f"  omega · tau_des = {np.dot(omega, tau_des):.2e} (negative = damping)")

# Build effective torque matrix
A_mtq = -skewsym(b_body) @ A_mtq_axes
A_total = np.hstack([A_rw, A_mtq])
print(f"\nA_total shape: {A_total.shape}")
print(f"A_total:\n{A_total}")

# What direction can MTQs produce torque?
print(f"\nMTQ torque plane (perpendicular to B):")
print(f"  B direction: {b_body / np.linalg.norm(b_body)}")
print(f"  MTQ can produce torque in x-z plane only (perp to y)")

# Check the constraint coefficient
C = omega @ A_total
print(f"\nConstraint gradient C = omega @ A_total:")
print(f"  C = {C}")

omega_dot_tau_des = np.dot(omega, tau_des)
print(f"\n  omega · tau_des = {omega_dot_tau_des:.2e}")

# For variant A: ω · τ ≤ max(0, ω · τ_des)
# If ω · τ_des < 0 (damping), then constraint is ω · τ ≤ 0
# This means C @ u ≤ 0

print(f"\nVariant A constraint: ω · τ ≤ max(0, {omega_dot_tau_des:.2e}) = 0")
print(f"  This constrains: {C} @ u ≤ 0")

# Let's test each allocator
allocators = [
    LPAllocator(),
    QPAllocator(),
    QPCAllocator("A"),
]

print("\n" + "=" * 60)
print("ALLOCATOR RESULTS")
print("=" * 60)

for alloc in allocators:
    result = alloc.allocate(
        tau_des=tau_des,
        b_body=b_body,
        A_rw=A_rw,
        A_mtq_axes=A_mtq_axes,
        u_rw_max=u_rw_max,
        u_mtq_max=u_mtq_max,
        omega=omega
    )
    
    print(f"\n{alloc.name}:")
    print(f"  u_rw: {result.u_rw}")
    print(f"  u_mtq: {result.u_mtq}")
    print(f"  tau_achieved: {result.tau_achieved}")
    print(f"  alpha: {result.alpha:.4f}")
    print(f"  direction_error: {result.direction_error_deg:.2f}°")
    
    # Check energy
    energy = np.dot(omega, result.tau_achieved)
    print(f"  omega · tau_achieved = {energy:.2e}")
    print(f"  (negative = removing energy = good for damping)")

# Now let's trace through QPC-A step by step
print("\n" + "=" * 60)
print("DETAILED QPC-A TRACE")
print("=" * 60)

from scipy.optimize import minimize, Bounds

# Bounds
lb = np.concatenate([-u_rw_max, -u_mtq_max])
ub = np.concatenate([u_rw_max, u_mtq_max])
n_act = 4

# Objective
def objective(u):
    r = A_total @ u - tau_des
    return 0.5 * np.dot(r, r)

def gradient(u):
    return A_total.T @ (A_total @ u - tau_des)

# Constraint for variant A
ub_constraint = max(0.0, omega_dot_tau_des)  # = 0 since omega_dot_tau_des < 0

constraint_A = {
    "type": "ineq",
    "fun": lambda u: ub_constraint - C @ u,  # This is: 0 - C @ u ≥ 0 → C @ u ≤ 0
    "jac": lambda u: -C
}

print(f"Constraint: {C} @ u ≤ {ub_constraint}")

# Test at u = 0
u_test = np.zeros(n_act)
print(f"\nAt u=0:")
print(f"  Objective: {objective(u_test):.2e}")
print(f"  Constraint value (should be ≥ 0 for feasible): {constraint_A['fun'](u_test):.2e}")

# Try to find optimal
u0 = np.zeros(n_act)
bounds_obj = Bounds(lb, ub)

print(f"\nSolving with SLSQP...")
res = minimize(objective, u0, jac=gradient, method='SLSQP',
              constraints=[constraint_A], bounds=bounds_obj,
              options={'disp': True})

print(f"\nResult:")
print(f"  Success: {res.success}")
print(f"  Message: {res.message}")
print(f"  x: {res.x}")
print(f"  Objective at solution: {objective(res.x):.2e}")
print(f"  Constraint at solution: {constraint_A['fun'](res.x):.2e}")

tau_ach = A_total @ res.x
print(f"  tau_achieved: {tau_ach}")
print(f"  omega · tau_achieved: {np.dot(omega, tau_ach):.2e}")

# The issue: we need to check if the constraint is TOO restrictive
# Let's find what torques CAN satisfy C @ u ≤ 0
print("\n" + "=" * 60)
print("ANALYSIS: What torques satisfy the constraint?")
print("=" * 60)

# For C @ u ≤ 0, we need omega @ A_total @ u ≤ 0
# This means omega · tau ≤ 0
# For tau_des = [-1e-5, 0, 0] and omega = [0.01, 0, 0]:
# omega · tau_des = -1e-7 < 0

# The constraint says: achieved power must be ≤ 0
# But wait - that's actually what we want for damping!

# Let's check what the unconstrained QP achieves
res_unconstrained = minimize(objective, u0, jac=gradient, method='SLSQP',
                            bounds=bounds_obj)
tau_unconstrained = A_total @ res_unconstrained.x
energy_unconstrained = np.dot(omega, tau_unconstrained)
print(f"\nUnconstrained QP:")
print(f"  tau_achieved: {tau_unconstrained}")
print(f"  omega · tau_achieved: {energy_unconstrained:.2e}")
print(f"  Constraint satisfied? {energy_unconstrained <= ub_constraint + 1e-9}")

# The issue might be numerical - let's check if it's actually finding a solution
print("\n" + "=" * 60)
print("CHECKING FEASIBILITY")
print("=" * 60)

# What if we manually find a feasible point that achieves some torque?
# We want tau in the -x direction, and omega · tau ≤ 0

# MTQ can produce torque in x-z plane (perp to B along y)
# To get -x torque: need torque = m × B = -[B]× m
# B = [0, 30e-6, 0], so B× = [[0, 0, 30e-6], [0, 0, 0], [-30e-6, 0, 0]]
# -B× = [[0, 0, -30e-6], [0, 0, 0], [30e-6, 0, 0]]
# To get tau_x < 0: need -30e-6 * m_z < 0, so m_z > 0

# Try u = [0, 0, 0, 0.1]  (MTQ on z-axis at 0.1 Am²)
u_manual = np.array([0, 0, 0, 0.1])
tau_manual = A_total @ u_manual
energy_manual = np.dot(omega, tau_manual)
constraint_manual = constraint_A['fun'](u_manual)

print(f"\nManual test with u = {u_manual}:")
print(f"  tau = {tau_manual}")
print(f"  omega · tau = {energy_manual:.2e}")
print(f"  Constraint value: {constraint_manual:.2e} (≥0 means feasible)")
print(f"  Objective: {objective(u_manual):.2e}")

# The problem is that SLSQP might be having trouble with the constraint
# Let's try a different approach - use the QP result but project it

print("\n" + "=" * 60)
print("FIX: Project QP solution to satisfy constraint")  
print("=" * 60)

# If QP gives us a solution that violates the constraint, we can project it
u_qp = res_unconstrained.x
energy_qp = np.dot(omega, A_total @ u_qp)

if energy_qp > ub_constraint + 1e-9:
    # Need to reduce energy
    # Find scaling factor β such that energy_qp * β ≤ ub_constraint
    # But scaling might not work directly...
    
    # Better: do constrained optimization properly
    # The issue is probably numerical tolerance
    pass

print(f"\nQP solution energy: {energy_qp:.2e}")
print(f"Constraint upper bound: {ub_constraint:.2e}")
print(f"Violates constraint by: {energy_qp - ub_constraint:.2e}")
