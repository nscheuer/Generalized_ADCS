"""
Deeper debugging of QP vs LP differences.
"""
import numpy as np
from scipy.optimize import lsq_linear, minimize, Bounds
from ADCS.helpers.math_helpers import skewsym

# Simple test case
np.random.seed(42)

# Actuator config
A_mtq_axes = np.eye(3)
u_mtq_max = np.array([0.2, 0.2, 0.2])
A_rw = np.array([[0], [0], [1.0]])
u_rw_max = np.array([0.001])

# Test scenario
omega = np.array([0.01, 0.0, 0.0])
tau_des = np.array([-1e-5, 0.0, 0.0])
b_body = np.array([0.0, 30e-6, 0.0])

# Build A_total
A_mtq = -skewsym(b_body) @ A_mtq_axes
A_total = np.hstack([A_rw, A_mtq])

print("A_total:")
print(A_total)

print(f"\ntau_des: {tau_des}")

# What's the pseudoinverse solution?
u_pinv = np.linalg.pinv(A_total) @ tau_des
print(f"\nPseudoinverse solution:")
print(f"  u = {u_pinv}")
print(f"  tau = {A_total @ u_pinv}")

# Check bounds
lb = np.concatenate([-u_rw_max, -u_mtq_max])
ub = np.concatenate([u_rw_max, u_mtq_max])
print(f"\nBounds: lb={lb}, ub={ub}")
print(f"  u_pinv within bounds? {np.all(u_pinv >= lb) and np.all(u_pinv <= ub)}")

# Try bounded least squares with explicit objective
def objective(u):
    r = A_total @ u - tau_des
    return 0.5 * np.dot(r, r)

def gradient(u):
    return A_total.T @ (A_total @ u - tau_des)

print("\n" + "=" * 60)
print("Testing different solvers")
print("=" * 60)

# 1. lsq_linear
res1 = lsq_linear(A_total, tau_des, bounds=(lb, ub), method='trf')
print(f"\nlsq_linear (TRF):")
print(f"  u = {res1.x}")
print(f"  tau = {A_total @ res1.x}")
print(f"  objective = {objective(res1.x):.2e}")

# 2. lsq_linear with bvls
res2 = lsq_linear(A_total, tau_des, bounds=(lb, ub), method='bvls')
print(f"\nlsq_linear (BVLS):")
print(f"  u = {res2.x}")
print(f"  tau = {A_total @ res2.x}")
print(f"  objective = {objective(res2.x):.2e}")

# 3. SLSQP
res3 = minimize(objective, np.zeros(4), jac=gradient, method='SLSQP',
               bounds=Bounds(lb, ub))
print(f"\nSLSQP:")
print(f"  u = {res3.x}")
print(f"  tau = {A_total @ res3.x}")
print(f"  objective = {objective(res3.x):.2e}")

# 4. L-BFGS-B
res4 = minimize(objective, np.zeros(4), jac=gradient, method='L-BFGS-B',
               bounds=[(lb[i], ub[i]) for i in range(4)])
print(f"\nL-BFGS-B:")
print(f"  u = {res4.x}")
print(f"  tau = {A_total @ res4.x}")
print(f"  objective = {objective(res4.x):.2e}")

# Now let's check: what IS the minimum?
# The problem is that A_total has a nontrivial nullspace
print("\n" + "=" * 60)
print("Matrix analysis")
print("=" * 60)

print(f"\nA_total rank: {np.linalg.matrix_rank(A_total)}")
print(f"A_total shape: {A_total.shape}")

# SVD
U, S, Vh = np.linalg.svd(A_total, full_matrices=True)
print(f"\nSingular values: {S}")

# The null space
null_space = Vh[3:, :]  # Last row since rank is 2 < 4
print(f"\nNull space basis (rows of Vh beyond rank):")
print(Vh)

# tau_des component in range vs orthogonal complement
tau_in_range = A_total @ np.linalg.pinv(A_total) @ tau_des
tau_perp = tau_des - tau_in_range
print(f"\ntau_des decomposition:")
print(f"  In range: {tau_in_range}")
print(f"  Perpendicular (unreachable): {tau_perp}")
print(f"  ||tau_perp|| = {np.linalg.norm(tau_perp):.2e}")

# So the minimum achievable error is ||tau_perp||
min_error_squared = np.dot(tau_perp, tau_perp)
print(f"\nMinimum achievable objective: {0.5 * min_error_squared:.2e}")

# And all solutions u that achieve this have the form u = u_pinv + null_space_combination
# Let's parametrize the nullspace

# Actually, let me look at A_total more carefully
print("\n" + "=" * 60)
print("Understanding the geometry")
print("=" * 60)

print("A_total columns (what each actuator does):")
print(f"  RW (col 0): {A_total[:, 0]} - torque along z")
print(f"  MTQ-x (col 1): {A_total[:, 1]} - torque along z (small)")
print(f"  MTQ-y (col 2): {A_total[:, 2]} - nothing! (parallel to B)")
print(f"  MTQ-z (col 3): {A_total[:, 3]} - torque along -x")

# So to get -x torque, we need MTQ-z > 0
# The bounds allow up to 0.2 Am²

# Maximum -x torque: 0.2 * 3e-5 = 6e-6 Nm
# We want 1e-5 Nm, which is more than achievable!
# Wait, tau_des is 1e-5, max achievable is 6e-6

print(f"\nMaximum achievable x-torque magnitude: {0.2 * 30e-6:.2e} Nm")
print(f"Requested x-torque magnitude: {np.abs(tau_des[0]):.2e} Nm")
print(f"Can achieve: {0.2 * 30e-6 >= np.abs(tau_des[0])}")

# Hmm, 6e-6 < 1e-5, so we can't fully achieve the desired torque
# The minimum error solution should use MTQ-z at saturation

# Let's verify
u_optimal = np.array([0, 0, 0, 0.2])  # MTQ-z at max
tau_optimal = A_total @ u_optimal
error_optimal = tau_des - tau_optimal
obj_optimal = 0.5 * np.dot(error_optimal, error_optimal)

print(f"\nExpected optimal:")
print(f"  u = {u_optimal}")
print(f"  tau = {tau_optimal}")
print(f"  error = {error_optimal}")
print(f"  objective = {obj_optimal:.2e}")

# Compare with what lsq_linear found
print(f"\nComparison:")
print(f"  lsq_linear objective: {objective(res1.x):.2e}")
print(f"  Expected optimal objective: {obj_optimal:.2e}")
print(f"  lsq_linear is worse by: {objective(res1.x) - obj_optimal:.2e}")

# AHA! The issue is the solvers are finding a LOCAL minimum
# Let's try different initial points

print("\n" + "=" * 60)
print("Testing multiple starting points")
print("=" * 60)

best_obj = float('inf')
best_u = None

for _ in range(100):
    u0 = np.random.uniform(lb, ub)
    res = minimize(objective, u0, jac=gradient, method='L-BFGS-B',
                  bounds=[(lb[i], ub[i]) for i in range(4)])
    if res.fun < best_obj:
        best_obj = res.fun
        best_u = res.x

print(f"\nBest from 100 random starts:")
print(f"  u = {best_u}")
print(f"  tau = {A_total @ best_u}")
print(f"  objective = {best_obj:.2e}")
