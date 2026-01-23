"""
Boundary Value Problem Derivation for Optimal Desaturation Trajectories
=======================================================================

Full mathematical derivation of the optimal control / calculus of variations
approach to trajectory generation.
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def full_derivation():
    """Complete mathematical derivation."""
    print("=" * 80)
    print("OPTIMAL TRAJECTORY VIA CALCULUS OF VARIATIONS")
    print("=" * 80)
    
    print("""
PROBLEM SETUP
=============

State: Attitude represented by quaternion q(t) ∈ S³ ⊂ ℝ⁴
       Or equivalently, rotation matrix R(t) ∈ SO(3)

Control: Angular velocity ω(t) ∈ ℝ³

Kinematics: q̇ = ½ q ⊗ [0, ω]ᵀ
            (quaternion derivative from angular velocity)

Desaturation potential: D(q, t) = sin(angle between B_body and a_rw)
    where B_body = R(q)ᵀ B_eci(t)
    
    D = 1 when B ⊥ a_rw (maximum torque capability)
    D = 0 when B ∥ a_rw (zero torque capability)

Boundary conditions:
    q(0) = q₀  (initial attitude)
    q(T) = qf  (final attitude)


OPTIMAL CONTROL FORMULATION
===========================

We want to find the trajectory that:
1. Connects q₀ to qf
2. Maximizes time-integrated desaturation potential
3. With reasonable path length (not too long)

Cost functional:
    J[q(·)] = ∫₀ᵀ L(q, q̇, t) dt

where the Lagrangian is:
    L(q, q̇, t) = ½||ω||² - λ·D(q, t)
                = (path length penalty) - λ·(desaturation reward)

Note: ||ω||² = ||q̇||² for unit quaternions (up to scaling)

The parameter λ > 0 controls the trade-off:
    λ small → shortest path (geodesic)
    λ large → maximize desaturation (may detour significantly)


EULER-LAGRANGE EQUATIONS
========================

For unconstrained optimization on ℝⁿ:
    d/dt(∂L/∂q̇) - ∂L/∂q = 0

For our Lagrangian L = ½||ω||² - λD(q,t):

1. Gradient of path length term:
   ∂/∂q̇(½||ω||²) = ω  (the angular velocity itself)
   
   Note: q̇ and ω are related by q̇ = ½ q ⊗ [0,ω]
   So ∂(½||ω||²)/∂q̇ involves the relationship.

2. Gradient of desaturation term:
   ∂D/∂q = gradient of sin(angle(B_body, a_rw)) w.r.t. attitude
   
Let me work this out more carefully in the rotation matrix formulation...


ROTATION MATRIX FORMULATION (cleaner math)
==========================================

State: R(t) ∈ SO(3)
Kinematics: Ṙ = R·[ω]× where [ω]× is skew-symmetric matrix of ω

Desaturation potential: 
    D(R) = ||B_body - (B_body·a)a|| / ||B||
    where B_body = Rᵀ·B_eci and a = a_rw

Let b = B_eci/||B_eci|| (unit vector in ECI).
Then: D(R) = ||Rᵀb - (Rᵀb·a)a|| = ||(I - aaᵀ)Rᵀb||

Let P_⊥ = I - aaᵀ (projection onto plane perpendicular to a_rw).
Then: D(R) = ||P_⊥ Rᵀ b||

Gradient of D with respect to R:
    ∂D/∂R = ∂/∂R ||P_⊥ Rᵀ b||

Let v = P_⊥ Rᵀ b, so D = ||v||.

∂D/∂R = (v/||v||)ᵀ · ∂v/∂R
       = (v/||v||)ᵀ · P_⊥ · ∂(Rᵀb)/∂R

Now, ∂(Rᵀb)/∂R is a 3rd-order tensor. But we only need the directional
derivative in the tangent space of SO(3).

For SO(3), tangent vectors are of the form R·[ξ]× for ξ ∈ ℝ³.
The directional derivative is:
    d/dε|_{ε=0} (R·exp(ε[ξ]×))ᵀ b = d/dε|_{ε=0} exp(-ε[ξ]×)·Rᵀ b
                                   = -[ξ]× Rᵀb = ξ × (Rᵀb) = -Rᵀb × ξ

So: ∂(Rᵀb)/∂ξ = -[Rᵀb]× (as a linear map from ξ to the derivative)

Therefore:
    ∂D/∂ξ = (v/||v||)ᵀ · P_⊥ · (-[Rᵀb]×) · ξ

The gradient (in body frame, as angular velocity direction) is:
    ∇_ω D = [Rᵀb]×ᵀ · P_⊥ᵀ · (v/||v||)
          = -[Rᵀb]× · P_⊥ · (v/||v||)  (since skew matrices are antisymmetric)
          = -(Rᵀb) × (P_⊥ v / ||v||)

Let's simplify. We have:
    v = P_⊥ Rᵀ b = (I - aaᵀ) Rᵀ b
    
So P_⊥ v = P_⊥² Rᵀ b = P_⊥ Rᵀ b = v (since P_⊥ is idempotent).

Therefore:
    ∇_ω D = -(Rᵀb) × (v/||v||) = -(Rᵀb) × v̂

where v̂ = v/||v|| is the unit vector in the direction of the perpendicular
component of B_body.


EULER-LAGRANGE IN BODY FRAME
============================

The Lagrangian in terms of ω:
    L = ½||ω||² - λD(R)

The Euler-Lagrange equation for rotational mechanics:
    d/dt(∂L/∂ω) + ω × (∂L/∂ω) = Rᵀ · (∂L/∂R)·R

For our L:
    ∂L/∂ω = ω
    d/dt(ω) = ω̇
    ω × ω = 0

The term Rᵀ·(∂L/∂R)·R extracts the "torque-like" gradient.
For our D(R), this is: -λ·∇_ω D = -λ·(-(Rᵀb) × v̂) = λ·(Rᵀb) × v̂

So the Euler-Lagrange equation becomes:
    ω̇ = λ · (Rᵀb) × v̂
    
where v̂ = normalize((I - aaᵀ)Rᵀb).


PHYSICAL INTERPRETATION
=======================

    ω̇ = λ · (B_body × v̂)

This says: accelerate angularly in the direction B_body × v̂.

What is this direction?
- B_body is the magnetic field in body frame
- v̂ is the perpendicular component of B_body (relative to RW axis)
- B_body × v̂ is perpendicular to both

This torque-like term rotates the spacecraft to make B_body more
perpendicular to a_rw, increasing D!


THE BOUNDARY VALUE PROBLEM
==========================

State: x = [q, ω] ∈ S³ × ℝ³ (7 components, but q is on S³)

Dynamics:
    q̇ = ½ q ⊗ [0, ω]
    ω̇ = λ · (Rᵀb) × v̂

Boundary conditions:
    q(0) = q₀
    q(T) = qf
    (no constraint on ω at boundaries - "natural" BC)

This is a two-point boundary value problem (BVP) with:
- 4 state components (quaternion)
- 3 control/costate components (angular velocity)
- 4 boundary conditions at t=0
- 4 boundary conditions at t=T
- Free ω at boundaries → additional conditions from transversality

Total: 7 ODEs with 8 boundary conditions... but quaternion has 1 constraint
(||q||=1), so effectively 6 DOF with 6 boundary conditions (attitude at each end).

This is well-posed for shooting methods.


SIMPLIFIED VERSION (for practical use)
======================================

The full BVP is complex. A simpler approach:

1. Parameterize the path by arc length s ∈ [0, 1]

2. Discretize: s = 0, h, 2h, ..., 1 with N+1 points

3. Represent attitude at each point: qᵢ for i = 0, ..., N

4. Define discrete energy:
   E = Σᵢ ||qᵢ₊₁ - qᵢ||² + μ Σᵢ (1 - D(qᵢ))²
   
   (path length) + (desaturation penalty)

5. Fix q₀ and qN, minimize E over q₁, ..., qN₋₁

6. This is unconstrained optimization over (N-1) × 4 variables
   (with quaternion normalization as soft constraint)


GRADIENT FOR NUMERICAL OPTIMIZATION
===================================

∂E/∂qᵢ = 2(qᵢ - qᵢ₋₁) + 2(qᵢ - qᵢ₊₁) - 2μ(1-D(qᵢ))·∂D/∂qᵢ

where ∂D/∂qᵢ is computed from the earlier derivation:
    ∂D/∂q = ... (involves rotation matrix gradients)

For numerical stability, use finite differences:
    ∂D/∂q ≈ (D(q + εeᵢ) - D(q - εeᵢ)) / (2ε)


SUMMARY OF BVP APPROACH
=======================

The mathematically optimal trajectory satisfies:

    q̇ = ½ q ⊗ [0, ω]
    ω̇ = λ · (R(q)ᵀ B_eci) × normalize((I - a_rw·a_rwᵀ) R(q)ᵀ B_eci)

with boundary conditions q(0) = q₀, q(T) = qf.

This is the Euler-Lagrange equation for the functional
    J = ∫ (½||ω||² - λD(q)) dt

Solving this BVP gives the trajectory that optimally trades off
path length and desaturation potential.
""")
    return


def numerical_bvp_solution():
    """Implement and solve the BVP numerically."""
    print("\n" + "=" * 80)
    print("NUMERICAL SOLUTION OF THE BVP")
    print("=" * 80)
    
    # Helper functions
    def quat_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_to_rotmat(q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
    def D_potential(q, B_eci, a_rw):
        """Desaturation potential."""
        R = quat_to_rotmat(q)
        B_body = R.T @ B_eci
        B_body_norm = B_body / np.linalg.norm(B_body)
        # D = ||(I - a·aᵀ)·B_body_norm||
        P_perp = np.eye(3) - np.outer(a_rw, a_rw)
        v = P_perp @ B_body_norm
        return np.linalg.norm(v)
    
    def D_gradient_omega(q, B_eci, a_rw):
        """Gradient of D w.r.t. angular velocity (body frame)."""
        R = quat_to_rotmat(q)
        B_body = R.T @ B_eci
        b = B_body / np.linalg.norm(B_body)
        
        P_perp = np.eye(3) - np.outer(a_rw, a_rw)
        v = P_perp @ b
        v_norm = np.linalg.norm(v)
        
        if v_norm < 1e-10:
            return np.zeros(3)
        
        v_hat = v / v_norm
        # ∇_ω D = -(B_body) × v̂
        return -np.cross(b, v_hat)
    
    # Problem setup
    q0 = np.array([1.0, 0, 0, 0])  # Identity
    qf = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])  # 90° around x
    B_eci = np.array([0.3, 0.1, 0.9])
    B_eci = B_eci / np.linalg.norm(B_eci)
    a_rw = np.array([0, 0, 1])  # z-axis wheel
    
    lam = 5.0  # Trade-off parameter
    T = 1.0    # Normalized time
    
    print(f"Setup:")
    print(f"  q₀ = {q0}")
    print(f"  qf = {qf}")
    print(f"  B_eci = {B_eci}")
    print(f"  a_rw = {a_rw}")
    print(f"  λ = {lam}")
    
    # State: y = [q (4), ω (3)] = 7 components
    def ode(t, y):
        q = y[:4]
        q = q / np.linalg.norm(q)  # Normalize
        omega = y[4:7]
        
        # q̇ = ½ q ⊗ [0, ω]
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quat_mult(q, omega_quat)
        
        # ω̇ = λ · ∇_ω D
        grad_D = D_gradient_omega(q, B_eci, a_rw)
        omega_dot = lam * grad_D
        
        return np.concatenate([q_dot, omega_dot])
    
    def bc(ya, yb):
        """Boundary conditions: q(0) = q0, q(T) = qf."""
        # 4 conditions at t=0 (quaternion)
        # 4 conditions at t=T (quaternion)
        # That's 8 conditions for 7 states... need to relax one.
        # Use 3 components of q at each end (the 4th is determined by ||q||=1)
        
        residual = np.zeros(7)
        residual[0:3] = ya[:3] - q0[:3]  # First 3 components of q at t=0
        residual[3:6] = yb[:3] - qf[:3]  # First 3 components of q at t=T
        # Natural BC for omega: ω(T) is free, or we can require ω(T) = 0
        residual[6] = yb[4]  # ω_x(T) = 0 (stopping condition)
        
        return residual
    
    # Initial guess: geodesic interpolation
    def slerp(q1, q2, t):
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta = np.arccos(dot)
        return (np.sin((1-t)*theta)*q1 + np.sin(t*theta)*q2) / np.sin(theta)
    
    N = 20
    t_init = np.linspace(0, T, N)
    y_init = np.zeros((7, N))
    
    for i, t in enumerate(t_init):
        y_init[:4, i] = slerp(q0, qf, t/T)
        # Initial omega from finite difference
        if i > 0:
            dq = y_init[:4, i] - y_init[:4, i-1]
            dt = t_init[i] - t_init[i-1]
            # Approximate omega (crude)
            y_init[4:7, i] = 2 * dq[1:4] / dt
    
    print(f"\nSolving BVP...")
    
    try:
        sol = solve_bvp(ode, bc, t_init, y_init, verbose=2, max_nodes=1000)
        
        if sol.success:
            print(f"BVP solved successfully!")
            
            # Evaluate solution
            t_eval = np.linspace(0, T, 50)
            y_eval = sol.sol(t_eval)
            
            # Compute D along trajectory
            D_trajectory = []
            for i in range(len(t_eval)):
                q = y_eval[:4, i]
                q = q / np.linalg.norm(q)
                D_trajectory.append(D_potential(q, B_eci, a_rw))
            
            D_trajectory = np.array(D_trajectory)
            
            # Compare to geodesic
            D_geodesic = []
            for t in t_eval:
                q = slerp(q0, qf, t/T)
                D_geodesic.append(D_potential(q, B_eci, a_rw))
            D_geodesic = np.array(D_geodesic)
            
            print(f"\nResults:")
            print(f"  Geodesic: D_avg = {np.mean(D_geodesic):.3f}, D_min = {np.min(D_geodesic):.3f}")
            print(f"  BVP:      D_avg = {np.mean(D_trajectory):.3f}, D_min = {np.min(D_trajectory):.3f}")
            print(f"  Improvement: {(np.mean(D_trajectory)-np.mean(D_geodesic))/np.mean(D_geodesic)*100:.1f}%")
            
        else:
            print(f"BVP failed: {sol.message}")
            
    except Exception as e:
        print(f"BVP solver error: {e}")
        print("Note: BVP on SO(3) is numerically challenging.")
        print("The discretized optimization approach is more robust.")
    
    return


def discretized_optimization():
    """Solve via discretized optimization (more robust)."""
    print("\n" + "=" * 80)
    print("DISCRETIZED OPTIMIZATION APPROACH")
    print("=" * 80)
    
    from scipy.optimize import minimize
    
    def quat_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_to_rotmat(q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
    def D_potential(q, B_eci, a_rw):
        R = quat_to_rotmat(q)
        B_body = R.T @ B_eci
        B_body_norm = B_body / np.linalg.norm(B_body)
        P_perp = np.eye(3) - np.outer(a_rw, a_rw)
        v = P_perp @ B_body_norm
        return np.linalg.norm(v)
    
    def slerp(q1, q2, t):
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta = np.arccos(dot)
        return (np.sin((1-t)*theta)*q1 + np.sin(t*theta)*q2) / np.sin(theta)
    
    # Setup
    q0 = np.array([1.0, 0, 0, 0])
    qf = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
    B_eci = np.array([0.3, 0.1, 0.9])
    B_eci = B_eci / np.linalg.norm(B_eci)
    a_rw = np.array([0, 0, 1])
    
    N = 10  # Number of interior points
    mu = 1.0  # Desaturation weight
    
    print(f"Discretized optimization with N={N} interior points, μ={mu}")
    
    def energy(x):
        """Total energy: path length + desaturation penalty."""
        # Reconstruct quaternions
        quats = [q0]
        for i in range(N):
            qi = x[4*i:4*(i+1)]
            qi = qi / np.linalg.norm(qi)  # Normalize
            quats.append(qi)
        quats.append(qf)
        
        # Path length (sum of squared distances)
        path_length = 0
        for i in range(len(quats)-1):
            # Quaternion distance
            dot = abs(np.dot(quats[i], quats[i+1]))
            angle = 2 * np.arccos(np.clip(dot, 0, 1))
            path_length += angle**2
        
        # Desaturation penalty
        desat_penalty = 0
        for i in range(1, len(quats)-1):  # Interior points only
            D = D_potential(quats[i], B_eci, a_rw)
            desat_penalty += (1 - D)**2
        
        return path_length + mu * desat_penalty
    
    # Initial guess: geodesic
    x0 = []
    for i in range(N):
        t = (i + 1) / (N + 1)
        qi = slerp(q0, qf, t)
        x0.extend(qi)
    x0 = np.array(x0)
    
    # Optimize
    print("Optimizing...")
    result = minimize(energy, x0, method='L-BFGS-B', 
                     options={'maxiter': 1000, 'disp': True})
    
    if result.success:
        print(f"Optimization converged!")
        
        # Extract optimized trajectory
        quats_opt = [q0]
        for i in range(N):
            qi = result.x[4*i:4*(i+1)]
            qi = qi / np.linalg.norm(qi)
            quats_opt.append(qi)
        quats_opt.append(qf)
        
        # Compare D values
        D_geodesic = []
        D_optimized = []
        
        for i in range(N+2):
            t = i / (N + 1)
            q_geo = slerp(q0, qf, t)
            D_geodesic.append(D_potential(q_geo, B_eci, a_rw))
            D_optimized.append(D_potential(quats_opt[i], B_eci, a_rw))
        
        print(f"\nResults:")
        print(f"  Geodesic:  D_avg = {np.mean(D_geodesic):.3f}, D_min = {np.min(D_geodesic):.3f}")
        print(f"  Optimized: D_avg = {np.mean(D_optimized):.3f}, D_min = {np.min(D_optimized):.3f}")
        print(f"  Improvement: {(np.mean(D_optimized)-np.mean(D_geodesic))/np.mean(D_geodesic)*100:.1f}%")
        
    else:
        print(f"Optimization failed: {result.message}")
    
    return


def summary():
    """Summary of BVP approach."""
    print("\n" + "=" * 80)
    print("SUMMARY: BVP FOR DESATURATION TRAJECTORIES")
    print("=" * 80)
    
    print("""
THE MATHEMATICS:
================

The optimal trajectory q(t) satisfies the Euler-Lagrange equation:

    q̇ = ½ q ⊗ [0, ω]ᵀ
    
    ω̇ = λ · (R(q)ᵀ b) × v̂

where:
    b = B_eci / ||B_eci||  (unit magnetic field vector)
    v = (I - a·aᵀ) R(q)ᵀ b  (perpendicular component of B in body frame)
    v̂ = v / ||v||          (unit vector)
    λ = trade-off parameter

This is derived from the cost functional:
    J = ∫₀ᵀ (½||ω||² - λ·D(q)) dt

The second equation says:
    "Angular acceleration is proportional to B_body × v̂"
    
This torque-like term naturally steers the spacecraft toward attitudes
where B is more perpendicular to the RW axis (higher D).


BOUNDARY VALUE PROBLEM:
=======================

Given: q(0) = q₀, q(T) = qf
Find: ω(0), ω(T) such that the trajectory connects q₀ to qf

This is a two-point BVP on SO(3) × ℝ³.

Methods to solve:
1. Shooting method (initial value → adjust until BC met)
2. Collocation (discretize, solve nonlinear system)
3. Discretized optimization (treat as unconstrained optimization)


PRACTICAL RECOMMENDATIONS:
==========================

1. For simple cases: Use the discretized optimization
   - More robust than full BVP
   - Easy to implement
   - Gives good results

2. For real-time: Use the closed-form geodesic perturbation
   - ε(s) = ε_max · 16s²(1-s)²
   - n(s) = direction toward higher D
   - Very fast, no optimization needed

3. For maximum performance: Solve full BVP with multiple shooting
   - Most accurate
   - Computationally expensive
   - May have convergence issues

The trade-off parameter λ should be chosen based on:
    - Available time for slew
    - Current momentum level
    - Importance of desaturation vs. speed
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    full_derivation()
    numerical_bvp_solution()
    discretized_optimization()
    summary()
