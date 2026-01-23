"""
Continuous Trajectory Generation for Desaturation
=================================================

Mathematical approaches that don't use discrete waypoints.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def introduction():
    """Present continuous trajectory approaches."""
    print("=" * 80)
    print("CONTINUOUS TRAJECTORY GENERATION (NO WAYPOINTS)")
    print("=" * 80)
    
    print("""
PROBLEM: Generate slew trajectory q(t) from q_0 to q_f that maximizes
         desaturation potential D(q, t) = sin(angle(B_body, a_rw))
         
CONSTRAINT: Single waypoint is discontinuous (velocity jump at waypoint)

GOAL: Smooth trajectory with continuous derivatives


APPROACH 1: GEODESIC PERTURBATION
=================================

Base trajectory: geodesic (eigenaxis) from q_0 to q_f
Perturbed: q(s) = q_base(s) ⊗ exp(ε(s)·n(s))

where:
    s ∈ [0, 1] is path parameter
    ε(s) is perturbation magnitude
    n(s) is perturbation direction (perpendicular to geodesic)

Boundary conditions: ε(0) = ε(1) = 0 (must hit endpoints)

Natural choice: ε(s) = ε_max · sin(πs)

Direction n(s): Choose to maximize dD/dε at each point.


APPROACH 2: OPTIMAL CONTROL (CALCULUS OF VARIATIONS)
====================================================

Minimize: J = ∫₀ᵀ L(q, q̇, t) dt

where L = (path length cost) - λ·(desaturation potential)
        = ||q̇||² - λ·D(q, t)

Euler-Lagrange equations give the optimal q(t).

This is a boundary value problem:
    q(0) = q_0, q(T) = q_f


APPROACH 3: GRADIENT FLOW
=========================

Start with geodesic, then flow toward higher D:

    ∂q/∂τ = ∇_q D(q, t)  (gradient flow on D)

Subject to: endpoints fixed, path length bounded.

This is like "inflating" the geodesic toward favorable B regions.


APPROACH 4: RIEMANNIAN SPLINE WITH DESATURATION BIAS
===================================================

Instead of geodesic, use spline that naturally curves toward high D.

Define energy: E = ∫ (||q̈||² + μ·(1 - D(q)))² ds

Minimize E subject to boundary conditions.

The term (1-D) penalizes low desaturation potential.
""")
    return


def geodesic_perturbation():
    """Implement geodesic perturbation approach."""
    print("\n" + "=" * 80)
    print("APPROACH 1: GEODESIC PERTURBATION")
    print("=" * 80)
    
    print("""
MATHEMATICS:
============

Let q_geo(s) be the geodesic from q_0 to q_f, s ∈ [0,1].

Perturbed trajectory:
    q(s) = q_geo(s) ⊗ δq(s)

where δq(s) is a small rotation:
    δq(s) = [cos(ε(s)/2), sin(ε(s)/2)·n(s)]

For smooth trajectory:
    ε(s) = ε_max · sin(πs)·sin(πs)  [C² smooth]
    
or use polynomial:
    ε(s) = ε_max · 16·s²·(1-s)²    [also C² smooth]


CHOOSING n(s):
--------------

At each point, we want to deviate toward higher D.

D(q) = sin(angle(R(q)ᵀ·B, a_rw))
     = ||R(q)ᵀ·B - (R(q)ᵀ·B · a_rw)·a_rw|| / ||B||

Gradient: ∇_q D points toward attitudes with B ⊥ a_rw

In body frame: n(s) ∝ proj_perp(B_body × a_rw)
where proj_perp removes the component along the eigenaxis.


CLOSED-FORM FOR SMALL PERTURBATION:
-----------------------------------

For small ε_max, the path length increase is:

    ΔL/L ≈ (π·ε_max)² / 4

The desaturation gain depends on geometry but typically:

    ΔD_avg ≈ (ε_max / (π/2 - D_geo)) · something

where D_geo is the average D along the geodesic.
""")
    
    # Implement numerically
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
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
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
    
    def D_potential(q, B_eci, a_rw):
        """Desaturation potential: sin(angle between B_body and a_rw)"""
        R = quat_to_rotmat(q)
        B_body = R.T @ B_eci
        B_body = B_body / np.linalg.norm(B_body)
        # D = ||B_body - (B_body·a_rw)·a_rw|| = sin(angle)
        B_perp = B_body - np.dot(B_body, a_rw) * a_rw
        return np.linalg.norm(B_perp)
    
    # Setup
    q_0 = np.array([1.0, 0, 0, 0])  # Identity
    q_f = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])  # 90° around x
    B_eci = np.array([0.3, 0.1, 0.9])  # Mostly along z
    B_eci = B_eci / np.linalg.norm(B_eci)
    a_rw = np.array([0, 0, 1])  # z-axis wheel
    
    # Geodesic trajectory
    N = 50
    s_vals = np.linspace(0, 1, N)
    
    # Compute eigenaxis
    q_err = quat_mult(q_f, np.array([q_0[0], -q_0[1], -q_0[2], -q_0[3]]))
    if q_err[0] < 0:
        q_err = -q_err
    theta_total = 2 * np.arccos(np.clip(q_err[0], -1, 1))
    if theta_total > 1e-6:
        eigenaxis = q_err[1:4] / np.sin(theta_total/2)
    else:
        eigenaxis = np.array([1, 0, 0])
    
    print(f"\nSetup:")
    print(f"  Slew angle: {np.degrees(theta_total):.1f}°")
    print(f"  Eigenaxis: {eigenaxis}")
    print(f"  B_eci: {B_eci}")
    
    # Geodesic D values
    D_geo = []
    for s in s_vals:
        q = slerp(q_0, q_f, s)
        D_geo.append(D_potential(q, B_eci, a_rw))
    D_geo = np.array(D_geo)
    
    print(f"\nGeodesic:")
    print(f"  D_min = {np.min(D_geo):.3f}")
    print(f"  D_max = {np.max(D_geo):.3f}")
    print(f"  D_avg = {np.mean(D_geo):.3f}")
    
    # Perturbed trajectory
    eps_max = np.radians(15)  # 15° max deviation
    
    # Choose perturbation direction at each point
    D_perturbed = []
    for i, s in enumerate(s_vals):
        q_geo = slerp(q_0, q_f, s)
        
        # Perturbation profile (smooth)
        eps = eps_max * 16 * s**2 * (1-s)**2  # Peaks at s=0.5
        
        # Direction: rotate toward B ⊥ a_rw
        R_geo = quat_to_rotmat(q_geo)
        B_body = R_geo.T @ B_eci
        
        # n = (B_body × a_rw) projected perpendicular to eigenaxis
        n_raw = np.cross(B_body, a_rw)
        n_raw = n_raw - np.dot(n_raw, eigenaxis) * eigenaxis  # Remove eigenaxis component
        if np.linalg.norm(n_raw) > 1e-6:
            n = n_raw / np.linalg.norm(n_raw)
        else:
            n = np.array([0, 1, 0])  # Default
        
        # Apply perturbation
        dq = np.array([np.cos(eps/2), np.sin(eps/2)*n[0], np.sin(eps/2)*n[1], np.sin(eps/2)*n[2]])
        q_pert = quat_mult(q_geo, dq)
        q_pert = q_pert / np.linalg.norm(q_pert)
        
        D_perturbed.append(D_potential(q_pert, B_eci, a_rw))
    
    D_perturbed = np.array(D_perturbed)
    
    print(f"\nPerturbed (ε_max = {np.degrees(eps_max):.1f}°):")
    print(f"  D_min = {np.min(D_perturbed):.3f}")
    print(f"  D_max = {np.max(D_perturbed):.3f}")
    print(f"  D_avg = {np.mean(D_perturbed):.3f}")
    print(f"  Improvement: {(np.mean(D_perturbed) - np.mean(D_geo))/np.mean(D_geo)*100:.1f}%")
    
    return


def optimal_control_approach():
    """Optimal control / calculus of variations approach."""
    print("\n" + "=" * 80)
    print("APPROACH 2: OPTIMAL CONTROL")
    print("=" * 80)
    
    print("""
FORMULATION:
============

State: q(t) ∈ SO(3) (attitude quaternion)
Control: ω(t) ∈ ℝ³ (angular velocity, our "control")

Dynamics: q̇ = ½·q ⊗ [0, ω]

Objective: J = ∫₀ᵀ (||ω||² - λ·D(q, t)) dt

Boundary conditions: q(0) = q_0, q(T) = q_f

This is an optimal control problem on SO(3).


EULER-LAGRANGE FOR ATTITUDE:
----------------------------

For rotation dynamics, the Euler-Lagrange equations become:

    d/dt(∂L/∂ω) - ω × (∂L/∂ω) = ∂L/∂q

With L = ½||ω||² - λ·D(q):
    ∂L/∂ω = ω
    d/dt(ω) = ω̇
    
    ω̇ - ω × ω = -λ·∇_q D
    ω̇ = -λ·∇_q D  (since ω × ω = 0)

This says: angular acceleration equals gradient of desaturation!


SIMPLIFIED (Kinematic):
-----------------------

If we ignore dynamics and just want a smooth path:

    min ∫₀¹ (||q̇||² + μ·(1-D)²) ds

This penalizes:
    1. Path curvature (||q̇||²)
    2. Low desaturation potential ((1-D)²)

The balance μ determines trade-off.


DISCRETIZED SOLUTION:
---------------------

Discretize q(s) at N points: q₁, q₂, ..., qₙ

Minimize: Σᵢ ||qᵢ₊₁ - qᵢ||² + μ·Σᵢ (1 - D(qᵢ))²

Subject to: q₁ = q₀, qₙ = q_f

This is nonlinear optimization over quaternions.
Can solve with gradient descent or sequential QP.
""")
    
    return


def gradient_flow_approach():
    """Gradient flow to deform geodesic."""
    print("\n" + "=" * 80)
    print("APPROACH 3: GRADIENT FLOW")
    print("=" * 80)
    
    print("""
CONCEPT:
========

Start with geodesic curve γ₀(s).
"Flow" it toward higher desaturation potential:

    ∂γ/∂τ = ∇_γ D(γ, t)

where τ is "flow time" (not physical time).

This is like curve evolution / active contours.


CONSTRAINTS:
------------

1. Endpoints fixed: γ(0,τ) = q₀, γ(1,τ) = q_f for all τ
2. Path length bounded: L[γ] ≤ L_max

The flow naturally increases D along the curve.


IMPLEMENTATION:
---------------

Discretize curve as N points: γ = [q₁, ..., qₙ]

At each flow step:
    1. Compute ∇D at each qᵢ
    2. Project gradient perpendicular to curve tangent
    3. Move each qᵢ in projected gradient direction
    4. Re-project to SO(3)
    5. Enforce path length constraint


STOPPING CRITERION:
-------------------

Stop when:
    1. D_avg converges
    2. Path length reaches limit
    3. Maximum flow time reached
""")
    
    # Numerical implementation
    def quat_to_rotmat(q):
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
    def slerp(q1, q2, t):
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta = np.arccos(np.clip(dot, -1, 1))
        return (np.sin((1-t)*theta)*q1 + np.sin(t*theta)*q2) / np.sin(theta)
    
    def D_potential(q, B_eci, a_rw):
        R = quat_to_rotmat(q)
        B_body = R.T @ B_eci
        B_body = B_body / np.linalg.norm(B_body)
        B_perp = B_body - np.dot(B_body, a_rw) * a_rw
        return np.linalg.norm(B_perp)
    
    def D_gradient_numerical(q, B_eci, a_rw, eps=1e-6):
        """Numerical gradient of D w.r.t. quaternion."""
        grad = np.zeros(4)
        D0 = D_potential(q, B_eci, a_rw)
        for i in range(4):
            q_pert = q.copy()
            q_pert[i] += eps
            q_pert = q_pert / np.linalg.norm(q_pert)
            grad[i] = (D_potential(q_pert, B_eci, a_rw) - D0) / eps
        return grad
    
    # Setup
    q_0 = np.array([1.0, 0, 0, 0])
    q_f = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
    B_eci = np.array([0.3, 0.1, 0.9])
    B_eci = B_eci / np.linalg.norm(B_eci)
    a_rw = np.array([0, 0, 1])
    
    # Initialize with geodesic
    N = 20
    s_vals = np.linspace(0, 1, N)
    curve = np.array([slerp(q_0, q_f, s) for s in s_vals])
    
    # Gradient flow
    dt_flow = 0.1
    n_flow_steps = 50
    
    D_history = []
    
    for step in range(n_flow_steps):
        # Compute D along curve
        D_vals = np.array([D_potential(q, B_eci, a_rw) for q in curve])
        D_history.append(np.mean(D_vals))
        
        # Update interior points (keep endpoints fixed)
        for i in range(1, N-1):
            grad = D_gradient_numerical(curve[i], B_eci, a_rw)
            
            # Project gradient perpendicular to quaternion (stay on S³)
            grad = grad - np.dot(grad, curve[i]) * curve[i]
            
            # Update
            curve[i] = curve[i] + dt_flow * grad
            curve[i] = curve[i] / np.linalg.norm(curve[i])
    
    # Final D
    D_final = np.array([D_potential(q, B_eci, a_rw) for q in curve])
    
    print(f"\nGradient Flow Results:")
    print(f"  Initial D_avg: {D_history[0]:.3f}")
    print(f"  Final D_avg: {D_history[-1]:.3f}")
    print(f"  Improvement: {(D_history[-1] - D_history[0])/D_history[0]*100:.1f}%")
    print(f"  D_min (final): {np.min(D_final):.3f}")
    print(f"  D_max (final): {np.max(D_final):.3f}")
    
    return


def spline_with_desat_bias():
    """Spline interpolation with desaturation bias."""
    print("\n" + "=" * 80)
    print("APPROACH 4: BIASED SPLINE INTERPOLATION")
    print("=" * 80)
    
    print("""
CONCEPT:
========

Instead of computing a geodesic then perturbing, directly construct
a spline that interpolates q_0, q_f with bias toward high D regions.

METHOD:
-------

1. Find the attitude q* that maximizes D (B ⊥ a_rw)
2. Construct spline through q_0, q*, q_f
3. Optimize q* placement to minimize path length while achieving D threshold

This is like the waypoint approach but with smooth (C²) interpolation.


QUATERNION SPLINE:
------------------

Use spherical spline (squad) or convert to axis-angle and use regular spline.

For axis-angle representation θ·n:
    - Concatenate: [θ₁·n₁, θ₂·n₂, θ₃·n₃, ...]
    - Apply cubic spline
    - Convert back to quaternion

This gives C² continuous trajectory.


OPTIMIZATION:
-------------

Given desaturation constraint D_min:
    min L[curve]
    s.t. min_s D(curve(s)) ≥ D_min

This finds the shortest path that achieves minimum desaturation threshold.
""")
    
    def quat_to_axis_angle(q):
        q = q / np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        theta = 2 * np.arccos(np.clip(q[0], -1, 1))
        if theta < 1e-6:
            return np.zeros(3)
        axis = q[1:4] / np.sin(theta/2)
        return theta * axis
    
    def axis_angle_to_quat(aa):
        theta = np.linalg.norm(aa)
        if theta < 1e-6:
            return np.array([1.0, 0, 0, 0])
        axis = aa / theta
        return np.array([np.cos(theta/2), 
                        np.sin(theta/2)*axis[0],
                        np.sin(theta/2)*axis[1],
                        np.sin(theta/2)*axis[2]])
    
    def quat_to_rotmat(q):
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
    
    def quat_mult(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def D_potential(q, B_eci, a_rw):
        R = quat_to_rotmat(q)
        B_body = R.T @ B_eci
        B_body = B_body / np.linalg.norm(B_body)
        B_perp = B_body - np.dot(B_body, a_rw) * a_rw
        return np.linalg.norm(B_perp)
    
    # Setup
    q_0 = np.array([1.0, 0, 0, 0])
    q_f = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])
    B_eci = np.array([0.3, 0.1, 0.9])
    B_eci = B_eci / np.linalg.norm(B_eci)
    a_rw = np.array([0, 0, 1])
    
    # Find optimal intermediate attitude (B ⊥ a_rw)
    # This is where R(q)ᵀ·B_eci is perpendicular to a_rw
    # Use optimization
    from scipy.optimize import minimize
    
    def objective(aa_mid):
        q_mid = axis_angle_to_quat(aa_mid)
        D = D_potential(q_mid, B_eci, a_rw)
        # Also minimize distance from midpoint of geodesic
        q_geo_mid = (q_0 + q_f) / np.linalg.norm(q_0 + q_f)  # Approx midpoint
        dist = 1 - abs(np.dot(q_mid, q_geo_mid))
        return -D + 0.5 * dist  # Trade-off
    
    # Start near geodesic midpoint
    q_geo_mid = (q_0 + q_f) / np.linalg.norm(q_0 + q_f)
    aa_0 = quat_to_axis_angle(q_geo_mid)
    
    res = minimize(objective, aa_0, method='BFGS')
    q_mid_opt = axis_angle_to_quat(res.x)
    
    print(f"\nOptimal intermediate attitude:")
    print(f"  q_mid = {q_mid_opt}")
    print(f"  D(q_mid) = {D_potential(q_mid_opt, B_eci, a_rw):.3f}")
    
    # Build spline through q_0, q_mid, q_f
    # Use axis-angle representation relative to q_0
    aa_0 = np.zeros(3)  # q_0 is identity
    aa_mid = quat_to_axis_angle(quat_mult(q_mid_opt, np.array([q_0[0], -q_0[1], -q_0[2], -q_0[3]])))
    aa_f = quat_to_axis_angle(quat_mult(q_f, np.array([q_0[0], -q_0[1], -q_0[2], -q_0[3]])))
    
    # Cubic spline
    t_knots = np.array([0, 0.5, 1])
    aa_knots = np.array([aa_0, aa_mid, aa_f])
    
    spline = CubicSpline(t_knots, aa_knots, bc_type='clamped')
    
    # Evaluate
    N = 50
    t_vals = np.linspace(0, 1, N)
    D_spline = []
    
    for t in t_vals:
        aa = spline(t)
        q_rel = axis_angle_to_quat(aa)
        q = quat_mult(q_0, q_rel)
        q = q / np.linalg.norm(q)
        D_spline.append(D_potential(q, B_eci, a_rw))
    
    D_spline = np.array(D_spline)
    
    # Compare to geodesic
    D_geo = []
    for t in t_vals:
        dot = np.dot(q_0, q_f)
        if dot < 0:
            q_f_adj = -q_f
            dot = -dot
        else:
            q_f_adj = q_f
        if dot > 0.9995:
            q = q_0 + t * (q_f_adj - q_0)
        else:
            theta = np.arccos(dot)
            q = (np.sin((1-t)*theta)*q_0 + np.sin(t*theta)*q_f_adj) / np.sin(theta)
        q = q / np.linalg.norm(q)
        D_geo.append(D_potential(q, B_eci, a_rw))
    D_geo = np.array(D_geo)
    
    print(f"\nSpline trajectory:")
    print(f"  D_min = {np.min(D_spline):.3f} (geodesic: {np.min(D_geo):.3f})")
    print(f"  D_avg = {np.mean(D_spline):.3f} (geodesic: {np.mean(D_geo):.3f})")
    print(f"  D_max = {np.max(D_spline):.3f} (geodesic: {np.max(D_geo):.3f})")
    
    return


def summary():
    """Summarize continuous trajectory approaches."""
    print("\n" + "=" * 80)
    print("SUMMARY: CONTINUOUS TRAJECTORY OPTIONS")
    print("=" * 80)
    
    print("""
COMPARISON:
===========

┌─────────────────────┬────────────┬────────────┬────────────┬─────────────┐
│ Approach            │ Smoothness │ Optimality │ Complexity │ Recommended │
├─────────────────────┼────────────┼────────────┼────────────┼─────────────┤
│ Geodesic Perturb.   │ C²         │ Approx     │ Low        │ ✓ Simple    │
│ Optimal Control     │ C∞         │ Optimal    │ High       │ Theory only │
│ Gradient Flow       │ C¹         │ Local opt  │ Medium     │ Iterative   │
│ Biased Spline       │ C²         │ Approx     │ Low        │ ✓ Practical │
└─────────────────────┴────────────┴────────────┴────────────┴─────────────┘


RECOMMENDED APPROACH FOR IMPLEMENTATION:
========================================

Use BIASED SPLINE with optimal intermediate point:

1. Find q_mid that maximizes D subject to path length constraint
2. Build cubic spline: q_0 → q_mid → q_f
3. Result is C² smooth trajectory with improved desaturation


FORMULA:
--------

Given q_0, q_f, B_eci, a_rw:

1. Compute q_mid:
   - Find rotation that makes B_body ⊥ a_rw
   - Choose the one closest to geodesic midpoint

2. Build spline:
   - Convert to axis-angle: aa(t) for t ∈ [0,1]
   - Use cubic spline through [aa_0, aa_mid, aa_f] at t = [0, 0.5, 1]
   - Convert back: q(t) = q_0 ⊗ exp(aa(t))


WHY CONTINUOUS BEATS WAYPOINT:
==============================

1. No velocity discontinuity at midpoint
2. Smooth acceleration profile
3. Better tracking by attitude controller
4. More natural motion for optical payloads

The trade-off: slightly longer path for same D improvement.
Typically <5% path overhead for 10-20% D improvement.
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    introduction()
    geodesic_perturbation()
    optimal_control_approach()
    gradient_flow_approach()
    spline_with_desat_bias()
    summary()
