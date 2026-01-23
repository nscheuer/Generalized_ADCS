"""
Mathematical Options for Desaturation-Favorable Trajectory Generation
=====================================================================

Generate slew trajectories that pass through attitudes favorable for
momentum desaturation, without requiring heavy optimization.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def introduction():
    """Present the problem and mathematical framework."""
    print("=" * 80)
    print("TRAJECTORY GENERATION FOR DESATURATION-FAVORABLE SLEWS")
    print("=" * 80)
    
    print("""
PROBLEM STATEMENT
=================

Given:
    - Current attitude: q_0
    - Target attitude: q_f  
    - Current stored momentum: h (in body frame)
    - Magnetic field: B(t) as function of time (from orbit model)
    - RW configuration: axis a_rw (e.g., [0,0,1] for z-axis RW)

Goal:
    Find a trajectory q(t) from q_0 to q_f that:
    1. Reaches target attitude
    2. Passes through attitudes where B ⊥ a_rw (desaturation possible)
    3. Minimizes additional path length/time


KEY INSIGHT
===========

For 3MTQ + 1RW (single z-axis wheel):
    - MTQ can produce τ in plane perpendicular to B
    - To desaturate h_z (RW momentum), need τ_z from MTQ
    - τ_z from MTQ exists when B has x or y component (B ⊥ z)
    - So we want attitudes where B_body is perpendicular to z

The "desaturation potential" at attitude q and time t is:
    D(q, t) = |B_body(q,t) × ẑ| / |B_body(q,t)|
            = √(B_x² + B_y²) / |B|
            = sin(angle between B_body and z-axis)

D = 1 when B ⊥ z (maximum desaturation potential)
D = 0 when B ∥ z (cannot desaturate at all)


APPROACH OPTIONS
================

1. Eigenaxis + offset: Rotate around eigenaxis, then offset toward favorable B
2. Two-stage slew: Slew to intermediate attitude, then to target  
3. Great circle deviation: Follow great circle but deviate toward favorable B
4. Waypoint insertion: Insert optimal waypoint(s) between start and end
5. Attitude-scheduled desaturation: Don't modify path, wait for favorable B
""")
    return


def option1_eigenaxis_offset():
    """Eigenaxis rotation with offset toward favorable B."""
    print("\n" + "=" * 80)
    print("OPTION 1: EIGENAXIS WITH OFFSET")
    print("=" * 80)
    
    print("""
CONCEPT
-------

The eigenaxis rotation is the shortest path between two attitudes.
We can "offset" the rotation axis toward a direction that will pass
through favorable B orientations.

MATHEMATICS
-----------

Standard eigenaxis from q_0 to q_f:
    q_err = q_f ⊗ q_0*
    θ = 2·arccos(q_err_w)
    e = q_err_v / sin(θ/2)  (eigenaxis)

The rotation is: q(s) = q_0 ⊗ [cos(sθ/2), sin(sθ/2)·e]  for s ∈ [0,1]

OFFSET MODIFICATION:
    e' = normalize(e + λ·e_offset)
    
where e_offset points toward favorable B orientations.

Choice of e_offset:
    - Compute B at current time (in ECI)
    - Want body z-axis to become perpendicular to B
    - So e_offset should rotate z toward (B × z) direction
    
    e_offset = normalize(B_eci × R(q_0)·ẑ)

Parameter λ controls deviation from shortest path:
    - λ = 0: pure eigenaxis
    - λ > 0: deviate toward favorable B
    - Optimal λ balances path length vs desaturation time

PATH LENGTH INCREASE:
    For small λ: Δpath ≈ λ²·θ/2
    
DESATURATION GAIN:
    Depends on geometry, but typically:
    - Extra 10° deviation → 30-50% more desaturation potential
    - Law of diminishing returns beyond ~20° deviation

PROS:
    + Simple modification to existing eigenaxis slew
    + Single parameter to tune
    + Smooth trajectory

CONS:
    - May not reach optimal B orientation
    - Fixed offset doesn't adapt to B field changes during slew
""")
    return


def option2_two_stage_slew():
    """Two-stage slew through optimal intermediate attitude."""
    print("\n" + "=" * 80)
    print("OPTION 2: TWO-STAGE SLEW")
    print("=" * 80)
    
    print("""
CONCEPT
-------

Instead of direct slew, go through an intermediate attitude q_mid
chosen to maximize desaturation.

MATHEMATICS
-----------

Stage 1: q_0 → q_mid (desaturation phase)
Stage 2: q_mid → q_f (pointing phase)

Total angle: θ_total = θ_1 + θ_2

For efficient path: θ_total ≤ θ_direct + Δθ_budget

CHOOSING q_mid:
    
Method A: Rotate body z toward B⊥
    
    At time t_mid (midpoint of slew):
        B_eci = B(t_mid)
        B_perp_desired = normalize(B_eci × any_vector_not_parallel_to_B)
        
    Choose q_mid such that:
        R(q_mid) · ẑ ⊥ B_eci
        
    This is a 1-DOF family of attitudes! (rotation around B_eci)
    Choose the one closest to the q_0 → q_f path.

Method B: Optimal placement on sphere
    
    Model the problem geometrically on S³ (quaternion space):
        - Start: q_0
        - End: q_f
        - Constraint: D(q_mid, t_mid) ≥ D_threshold
        
    Find q_mid minimizing θ_1 + θ_2 subject to constraint.
    
    This is a constrained optimization but with clear geometry:
        - Feasible set is a "band" of attitudes where B ⊥ z
        - Find point in band closest to geodesic q_0 → q_f

CLOSED-FORM APPROXIMATION:
    
Let n = normalize(z × B_eci)  (axis to rotate z toward B⊥)
    φ = angle to make z ⊥ B
    
q_mid ≈ [cos(φ/2), sin(φ/2)·n] ⊗ q_0

Then adjust along null direction (rotation around B) to minimize path.


PROS:
    + Clear desaturation phase
    + Can achieve optimal B orientation
    + Analytically tractable

CONS:
    - Discontinuous acceleration at q_mid (need blending)
    - May significantly extend path
    - Requires knowledge of B at future time
""")
    return


def option3_great_circle_deviation():
    """Deviate from great circle toward favorable B."""
    print("\n" + "=" * 80)
    print("OPTION 3: GREAT CIRCLE DEVIATION")
    print("=" * 80)
    
    print("""
CONCEPT
-------

The eigenaxis rotation traces a "great circle" on the attitude sphere.
We can parameterize deviations from this as a perturbation function.

MATHEMATICS
-----------

Base trajectory: q_base(s) for s ∈ [0,1]

Deviated trajectory:
    q(s) = q_base(s) ⊗ δq(s)
    
where δq(s) is a small perturbation quaternion.

For perpendicular perturbation:
    δq(s) = [1, ε(s)·n_perp(s)]  (small angle approx)
    
where n_perp(s) ⊥ eigenaxis and ε(s) is the deviation magnitude.

SHAPE OF ε(s):
    
To return to endpoints: ε(0) = ε(1) = 0

Simple choice: ε(s) = ε_max · sin(πs)
    - Maximum deviation at midpoint
    - Smooth return to nominal path

Better choice: ε(s) = ε_max · w(s) where w(s) peaks where D is best improved

CHOOSING n_perp(s):
    
At each s, we want to deviate in the direction that maximizes dD/dε.

    dD/dε = ∂D/∂q · ∂q/∂ε
    
For our D = sin(angle(B_body, z)):
    The gradient points toward attitudes where B_body → perpendicular to z.
    
    n_perp(s) = normalize(proj_perp(B_body × ẑ))
    
where proj_perp removes component along eigenaxis.


PATH LENGTH COST:

For small deviation:
    Δpath/path ≈ (π·ε_max)² / (8·θ)
    
So for θ = 90° and ε_max = 10°:
    Δpath ≈ (π·10°)² / (8·90°) ≈ 4.4%


PROS:
    + Smooth trajectory
    + Continuous acceleration
    + Predictable path length increase

CONS:
    - Requires computing gradient of D
    - May miss optimal B region if far from path
    - Adds complexity to trajectory generation
""")
    return


def option4_waypoint_insertion():
    """Insert optimal waypoints for desaturation."""
    print("\n" + "=" * 80)
    print("OPTION 4: WAYPOINT INSERTION")
    print("=" * 80)
    
    print("""
CONCEPT
-------

Insert one or more waypoints between start and end that pass through
favorable B orientations.

MATHEMATICS
-----------

For single waypoint q_w:
    Path: q_0 → q_w → q_f
    
    Total angle: θ_1 + θ_2 where θ_1 = angle(q_0, q_w), θ_2 = angle(q_w, q_f)
    
    Constraint: θ_1 + θ_2 ≤ θ_direct + Δθ_budget

OPTIMAL SINGLE WAYPOINT:

Objective: max D(q_w, t_w)
Constraint: θ_1(q_w) + θ_2(q_w) ≤ θ_max

This is optimization over SO(3), but can be simplified:

1. Parameterize q_w as deviation from midpoint of eigenaxis:
    q_mid = slerp(q_0, q_f, 0.5)
    q_w = q_mid ⊗ δq
    
2. δq is a small rotation, parameterized by axis-angle (3 params)

3. Constraint becomes quadratic in δq

4. Objective D(q_w) is nonlinear but smooth

Can solve with gradient descent or analytically for special cases.


ANALYTICAL SOLUTION FOR SPECIAL CASE:

If B is constant during slew (short slew or low orbit):
    
    Optimal q_w has body z-axis perpendicular to B_eci.
    
    This defines a great circle of attitudes: {q : R(q)·ẑ ⊥ B}
    
    Find intersection of this circle with the "tube" of attitudes
    within Δθ_budget of the direct path.
    
    Solution is at most 2 points (may be 0 if tube doesn't reach circle).


MULTIPLE WAYPOINTS:

For long slews where B changes significantly:
    
    Discretize: t_1, t_2, ..., t_k
    
    At each t_i, compute optimal B orientation
    
    Insert waypoints at q_i that approximately achieve these orientations
    
    Smooth the resulting path with spline interpolation


PROS:
    + Can precisely hit optimal B orientations
    + Clear physical interpretation
    + Can handle multiple B conditions

CONS:
    - Discontinuities at waypoints (need smoothing)
    - Computational cost for optimization
    - May significantly extend path for tight constraints
""")
    return


def option5_scheduled_desaturation():
    """Don't modify trajectory, wait for favorable B."""
    print("\n" + "=" * 80)
    print("OPTION 5: SCHEDULED DESATURATION (NO PATH MODIFICATION)")
    print("=" * 80)
    
    print("""
CONCEPT
-------

Accept that some attitudes have poor desaturation potential.
Schedule dedicated desaturation maneuvers during favorable orbital phases.

MATHEMATICS
-----------

Orbital analysis:
    - B_eci(t) varies with orbit (period ~90 min for LEO)
    - For any fixed body attitude, D varies periodically
    
Time to favorable geometry:
    t_favorable = min{t : D(q_current, t) ≥ D_threshold}
    
Typically t_favorable < T_orbit/4 ≈ 22 min for LEO.


DESATURATION WINDOWS:

Given pointing requirement q_required and tolerance:
    
    Desaturation window = {t : D(q_required, t) ≥ D_min}
    
For 3MTQ + 1RW with z-axis wheel:
    - Window occurs twice per orbit (B rotates around orbit normal)
    - Duration depends on D_min threshold
    - Typical: 20-40 min windows with D > 0.5

HYBRID APPROACH:

1. During nominal pointing: Apply torque-free desaturation when D > D_min
2. During slews: No modification, just execute slew
3. If h exceeds threshold: Schedule dedicated desat maneuver

This is the simplest approach and may be sufficient for many missions.


ANALYSIS: When is path modification worth it?

Path modification helps when:
    - h is critically high
    - Next natural desaturation window is far away
    - Slew geometry happens to pass near favorable B
    
Path modification NOT worth it when:
    - h is low
    - Current attitude already has D > D_min
    - Path modification cost exceeds waiting cost


PROS:
    + Simplest implementation
    + No trajectory modification needed
    + Natural for operational planning

CONS:
    - May have periods with no desaturation
    - Requires momentum margin
    - Less efficient use of MTQ authority
""")
    return


def analytical_waypoint_formula():
    """Derive analytical formula for single optimal waypoint."""
    print("\n" + "=" * 80)
    print("ANALYTICAL SINGLE WAYPOINT FORMULA")
    print("=" * 80)
    
    print("""
DERIVATION
==========

Setup:
    - Start: q_0, End: q_f
    - Direct angle: θ_direct = 2·arccos(|q_0 · q_f|)
    - Eigenaxis: e = normalize(q_f ⊗ q_0*)_vector
    - B field: B_eci (assumed constant for short slew)
    - RW axis in body: a_rw = [0, 0, 1] (z-axis)

Goal: Find q_w that maximizes D while keeping path overhead ≤ Δθ_budget


STEP 1: Parameterize optimal desaturation attitude

For D = 1, we need: R(q_w) · a_rw ⊥ B_eci

In ECI frame: R(q_w) · ẑ must be perpendicular to B.

This is satisfied by all q_w such that:
    R(q_w) · ẑ = v_perp where v_perp · B = 0

The set of such attitudes forms a great circle on S³.


STEP 2: Find closest point on great circle to path

The "direct path" is a geodesic from q_0 to q_f on S³.

The "desaturation circle" is {q : R(q)·ẑ ⊥ B}.

Distance from path point q(s) to desaturation circle:
    d(s) = arcsin(|B · R(q(s))·ẑ| / |B|)

Minimum is where B · R(q(s))·ẑ is smallest.


STEP 3: Compute optimal waypoint

Let's work in the body frame at the midpoint of the slew:
    q_mid = slerp(q_0, q_f, 0.5)
    B_body_mid = R(q_mid)ᵀ · B_eci

Deviation needed to make ẑ ⊥ B_body_mid:
    Current angle: φ_current = arccos(|B_body_mid · ẑ| / |B_body_mid|)
    Needed rotation: φ_needed = 90° - φ_current

Rotation axis (to make ẑ → ⊥B):
    n = normalize(B_body_mid × ẑ)

Small rotation:
    δq = [cos(φ_needed/2), sin(φ_needed/2)·n]

Waypoint:
    q_w = q_mid ⊗ δq

Path overhead:
    θ_new = angle(q_0, q_w) + angle(q_w, q_f)
    Δθ = θ_new - θ_direct


CLOSED-FORM RESULT:
==================

Given:
    B_eci, q_0, q_f, a_rw = [0,0,1]

Compute:
    q_mid = slerp(q_0, q_f, 0.5)
    R_mid = rotation_matrix(q_mid)
    B_body = R_midᵀ · B_eci
    
    # Angle from z to B
    φ = arccos(|B_body_z| / |B_body|)
    
    # Rotation axis
    n = normalize([B_body_y, -B_body_x, 0])  # = normalize(B_body × ẑ) projected to xy
    
    # Amount to rotate
    δφ = π/2 - φ  (to make B ⊥ z)
    
    # Limit by budget
    δφ_limited = min(δφ, Δθ_budget / 2)
    
    # Waypoint
    δq = [cos(δφ_limited/2), sin(δφ_limited/2)·n]
    q_w = q_mid ⊗ δq

This gives a waypoint that:
    1. Is close to the midpoint of direct path
    2. Rotates toward B ⊥ z configuration
    3. Respects the path overhead budget
""")
    
    # Numerical example
    print("\n" + "-" * 70)
    print("NUMERICAL EXAMPLE")
    print("-" * 70)
    
    def quat_mult(q1, q2):
        """Quaternion multiplication [w, x, y, z] convention."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quat_to_rotmat(q):
        w, x, y, z = q
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
            return q1 + t * (q2 - q1)
        theta = np.arccos(dot)
        return (np.sin((1-t)*theta)*q1 + np.sin(t*theta)*q2) / np.sin(theta)
    
    def quat_angle(q1, q2):
        dot = abs(np.dot(q1, q2))
        return 2 * np.arccos(min(dot, 1.0))
    
    # Example setup
    q_0 = np.array([1.0, 0, 0, 0])  # Identity
    q_f = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])  # 90° around x
    B_eci = np.array([0.3, 0.1, 0.9])  # Mostly along z in ECI
    B_eci = B_eci / np.linalg.norm(B_eci)
    
    print(f"q_0 = {q_0}")
    print(f"q_f = {q_f}")
    print(f"B_eci = {B_eci}")
    
    # Direct path
    theta_direct = quat_angle(q_0, q_f)
    print(f"\nDirect path angle: {np.degrees(theta_direct):.1f}°")
    
    # Midpoint
    q_mid = slerp(q_0, q_f, 0.5)
    R_mid = quat_to_rotmat(q_mid)
    B_body_mid = R_mid.T @ B_eci
    
    print(f"B_body at midpoint: {B_body_mid}")
    
    # Angle from z to B
    phi = np.arccos(abs(B_body_mid[2]) / np.linalg.norm(B_body_mid))
    print(f"Angle from z to B: {np.degrees(phi):.1f}° (want 90°)")
    
    # Desaturation potential at midpoint
    D_mid = np.sqrt(B_body_mid[0]**2 + B_body_mid[1]**2) / np.linalg.norm(B_body_mid)
    print(f"Desaturation potential at midpoint: D = {D_mid:.3f}")
    
    # Rotation axis to make z ⊥ B
    n_unnorm = np.array([B_body_mid[1], -B_body_mid[0], 0])
    if np.linalg.norm(n_unnorm) < 1e-10:
        n = np.array([1, 0, 0])  # Arbitrary if B already along z
    else:
        n = n_unnorm / np.linalg.norm(n_unnorm)
    
    # Amount to rotate
    delta_phi = np.pi/2 - phi
    print(f"Needed rotation: {np.degrees(delta_phi):.1f}°")
    
    # Budget
    budget_deg = 20
    budget_rad = np.radians(budget_deg)
    delta_phi_limited = min(delta_phi, budget_rad / 2)
    print(f"Limited by budget ({budget_deg}°): {np.degrees(delta_phi_limited):.1f}°")
    
    # Waypoint quaternion
    delta_q = np.array([np.cos(delta_phi_limited/2), 
                        np.sin(delta_phi_limited/2)*n[0],
                        np.sin(delta_phi_limited/2)*n[1],
                        np.sin(delta_phi_limited/2)*n[2]])
    q_w = quat_mult(q_mid, delta_q)
    q_w = q_w / np.linalg.norm(q_w)
    
    print(f"\nWaypoint: q_w = {q_w}")
    
    # Check desaturation potential at waypoint
    R_w = quat_to_rotmat(q_w)
    B_body_w = R_w.T @ B_eci
    D_w = np.sqrt(B_body_w[0]**2 + B_body_w[1]**2) / np.linalg.norm(B_body_w)
    print(f"Desaturation potential at waypoint: D = {D_w:.3f}")
    
    # Path overhead
    theta_new = quat_angle(q_0, q_w) + quat_angle(q_w, q_f)
    overhead = theta_new - theta_direct
    print(f"\nPath overhead: {np.degrees(overhead):.1f}° ({overhead/theta_direct*100:.1f}%)")
    
    return


def simple_implementation():
    """Provide a simple implementation formula."""
    print("\n" + "=" * 80)
    print("SIMPLE IMPLEMENTATION FORMULA")
    print("=" * 80)
    
    print("""
FOR PRACTICAL USE:
==================

Given: q_0, q_f, B_eci, a_rw (RW axis in body, typically [0,0,1])

ALGORITHM:

1. Compute midpoint:
   q_mid = slerp(q_0, q_f, 0.5)

2. Get B in body frame at midpoint:
   B_body = R(q_mid)ᵀ · B_eci

3. Compute current desaturation potential:
   D_current = ||B_body - (B_body·a_rw)·a_rw|| / ||B_body||
   
   If D_current > 0.9: # Already good, no waypoint needed
       return direct path

4. Compute rotation to improve D:
   n = normalize(B_body × a_rw)
   δφ = π/2 - arcsin(D_current)

5. Limit rotation by budget:
   δφ = min(δφ, θ_budget/2)

6. Create waypoint:
   δq = [cos(δφ/2), sin(δφ/2)·n]
   q_w = q_mid ⊗ δq

7. Return path: q_0 → q_w → q_f


COMPUTATIONAL COST:
- 2 quaternion operations
- 1 matrix-vector multiply
- Basic trig functions
- No optimization required!


WHEN TO USE WAYPOINT:
- D_current < 0.5 (poor desaturation)
- h_rw close to saturation
- Path overhead is acceptable (< 20% of direct)

WHEN NOT TO USE:
- D_current > 0.8 (already good)
- h_rw is low
- Time-critical slew
""")
    return


def summary():
    """Summarize the mathematical findings."""
    print("\n" + "=" * 80)
    print("SUMMARY: TRAJECTORY GENERATION OPTIONS")
    print("=" * 80)
    
    print("""
RECOMMENDED APPROACH: Single Waypoint Insertion
===============================================

For most cases, a single optimal waypoint provides the best balance of:
- Simplicity (closed-form solution)
- Effectiveness (can achieve D → 1)
- Efficiency (minimal path overhead)

FORMULA:
    q_w = slerp(q_0, q_f, 0.5) ⊗ [cos(δφ/2), sin(δφ/2)·n]
    
where:
    n = normalize(B_body × a_rw)
    δφ = min(π/2 - arcsin(D_current), θ_budget/2)


COMPARISON TABLE:
                        Complexity   Max D   Path Overhead   Implementation
Option 1 (offset)       Low         ~0.8    Low            Modify eigenaxis
Option 2 (two-stage)    Medium      1.0     Medium-High    Needs blending
Option 3 (deviation)    High        ~0.9    Low            Complex math
Option 4 (waypoint)     Medium      1.0     Medium         Simple formula ✓
Option 5 (scheduled)    Very low    varies  None           Orbit planning


CASES WHERE THIS DOESN'T WORK:
==============================

1. B ∥ a_rw for entire slew
   - No path modification can help
   - Must wait for orbital geometry to change
   - Flag this case and use scheduled desaturation

2. Very long slews (B changes significantly)
   - Single waypoint may not be optimal
   - Consider multiple waypoints or time-varying analysis
   - But often single waypoint at midpoint is still "good enough"

3. Very short slews
   - Not enough time for desaturation anyway
   - Don't bother with path modification
   - Let momentum accumulate, desaturate later


GRACEFUL DEGRADATION:
====================

The approach degrades gracefully:
- If optimal waypoint is outside budget → use budget-limited version
- If B is unfavorable → waypoint helps but doesn't fully solve
- If path is already optimal → algorithm returns direct path

It NEVER makes things worse - worst case is slightly longer path with
same or better desaturation potential.
""")
    return


if __name__ == "__main__":
    introduction()
    option1_eigenaxis_offset()
    option2_two_stage_slew()
    option3_great_circle_deviation()
    option4_waypoint_insertion()
    option5_scheduled_desaturation()
    analytical_waypoint_formula()
    simple_implementation()
    summary()
