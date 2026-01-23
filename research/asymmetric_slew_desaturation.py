"""
Asymmetric and Curvy Slews for Desaturation
===========================================

Questions:
1. Do we need symmetric slews? Can we move into a different axis?
2. What if B-field doesn't allow full desaturation (3MTQ+1RW)?
3. Can we do a "curvy" slew to desaturate?

Key insight: The physics of momentum exchange depends on:
- Torque direction
- Time spent at that torque
- Symmetry (or lack thereof) of the profile
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    return np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])


# ============== ASYMMETRIC SLEW ANALYSIS ==============

def analyze_symmetric_vs_asymmetric():
    """Understand why symmetric slews don't transfer momentum."""
    print("=" * 80)
    print("WHY SYMMETRIC SLEWS DON'T TRANSFER MOMENTUM")
    print("=" * 80)
    
    print("""
SYMMETRIC SLEW (Trapezoidal velocity):
--------------------------------------
Phase 1: Accelerate  τ = +τ_max   for t_accel
Phase 2: Coast       τ = 0        for t_coast  
Phase 3: Decelerate  τ = -τ_max   for t_decel (= t_accel)

RW momentum change:
  Δh = ∫ τ_rw dt = ∫(-τ_body) dt  (Newton's 3rd law)
  
  Phase 1: Δh₁ = -τ_max * t_accel
  Phase 2: Δh₂ = 0
  Phase 3: Δh₃ = +τ_max * t_accel
  
  Total: Δh = Δh₁ + Δh₂ + Δh₃ = 0  ← Symmetric cancellation!


ASYMMETRIC SLEW (Different accel/decel times):
----------------------------------------------
Phase 1: Accelerate  τ = +τ₁    for t₁
Phase 2: Decelerate  τ = -τ₂    for t₂ (≠ t₁)

For same position change (angle θ):
  θ = ½α₁t₁² + ½α₂t₂² (for triangular)
  
  With ω_peak = α₁t₁ = α₂t₂ (continuity)
  
RW momentum change:
  Δh = -τ₁*t₁ + τ₂*t₂
     = -J*α₁*t₁ + J*α₂*t₂
     = -J*ω_peak + J*ω_peak = 0  ← Still zero!

The constraint that ω must return to zero forces Δh = 0.


THE REAL INSIGHT:
-----------------
For momentum transfer, we need NET torque impulse:
  ∫ τ dt ≠ 0

This happens when:
1. External torque (like MTQ) doesn't require RW reaction
2. Slew doesn't return to zero angular velocity
3. Something else absorbs the momentum (like MTQ dumping to B-field)
""")
    
    return


def analyze_curvy_slew():
    """Analyze "curvy" slews with changing rotation axis."""
    print("\n" + "=" * 80)
    print("CURVY SLEWS: CHANGING ROTATION AXIS")
    print("=" * 80)
    
    print("""
IDEA: Instead of rotating about a fixed axis, curve through attitude space.

Example: Start at q₀, end at q_f, but follow a curved path.

For a straight (eigenaxis) path:
  q(t) = slerp(q₀, q_f, s(t))
  Rotation axis constant throughout

For a curved path:
  q(t) = some nonlinear function
  Rotation axis CHANGES over time
  
At any instant: ω = 2 * q̇ ⊗ q*
The instantaneous axis is ω/|ω|


MOMENTUM IMPLICATIONS:
----------------------
Even for curved paths, if ω goes 0 → ω_max → 0:
  ∫ τ dt = J * ∫ ω̇ dt = J * (ω_final - ω_initial) = J * (0 - 0) = 0

The NET momentum change is STILL ZERO for any slew that starts and ends
at rest, regardless of the path shape!


BUT WAIT - DIFFERENT ACTUATOR CONTRIBUTIONS:
--------------------------------------------
The TOTAL Δh = 0, but individual actuators can have different Δh!

Example with RW + MTQ:
  τ_total = τ_rw + τ_mtq
  ∫ τ_total dt = 0 (body returns to rest)
  
  But: ∫ τ_rw dt might not equal -∫ τ_mtq dt if MTQ can dump to B-field!
  
  Actually: τ_mtq = m × B (external torque from Earth's field)
  This means: ∫ τ_mtq dt can be "absorbed" by the external field
  
  Result: ∫ τ_rw dt = -∫ τ_total dt + ∫ τ_mtq dt
                    = 0 + (something dumped to B-field)
                    ≠ 0 !

THIS IS THE KEY: MTQ can "donate" torque impulse from outside!
""")
    
    return


def demonstrate_asymmetric_with_mtq():
    """Demonstrate desaturation during slew using MTQ."""
    print("\n" + "=" * 80)
    print("ASYMMETRIC SLEW WITH MTQ DESATURATION")
    print("=" * 80)
    
    print("""
SETUP:
- 3 RW along body axes
- 3 MTQ along body axes
- B-field varies (LEO orbit)
- Want to slew 90° while reducing h_z

STRATEGY:
During slew, overlay torque-free desaturation:
  τ_mtq = desaturation torque (perpendicular to B)
  τ_rw_desat = -τ_mtq (cancels MTQ torque on body)
  
  Net body torque = τ_rw_slew (just the slew torque)
  But: Δh_rw = ∫ (τ_rw_slew + τ_rw_desat) dt
             = 0 + ∫ τ_rw_desat dt
             = -∫ τ_mtq dt
             ≠ 0 !

The MTQ "injects" torque impulse from the environment, allowing
RW momentum to change even though the body motion is unaffected.
""")
    
    # Simulation
    print("\nSimulation: 90° slew with MTQ desat overlay")
    print("-" * 50)
    
    J = np.diag([0.022, 0.022, 0.004])
    h_init = np.array([0, 0, 0.005])  # 5 mNm·s on z-axis
    
    # Slew parameters
    theta = np.radians(90)
    t_slew = 60  # seconds
    slew_axis = normalize([1, 0, 0])  # x-axis rotation
    
    # Trapezoidal velocity profile
    t_accel = t_slew / 3
    t_coast = t_slew / 3
    omega_max = theta / (t_accel + t_coast)
    alpha = omega_max / t_accel
    
    dt = 0.5
    t = 0
    h = h_init.copy()
    
    h_history = [h.copy()]
    t_history = [0]
    
    # B-field varies with orbit
    def get_B(t):
        phase = 2 * np.pi * t / 5400  # 90 min orbit
        return 30e-6 * np.array([np.cos(phase), 0.5*np.sin(phase), 0.3])
    
    while t < t_slew:
        # Slew torque (along slew axis)
        if t < t_accel:
            tau_slew = J @ (slew_axis * alpha)
        elif t < t_accel + t_coast:
            tau_slew = np.zeros(3)
        else:
            tau_slew = J @ (slew_axis * (-alpha))
        
        # Desaturation overlay
        B = get_B(t)
        b_hat = B / np.linalg.norm(B)
        
        # Desired desat: reduce h_z
        k_desat = 0.1
        tau_desat_des = -k_desat * h
        
        # Project to MTQ achievable (perpendicular to B)
        tau_mtq = tau_desat_des - np.dot(tau_desat_des, b_hat) * b_hat
        
        # RW cancels MTQ (torque-free on body)
        tau_rw_desat = -tau_mtq
        
        # Total RW torque: slew + desat
        tau_rw_total = tau_slew + tau_rw_desat
        
        # Update momentum (Δh = -τ_rw * dt, but we track total)
        h += -tau_rw_total * dt
        
        t += dt
        h_history.append(h.copy())
        t_history.append(t)
    
    h_history = np.array(h_history)
    
    print(f"Initial h: {h_init * 1000} mNm·s")
    print(f"Final h:   {h * 1000} mNm·s")
    print(f"h_z reduction: {(h_init[2] - h[2]) / h_init[2] * 100:.1f}%")
    print(f"h_x, h_y change: {(h[:2] - h_init[:2]) * 1000} mNm·s")
    
    return


def asymmetric_slew_different_axis():
    """Explore slewing to a DIFFERENT final attitude that helps desat."""
    print("\n" + "=" * 80)
    print("SLEWING TO DIFFERENT ATTITUDE FOR DESATURATION")
    print("=" * 80)
    
    print("""
IDEA: What if we don't just change the PATH, but the GOAL?

Original goal: Point at target A
Modified goal: Point at target A' (near A) that requires slew axis aligned with h

Example:
- Want to point camera at star (defines 2 DOF)
- Have freedom in roll axis (1 DOF)
- Choose roll that:
  1. Aligns slew axis with momentum h
  2. Enables better MTQ desaturation geometry

This is "attitude trajectory optimization" - the planner already does this!


FOR POINTING (REDUCED ATTITUDE):
--------------------------------
If goal is just boresight direction:
  - Infinite attitudes satisfy this (any roll)
  - Choose roll that maximizes desat potential
  - Slew to that specific attitude

FOR FULL ATTITUDE:
------------------
If goal is specific attitude (no freedom):
  - Can't change goal, but can change path
  - Use waypoint that touches favorable geometry
  - Accept longer path for better desat
""")
    
    return


def curvy_slew_for_partial_bfield():
    """Handle the case where B-field can't fully desaturate."""
    print("\n" + "=" * 80)
    print("CURVY SLEW FOR PARTIAL B-FIELD GEOMETRY")
    print("=" * 80)
    
    print("""
PROBLEM: 3MTQ + 1RW (z-axis)
---------------------------
- RW stores momentum along z
- MTQ produces torque ⊥ to B
- If B ∥ z, MTQ can't produce z-torque, can't desaturate!

CURVY SLEW SOLUTION:
-------------------
Instead of eigenaxis slew, curve through attitudes where B geometry is favorable.

Example: 
- Direct path has B ∥ z the whole time (bad)
- Curved path passes through attitude where B ⊥ z (good!)

Implementation:
1. Compute B-field for direct path attitude history
2. Find "desaturation potential" along path: |B_xy| / |B|
3. If potential is low, add waypoints to visit better geometries
4. Optimize waypoints for max desat at min path cost


MATHEMATICAL FORMULATION:
-------------------------
Let q(s) be the path parameterized by s ∈ [0, 1]
B(s) = R(q(s))ᵀ @ B_inertial(t(s))  (B in body frame)

Desaturation potential: D(s) = |B(s) × ẑ| / |B(s)|
                             = sin(angle between B and z)
                             
Integrated desat potential: ∫₀¹ D(s) ds

Want to maximize this while keeping path length reasonable.


CURVY PATH PARAMETERIZATION:
---------------------------
Use Bezier curves in quaternion space (on S³):

q(s) = deCasteljau(q₀, q₁, ..., qₙ, s)

Where q₁, ..., qₙ₋₁ are control points to optimize.

Optimization:
  max  ∫ D(s) ds  (desat potential)
  s.t. path_length(q) ≤ (1 + ε) * eigenaxis_length
       q(0) = q_start
       q(1) = q_goal
""")
    
    # Demonstrate with specific geometry
    print("\nExample: Slew where direct path has poor geometry")
    print("-" * 50)
    
    # Setup where B is mostly along z for direct path
    q_start = np.array([1, 0, 0, 0])
    q_goal = quat_from_axis_angle([0, 0, 1], np.radians(90))  # 90° about z
    
    # B-field in inertial frame (happens to align poorly)
    B_inertial = normalize([0.1, 0.1, 1.0]) * 30e-6
    
    # For this slew (pure z-rotation), body frame doesn't change B much
    # B_body ≈ B_inertial throughout (bad for z-desat)
    
    print(f"Slew: 90° about z-axis")
    print(f"B_inertial ≈ [0.1, 0.1, 1.0] (mostly z)")
    print(f"Direct path: B stays mostly along z → poor z-desat geometry")
    
    # Alternative: curve through x-y rotation first
    # This changes body frame orientation so B appears more in x-y plane
    
    print(f"\nCurved path: First rotate about x or y")
    print(f"This temporarily makes B appear perpendicular to z")
    print(f"During that phase, MTQ can produce z-torque for desat")
    
    return


def practical_curvy_slew_implementation():
    """Practical implementation of curvy slew for desat."""
    print("\n" + "=" * 80)
    print("PRACTICAL IMPLEMENTATION: CURVY SLEW")
    print("=" * 80)
    
    print("""
ALGORITHM: Desaturation-Optimized Curved Slew
=============================================

INPUTS:
  q_start, q_goal: Start and end attitudes
  h_current: Current RW momentum (to reduce)
  B_model(t, q): Function giving B-field in body frame
  t_available: Time budget

STEP 1: Evaluate direct path geometry
--------------------------------------
  For s ∈ [0, 1]:
    q_s = slerp(q_start, q_goal, s)
    B_s = B_model(t(s), q_s)
    D_s = desat_potential(B_s, h_direction)
    
  D_direct = ∫ D_s ds

STEP 2: Check if curvy path would help
--------------------------------------
  If D_direct > threshold:
    return direct_path  # Already good geometry
  
STEP 3: Find optimal waypoints
------------------------------
  # Binary search / optimization for waypoint
  For waypoint_axis in candidate_axes:
    For waypoint_angle in [10°, 20°, 30°, ...]:
      q_wp = create_waypoint(q_mid, waypoint_axis, waypoint_angle)
      path = [q_start, q_wp, q_goal]
      
      # Evaluate this path
      D_curved = ∫ D(path, s) ds
      path_cost = path_length(path) / eigenaxis_length
      
      score = D_curved / path_cost
      
      if score > best_score and path_time < t_available:
        best_path = path
        
STEP 4: Execute curved slew with desat overlay
----------------------------------------------
  For each segment of best_path:
    Execute slew segment
    Continuously apply torque-free desat:
      τ_mtq = k * project(-h, perpendicular_to_B)
      τ_rw_desat = -τ_mtq
      
OUTPUT:
  best_path: List of quaternion waypoints
  expected_desat: Estimated Δh during slew


CANDIDATE WAYPOINT AXES:
-----------------------
Good candidates are axes that:
1. Create attitude where B ⊥ h
2. Don't add too much path length
3. Are smooth to traverse

Heuristic: Try axes perpendicular to both eigenaxis and h direction.
""")
    
    return


def summary():
    """Print summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: ASYMMETRIC AND CURVY SLEWS")
    print("=" * 80)
    
    print("""
KEY INSIGHTS:

1. SYMMETRIC SLEWS DON'T TRANSFER MOMENTUM
   - ∫τ dt = J*(ω_final - ω_initial) = 0 for any rest-to-rest slew
   - This is true regardless of path shape!
   - The "symmetry" that matters is ω returning to zero, not the profile shape

2. MTQ ENABLES MOMENTUM TRANSFER
   - MTQ exchanges momentum with Earth's magnetic field
   - τ_mtq = m × B is an EXTERNAL torque (not internal like RW)
   - ∫τ_mtq dt can be nonzero even for rest-to-rest slew
   - This is what enables desaturation!

3. CURVY SLEWS HELP BY CHANGING B GEOMETRY
   - Direct path might have B ∥ h (bad for desat)
   - Curved path can visit attitudes where B ⊥ h (good!)
   - Trade-off: longer path for better desat geometry

4. FOR 3MTQ + 1RW:
   - Can't always make τ_mtq exactly along z
   - But curvy path can maximize time in favorable geometry
   - Partial desaturation is still valuable

5. IMPLEMENTATION:
   - Use waypoints to curve through favorable geometries
   - Optimize waypoints for desat potential / path cost
   - Apply torque-free desat overlay throughout slew
   - Any momentum reduction is "free" vs dedicated desat time

6. ASYMMETRIC ≠ NET MOMENTUM CHANGE
   - Asymmetric velocity profiles still return ω to zero
   - It's the EXTERNAL torque (MTQ) that enables Δh ≠ 0
   - Think of curvy path as maximizing external torque opportunity
""")


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_symmetric_vs_asymmetric()
    analyze_curvy_slew()
    demonstrate_asymmetric_with_mtq()
    asymmetric_slew_different_axis()
    curvy_slew_for_partial_bfield()
    practical_curvy_slew_implementation()
    summary()
