"""
Analytical Approaches to Slew-Based Desaturation
================================================

Question: Can we choose slew geometry to desaturate rather than 
desaturating during pointing holds?

Key insight: During slew, we need torque anyway. If we can route
the torque through the RW in the right direction, we can "spend"
momentum on the maneuver.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, minimize_scalar
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import normalize, rot_mat


def quat_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle."""
    axis = normalize(axis)
    return np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])


def quat_mult(q1, q2):
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_inv(q):
    """Quaternion inverse."""
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def eigenaxis_slew(q_start, q_goal):
    """Compute eigenaxis (shortest path) rotation."""
    q_err = quat_mult(quat_inv(q_start), q_goal)
    if q_err[0] < 0:
        q_err = -q_err
    
    angle = 2 * np.arccos(np.clip(q_err[0], -1, 1))
    if angle < 1e-6:
        return np.array([0, 0, 1]), 0
    
    axis = q_err[1:4] / np.sin(angle/2)
    return normalize(axis), angle


# ============== ANALYTICAL APPROACH ==============

def analyze_slew_momentum_exchange():
    """
    Analyze how momentum changes during a slew.
    
    Key physics:
    - τ_rw = -ḣ_rw (reaction wheel torque = negative momentum rate)
    - During slew: τ_body = J·α (body needs angular acceleration)
    - If RW provides this torque: ḣ_rw = -τ_body
    
    For eigenaxis slew with trapezoidal profile:
    - Accel phase: τ = J·α along rotation axis, h decreases
    - Coast phase: τ = 0, h constant
    - Decel phase: τ = -J·α along rotation axis, h increases
    - Net: Δh = 0 (symmetric)
    
    For NON-eigenaxis path:
    - Different torque directions in different phases
    - Can have net Δh ≠ 0
    """
    print("=" * 80)
    print("MOMENTUM EXCHANGE DURING SLEW")
    print("=" * 80)
    
    J = np.diag([0.022, 0.022, 0.004])  # Typical CubeSat inertia
    
    # Consider a slew about axis n
    n = normalize([1, 0, 0])  # x-axis rotation
    angle = np.radians(90)
    
    # Angular acceleration needed
    t_slew = 60  # seconds
    # Trapezoidal: accel for t/3, coast for t/3, decel for t/3
    t_accel = t_slew / 3
    omega_max = angle / (2 * t_accel)  # Peak angular velocity
    alpha = omega_max / t_accel
    
    print(f"\nSlew parameters:")
    print(f"  Rotation: {np.degrees(angle):.0f}° about {n}")
    print(f"  Duration: {t_slew} s")
    print(f"  Peak omega: {np.degrees(omega_max):.2f} °/s")
    print(f"  Alpha: {np.degrees(alpha):.4f} °/s²")
    
    # Torque needed (about rotation axis)
    # τ = J·α·n (simplified - ignoring gyroscopic terms for analysis)
    tau_mag = np.dot(J @ n, n) * alpha
    
    print(f"  Torque magnitude: {tau_mag * 1e6:.2f} μNm")
    
    # Momentum change during accel phase
    # Δh = ∫τ dt = τ·t_accel (for constant τ)
    delta_h_accel = tau_mag * t_accel
    
    print(f"\nMomentum exchange:")
    print(f"  During accel: Δh = {delta_h_accel * 1e3:.2f} mNm·s")
    print(f"  During decel: Δh = {-delta_h_accel * 1e3:.2f} mNm·s")
    print(f"  Net: Δh = 0")
    
    return


def analyze_non_eigenaxis_desat():
    """
    Analyze non-eigenaxis paths for desaturation.
    
    Key idea: If we break the slew into segments with different
    rotation axes, we can have net momentum transfer.
    
    Example: Want to rotate 90° about x, but have h along z.
    
    Option A (eigenaxis):
      - Rotate 90° about x
      - Torque is along x
      - h_z unchanged
    
    Option B (via waypoint):
      - Rotate θ1 about axis that includes z
      - Rotate θ2 to final orientation
      - Some torque along z → h_z changes
    """
    print("\n" + "=" * 80)
    print("NON-EIGENAXIS PATHS FOR DESATURATION")
    print("=" * 80)
    
    # Setup
    q_start = np.array([1, 0, 0, 0])  # Identity
    q_goal = quat_from_axis_angle([1, 0, 0], np.radians(90))  # 90° about x
    
    h_current = np.array([0, 0, 0.005])  # 5 mNm·s along z
    
    # Eigenaxis parameters
    axis_direct, angle_direct = eigenaxis_slew(q_start, q_goal)
    
    print(f"\nDirect eigenaxis:")
    print(f"  Axis: {axis_direct}")
    print(f"  Angle: {np.degrees(angle_direct):.1f}°")
    print(f"  h·axis = {np.dot(h_current, axis_direct):.4f}")
    print(f"  (Torque along x, h along z → no h_z change)")
    
    # Now consider waypoint paths
    print(f"\nWaypoint paths:")
    print("-" * 60)
    
    # Parameterize waypoint as rotation from midpoint
    # The waypoint adds a rotation that uses z-torque
    
    def evaluate_waypoint(wp_angle_deg, wp_axis):
        """
        Create waypoint path and evaluate z-torque usage.
        
        Returns: (total_path_angle, z_torque_integral)
        """
        wp_angle = np.radians(wp_angle_deg)
        wp_axis = normalize(wp_axis)
        
        # Midpoint of direct path
        q_mid = quat_mult(q_start, quat_from_axis_angle(axis_direct, angle_direct/2))
        
        # Waypoint: rotate from midpoint
        q_wp = quat_mult(q_mid, quat_from_axis_angle(wp_axis, wp_angle))
        q_wp = normalize(q_wp)
        
        # Path: start -> waypoint -> goal
        axis1, angle1 = eigenaxis_slew(q_start, q_wp)
        axis2, angle2 = eigenaxis_slew(q_wp, q_goal)
        
        total_angle = angle1 + angle2
        
        # z-torque usage (proportional to how much rotation is about z)
        z_torque_1 = abs(axis1[2]) * angle1
        z_torque_2 = abs(axis2[2]) * angle2
        z_torque_total = z_torque_1 + z_torque_2
        
        return total_angle, z_torque_total, (axis1, angle1), (axis2, angle2)
    
    # Test waypoints along different axes
    for wp_axis_name, wp_axis in [
        ("z-axis", [0, 0, 1]),
        ("y-axis", [0, 1, 0]),
        ("45° yz", [0, 1, 1]),
    ]:
        print(f"\nWaypoint axis: {wp_axis_name}")
        print(f"{'WP angle':>10} {'Total path':>12} {'z-torque':>12} {'Overhead':>10}")
        print("-" * 50)
        
        for wp_angle_deg in [0, 15, 30, 45, 60]:
            total, z_torque, (a1, an1), (a2, an2) = evaluate_waypoint(wp_angle_deg, wp_axis)
            overhead = (total - angle_direct) / angle_direct * 100
            print(f"{wp_angle_deg:>10}° {np.degrees(total):>12.1f}° {z_torque:>12.3f} {overhead:>10.1f}%")
    
    return


def optimal_desaturating_path():
    """
    Find the optimal waypoint that maximizes desaturation per unit path length.
    """
    print("\n" + "=" * 80)
    print("OPTIMAL DESATURATING PATH")
    print("=" * 80)
    
    q_start = np.array([1, 0, 0, 0])
    q_goal = quat_from_axis_angle([1, 0, 0], np.radians(90))
    
    h_current = np.array([0, 0, 0.005])  # Want to reduce z-momentum
    h_dir = normalize(h_current)
    
    axis_direct, angle_direct = eigenaxis_slew(q_start, q_goal)
    
    def evaluate_path(params):
        """
        params: [wp_angle, wp_axis_theta, wp_axis_phi]
        Returns: -efficiency (for minimization)
        """
        wp_angle = params[0]
        theta = params[1]
        phi = params[2]
        
        # Spherical coordinates for waypoint axis
        wp_axis = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Create waypoint
        q_mid = quat_mult(q_start, quat_from_axis_angle(axis_direct, angle_direct/2))
        q_wp = quat_mult(q_mid, quat_from_axis_angle(wp_axis, wp_angle))
        q_wp = normalize(q_wp)
        
        # Compute path
        axis1, angle1 = eigenaxis_slew(q_start, q_wp)
        axis2, angle2 = eigenaxis_slew(q_wp, q_goal)
        
        total_angle = angle1 + angle2
        
        # Desaturation metric: how much torque is along h direction
        desat_1 = abs(np.dot(axis1, h_dir)) * angle1
        desat_2 = abs(np.dot(axis2, h_dir)) * angle2
        desat_total = desat_1 + desat_2
        
        # Efficiency: desaturation per unit path length overhead
        overhead = total_angle - angle_direct
        if overhead < 0.01:
            return 0  # No overhead, no benefit
        
        efficiency = desat_total / overhead
        return -efficiency  # Negative for minimization
    
    # Optimize
    from scipy.optimize import differential_evolution
    
    bounds = [
        (0.1, 1.0),  # wp_angle: 0.1 to 1 radian
        (0, np.pi),  # theta: 0 to pi
        (0, 2*np.pi),  # phi: 0 to 2pi
    ]
    
    result = differential_evolution(evaluate_path, bounds, seed=42, maxiter=100)
    
    opt_params = result.x
    wp_angle = opt_params[0]
    theta, phi = opt_params[1], opt_params[2]
    opt_axis = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    print(f"\nOptimal waypoint:")
    print(f"  Angle: {np.degrees(wp_angle):.1f}°")
    print(f"  Axis: {opt_axis}")
    print(f"  Efficiency: {-result.fun:.3f} (desat per radian overhead)")
    
    # Evaluate this path
    q_mid = quat_mult(q_start, quat_from_axis_angle(axis_direct, angle_direct/2))
    q_wp = quat_mult(q_mid, quat_from_axis_angle(opt_axis, wp_angle))
    q_wp = normalize(q_wp)
    
    axis1, angle1 = eigenaxis_slew(q_start, q_wp)
    axis2, angle2 = eigenaxis_slew(q_wp, q_goal)
    
    total_angle = angle1 + angle2
    overhead_pct = (total_angle - angle_direct) / angle_direct * 100
    
    desat_1 = abs(np.dot(axis1, h_dir)) * angle1
    desat_2 = abs(np.dot(axis2, h_dir)) * angle2
    
    print(f"\nPath details:")
    print(f"  Leg 1: {np.degrees(angle1):.1f}° about {axis1}")
    print(f"  Leg 2: {np.degrees(angle2):.1f}° about {axis2}")
    print(f"  Total: {np.degrees(total_angle):.1f}° ({overhead_pct:.1f}% overhead)")
    print(f"  Desaturation: {desat_1 + desat_2:.3f} rad·(h-aligned)")
    
    return


def practical_slew_planning():
    """
    Practical algorithm for slew planning with desaturation.
    """
    print("\n" + "=" * 80)
    print("PRACTICAL SLEW PLANNING ALGORITHM")
    print("=" * 80)
    
    print("""
ALGORITHM: Desaturation-Aware Slew Planner
==========================================

INPUT:
  q_current, q_goal  : Start and end orientations
  h_rw               : Current RW momentum
  t_available        : Time budget for maneuver
  h_threshold        : Momentum level requiring desaturation

OUTPUT:
  path               : List of quaternion waypoints
  expected_desat     : Estimated momentum reduction

PROCEDURE:

1. COMPUTE DIRECT PATH
   axis, angle = eigenaxis(q_current, q_goal)
   t_direct = slew_time(angle)
   
2. CHECK IF DESATURATION NEEDED
   if ||h_rw|| < h_threshold:
       return [q_current, q_goal]  # Direct path
       
3. CHECK NATURAL ALIGNMENT
   alignment = |h_rw · axis| / ||h_rw||
   if alignment > 0.8:
       return [q_current, q_goal]  # Already aligned, natural desat
       
4. COMPUTE OPTIMAL WAYPOINT
   For waypoint to be useful, it should:
   - Add rotation about h_rw direction
   - Not increase path length too much
   
   h_dir = h_rw / ||h_rw||
   
   # Simple heuristic: add waypoint along h direction
   q_mid = slerp(q_current, q_goal, 0.5)
   
   # Find optimal waypoint angle
   for wp_angle in [15°, 30°, 45°]:
       q_wp = q_mid ⊗ quat(h_dir, wp_angle)
       
       # Check path length
       path = [q_current, q_wp, q_goal]
       total_length = path_length(path)
       t_path = slew_time(total_length)
       
       if t_path <= t_available:
           desat = estimate_desat(path, h_rw)
           candidates.append((wp_angle, path, desat))
   
   # Select best candidate
   if candidates:
       return best by desat / overhead
   else:
       return [q_current, q_goal]  # Fallback to direct

5. ESTIMATE DESATURATION
   For each leg of path:
   - Compute rotation axis
   - Torque ~ J @ axis @ alpha
   - Desat contribution ~ |torque · h_dir| * time
""")
    
    return


def closed_form_analysis():
    """
    Closed-form analysis of slew desaturation.
    """
    print("\n" + "=" * 80)
    print("CLOSED-FORM ANALYSIS")
    print("=" * 80)
    
    print("""
THEOREM: Momentum Change During Slew
------------------------------------

For a slew with rotation axis n and angle θ, using trapezoidal velocity:

  Δh_net = 0  (symmetric accel/decel cancels)

But the MAXIMUM momentum excursion during slew is:

  |Δh_max| = (J·n·n^T) · (θ / t_slew) · (t_slew/3)
           = J_nn · θ / 3

where J_nn = n^T · J · n is the moment of inertia about axis n.

For a two-leg path with axes n1, n2:

  Leg 1: accel along n1, coast, decel along n1 → Δh = 0
  Leg 2: accel along n2, coast, decel along n2 → Δh = 0
  
  Total: Δh = 0 (still!)

The key insight: SYMMETRIC slews don't transfer momentum.

To get NET momentum transfer, we need ASYMMETRIC torque profile:
  - Different accel vs decel times
  - Different axis during accel vs decel
  - External torque (MTQ) during part of slew

HYBRID APPROACH: Use MTQ during slew for torque-free desat
----------------------------------------------------------

During slew, add torque-free desaturation overlay:
  - RW provides slew torque + desat torque
  - MTQ provides -desat torque
  - Net: slew torque only (no extra body torque)
  - But h_rw changes!

This is the CLEANEST approach:
  1. Plan eigenaxis slew (shortest time)
  2. Overlay torque-free desat during slew
  3. Get desaturation "for free" during maneuver time
""")
    
    return


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_slew_momentum_exchange()
    analyze_non_eigenaxis_desat()
    optimal_desaturating_path()
    practical_slew_planning()
    closed_form_analysis()
    
    print("\n" + "=" * 80)
    print("KEY CONCLUSIONS")
    print("=" * 80)
    print("""
1. SYMMETRIC SLEWS DON'T DESATURATE
   - Eigenaxis slew: accel/decel cancel → Δh = 0
   - Even multi-leg paths: each leg is symmetric → Δh = 0
   
2. PATH GEOMETRY AFFECTS AVAILABLE DESAT
   - Non-eigenaxis paths use different torque directions
   - More path length along h direction = more desat opportunity
   - But NOT automatic momentum transfer (still symmetric)
   
3. THE RIGHT APPROACH: TORQUE-FREE OVERLAY
   - During any slew, overlay torque-free desaturation
   - MTQ provides desat torque, RW cancels it
   - Slew proceeds normally, but h decreases
   - This works with ANY slew geometry
   
4. WAYPOINTS ARE USEFUL FOR:
   - Getting torque along h direction (for torque-free overlay)
   - When eigenaxis slew axis is perpendicular to h
   - Trade: longer path time for more desat opportunity
""")
