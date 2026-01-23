"""
Slew Geometry for Desaturation
==============================

Question: Can we choose slew paths that naturally desaturate momentum
rather than desaturating during pointing holds?

Key insight: During a slew, we're applying large torques anyway.
If we can route the slew path to use RW torque in the right direction,
we can "spend" momentum on the maneuver instead of storing it.
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.spatial.transform import Rotation
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from ADCS.helpers.math_helpers import skewsym, normalize, rot_mat, quat_inv, quat_mult


def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    return np.concatenate([[np.cos(angle/2)], axis * np.sin(angle/2)])


def quat_err(q, q_goal):
    q, q_goal = normalize(q), normalize(q_goal)
    qe = quat_mult(quat_inv(q_goal), q)
    if qe[0] < 0: qe = -qe
    return 2*qe[1:4]


def full_err_deg(q, q_goal):
    q, q_goal = normalize(q), normalize(q_goal)
    qe = quat_mult(quat_inv(q_goal), q)
    if qe[0] < 0: qe = -qe
    return np.degrees(2*np.arccos(np.clip(qe[0], -1, 1)))


def eigenaxis_slew(q_start, q_goal):
    """Standard eigenaxis (shortest path) slew."""
    q_err = quat_mult(quat_inv(q_start), q_goal)
    if q_err[0] < 0:
        q_err = -q_err
    
    angle = 2 * np.arccos(np.clip(q_err[0], -1, 1))
    if angle < 1e-6:
        return np.array([0, 0, 1]), 0
    
    axis = q_err[1:4] / np.sin(angle/2)
    return normalize(axis), angle


def slew_with_waypoint(q_start, q_goal, waypoint_axis, waypoint_angle):
    """
    Slew via a waypoint: 
    q_start -> q_waypoint -> q_goal
    
    The waypoint_axis and waypoint_angle define a detour from the direct path.
    """
    # Create waypoint as a rotation from midpoint
    axis_direct, angle_direct = eigenaxis_slew(q_start, q_goal)
    
    # Midpoint quaternion
    q_mid = quat_mult(q_start, quat_from_axis_angle(axis_direct, angle_direct/2))
    
    # Apply waypoint offset
    q_waypoint = quat_mult(q_mid, quat_from_axis_angle(waypoint_axis, waypoint_angle))
    
    # Return path: start -> waypoint -> goal
    return [q_start, normalize(q_waypoint), q_goal]


def compute_slew_momentum_change(q_path, J, omega_max, t_slew):
    """
    Estimate momentum change during a slew following q_path.
    
    For a trapezoidal velocity profile:
    - Accelerate, coast, decelerate
    - Momentum change from RW during accel/decel
    
    Returns: Δh (change in RW momentum during slew)
    """
    # Total rotation
    total_angle = 0
    for i in range(len(q_path) - 1):
        _, angle = eigenaxis_slew(q_path[i], q_path[i+1])
        total_angle += angle
    
    # For eigenaxis slew, ω is along the rotation axis
    # Torque τ = J·α (angular acceleration)
    # For trapezoidal: accel for t_accel, coast, decel for t_decel
    
    # Simplified: assume triangular profile (no coast)
    # ω_max at midpoint, ω = 0 at start/end
    # α = ω_max / (t_slew/2)
    
    t_half = t_slew / 2
    alpha_mag = omega_max / t_half  # Angular acceleration magnitude
    
    # For each segment, compute the required torque direction
    delta_h = np.zeros(3)
    
    for i in range(len(q_path) - 1):
        axis, angle = eigenaxis_slew(q_path[i], q_path[i+1])
        
        # Torque along axis during accel, opposite during decel
        # τ = J·α·axis
        tau_mag = np.dot(J @ axis, axis) * alpha_mag
        
        # Momentum change: Δh = τ·Δt
        # During accel: RW provides positive torque, h decreases
        # During decel: RW provides negative torque, h increases
        # Net effect depends on RW configuration
        
        # For simplicity, assume RW aligned with body axes
        # τ_rw = -Δh_rw (reaction wheel dynamics)
        
        # The key insight: torque direction during slew is along rotation axis
        delta_h += tau_mag * axis * t_half  # Accel phase
        delta_h -= tau_mag * axis * t_half  # Decel phase (opposite)
    
    # Net: symmetric slew has zero net momentum change
    # But NON-symmetric slews can have net momentum change!
    
    return delta_h


def find_desaturating_slew(q_start, q_goal, h_current, J, A_rw):
    """
    Find a slew geometry that helps desaturate.
    
    Strategy: Choose slew axis that:
    1. Gets us from start to goal
    2. Aligns with current momentum to "spend" it
    """
    
    # Direct slew
    axis_direct, angle_direct = eigenaxis_slew(q_start, q_goal)
    
    # Current momentum direction (in body frame)
    h_body = A_rw @ h_current
    h_dir = normalize(h_body) if np.linalg.norm(h_body) > 1e-10 else np.array([0, 0, 1])
    
    print(f"Direct slew axis: {axis_direct}")
    print(f"Momentum direction: {h_dir}")
    print(f"Alignment (dot product): {np.dot(axis_direct, h_dir):.3f}")
    
    # If slew axis aligns with h, we can "spend" momentum during slew
    # If misaligned, we'll build more momentum
    
    # For eigenaxis slew, we don't have freedom in axis choice
    # But we can add a waypoint to create a path that uses the h direction
    
    # Option 1: Decompose into two rotations
    # One along h, one perpendicular
    
    # Let's just analyze the geometry
    return axis_direct, h_dir


def analyze_slew_desaturation():
    """
    Analyze when slew geometry can help with desaturation.
    """
    print("=" * 80)
    print("SLEW GEOMETRY FOR DESATURATION")
    print("=" * 80)
    
    print("""
QUESTION: Can we choose slew paths that naturally desaturate?

KEY INSIGHT:
During a slew, the spacecraft accelerates and decelerates.
- Acceleration phase: RW provides positive torque → h decreases
- Deceleration phase: RW provides negative torque → h increases
- For eigenaxis (symmetric) slew: net Δh ≈ 0

BUT: Non-eigenaxis paths can have asymmetric torque profiles!

EXAMPLE:
Current state: h = [0, 0, +5] mNm·s (z-axis momentum)
We want to slew 90° about x-axis.

Standard eigenaxis: torque about x-axis, z-momentum unchanged.

Alternative path:
1. First rotate 45° about axis that mixes x and z
2. Then rotate to final orientation

This uses z-torque during the slew, reducing z-momentum.
""")
    
    # Example setup
    J = np.diag([0.022, 0.022, 0.004])
    A_rw = np.eye(3)  # 3-axis RW
    
    # Current momentum
    h_current = np.array([0.001, -0.002, 0.005])  # Significant z-momentum
    
    # Slew goal: 90° about x
    q_start = np.array([1, 0, 0, 0])
    q_goal = quat_from_axis_angle([1, 0, 0], np.pi/2)
    
    print(f"\nSetup:")
    print(f"  Initial momentum: {h_current * 1000} mNm·s")
    print(f"  Slew: 90° about x-axis")
    
    axis_direct, h_dir = find_desaturating_slew(q_start, q_goal, h_current, J, A_rw)
    
    print("""
ANALYSIS:
---------
For a 90° x-axis slew:
- Direct eigenaxis uses torque about x
- h_z is unchanged (no z-torque)

To reduce h_z during slew, we need z-torque.
Options:
1. Add a waypoint that requires z-rotation
2. Use non-eigenaxis path (longer but desaturating)
3. Accept slower slew with explicit desat overlay
""")
    
    # Compute different path options
    print("\nPath options:")
    print("-" * 40)
    
    # Option A: Direct eigenaxis
    print("A) Direct eigenaxis:")
    print(f"   Rotation axis: {axis_direct}")
    print(f"   Angle: 90°")
    print(f"   Momentum change: negligible z-change")
    
    # Option B: Via waypoint
    print("\nB) Via z-aligned waypoint:")
    
    # Create a path that goes via a 45° rotation about z first
    q_waypoint = quat_mult(q_start, quat_from_axis_angle([0, 0, 1], np.pi/4))
    
    axis1, angle1 = eigenaxis_slew(q_start, q_waypoint)
    axis2, angle2 = eigenaxis_slew(q_waypoint, q_goal)
    
    print(f"   Leg 1: {np.degrees(angle1):.1f}° about {axis1}")
    print(f"   Leg 2: {np.degrees(angle2):.1f}° about {axis2}")
    print(f"   Total path length: {np.degrees(angle1 + angle2):.1f}° (vs 90° direct)")
    
    # The waypoint leg uses z-torque which can reduce h_z
    print(f"   Leg 1 uses z-torque: good for h_z reduction!")
    
    print("""
TRADE-OFF:
----------
- Direct path: 90°, no desaturation
- Via waypoint: ~130°+, some desaturation

The extra slew time/distance is the "cost" of desaturation.

WHEN IS THIS WORTHWHILE?
1. When momentum is high and needs reduction
2. When time permits the longer path
3. When desaturation windows are limited

IMPLEMENTATION:
1. Check momentum state before each slew
2. If h > threshold and time permits, use waypoint path
3. Choose waypoint axis aligned with h to maximize dump
""")
    
    return


def optimal_desaturating_slew():
    """
    Find optimal waypoint that balances slew time and desaturation.
    """
    print("\n" + "=" * 80)
    print("OPTIMAL DESATURATING SLEW")
    print("=" * 80)
    
    q_start = np.array([1, 0, 0, 0])
    q_goal = quat_from_axis_angle([1, 0, 0], np.pi/2)  # 90° about x
    
    h_current = np.array([0.001, -0.002, 0.005])  # Body momentum
    h_norm = np.linalg.norm(h_current)
    h_dir = h_current / h_norm
    
    print(f"Goal: Slew 90° about x while reducing h = {h_current * 1000} mNm·s")
    
    # Parameterize waypoint as offset from midpoint
    def evaluate_path(waypoint_angle):
        """
        Waypoint at angle 'waypoint_angle' along h direction from midpoint.
        Returns: (total_path_length, z_desat_fraction)
        """
        # Create waypoint
        q_mid = quat_mult(q_start, quat_from_axis_angle([1, 0, 0], np.pi/4))
        q_wp = quat_mult(q_mid, quat_from_axis_angle(h_dir, waypoint_angle))
        q_wp = normalize(q_wp)
        
        # Compute path lengths
        _, a1 = eigenaxis_slew(q_start, q_wp)
        _, a2 = eigenaxis_slew(q_wp, q_goal)
        total_length = a1 + a2
        
        # Compute how much z-torque is used
        axis1, _ = eigenaxis_slew(q_start, q_wp)
        axis2, _ = eigenaxis_slew(q_wp, q_goal)
        
        # z-component of rotation axes (proportional to z-torque used)
        z_usage = abs(axis1[2]) * a1 + abs(axis2[2]) * a2
        
        return total_length, z_usage
    
    # Scan waypoint angles
    print("\nWaypoint angle vs path length and z-torque usage:")
    print(f"{'Angle':>10} {'Path len':>12} {'z-usage':>12}")
    print("-" * 40)
    
    for angle_deg in [0, 15, 30, 45, 60]:
        angle_rad = np.radians(angle_deg)
        path_len, z_use = evaluate_path(angle_rad)
        print(f"{angle_deg:>10}° {np.degrees(path_len):>12.1f}° {z_use:>12.3f}")
    
    print("""
INTERPRETATION:
- Larger waypoint angle → more z-torque usage → more desaturation
- But also longer path → more time
- Optimal choice depends on mission constraints

RECOMMENDATION:
1. Pre-compute "desaturation potential" for planned slews
2. If current h aligns with slew axis: free desaturation
3. If misaligned: evaluate waypoint cost vs explicit desat time
4. For small h: just use eigenaxis (fastest)
5. For large h: consider waypoint if desat window scarce
""")


def practical_implementation():
    """
    Practical algorithm for slew desaturation.
    """
    print("\n" + "=" * 80)
    print("PRACTICAL IMPLEMENTATION")
    print("=" * 80)
    
    print("""
ALGORITHM: Desaturation-Aware Slew Planning
===========================================

INPUTS:
- q_current, q_goal: Current and target quaternions
- h_rw: Current RW momentum
- t_available: Available time for maneuver
- h_threshold: Momentum threshold for desaturation

LOGIC:

1. Compute eigenaxis slew:
   axis_direct, angle_direct = eigenaxis(q_current, q_goal)
   t_direct = estimate_slew_time(angle_direct, omega_max, alpha_max)

2. Check if desaturation needed:
   if ||h_rw|| < h_threshold:
       return eigenaxis slew (fastest)

3. Check alignment:
   h_dir = h_rw / ||h_rw||
   alignment = |dot(axis_direct, h_dir)|
   
   if alignment > 0.8:  # Already well aligned
       return eigenaxis slew (will naturally desaturate)

4. If misaligned and time permits:
   # Find waypoint that uses h_dir axis
   for waypoint_angle in [15°, 30°, 45°]:
       path = compute_waypoint_path(q_current, q_goal, h_dir, waypoint_angle)
       t_path = estimate_path_time(path)
       
       if t_path <= t_available:
           desat_benefit = estimate_desaturation(path, h_rw)
           if desat_benefit > threshold:
               return waypoint path

5. Default: eigenaxis + explicit desaturation during hold

TRADE-OFFS:
- Waypoint path: ~10-50% longer but "free" desaturation
- Explicit desat: Adds time after slew but optimal path
- Hybrid: Some desat during slew, finish during hold

OUTPUT:
- Slew path (quaternion waypoints)
- Expected Δh during slew
- Remaining desat needed post-slew
""")
    
    return


if __name__ == "__main__":
    np.random.seed(42)
    
    analyze_slew_desaturation()
    optimal_desaturating_slew()
    practical_implementation()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
SLEW GEOMETRY FOR DESATURATION: KEY FINDINGS

1. EIGENAXIS SLEWS ARE MOMENTUM-NEUTRAL
   - Symmetric accel/decel → net Δh ≈ 0
   - Unless slew axis aligns with h, no desaturation

2. WAYPOINT PATHS CAN DESATURATE
   - Add intermediate point that requires torque along h
   - Trade: longer path (~10-50%) for momentum reduction
   - Most effective when h is large and misaligned with slew

3. IMPLEMENTATION STRATEGY
   a) Check if natural alignment gives free desat
   b) If h critical and time available, consider waypoints
   c) Pre-compute desat potential for upcoming slews
   d) Integrate with mission planning (observation schedule)

4. WHEN TO USE
   - Large accumulated momentum
   - Limited desat windows (orbit geometry)
   - Flexible slew timing
   
5. WHEN NOT TO USE
   - Time-critical slews
   - Small momentum
   - Slew naturally aligned with h
""")
