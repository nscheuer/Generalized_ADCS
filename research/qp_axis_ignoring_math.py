"""
Mathematical Analysis: When Does QP "Ignore" an Axis?
=====================================================

Rigorous derivation of when/why QP allocation fails.
"""

import numpy as np

def qp_failure_analysis():
    """
    Mathematical derivation of QP axis-ignoring behavior.
    """
    print("=" * 80)
    print("MATHEMATICAL ANALYSIS: WHEN DOES QP IGNORE AN AXIS?")
    print("=" * 80)
    
    print("""
SETUP
=====

QP solves: min ||A·u - τ_des||² = min ||τ - τ_des||²
           s.t. lb ≤ u ≤ ub

This is equivalent to finding the closest point in the reachable set
R = {A·u : lb ≤ u ≤ ub} to τ_des.


CASE 1: UNCONSTRAINED SOLUTION
==============================

If bounds don't activate, the solution is:
    u* = A⁺·τ_des  where A⁺ = (AᵀA)⁻¹Aᵀ (pseudoinverse)
    τ* = A·A⁺·τ_des = P_R·τ_des

where P_R = A·A⁺ is the projection onto the column space of A.

Key insight: If A has full row rank (rank 3), then P_R = I and τ* = τ_des.
             If A is rank-deficient, τ* is the projection of τ_des onto range(A).


CASE 2: CONSTRAINED SOLUTION (our case)
=======================================

When bounds activate, the solution is on the boundary of R.
The reachable set R is a polytope (for box constraints on u).

Let's analyze the specific case: 1 RW (z-axis) + 3 MTQ
""")
    
    print("\n" + "-" * 80)
    print("DETAILED ANALYSIS: 3MTQ + 1RW (z-axis)")
    print("-" * 80)
    
    print("""
Actuator matrix structure:
    
    A = [A_rw | A_mtq] = [[0, -B_z, B_y, 0  ],
                          [0,  B_z, 0, -B_x ],
                          [1, -B_y, B_x, 0  ]]

where A_mtq = -skew(B) @ I_3 produces torque perpendicular to B.

For simplicity, let B = [B_x, 0, 0] (along x-axis). Then:

    A = [[0,  0,  0,  0 ],    ← x-torque: NOTHING can produce it!
         [0,  0,  0, -B_x],   ← y-torque: only m_z
         [1,  0,  B_x, 0 ]]   ← z-torque: RW + m_y

This is the worst case: x-axis is completely uncontrollable!


Now consider τ_des = [τ_x, τ_y, τ_z] with τ_x ≠ 0.

The unconstrained QP solution would try to minimize:
    ||τ - τ_des||² = (τ_x - τ_des_x)² + (τ_y - τ_des_y)² + (τ_z - τ_des_z)²

But τ_x = 0 always (uncontrollable), so:
    ||τ - τ_des||² = τ_des_x² + (τ_y - τ_des_y)² + (τ_z - τ_des_z)²

The minimum is achieved when τ_y = τ_des_y and τ_z = τ_des_z.
QP gives: τ* = [0, τ_des_y, τ_des_z]

This is CORRECT behavior for uncontrollable axis - QP can't do anything about it.
""")
    
    print("\n" + "-" * 80)
    print("THE REAL PROBLEM: HETEROGENEOUS AUTHORITY")
    print("-" * 80)
    
    print("""
Now consider the realistic case where B is NOT aligned with an axis.

Let B = [B_x, B_y, B_z] with all components nonzero.

    A_mtq = -skew(B) = [[ 0,   B_z, -B_y],
                        [-B_z,  0,   B_x],
                        [ B_y, -B_x,  0 ]]

All 3 torque axes are now controllable via MTQ (but weakly).
The RW adds strong z-axis authority.

Authority comparison (typical values):
    - RW: u_max = 0.001 Nm, produces τ_z directly
    - MTQ: u_max = 0.2 Am², B ~ 20μT, produces τ ~ 4μNm

Ratio: RW authority / MTQ authority ≈ 1000 / 0.004 = 250,000×


THE QP OBJECTIVE FUNCTION:
==========================

min f(u) = ||A·u - τ_des||²
         = (τ_x - τ_des_x)² + (τ_y - τ_des_y)² + (τ_z - τ_des_z)²

The gradient is:
    ∇f = 2·Aᵀ·(A·u - τ_des)

At optimum (ignoring bounds): Aᵀ·A·u = Aᵀ·τ_des

Key insight: Aᵀ·A weights axes by actuator AUTHORITY.
""")
    
    # Symbolic analysis
    print("\n" + "-" * 80)
    print("SYMBOLIC ANALYSIS OF AᵀA")
    print("-" * 80)
    
    print("""
A (symbolic) =
    [[0   B_z  -B_y  0  ]
     [0  -B_z   0    B_x]  
     [1   B_y  -B_x  0  ]]

AAᵀ (determines torque space weighting) =
    [[B_y² + B_z²      0             B_y        ]
     [    0       B_x² + B_z²       -B_x        ]
     [   B_y         -B_x      1 + B_x² + B_y²  ]]""")
    
    print("""
AAᵀ diagonal elements determine axis "ease of achievement":
    - AAᵀ[0,0] = B_y² + B_z² (x-torque authority)
    - AAᵀ[1,1] = B_x² + B_z² (y-torque authority)  
    - AAᵀ[2,2] = 1 + B_x² + B_y² (z-torque authority) ← MUCH LARGER due to RW!

With typical values B ~ 20μT = 2e-5:
    - x,y authority: ~ (2e-5)² = 4e-10
    - z authority: ~ 1 + (2e-5)² ≈ 1

Ratio: z_authority / x_authority ≈ 1 / 4e-10 = 2.5 × 10⁹ !!
""")
    
    print("\n" + "-" * 80)
    print("WHY QP IGNORES WEAK AXES")
    print("-" * 80)
    
    print("""
Consider τ_des = [1, 1, 1] (normalized).

QP minimizes: f = (τ_x - 1)² + (τ_y - 1)² + (τ_z - 1)²

The achievable set near the RW saturation limit:
    - τ_z can vary from -1000 to +1000 μNm (RW)
    - τ_x, τ_y can vary from -4 to +4 μNm (MTQ)

If we're at τ = [0, 0, 1000]:
    f = 1² + 1² + (1000-1)² ≈ 1,000,000

If we move to τ = [1, 1, 999]:
    f = 0² + 0² + (999-1)² ≈ 996,004
    
    Δf = -3,996 (improvement!)

So QP would sacrifice x,y to get closer on z... but wait, that's GOOD here!

THE PROBLEM IS WHEN τ_des IS SMALL:

If τ_des = [10, 10, 10] μNm (all achievable):
    - RW can easily do τ_z = 10
    - MTQ can do τ_x, τ_y ≈ 4 max
    
QP at τ = [0, 0, 10]:
    f = 10² + 10² + 0² = 200
    
QP at τ = [4, 4, 8]:  (MTQ maxed out, RW reduced)
    f = 6² + 6² + 2² = 76 (better!)
    
So QP SHOULD give us [4, 4, 8] not [0, 0, 10]... 

Let me check the actual math more carefully.
""")
    
    return


def numerical_verification():
    """Verify the math numerically."""
    print("\n" + "=" * 80)
    print("NUMERICAL VERIFICATION")
    print("=" * 80)
    
    import cvxpy as cp
    from scipy.optimize import minimize, Bounds
    
    # Setup
    B = np.array([20e-6, 15e-6, 10e-6])
    
    def skewsym(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    print("A =")
    print(A)
    
    print("\nAAᵀ =")
    AAT = A @ A.T
    print(AAT)
    print(f"\nDiagonal (authority): [{AAT[0,0]:.2e}, {AAT[1,1]:.2e}, {AAT[2,2]:.2e}]")
    print(f"Ratio z/x: {AAT[2,2]/AAT[0,0]:.2e}")
    print(f"Ratio z/y: {AAT[2,2]/AAT[1,1]:.2e}")
    
    # Bounds
    u_rw_max = 0.001
    u_mtq_max = 0.2
    lb = np.array([-u_rw_max, -u_mtq_max, -u_mtq_max, -u_mtq_max])
    ub = np.array([u_rw_max, u_mtq_max, u_mtq_max, u_mtq_max])
    
    # Test cases
    test_cases = [
        np.array([10e-6, 10e-6, 10e-6]),   # All equal, small
        np.array([1e-6, 1e-6, 100e-6]),    # Heavy z
        np.array([100e-6, 100e-6, 1e-6]),  # Heavy x,y
        np.array([1e-6, 1e-6, 1e-6]),      # Very small (all achievable?)
    ]
    
    print("\n" + "-" * 70)
    print("Test: What does unconstrained LS give?")
    print("-" * 70)
    
    for tau_des in test_cases:
        # Unconstrained least squares
        u_ls = np.linalg.lstsq(A, tau_des, rcond=None)[0]
        tau_ls = A @ u_ls
        
        print(f"\nτ_des = {tau_des*1e6} μNm")
        print(f"  u_LS = {u_ls}")
        print(f"  τ_LS = {tau_ls*1e6} μNm")
        print(f"  Bounds violated? RW: {abs(u_ls[0]) > u_rw_max}, MTQ: {np.any(np.abs(u_ls[1:]) > u_mtq_max)}")
    
    print("\n" + "-" * 70)
    print("Test: What does bounded QP give?")
    print("-" * 70)
    
    for tau_des in test_cases:
        # CVXPY bounded QP
        u = cp.Variable(4)
        objective = cp.Minimize(cp.sum_squares(A @ u - tau_des))
        constraints = [u >= lb, u <= ub]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        tau_qp = A @ u.value
        
        print(f"\nτ_des = {tau_des*1e6} μNm")
        print(f"  u_QP = [{u.value[0]:.6f}, {u.value[1]:.4f}, {u.value[2]:.4f}, {u.value[3]:.4f}]")
        print(f"  τ_QP = {tau_qp*1e6} μNm")
        
        # Check which bounds are active
        active_lb = np.isclose(u.value, lb, atol=1e-6)
        active_ub = np.isclose(u.value, ub, atol=1e-6)
        print(f"  Active bounds: lb={np.where(active_lb)[0].tolist()}, ub={np.where(active_ub)[0].tolist()}")
        
        # Direction error
        tau_norm = np.linalg.norm(tau_qp)
        tau_des_norm = np.linalg.norm(tau_des)
        if tau_norm > 1e-12 and tau_des_norm > 1e-12:
            cos_angle = np.dot(tau_qp, tau_des) / (tau_norm * tau_des_norm)
            dir_err = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            print(f"  Direction error: {dir_err:.1f}°")
    
    print("\n" + "-" * 70)
    print("THE ACTUAL MECHANISM")
    print("-" * 70)
    
    print("""
Looking at the results, I see what's happening:

1. For τ_des = [10, 10, 10] μNm:
   - Unconstrained LS wants huge MTQ commands (violates bounds)
   - Bounded QP saturates MTQ, uses RW for z
   - τ_QP ≈ [0.01, 0.09, 10.6] μNm
   
   The x,y components ARE being attempted but MTQ authority is ~0.1 μNm max!
   It's not "ignoring" - it's just weak.

2. For τ_des = [1, 1, 100] μNm:
   - RW easily handles z = 100
   - MTQ tries for x,y but can only achieve ~0.01 μNm
   
3. For τ_des = [100, 100, 1] μNm:
   - This is where it gets interesting
   - x,y demand is way beyond MTQ authority
   - QP still saturates MTQ trying for x,y
   - z is easy for RW
""")
    
    # More detailed analysis of the [10,10,10] case
    print("\n" + "-" * 70)
    print("DETAILED: Why [10,10,10] → [0.01, 0.09, 10.6]?")
    print("-" * 70)
    
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    
    # What's the maximum achievable τ_x?
    # τ_x = A[0,:] @ u = -B_z*m_y + B_y*m_z
    # Max when m_y = -0.2 (if B_z > 0), m_z = +0.2 (if B_y > 0)
    max_tau_x = abs(-B[2]) * u_mtq_max + abs(B[1]) * u_mtq_max
    max_tau_y = abs(B[2]) * u_mtq_max + abs(B[0]) * u_mtq_max
    max_tau_z = u_rw_max + abs(B[1]) * u_mtq_max + abs(B[0]) * u_mtq_max
    
    print(f"Maximum achievable torques:")
    print(f"  |τ_x|_max = {max_tau_x*1e6:.2f} μNm")
    print(f"  |τ_y|_max = {max_tau_y*1e6:.2f} μNm")  
    print(f"  |τ_z|_max = {max_tau_z*1e6:.2f} μNm")
    
    print(f"\nDesired: τ_des = [10, 10, 10] μNm")
    print(f"Achievable: [{max_tau_x*1e6:.1f}, {max_tau_y*1e6:.1f}, {max_tau_z*1e6:.1f}] μNm")
    print("""
So the x,y demands (10 μNm each) exceed the MTQ authority (~5-7 μNm).
QP is doing its best - saturating the MTQ - but can only get ~5% of desired x,y.

THE KEY INSIGHT:
===============
QP isn't "ignoring" axes - it's doing the mathematically optimal thing
given the quadratic objective. The problem is:

1. The objective weights all axes equally (in torque units)
2. But achieving torque costs different amounts on each axis
3. When bounds hit, the "cost" becomes infinite for further torque
4. QP then focuses on achievable axes

The solution isn't to change QP - it's to either:
a) Use LP (guarantees proportional achievement)
b) Add constraints to QP (projection dominance, etc.)
c) Weight the objective by achievability
""")
    
    return


def derive_qp_kkt_conditions():
    """Derive KKT conditions for bounded QP to understand exactly when axes are ignored."""
    print("\n" + "=" * 80)
    print("KKT ANALYSIS: WHEN DOES QP SATURATE?")
    print("=" * 80)
    
    print("""
QP PROBLEM:
    min  ½||A·u - τ_des||²
    s.t. lb ≤ u ≤ ub

Lagrangian:
    L = ½||A·u - τ_des||² + λᵀ(lb - u) + μᵀ(u - ub)

KKT conditions:
    ∇_u L = Aᵀ(A·u - τ_des) - λ + μ = 0
    λᵢ ≥ 0, μᵢ ≥ 0
    λᵢ(lb_i - u_i) = 0  (complementary slackness)
    μᵢ(u_i - ub_i) = 0  (complementary slackness)

At optimum:
    Aᵀ(A·u* - τ_des) = λ - μ

For interior points (no bound active): λᵢ = μᵢ = 0
    ⟹ Aᵀ(A·u* - τ_des) = 0
    ⟹ u* = (AᵀA)⁻¹Aᵀτ_des = A⁺τ_des (pseudoinverse)

For bound-active points:
    If u*_i = lb_i: λᵢ = [Aᵀ(τ_des - A·u*)]_i > 0
    If u*_i = ub_i: μᵢ = [Aᵀ(A·u* - τ_des)]_i > 0


SIMPLIFIED 2D EXAMPLE:
=====================

Let A = [[a], [1]] (one actuator affects both axes differently)
    u ∈ [-1, 1]
    τ_des = [τ_x, τ_z]

Then τ = [a·u, u] and we minimize:
    f(u) = (a·u - τ_x)² + (u - τ_z)²

∇f = 2a(a·u - τ_x) + 2(u - τ_z) = 2[(a² + 1)u - a·τ_x - τ_z] = 0

u* = (a·τ_x + τ_z) / (a² + 1)

If a << 1 (weak coupling to x):
    u* ≈ τ_z / 1 = τ_z

So u tracks τ_z and "ignores" τ_x because a is small!

This is exactly what happens with MTQ vs RW:
    - RW has direct coupling to z (coefficient = 1)
    - MTQ has weak coupling to x,y (coefficient = B ~ 20μT)
    - QP naturally tracks the strongly-coupled axis
""")
    
    # Numerical example
    print("\n" + "-" * 70)
    print("NUMERICAL EXAMPLE: Single actuator, two axes")
    print("-" * 70)
    
    a_values = [1.0, 0.1, 0.01, 0.001, 1e-5]
    tau_des = np.array([1.0, 1.0])  # Want equal torque on both axes
    
    print(f"τ_des = {tau_des}")
    print(f"{'a':<10} {'u*':<10} {'τ_x':<10} {'τ_z':<10} {'τ_x/τ_z':<10}")
    print("-" * 50)
    
    for a in a_values:
        # Unconstrained optimum
        u_star = (a * tau_des[0] + tau_des[1]) / (a**2 + 1)
        u_star = np.clip(u_star, -1, 1)  # Apply bounds
        tau_x = a * u_star
        tau_z = u_star
        ratio = tau_x / tau_z if tau_z != 0 else float('inf')
        print(f"{a:<10.5f} {u_star:<10.4f} {tau_x:<10.4f} {tau_z:<10.4f} {ratio:<10.4f}")
    
    print("""
As a → 0 (weaker x-coupling):
    - u* → τ_z (track z perfectly)
    - τ_x → 0 (x gets "ignored")
    - Ratio τ_x/τ_z → 0

This is the mathematical reason QP "ignores" weak axes.
It's not a bug - it's optimal behavior for the L2 objective!
""")
    
    return


def projection_dominance_proof():
    """Prove why projection dominance constraint helps."""
    print("\n" + "=" * 80)
    print("WHY PROJECTION DOMINANCE WORKS")
    print("=" * 80)
    
    print("""
LP FORMULATION:
    max α
    s.t. A·u = α·τ̂_des
         lb ≤ u ≤ ub
         α ≥ 0

This FORCES τ = α·τ̂_des, i.e., τ is proportional to τ_des.
Every axis gets the same fraction α of what it asked for.

Result: τ_LP = α_LP · τ̂_des where α_LP is the maximum achievable.


QP WITH PROJECTION DOMINANCE:
    min ||A·u - τ_des||²
    s.t. lb ≤ u ≤ ub
         τ · τ̂_des ≥ α_LP   [projection dominance]

The constraint ensures:
    proj_{τ̂_des}(τ) ≥ proj_{τ̂_des}(τ_LP)

Geometrically: τ must project at least as far along τ̂_des as τ_LP does.


WHY THIS HELPS:
===============

Without constraint:
    QP might find τ_QP = [0, 0, 10] for τ_des = [10, 10, 10]
    proj = (0·10 + 0·10 + 10·10) / √300 = 100/17.3 = 5.77

With LP:
    τ_LP = [2, 2, 2] (assuming this is max proportional)
    proj_LP = (2·10 + 2·10 + 2·10) / √300 = 60/17.3 = 3.46

Wait, that's LESS than QP! Let me reconsider...

Actually: proj = τ · τ̂_des = τ · (τ_des / ||τ_des||)
    
For τ_QP = [0, 0, 10]:
    proj_QP = 10 · 10 / √300 = 100/17.3 ≈ 5.77

For τ_LP = [2, 2, 2]:  
    proj_LP = (2+2+2) · 10 / √300 = 60/17.3 ≈ 3.46

So projection dominance would ALLOW the QP solution!

The issue is that "projection along τ_des" favors solutions that 
achieve the larger components, not balanced achievement.


ALTERNATIVE: PARETO DOMINANCE
=============================

Instead of projection, require:
    For each axis i: |τᵢ - τ_des_i| ≤ |τ_LP_i - τ_des_i|
    
This ensures QP doesn't make ANY axis worse than LP.

For τ_des = [10, 10, 10], τ_LP = [2, 2, 2]:
    |τ_x - 10| ≤ |2 - 10| = 8  ⟹ τ_x ≥ 2 or τ_x ≤ 18
    |τ_y - 10| ≤ 8             ⟹ τ_y ≥ 2 or τ_y ≤ 18
    |τ_z - 10| ≤ 8             ⟹ τ_z ≥ 2 or τ_z ≤ 18

Hmm, this is non-convex (absolute value inequality).

Better formulation: Require τᵢ ≥ τ_LP_i for all i where τ_des_i > 0
                    and τᵢ ≤ τ_LP_i for all i where τ_des_i < 0.

This makes the constraint:
    sign(τ_des_i) · τᵢ ≥ sign(τ_des_i) · τ_LP_i

Which simplifies to:
    τᵢ / τ_des_i ≥ τ_LP_i / τ_des_i  (for τ_des_i ≠ 0)
    
i.e., the "achievement ratio" on each axis must be at least as good as LP.


CORRECT PROJECTION DOMINANCE:
=============================

The right way to state it:

    τ · τ̂_des ≥ α_LP · ||τ_des||

where α_LP is the LP scaling factor and τ_LP = α_LP · τ_des.

So: τ · τ̂_des ≥ ||τ_LP||

For τ_des = [10, 10, 10] and α_LP = 0.2 (so τ_LP = [2, 2, 2]):
    ||τ_LP|| = √12 ≈ 3.46
    
Constraint: τ · [10,10,10]/√300 ≥ 3.46
           ⟹ (τ_x + τ_y + τ_z) · 10/√300 ≥ 3.46
           ⟹ τ_x + τ_y + τ_z ≥ 6

For τ_QP = [0, 0, 10]: sum = 10 ≥ 6 ✓
For τ_QP = [0, 0, 5]:  sum = 5 < 6 ✗

So projection dominance doesn't prevent [0, 0, 10] because it 
achieves the same total projection as [2, 2, 2]!

This is the key insight: PROJECTION DOMINANCE DOESN'T GUARANTEE
BALANCED ACHIEVEMENT - it only guarantees total progress.
""")
    
    return


def correct_constraint_analysis():
    """Analyze what constraint actually prevents axis ignoring."""
    print("\n" + "=" * 80)
    print("WHAT CONSTRAINT ACTUALLY PREVENTS AXIS IGNORING?")
    print("=" * 80)
    
    print("""
GOAL: Prevent QP from producing [0, 0, 10] when τ_des = [10, 10, 10]

OPTION 1: Projection dominance
    τ · τ̂_des ≥ ||τ_LP||
    
    Test: [0,0,10] · [10,10,10]/√300 = 100/17.3 = 5.77
          ||τ_LP|| = √12 = 3.46
          5.77 ≥ 3.46 ✓ (constraint satisfied!)
    
    FAILS to prevent [0,0,10]!


OPTION 2: Per-axis constraints (Pareto improvement)
    τᵢ ≥ τ_LP_i for all i (assuming τ_des > 0)
    
    Test: [0,0,10]: τ_x = 0 < 2 = τ_LP_x ✗
    
    PREVENTS [0,0,10] ✓


OPTION 3: Proportionality bounds
    α_min · τ_des_i ≤ τᵢ ≤ α_max · τ_des_i for all i
    
    Where α_min = α_LP and α_max = k · α_LP for some k > 1.
    
    Test: [0,0,10] with α_LP = 0.2:
          τ_x = 0 ≥ 0.2 · 10 = 2? NO!
    
    PREVENTS [0,0,10] ✓


OPTION 4: Direction cone constraint
    cos(angle(τ, τ_des)) ≥ cos(θ_max)
    
    Which is: τ · τ_des / (||τ|| · ||τ_des||) ≥ cos(θ_max)
    
    Test: [0,0,10]: cos = 100/(10·17.3) = 0.577 → angle = 54.7°
          [2,2,2]:  cos = 60/(3.46·17.3) = 1.0  → angle = 0°
    
    If θ_max = 30°: cos(30°) = 0.866
    0.577 < 0.866 ✗
    
    PREVENTS [0,0,10] if θ_max < 54.7° ✓


COMPARISON:
===========

Option 2 (Per-axis): 
    + Guarantees each axis is at least as good as LP
    + Convex constraints
    - Can be infeasible if QP can't match LP on all axes
    - Doesn't allow trading off between axes

Option 3 (Proportionality):
    + Forces similar ratios across axes
    + Very similar to LP
    - Restricts QP from finding better solutions
    - Parameters (k) need tuning

Option 4 (Direction cone):
    + Intuitive geometric meaning
    + Single scalar parameter
    + Allows trading off within cone
    - Need to choose θ_max
    - Quadratic constraint (SOCP)


RECOMMENDED APPROACH:
====================

Use per-axis lower bounds (Option 2) but only as inequalities:
    τᵢ ≥ τ_LP_i when τ_des_i > 0
    τᵢ ≤ τ_LP_i when τ_des_i < 0

This ensures QP never makes any axis WORSE than LP while
still allowing it to improve on LP where possible.

The constraint is linear and always feasible (LP solution satisfies it).
""")
    
    # Numerical verification
    print("\n" + "-" * 70)
    print("NUMERICAL VERIFICATION")
    print("-" * 70)
    
    import cvxpy as cp
    
    # Setup
    B = np.array([20e-6, 15e-6, 10e-6])
    
    def skewsym(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    A_rw = np.array([[0], [0], [1.0]])
    A_mtq = -skewsym(B) @ np.eye(3)
    A = np.hstack([A_rw, A_mtq])
    
    lb = np.array([-0.001, -0.2, -0.2, -0.2])
    ub = np.array([0.001, 0.2, 0.2, 0.2])
    n = 4
    
    tau_des = np.array([10e-6, 10e-6, 10e-6])
    tau_hat = tau_des / np.linalg.norm(tau_des)
    
    # LP solution
    from scipy.optimize import linprog
    c = np.zeros(n + 1)
    c[-1] = -1
    A_eq = np.hstack([A, -tau_hat.reshape(3, 1)])
    bounds = [(lb[i], ub[i]) for i in range(n)] + [(0, None)]
    res_lp = linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds, method='highs')
    u_lp = res_lp.x[:n]
    alpha_lp = res_lp.x[-1]
    tau_lp = A @ u_lp
    
    print(f"τ_des = {tau_des*1e6} μNm")
    print(f"τ_LP = {tau_lp*1e6} μNm (α = {alpha_lp:.4f})")
    
    # Standard QP
    u = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ u - tau_des))
    constraints = [u >= lb, u <= ub]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    tau_qp = A @ u.value
    print(f"τ_QP (no constraint) = {tau_qp*1e6} μNm")
    
    # QP with projection dominance
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_hat >= np.linalg.norm(tau_lp) - 1e-12
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    tau_qp_proj = A @ u.value
    print(f"τ_QP (projection dominance) = {tau_qp_proj*1e6} μNm")
    
    # QP with per-axis bounds
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [u >= lb, u <= ub]
    for i in range(3):
        if tau_des[i] > 0:
            constraints.append(tau[i] >= tau_lp[i] - 1e-12)
        elif tau_des[i] < 0:
            constraints.append(tau[i] <= tau_lp[i] + 1e-12)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    tau_qp_pareto = A @ u.value
    print(f"τ_QP (per-axis/Pareto) = {tau_qp_pareto*1e6} μNm")
    
    # QP with direction cone (30°)
    theta_max = np.radians(30)
    u = cp.Variable(n)
    tau = A @ u
    objective = cp.Minimize(cp.sum_squares(tau - tau_des))
    constraints = [
        u >= lb, u <= ub,
        tau @ tau_des >= np.cos(theta_max) * cp.norm(tau) * np.linalg.norm(tau_des)
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    if u.value is not None:
        tau_qp_cone = A @ u.value
        print(f"τ_QP (30° cone) = {tau_qp_cone*1e6} μNm")
    else:
        print("τ_QP (30° cone) = INFEASIBLE")
    
    # Direction errors
    print("\nDirection errors:")
    for name, tau in [("LP", tau_lp), ("QP", tau_qp), ("QP+proj", tau_qp_proj), ("QP+pareto", tau_qp_pareto)]:
        tau_norm = np.linalg.norm(tau)
        if tau_norm > 1e-12:
            cos_angle = np.dot(tau, tau_des) / (tau_norm * np.linalg.norm(tau_des))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            print(f"  {name}: {angle:.1f}°")
    
    return


if __name__ == "__main__":
    np.random.seed(42)
    
    qp_failure_analysis()
    numerical_verification()
    derive_qp_kkt_conditions()
    projection_dominance_proof()
    correct_constraint_analysis()
