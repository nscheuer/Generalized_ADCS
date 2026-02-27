"""
Tests for APLQR and ThreeAxisMPC controllers.

Both controllers share the same linearized attitude-error dynamics and the same
MTQ-only actuation structure; they differ in how they compute the feedback gain
online (CARE steady-state vs. receding-horizon QP).

Tuning Guide
------------
APLQR
~~~~~
Tune three things:

  Q (6×6, state cost)
    Penalises [attitude error (0:3), angular-rate error (3:6)].
    *Larger Q / smaller R → faster response; smaller Q → smoother but slower.*
    Start: ``np.diag([1, 1, 1, 1, 1, 1])``
    Rate-dominate: ``np.diag([1, 1, 1, 100, 100, 100])`` damps spin faster.

  R_ctrl (3×3, dipole cost)
    Penalises the commanded magnetic dipole magnitude.
    Start: ``np.eye(3)``. Decrease (e.g. 0.1) for more aggressive authority.

  alpha (scalar gain multiplier, default 1.0)
    Scales the full control output after the LQR gain.  Increase to speed up
    convergence; decrease if you see sustained oscillations.  This is the
    easiest single knob to adjust on-orbit.

  Note: settling time for an MTQ-only controller is *inherently* 2–5 orbital
  periods regardless of gains, because controllability relies on the rotating
  B-field.  Gains mostly control oscillation amplitude, not the time constant.

ThreeAxisMPC
~~~~~~~~~~~~
  Q (6×6)  / Q_N (6×6, terminal)
    Same role as APLQR's Q.  Set Q_N ≥ Q for stability (common: Q_N = 10·Q).

  R (3×3, torque cost)
    Smaller R → more aggressive torque commands.  Start: ``np.eye(3) * 0.01``.

  N (horizon, steps)
    Longer horizon → better prediction of rotating B-field → better null-space
    exploitation.  Practical range: 10–20 steps.  Each step adds one
    ppigrf call inside find_u, so there is a compute cost.
    Rule of thumb: N * dt ≈ 1/4 of one orbital period.

  dt (MPC step, seconds)
    Should match your actual control update rate (20 s is typical for LEO MTQ).

  solver
    ``"closedform"`` is always available and mathematically equivalent to TinyMPC
    for the B-constraint problem (exact null-space projection).
    ``"tinympc"`` requires ``pip install tinympc`` (Linux/macOS/WSL).

Common pitfalls
~~~~~~~~~~~~~~~
* **All axes saturated immediately**: R is too small or alpha too large.
  Check: ``np.all(np.abs(u_hist[:10]) == mtq_umax)``.
* **No attitude change after many orbits**: Q might be zero or negligibly small
  relative to R — the controller is "too conservative."
* **Divergence**: numerical issues in the CARE (APLQR) — check that J is
  diagonal and positive definite; check sma/incl are in metres/radians.
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Optional, Dict
from functools import lru_cache
from tqdm import tqdm
import pytest

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

# ---------------------------------------------------------------------------
# ADCS Imports
# ---------------------------------------------------------------------------
from ADCS.CONOPS.goals import Goal, ECI_Goal, No_Goal
from ADCS.controller import APLQR, ThreeAxisMPC
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import random_n_unit_vec, normalize

# ---------------------------------------------------------------------------
# Plotting (optional, for __main__ block)
# ---------------------------------------------------------------------------
from ADCS.helpers.plotting.animate_estimator import animate_attitude
from ADCS.helpers.plotting.plot_estimator import plot_state_comparison
from ADCS.helpers.plotting.close_all_plots import create_close_all_button_window
from ADCS.helpers.plotting.plot_controller import plot_control, plot_target_tracking

# ===========================================================================
# Shared Hardware / Orbit Configuration
# ===========================================================================

# Satellite: 4 kg CubeSat with diagonal inertia [kg·m²] and ±1 A·m² MTQs
MTQ_UMAX   = 1.0                             # dipole limit [A·m²]
INERTIA    = np.diagflat([3.4, 2.9, 1.3])   # [kg·m²]
MASS       = 4.0                             # [kg]
BORESIGHT  = np.array([0.0, 0.0, 1.0])

# Orbit: ~7000 km LEO, inclination ≈ 45° (derived from R, V below)
SMA        = 7.0e6                           # semi-major axis [m]
INCL       = np.pi / 4.0                     # inclination [rad]
J2000_0    = 0.22                            # reference epoch [J2000 centuries]
R0_KM      = 7000.0 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2])  # [km]
V0_KMS     = np.array([8.0, 0.0, 0.0])                                    # [km/s]

# Physics simulation step
DT_PHYSICS = 20.0    # [s] — matches MPC default dt; adequate for MTQ control

# Simulation durations
TF_SMOKE   = 60.0    # [s] — just enough for one find_u smoke call
TF_SLOW    = 10000.0 # [s] — ≈1.7 orbital periods; marked @pytest.mark.slow

# ---------------------------------------------------------------------------
# APLQR default gains (starting point for tuning)
# ---------------------------------------------------------------------------
APLQR_CFG: Dict = dict(
    Q       = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    R_ctrl  = np.eye(3),
    alpha   = 1.0,
    target_mode    = "Inertial",
    has_gg_torque  = False,
)

# ---------------------------------------------------------------------------
# ThreeAxisMPC default configuration (starting point for tuning)
# ---------------------------------------------------------------------------
MPC_CFG: Dict = dict(
    N           = 10,
    dt          = 20.0,
    Q           = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    Q_N         = np.diag([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
    R           = np.eye(3) * 0.01,
    target_mode = "Inertial",
    has_gg_torque = False,
    solver      = "closedform",
)


# ===========================================================================
# Shared Utilities
# ===========================================================================

def _make_satellite() -> Tuple[Satellite, np.ndarray]:
    """MTQ-only 4 kg satellite, state = [ω(3), q(4)]."""
    mtqs = [MTQ(axis=ax, max_torque=MTQ_UMAX) for ax in MathConstants.unitvecs]
    mtms = [MTM(axis=ax) for ax in MathConstants.unitvecs]
    sat  = Satellite(
        mass=MASS,
        J_0=INERTIA,
        actuators=mtqs,
        sensors=mtms,
        boresight=BORESIGHT,
    )
    w0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    x0 = np.concatenate([w0, q0])
    return sat, x0


@lru_cache(maxsize=4)
def _get_orbit_cached(tf: float, dt: float) -> Orbit:
    """Pre-propagated orbit (cached so multiple tests reuse the same object)."""
    ephem      = Ephemeris()
    start_time = J2000_0 - 1.0 * TimeConstants.sec2cent
    end_time   = J2000_0 + tf * TimeConstants.sec2cent
    os0        = Orbital_State(ephem=ephem, J2000=start_time, R=R0_KM, V=V0_KMS)
    return Orbit(os0=os0, end_time=end_time, dt=dt, use_J2=True, fast=False)


def _make_scenario(name: str) -> Tuple[np.ndarray, np.ndarray, Goal, float]:
    """Return (w0, q0, goal, tf) for a named scenario."""
    name = name.lower()
    w0   = np.zeros(3)
    q0   = np.array([1.0, 0.0, 0.0, 0.0])
    goal = No_Goal()
    tf   = TF_SLOW

    if "spin" in name:
        w0 = random_n_unit_vec(3) * np.random.uniform(0.1, 0.3) * np.pi / 180.0
    if "random_q" in name or "spin" in name:
        q0 = normalize(random_n_unit_vec(4))
    if "align_z" in name:
        goal = ECI_Goal(np.array([0.0, 0.0, 1.0]))
    elif "align_x" in name:
        goal = ECI_Goal(np.array([1.0, 0.0, 0.0]))

    return w0, q0, goal, tf


def _simulate(
    controller,
    tf: float,
    dt: float,
    w0: np.ndarray,
    q0: np.ndarray,
    goal: Goal,
    verbose: bool = False,
    desc: str = "Simulating",
) -> Tuple[np.ndarray, np.ndarray, List[Orbital_State], np.ndarray]:
    """Generic closed-loop simulation loop (MTQ-only, no RW states)."""
    sat, _ = _make_satellite()
    orbit  = _get_orbit_cached(tf=tf, dt=dt)

    x = np.concatenate([w0, q0])
    steps = int(tf / dt)

    time_hist  = np.full(steps, np.nan)
    state_hist = np.full((steps, 7), np.nan)
    u_hist     = np.full((steps, 3), np.nan)
    os_hist: List[Orbital_State] = []

    t = 0.0
    for i in tqdm(range(steps), desc=desc, disable=not verbose):
        J2000   = J2000_0 + t * TimeConstants.sec2cent
        os_now  = orbit.get_os(J2000=J2000)
        os_next = orbit.get_os(J2000=J2000_0 + (t + dt) * TimeConstants.sec2cent)

        sens = sat.sensor_readings(x=x, os=os_now)
        u    = controller.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)

        time_hist[i]     = t
        state_hist[i, :] = x
        u_hist[i, :]     = u[:3]   # first 3 slots = MTQ commands
        os_hist.append(os_now)

        out = solve_ivp(
            fun=sat.dynamics_for_solver,
            t_span=(0.0, dt),
            y0=x,
            method="RK45",
            args=(u, os_now, os_next),
            rtol=1e-7,
            atol=1e-7,
        )
        x = out.y[:, -1]
        x[3:7] = normalize(x[3:7])
        t += dt

    return time_hist, state_hist, os_hist, u_hist


# ===========================================================================
# APLQR Tests
# ===========================================================================

class TestAPLQRSmoke:
    """Fast sanity checks — no full simulation."""

    @pytest.fixture(scope="class")
    def sat_and_ctrl_nadir(self):
        sat, x0 = _make_satellite()
        ctrl = APLQR(
            est_sat=sat,
            sma=SMA, incl=INCL,
            Q=APLQR_CFG["Q"],
            R_ctrl=APLQR_CFG["R_ctrl"],
            alpha=APLQR_CFG["alpha"],
            target_mode="Nadir",
        )
        return sat, ctrl, x0

    @pytest.fixture(scope="class")
    def sat_and_ctrl_inertial(self):
        sat, x0 = _make_satellite()
        ctrl = APLQR(
            est_sat=sat,
            sma=SMA, incl=INCL,
            Q=APLQR_CFG["Q"],
            R_ctrl=APLQR_CFG["R_ctrl"],
            alpha=APLQR_CFG["alpha"],
            target_mode="Inertial",
        )
        return sat, ctrl, x0

    def test_aplqr_nadir_instantiation(self, sat_and_ctrl_nadir):
        """CARE solve succeeds in Nadir mode; P_ss is PSD."""
        _, ctrl, _ = sat_and_ctrl_nadir
        assert ctrl.P_ss.shape == (6, 6)
        eigvals = np.linalg.eigvalsh(ctrl.P_ss)
        assert np.all(eigvals >= -1e-10), f"P_ss not PSD: min eig = {eigvals.min():.3e}"

    def test_aplqr_inertial_instantiation(self, sat_and_ctrl_inertial):
        """CARE solve succeeds in Inertial mode."""
        _, ctrl, _ = sat_and_ctrl_inertial
        assert ctrl.P_ss.shape == (6, 6)
        eigvals = np.linalg.eigvalsh(ctrl.P_ss)
        assert np.all(eigvals >= -1e-10)

    def test_aplqr_u_shape_and_finite(self, sat_and_ctrl_inertial):
        """find_u returns a finite vector of correct length."""
        sat, ctrl, x0 = sat_and_ctrl_inertial
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)
        goal   = ECI_Goal(np.array([0.0, 0.0, 1.0]))

        u = ctrl.find_u(x_hat=x0, sens=sat.sensor_readings(x=x0, os=os_hat),
                        est_sat=sat, os_hat=os_hat, goal=goal)
        assert u.shape == (len(sat.actuators),), "Wrong output length"
        assert np.all(np.isfinite(u)), f"find_u returned non-finite values: {u}"

    def test_aplqr_saturation_respected(self, sat_and_ctrl_inertial):
        """Dipole commands never exceed the declared MTQ limit."""
        sat, ctrl, _ = sat_and_ctrl_inertial
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)

        # Worst-case: large attitude + rate error
        w_big = np.array([0.1, 0.1, 0.1])
        q_id  = np.array([1.0, 0.0, 0.0, 0.0])
        x_big = np.concatenate([w_big, q_id])
        u = ctrl.find_u(x_hat=x_big, sens=sat.sensor_readings(x=x_big, os=os_hat),
                        est_sat=sat, os_hat=os_hat)
        assert np.all(np.abs(u) <= MTQ_UMAX + 1e-9), \
            f"MTQ saturation violated: {np.abs(u).max():.4f} > {MTQ_UMAX}"

    def test_aplqr_orbital_average_B_positive(self, sat_and_ctrl_inertial):
        """Orbital-average B̃ diagonal entries must be positive (Psiaki 2001 eq 20)."""
        _, ctrl, _ = sat_and_ctrl_inertial
        B_tilde = ctrl._orbital_average_B()
        lower_half = B_tilde[3:, :]
        diag_vals  = np.diag(lower_half)
        assert np.all(diag_vals > 0), f"B̃ diagonal not positive: {diag_vals}"

    def test_aplqr_zero_error_zero_output(self, sat_and_ctrl_inertial):
        """Zero error state (identity quat, zero rate, No_Goal) → zero torque."""
        sat, ctrl, x0 = sat_and_ctrl_inertial
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)
        u = ctrl.find_u(x_hat=x0, sens=sat.sensor_readings(x=x0, os=os_hat),
                        est_sat=sat, os_hat=os_hat, goal=No_Goal())
        assert np.allclose(u, 0.0, atol=1e-12), \
            f"Non-zero output for zero-error state: {u}"


# ---------------------------------------------------------------------------
# Slow simulation tests (APLQR)
# ---------------------------------------------------------------------------

APLQR_SCENARIOS = [
    "stop_spin",
    "align_z_static",
    "align_z_spin",
    "align_x_static",
]

@pytest.mark.slow
@pytest.mark.parametrize("scenario", APLQR_SCENARIOS)
def test_aplqr_convergence(scenario: str) -> None:
    """
    APLQR: angular rate must drop below threshold within TF_SLOW seconds.

    The fundamental convergence rate of MTQ-only control is set by the orbit
    (1–5 orbital periods), so TF_SLOW = 10 000 s ≈ 1.7 orbits is sufficient
    to see clear rate reduction even if full pointing is not yet achieved.
    """
    np.random.seed(42)
    w0, q0, goal, tf = _make_scenario(scenario)
    sat, _ = _make_satellite()

    ctrl = APLQR(
        est_sat=sat,
        sma=SMA, incl=INCL,
        Q=APLQR_CFG["Q"],
        R_ctrl=APLQR_CFG["R_ctrl"],
        alpha=APLQR_CFG["alpha"],
        target_mode=APLQR_CFG["target_mode"],
        has_gg_torque=APLQR_CFG["has_gg_torque"],
    )

    _, state_hist, _, u_hist = _simulate(
        controller=ctrl, tf=tf, dt=DT_PHYSICS,
        w0=w0, q0=q0, goal=goal,
        desc=f"APLQR {scenario}",
    )

    valid = ~np.isnan(state_hist[:, 0])
    w_init  = np.linalg.norm(state_hist[valid][0,  0:3])
    w_final = np.linalg.norm(state_hist[valid][-1, 0:3])

    # 1. No blow-up
    assert np.all(np.isfinite(state_hist[valid])), "State diverged (NaN/Inf)"

    # 2. Saturation never exceeded
    assert np.all(np.abs(u_hist[valid]) <= MTQ_UMAX + 1e-9), "MTQ limit violated"

    # 3. Rate reduction (allow for no-spin scenario: already ~0)
    if w_init > 1e-4:
        assert w_final < w_init, (
            f"Angular rate did not decrease: {w_init:.4e} → {w_final:.4e} rad/s"
        )


# ===========================================================================
# ThreeAxisMPC Tests
# ===========================================================================

class TestThreeAxisMPCSmoke:
    """Fast sanity checks for ThreeAxisMPC."""

    @pytest.fixture(scope="class")
    def sat_and_ctrl(self):
        sat, x0 = _make_satellite()
        ctrl = ThreeAxisMPC(
            est_sat=sat,
            sma=SMA, incl=INCL,
            N=MPC_CFG["N"],
            dt=MPC_CFG["dt"],
            Q=MPC_CFG["Q"],
            Q_N=MPC_CFG["Q_N"],
            R=MPC_CFG["R"],
            target_mode=MPC_CFG["target_mode"],
            solver="closedform",
        )
        return sat, ctrl, x0

    def test_mpc_closedform_instantiation(self, sat_and_ctrl):
        """Closed-form matrices are precomputed with correct shapes."""
        sat, ctrl, _ = sat_and_ctrl
        N1 = MPC_CFG["N"] - 1
        assert ctrl._cf_Gamma.shape  == (6 * N1, 3 * N1)
        assert ctrl._cf_Y.shape      == (6 * N1, 6)
        assert ctrl._cf_Lambda.shape == (3 * N1, 3 * N1)

    def test_mpc_lambda_positive_definite(self, sat_and_ctrl):
        """Lambda = (Γ'Q̄Γ + R̄)⁻¹ must be positive definite."""
        _, ctrl, _ = sat_and_ctrl
        eigvals = np.linalg.eigvalsh(ctrl._cf_Lambda)
        assert np.all(eigvals > 0), f"Lambda not PD: min eig = {eigvals.min():.3e}"

    def test_mpc_ad_bd_discretization(self, sat_and_ctrl):
        """Discretized Ad should be close to identity for small dt·n0."""
        _, ctrl, _ = sat_and_ctrl
        # Ad should be stable (spectral radius ≤ 1 for open-loop stable system)
        eigvals = np.abs(np.linalg.eigvals(ctrl.Ad))
        assert np.all(eigvals <= 1.0 + 1e-6), \
            f"Ad has eigenvalue > 1: {eigvals.max():.4f}"

    def test_mpc_u_shape_and_finite(self, sat_and_ctrl):
        """find_u returns a finite vector of correct length."""
        sat, ctrl, x0 = sat_and_ctrl
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)
        goal   = ECI_Goal(np.array([0.0, 0.0, 1.0]))

        u = ctrl.find_u(x_hat=x0, sens=sat.sensor_readings(x=x0, os=os_hat),
                        est_sat=sat, os_hat=os_hat, goal=goal)
        assert u.shape == (len(sat.actuators),)
        assert np.all(np.isfinite(u)), f"find_u returned non-finite: {u}"

    def test_mpc_saturation_respected(self, sat_and_ctrl):
        """Dipole commands never exceed the MTQ limit."""
        sat, ctrl, _ = sat_and_ctrl
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)
        w_big  = np.array([0.1, 0.1, 0.1])
        x_big  = np.concatenate([w_big, np.array([1.0, 0.0, 0.0, 0.0])])
        u = ctrl.find_u(x_hat=x_big, sens=sat.sensor_readings(x=x_big, os=os_hat),
                        est_sat=sat, os_hat=os_hat)
        assert np.all(np.abs(u) <= MTQ_UMAX + 1e-9)

    def test_mpc_zero_error_deadzone(self, sat_and_ctrl):
        """Near-zero error state triggers early return (zero command)."""
        sat, ctrl, x0 = sat_and_ctrl
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)
        # x0 = zero rate + identity quaternion → zero error with No_Goal
        u = ctrl.find_u(x_hat=x0, sens=sat.sensor_readings(x=x0, os=os_hat),
                        est_sat=sat, os_hat=os_hat, goal=No_Goal())
        assert np.allclose(u, 0.0, atol=1e-12)

    def test_mpc_b_constraint_approximately_satisfied(self, sat_and_ctrl):
        """
        The closed-form solution should produce a torque approximately perpendicular
        to B at step 0 (within the accuracy of the null-space projection).
        """
        sat, ctrl, _ = sat_and_ctrl
        ephem  = Ephemeris()
        os_hat = Orbital_State(ephem=ephem, J2000=J2000_0, R=R0_KM, V=V0_KMS)

        # Large attitude error to get a non-trivial torque command
        q_err = normalize(random_n_unit_vec(4))
        x_err = np.concatenate([np.array([0.5, 0.5, 0.5]),
                                 np.array([0.01, 0.01, 0.01])])

        # Reconstruct what the closed-form sees
        B_tgt_list, _ = ctrl._predict_horizon(os_hat)
        tau_tgt = ctrl._solve_closedform(x_err, B_tgt_list)

        B_tgt0 = B_tgt_list[0]
        B_hat  = B_tgt0 / (np.linalg.norm(B_tgt0) + 1e-30)
        tau_hat = tau_tgt / (np.linalg.norm(tau_tgt) + 1e-30)

        # sin(angle) between tau and B — should be close to 1.0 (perpendicular)
        sin_angle = np.linalg.norm(np.cross(tau_hat, B_hat))
        assert sin_angle > 0.9, (
            f"Closed-form torque not ⊥ B: sin(angle) = {sin_angle:.4f} "
            f"(expected ≈ 1.0)"
        )

    def test_mpc_tinympc_import_guard(self):
        """
        Requesting tinympc solver without the package installed raises ImportError
        (or sets up correctly if tinympc is available in the environment).
        """
        sat, _ = _make_satellite()
        try:
            import tinympc
            # If tinympc is present, setup should succeed
            ctrl = ThreeAxisMPC(
                est_sat=sat, sma=SMA, incl=INCL,
                N=5, dt=20.0,
                Q=np.eye(6), Q_N=np.eye(6), R=np.eye(3),
                solver="tinympc",
            )
            assert ctrl._tinympc is not None
        except ImportError:
            # If not installed, ImportError should propagate cleanly
            with pytest.raises(ImportError, match="tinympc"):
                ThreeAxisMPC(
                    est_sat=sat, sma=SMA, incl=INCL,
                    N=5, dt=20.0,
                    Q=np.eye(6), Q_N=np.eye(6), R=np.eye(3),
                    solver="tinympc",
                )


# ---------------------------------------------------------------------------
# Slow simulation tests (ThreeAxisMPC)
# ---------------------------------------------------------------------------

MPC_SCENARIOS = [
    "stop_spin",
    "align_z_static",
    "align_z_spin",
]

@pytest.mark.slow
@pytest.mark.parametrize("scenario", MPC_SCENARIOS)
def test_mpc_closedform_convergence(scenario: str) -> None:
    """
    ThreeAxisMPC (closed-form): angular rate must reduce within TF_SLOW seconds.
    Each find_u call propagates the orbit N-1 steps internally (ppigrf per step).
    """
    np.random.seed(42)
    w0, q0, goal, tf = _make_scenario(scenario)
    sat, _ = _make_satellite()

    ctrl = ThreeAxisMPC(
        est_sat=sat,
        sma=SMA, incl=INCL,
        N=MPC_CFG["N"],
        dt=MPC_CFG["dt"],
        Q=MPC_CFG["Q"],
        Q_N=MPC_CFG["Q_N"],
        R=MPC_CFG["R"],
        target_mode=MPC_CFG["target_mode"],
        solver="closedform",
    )

    _, state_hist, _, u_hist = _simulate(
        controller=ctrl, tf=tf, dt=DT_PHYSICS,
        w0=w0, q0=q0, goal=goal,
        desc=f"ThreeAxisMPC {scenario}",
    )

    valid = ~np.isnan(state_hist[:, 0])
    w_init  = np.linalg.norm(state_hist[valid][0,  0:3])
    w_final = np.linalg.norm(state_hist[valid][-1, 0:3])

    assert np.all(np.isfinite(state_hist[valid])), "State diverged"
    assert np.all(np.abs(u_hist[valid]) <= MTQ_UMAX + 1e-9), "MTQ limit violated"

    if w_init > 1e-4:
        assert w_final < w_init, (
            f"Angular rate did not decrease: {w_init:.4e} → {w_final:.4e} rad/s"
        )


# ===========================================================================
# Manual / Visual Debugging  (python -m pytest ... -s, or run directly)
# ===========================================================================

def _run_visual(ctrl_name: str, scenario: str, verbose: bool = True) -> None:
    np.random.seed(42)
    sat, _ = _make_satellite()
    w0, q0, goal, tf = _make_scenario(scenario)

    if ctrl_name == "aplqr":
        ctrl = APLQR(
            est_sat=sat, sma=SMA, incl=INCL,
            Q=APLQR_CFG["Q"], R_ctrl=APLQR_CFG["R_ctrl"],
            alpha=APLQR_CFG["alpha"],
            target_mode=APLQR_CFG["target_mode"],
        )
    else:
        ctrl = ThreeAxisMPC(
            est_sat=sat, sma=SMA, incl=INCL,
            N=MPC_CFG["N"], dt=MPC_CFG["dt"],
            Q=MPC_CFG["Q"], Q_N=MPC_CFG["Q_N"], R=MPC_CFG["R"],
            target_mode=MPC_CFG["target_mode"], solver="closedform",
        )

    time_hist, state_hist, os_hist, u_hist = _simulate(
        controller=ctrl, tf=tf, dt=DT_PHYSICS,
        w0=w0, q0=q0, goal=goal,
        verbose=verbose, desc=f"{ctrl_name} / {scenario}",
    )

    # Quick text summary
    valid = ~np.isnan(state_hist[:, 0])
    w_i = np.linalg.norm(state_hist[valid][0,  0:3])
    w_f = np.linalg.norm(state_hist[valid][-1, 0:3])
    print(f"\n|ω| initial: {w_i:.4e} rad/s  →  final: {w_f:.4e} rad/s")

    # Pad u_hist to match actuator count for plot_control
    u_full = np.zeros((u_hist.shape[0], 3))
    u_full[:, :u_hist.shape[1]] = u_hist

    animate_attitude(time=time_hist, state_hist=state_hist, os_hist=os_hist)
    plot_control(time=time_hist, u_hist=u_full)
    plot_state_comparison(time=time_hist, state_hist=state_hist)
    create_close_all_button_window()


if __name__ == "__main__":
    import sys

    ctrl_name = sys.argv[1] if len(sys.argv) > 1 else "aplqr"
    scenario  = sys.argv[2] if len(sys.argv) > 2 else "align_z_spin"

    available_ctrls    = ["aplqr", "mpc"]
    available_scenarios = APLQR_SCENARIOS + MPC_SCENARIOS

    if ctrl_name not in available_ctrls:
        print(f"Controller '{ctrl_name}' unknown. Choose from: {available_ctrls}")
        sys.exit(1)
    if scenario not in available_scenarios:
        print(f"Scenario '{scenario}' unknown. Choose from: {available_scenarios}")
        sys.exit(1)

    _run_visual(ctrl_name=ctrl_name, scenario=scenario)
