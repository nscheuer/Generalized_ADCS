"""P2.6 (REDO) -- Cold-gas leak with mid-mission onset, 3MTQ+3RW.

Replaces the +30deg impulsive-kick demo. Shows the loop absorbing a *routine
persistent fault* through the normal replan schedule, on a well-actuated config.
Single-trajectory (not MC).

Architecture (per the v6 discussion):
  * Tracker: plain Plan_and_Track_LQR (does NOT need the disturbance).
  * Estimator: SRUAKF with an estimable General_Disturbance on est_sat -- the
    augmented-state filter estimates the constant leak torque (initial
    disturbance state = 0).
  * Planner: the estimated constant disturbance is fed into the planner via
    plan_for_gendist / gendist_torq on each replan, so the replanned trajectory
    feed-forwards the leak.

Scenario:
  * Config 3MTQ+3RW, anti-ram pointing, tf=2000 s, dt=1 s.
  * Replan schedule: initial plan t=0, replans at 500 / 1000 / 1500 s.
  * Cold-gas leak onset t=600 s: tau_leak = M * normalize([1,1,0.5]) N.m,
    persistent body-fixed. Run BOTH magnitudes; present both, don't pick.
       M_small = 5e-5 N.m,  M_med = 2e-4 N.m

Per magnitude, 4 stacked panels: (a) pointing error (markers at replans + leak),
(b) estimator disturbance state vs true (3 body comps), (c) wheel momentum,
(d) wheel commands. Outputs: P2.6_replan_{small,med}_<ts>.json,
fig_replan_{small,med}.{png,pdf}.
"""

import os
import sys
import json
import time
import datetime as _dt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ADCS as ADCS
from ADCS.helpers.math_helpers import normalize
from ADCS.helpers.plot.control.targetplot import _angle_deg, _boresight_eci
from ADCS.satellite_hardware.disturbances import Prop_Disturbance, General_Disturbance
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors.star_tracker_quaternion import StarTrackerQuaternion
from ADCS.satellite_hardware.errors.noise import Noise
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.orbits.orbit import Orbit
from ADCS.controller.saltro.SALTRO_planner_settings import PlannerSettings as SALTRO_PlannerSettings
from ADCS.controller.saltro.SALTRO_pass_settings import PassConfig, CostConfig
from papers.Planner.paper import _paper2_sim as P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
CONFIG = "3+3"
TF = 2000.0
DT = float(os.environ.get('P26_DT', '1.0'))
REPLAN_TIMES = [0.0, 500.0, 1000.0, 1500.0]
PLAN_HORIZON = 600.0          # receding-horizon plan length per replan [s]
LEAK_ONSET = 600.0
LEAK_AXIS = normalize(np.array([1.0, 1.0, 0.5]))
SEC2CENT = TimeConstants.sec2cent
MAGS = {"small": 5e-5, "med": 2e-4}
# Capability-boundary sweep (P26_SWEEP=1): find the largest leak OldPlanner holds
# cleanly. Labels are leak magnitude in micro-N.m (5e-5 N.m = "m050").
if os.environ.get('P26_SWEEP'):
    MAGS = {f"m{int(round(v * 1e6)):03d}": v
            for v in [5e-5, 8e-5, 1.1e-4, 1.4e-4, 1.7e-4, 2.0e-4]}

# Scenario epoch offset: the default epoch starts in a ~760 s eclipse (sun
# sensors dead -> mag-only attitude drift). Start in sunlight so the star
# tracker + sun sensors give good attitude; the whole TF window stays sunlit.
SUNLIT_OFFSET_S = 800.0

# --- Serious ESPA-class smallsat (mission-grade ADCS) ---------------------
# A 3U with mN-scale wheels saturates against any real leak in seconds; a
# serious ~50 kg vehicle keeps the wheels in-envelope over the scenario and
# carries mission-grade sensing. Public-part-class numbers (BCT/Sinclair RW,
# BCT NST star tracker, LN-200-class FOG gyro, NewSpace MTQ).
MASS = 50.0
J_ESPA = np.diag([1.0, 1.2, 0.7])      # kg m^2
RW_HMAX = float(os.environ.get('P26_HMAX', '0.5'))     # N m s (RWp500 / RW-0.4 class)
RW_TAU = 0.025                         # N m
RW_JROTOR = 1e-3                       # kg m^2
MTQ_DIPOLE = float(os.environ.get('P26_MTQ', '10.0'))  # A m^2 (torque-rod class)
GYRO_STD = 2e-5                        # rad/s  (LN-200-class FOG)
ST_ARCSEC = 6.0                        # star-tracker per-axis noise (BCT NST)

# Estimator tuning for the augmented leak state: drop the (degenerate) RW
# actuator-bias states and keep the disturbance state "live" with a loose
# process noise so its covariance does not collapse before/after onset
# (3e-5 keeps the square-root filter numerically stable; 1e-4 caused
# occasional covariance reconditioning).
P_DIST = 5e-4
Q_DIST = float(os.environ.get('P26_QDIST', '3e-5'))

# Planner backend. Default 'saltro'. HISTORY (read before changing): OldPlanner
# ('old') *appeared* to hold the leak to ~0.3 deg, but that was an ARTIFACT -- its
# disturbance feedforward (plan_for_gendist/prop) was a silent no-op, so it planned
# the UNDISTURBED problem and the leak was handled entirely by the proportional
# TVLQR (a constant offset, no feedforward). The real fix
# (PlanAndTrackBase._refresh_planner_if_dirty: rebuild the C++ Planner when the
# disturbance changes, since it copies the satellite at construction) makes the
# disturbance actually reach the OldPlanner optimizer -- which then DIVERGES on the
# disturbed problem (a 174 deg warm-start spike). So 'old' can no longer reject the
# leak honestly. 'saltro' is the working path: its optimizer applies the
# disturbance (prop feedforward), the PD warm start avoids the spike, and with
# rw_AM_weight=1e1 (light momentum penalty -> wheels HOLD the leak instead of
# dumping it in a body-torquing lump) it acquires anti-ram to ~0.3 deg, takes a
# clear hit when the leak appears, then recovers on the replan that feeds it
# forward. Reproduce the demo: P26_PLANNER=saltro P26_GOAL=antivel (defaults).
PLANNER = os.environ.get('P26_PLANNER', 'saltro').lower()
# Planner knot spacing. SALTRO tracks with its OWN per-knot iLQR feedback gains, so
# spacing the knots at the sim step (1 s) makes those gains dense -- closing the
# "loose tracking ~5 deg" gap WITHOUT a separate dense-TVLQR module. OldPlanner has
# its own dense TVLQR, so it stays coarse (10 s). (Only used on the SALTRO path.)
PLAN_DT = float(os.environ.get('P26_PLAN_DT', '1.0' if PLANNER == 'saltro' else '10.0'))  # [s]
# Pointing goal. 'antivel' = anti-ram (the operational P2.6 goal, but the target
# rotates with the orbit, so x0=identity is ~60deg off and the run opens with an
# acquisition slew). 'identity' = hold the identity inertial attitude -- x0 IS
# identity, so it starts on-target and isolates the disturbance-rejection story
# cleanly: converge -> spike at leak onset -> constant LQR offset (no integral)
# -> shrink on the replan that feeds the leak forward.
GOAL_MODE = os.environ.get('P26_GOAL', 'antivel').lower()


class FastStarTracker(StarTrackerQuaternion):
    """Always-available quaternion attitude sensor. Skips the star-catalog /
    moon-sun ephemeris visibility check (we run permissively), which the UKF
    otherwise pays per sigma point per step (~1/3 of the update cost was JPL
    ephemeris lookups). Measurement is identical: attitude quaternion + noise."""

    def clean_reading(self, x, os):
        q = np.asarray(x[3:7], float).copy()
        return -q if q[0] < 0 else q


def bore_unit(sat):
    try:
        b = sat.get_boresight(None)
    except Exception:
        b = sat.boresight
        b = b.get("default", next(iter(b.values()))) if isinstance(b, dict) else b
    return normalize(np.asarray(b, float).reshape(3))


def make_serious_sat(estimated=False):
    """ESPA-class smallsat built fresh from the 3+3 factory: mission-grade
    wheels/MTQ/gyro, a quaternion star tracker, and ESPA inertia/mass."""
    base = P.make_sat(CONFIG, estimated=estimated)
    for a in base.actuators:
        cn = a.__class__.__name__
        if cn == "RW":
            a.h_max = RW_HMAX; a.u_max = RW_TAU
            if hasattr(a, "max_torque"):
                a.max_torque = RW_TAU
            a.J = RW_JROTOR
        elif cn == "MTQ":
            a.u_max = MTQ_DIPOLE
    sensors = list(base.sensors)
    for s in sensors:
        if s.__class__.__name__ == "Gyro" and getattr(s, "noise", None) is not None:
            s.noise.std_noise = np.array([GYRO_STD])
    arc = ST_ARCSEC * np.pi / (180.0 * 3600.0)
    # Always-available quaternion star tracker (no sun-exclusion/min-stars
    # gating) so sigma-point NaNs cannot poison the UKF covariance.
    st = FastStarTracker(boresight=np.array([0.0, 0.0, 1.0]),
                         fov=np.deg2rad(179.0), sun_exclusion=0.0, min_stars=0,
                         noise=Noise(noise=np.zeros(4), std_noise=np.array([arc] * 4)))
    sensors = sensors + [st]
    cls = EstimatedSatellite if estimated else Satellite
    return cls(mass=MASS, COM=base.COM, J_0=J_ESPA.copy(),
               disturbances=list(base.disturbances), sensors=sensors,
               actuators=base.actuators, boresight=base.boresight)


def make_estimator(n_rw):
    """SRUAKF with an estimable General_Disturbance (3-vector leak torque).
    RW actuator-bias states are dropped (degenerate with the disturbance)."""
    base = make_serious_sat(estimated=True)
    for a in base.actuators:                      # drop degenerate RW act-bias
        if a.__class__.__name__ == "RW":
            a.estimate_bias = False
    gd = General_Disturbance(estimated_vector_length=3, estimate_dist=True,
                             parameter_std_rate=Q_DIST)
    est_sat = EstimatedSatellite(
        mass=base.mass, COM=base.COM, J_0=base.J_0,
        disturbances=list(base.disturbances) + [gd],
        sensors=base.sensors, actuators=base.actuators, boresight=base.boresight)
    nb_a, nb_s, nd = est_sat.act_bias_len, est_sat.att_sens_bias_len, est_sat.dist_param_len
    N = est_sat.state_len + nb_a + nb_s + nd
    x0 = np.zeros(N); x0[3] = 1.0
    P_hat = block_diag(np.eye(3) * 1e-4 ** 2, np.eye(3) * 1e-2,
                       np.eye(n_rw) * 1e-3 ** 2, np.eye(nb_a) * 1e-2 ** 2,
                       np.eye(nb_s) * 1e-2 ** 2, np.eye(nd) * P_DIST ** 2)
    Q_hat = block_diag(np.eye(3) * 1e-10 ** 2, np.eye(3) * 1e-10,
                       np.eye(n_rw) * 1e-10 ** 2, np.eye(nb_a) * 1e-11 ** 2,
                       np.eye(nb_s) * 1e-11 ** 2, np.eye(nd) * Q_DIST ** 2)
    os0 = P.default_os0()
    est = ADCS.SRUAKF(est_sat=est_sat, J2000=os0.J2000, x_hat=x0, P_hat=P_hat,
                      Q_hat=Q_hat, dt=DT, cross_term=True, quat_as_vec=False)
    return est, N, nd


def circular_os0():
    """Circular orbit at the default_os0 radius.

    SALTRO's trajOpt validates the propagated orbit against a 6000-8000 km
    altitude window; ``_paper2_sim.default_os0`` (R≈7000 km, V=[8,0,0]) is
    eccentric to a ~9000 km apogee, so later receding-horizon replans climb out
    of range and trajOpt raises "failed to generate orbit". A circular orbit at
    the same radius stays in-window. (OldPlanner had no such validator.)
    """
    from ADCS.orbits.universal_constants import EarthConstants
    base = P.default_os0()
    R = np.asarray(base.R, float).reshape(3)
    vc = np.sqrt(EarthConstants.mu_e / np.linalg.norm(R))
    Vhat = np.asarray(base.V, float).reshape(3)
    Vhat = Vhat / np.linalg.norm(Vhat)
    return ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=base.J2000,
                              R=R, V=vc * Vhat)


def make_saltro_settings(ctrl_sat):
    """SALTRO planner settings for the P2.6 tracking + momentum-dump task.

    Uses the multi-RW-safe cost pairing now baked into CostConfig defaults
    (ang_cost_func_type=4, cost_hess_gauss_newton=True) so the 3MTQ+3RW backward
    pass stays PSD.  rw_AM_weight penalizes stored wheel momentum so the planner
    offloads the leak to the magnetorquers instead of saturating the wheels.
    """
    ps = SALTRO_PlannerSettings(est_sat=ctrl_sat)
    # PD-to-goal warm start (3), NOT B-dot (2). B-dot detumbles toward 180deg and
    # loops to the antipode, seeding a 360deg-slew spike the solver then has to
    # unwind; PD aims straight at the anti-velocity goal (and pre-cancels the prop
    # leak as feedforward) -> no loop, clean convergence.
    ps.init_traj.initcontroller = 3
    ps.init_traj.pd_gain_scale = 1.0
    p = ps.passes[0]
    p.dt = PLAN_DT
    p.cost = CostConfig(
        angle=1e3, angle_N=1e5,
        ang_vel=1e3, ang_vel_N=1e5,
        # Tight attitude cost (type 3 = 0.5*acos^2, quadratic in angle -> sub-degree
        # pointing) instead of the loose quartic default (type 4, ~10deg floor). The
        # c=1 singularity is Taylor-protected and cost_hess_gauss_newton keeps the
        # backward pass PSD (both live in the PKMN_antispike submodule C++).
        ang_cost_func_type=3,
        # ang_vel_err_dir is INTENTIONALLY 0: the Lyapunov cross-term has an
        # incomplete backward-pass Hessian (known gap), and a nonzero value makes
        # the 3MTQ+3RW solve return ok=False. fig_spin zeroes it for the same
        # reason. Rate damping is covered by ang_vel above.
        control_mult=1.0,
        mtq_control_weight=1.0, rw_control_weight=1.0,
        # Wheel-momentum penalty. NOTE: too stiff (1e3) is actively harmful here --
        # once a leak appears, the wheels absorb it (their momentum ramps), and a
        # stiff AM weight makes the planner DUMP that momentum back to the MTQ on
        # the very replan that should be holding the goal. That dump torques the
        # body -> a ~4.5deg transient spike that the tracker faithfully follows.
        # A light penalty (1e1) lets the wheels HOLD the leak: no spike, and the
        # post-leak hold settles to ~0.1deg (vs ~1.7deg at 1e3). Env-tunable.
        rw_AM_weight=float(os.environ.get('P26_SAL_AM_WEIGHT', '1e1')),
        # ang_cost_func_type=4 + cost_hess_gauss_newton=True come from defaults
    )
    return ps


class MomentumRateObserver:
    """Constant-leak observer from the rate of TOTAL angular momentum.

    Euler in the body frame: ``d/dt(Jw + h) + w x (Jw + h) = tau_ext``, where the
    wheels are internal (their momentum is in ``h``), so the only external
    actuator torque is the magnetorquers. Hence

        tau_leak = d/dt(Jw + h) + w x (Jw + h) - tau_mtq.

    The wheel term ``h`` integrates the disturbance into a large, clean signal
    (~0.06 N.m.s ramp), whereas in the gyro/omega channel the leak (~5e-5 rad/s)
    sits below the residual gyro-bias error (~1e-4 rad/s) even after the star
    tracker quashes the bias -- so the augmented UKF cannot resolve the leak
    direction but this observer can (~6-10% vs UKF ~50-150%). EMA-smoothed
    because the leak is constant. ``tau_mtq`` comes from the actuators' own
    ``torque(u, x, os)`` methods (exact sign/frame, no re-derivation)."""

    def __init__(self, J, ctrl_sat, n_rw, dt, tau=90.0):
        self.J = np.asarray(J, float)
        self.sat = ctrl_sat
        self.n_rw = n_rw
        self.dt = float(dt)
        self.mtq_idx = [i for i, a in enumerate(ctrl_sat.actuators)
                        if a.__class__.__name__ == "MTQ"]
        self.rw_ax = [np.asarray(a.axis, float).ravel() for a in ctrl_sat.actuators
                      if a.__class__.__name__ == "RW"]
        self.alpha = float(np.exp(-self.dt / tau))
        self.L_prev = None
        self.est = np.zeros(3)
        self.inited = False

    def _mtq_torque(self, u, x_phys, os):
        t = np.zeros(3)
        for i in self.mtq_idx:
            t = t + np.asarray(self.sat.actuators[i].torque(u[i], x_phys, os)).ravel()
        return t

    def update(self, x_phys, u, os):
        w = np.asarray(x_phys[:3], float)
        h = np.asarray(x_phys[7:7 + self.n_rw], float)
        hvec = sum(h[j] * self.rw_ax[j] for j in range(self.n_rw))
        L = self.J @ w + hvec
        if self.L_prev is not None:
            tau_obs = ((L - self.L_prev) / self.dt + np.cross(w, L)
                       - self._mtq_torque(u, x_phys, os))
            self.est = tau_obs if not self.inited else (
                self.alpha * self.est + (1.0 - self.alpha) * tau_obs)
            self.inited = True
        self.L_prev = L
        return self.est.copy()


# Disturbance feedforward source for the planner: 'obs' (momentum-rate observer,
# ~6-10% on the constant leak) or 'ukf' (augmented SRUAKF, ~50-150% - direction
# unobservable in the gyro channel). The UKF still runs either way (attitude).
DIST_SRC = os.environ.get('P26_DIST_SRC', 'obs').lower()


def run_leak(M):
    # Optional fixed seed: the gyro/sensor biases are drawn per call, so seeding
    # makes the bias/noise realization identical across runs -> clean A/B of
    # controller knobs (otherwise ~0.1deg run-to-run pointing noise swamps the
    # comparison). P26_ANGLE_SCALE scales the TVLQR attitude weight (stiffer
    # feedback -> smaller constant-leak offset, down to the attitude-est floor).
    _seed = os.environ.get('P26_SEED')
    if _seed is not None:
        np.random.seed(int(_seed))
    true_sat = make_serious_sat(estimated=False)
    n_rw = len([a for a in true_sat.actuators if a.__class__.__name__ == "RW"])
    bu = bore_unit(true_sat)
    leak = Prop_Disturbance(torque_nominal=np.zeros(3))
    true_sat.disturbances = list(true_sat.disturbances) + [leak]

    ctrl_sat = make_serious_sat(estimated=False)       # planner/tracker model
    if PLANNER == 'saltro':
        ctrl = ADCS.controller.SALTRO(
            est_sat=ctrl_sat, planner_settings=make_saltro_settings(ctrl_sat))
    else:
        ps_old = P.make_planner_settings(ctrl_sat)
        # Planner integration step. The shared _paper2_sim default (dt_tp=50s) is
        # too coarse for the stiff high-RW-momentum dynamics (w x (Jw+h)): the
        # warm-start RK4 rollout DIVERGES to Inf, producing NaN cost/Jacobians and
        # an infinite, regularization-proof backward-pass reset loop (the "grind"
        # on bigger leaks). dt_tp=10 keeps the rollout finite -> converges in ~2s.
        ps_old.dt_tp = float(os.environ.get('P26_DTTP', '10'))
        # Gauss-Newton (default): the shared _paper2_sim config forces full DDP
        # (use_dynamics_hess=1 + use_full_cost_hessian=True), which grinds at 99%
        # CPU for minutes on the med-leak high-momentum replans. GN is robust and
        # fast (same lesson as the SALTRO multi-RW case). Set P26_OLD_GN=0 to
        # restore DDP.
        if os.environ.get('P26_OLD_GN', '1') == '1':
            ps_old.pass1.regularization.use_dynamics_hess = 0
            ps_old.cost_main.use_full_cost_hessian = False
        # Stiffen the attitude weight to shrink the constant-leak offset (no integral
        # term in the LQR -> steady error ~ tau_leak / K_angle). Scale EVERY cost
        # stage, not just the TVLQR: cost_main/second set how tightly the *nominal*
        # planned trajectory holds the goal under the fed-forward leak, while
        # cost_tvlqr sets the tracking stiffness around it. These may alias the same
        # CostWeights object, so dedup by id to avoid double-scaling.
        ascale = float(os.environ.get('P26_ANGLE_SCALE', '1'))
        if ascale != 1.0:
            _seen_cw = set()
            for _cw in (ps_old.cost_main, ps_old.cost_second, ps_old.cost_tvlqr):
                if id(_cw) in _seen_cw:
                    continue
                _seen_cw.add(id(_cw))
                _cw.angle *= ascale
                _cw.angle_N *= ascale
        # RW angular-momentum management. Keep the AM cost (penalizes stored wheel
        # momentum -> planner dumps via MTQ) but DROP stiction (its smoothstep''
        # Hessian can be indefinite and it isn't needed here). Lower the AM
        # deadband (RWh_ok_mult) so dumping engages well before the wheels fill,
        # instead of the default 0.5*h_max where momentum is already high and the
        # gyroscopic coupling has stiffened.
        ps_old.rw_AM_weight = float(os.environ.get('P26_AM_WEIGHT', '1e4'))
        ps_old.rw_stic_weight = 0.0
        ps_old.RWh_ok_mult = float(os.environ.get('P26_AM_OK_MULT', '0.5'))
        ps_old.RWh_max_mult = float(os.environ.get('P26_AM_MAX_MULT', '0.8'))
        # Make MTQ torque cheaper so the planner spends it on dumping rather than
        # hoarding it (default mtq_control_weight=1e3 is stingy).
        ps_old.mtq_control_weight = float(os.environ.get('P26_MTQ_WEIGHT', '1e3'))
        # Iteration cap (MPC-standard): the high-momentum med-leak replans do not
        # fully converge (grind for minutes at the 7000-iter budget, GN or DDP).
        # Capping max_total_iter bounds each replan to seconds and returns the
        # best trajectory found so far. Set P26_MAXITER (e.g. 300) to enable.
        cap = os.environ.get('P26_MAXITER')
        if cap:
            c = int(cap)
            for pc in (ps_old.pass1.convergence, ps_old.pass2.convergence):
                pc.max_total_iter = min(pc.max_total_iter, c)
                pc.max_inner_iter = min(pc.max_inner_iter, max(20, c // 5))
        ctrl = ADCS.controller.Plan_and_Track_LQR(
            est_sat=ctrl_sat, planner_settings=ps_old)
    estimator, Nfull, nd = make_estimator(n_rw)

    os0 = circular_os0() if PLANNER == 'saltro' else P.default_os0()
    orb = Orbit(os0=os0, end_time=os0.J2000 + (SUNLIT_OFFSET_S + TF + 5) * SEC2CENT,
                dt=DT, use_J2=True, fast=False, verbose=False)
    N = int(TF / DT)
    os_seq = [orb.get_os(J2000=os0.J2000 + (SUNLIT_OFFSET_S + k * DT) * SEC2CENT)
              for k in range(N + 1)]
    if GOAL_MODE == 'identity':
        goal = ADCS.goals.Fixed_Attitude_Goal(np.array([1.0, 0.0, 0.0, 0.0]))
    else:
        goal = ADCS.goals.AntiVelocity_Goal()
    gl = ADCS.GoalList(goal_timeline={0.0: goal}, time_units="seconds",
                       start_juliantime=os_seq[0].J2000)

    x = P.x0(n_rw)                       # true plant state
    u = np.zeros(true_sat.control_len)
    rec = {k: [] for k in ("t", "err", "dist_est", "dist_obs", "dist_true", "rw_h", "u")}
    replan_solve_ms = []
    next_replan = list(REPLAN_TIMES)
    leak_active = False
    dist_est = np.zeros(3)
    dist_obs = np.zeros(3)
    observer = MomentumRateObserver(J_ESPA, ctrl_sat, n_rw, DT)

    for k in range(N):
        os_k = os_seq[k]; ct = os_k.J2000; t_s = k * DT

        # leak onset (mutate the plant disturbance)
        if not leak_active and t_s >= LEAK_ONSET:
            leak.torque_nominal = M * LEAK_AXIS
            leak.current_torque = (M * LEAK_AXIS).copy()
            leak_active = True

        # measure + estimate (augmented SRUAKF -> disturbance state)
        y = true_sat.sensor_readings(x=x, os=os_k)
        x_hat = np.asarray(estimator.update(u=u, sensors=y, os=os_k), float)
        dist_est = x_hat[-nd:] if nd > 0 else np.zeros(3)
        x_hat_phys = x_hat[:7 + n_rw]

        # momentum-rate observer (the disturbance the wheels are absorbing); `u`
        # here is the control applied over the last step, which produced this h.
        dist_obs = observer.update(x_hat_phys, u, os_k)
        if DIST_SRC == 'true':            # oracle feedforward (debug): exact leak
            dist_fb = (M * LEAK_AXIS).copy() if leak_active else np.zeros(3)
        elif DIST_SRC == 'none':          # no disturbance feedforward (bare LQR)
            dist_fb = np.zeros(3)
        elif DIST_SRC == 'obs':
            dist_fb = dist_obs
        else:
            dist_fb = dist_est

        # scheduled replan: feed the estimated constant disturbance to the planner
        if next_replan and t_s >= next_replan[0] - 1e-9:
            if PLANNER == 'saltro':
                # Feed the estimated leak as a constant prop torque. SALTRO's
                # plan_for_gendist path is currently a no-op in C++ (the flag
                # reaches the binding but trajOpt ignores it -> file a GenADCS
                # issue), whereas plan_for_prop actually applies the torque in
                # the planning dynamics (proven in generate_fig_spin, PR #27).
                # The leak IS a Prop_Disturbance, so this is physically exact.
                ctrl.planner_settings.disturbances.plan_for_prop = 1
                ctrl.planner_settings.disturbances.prop_torque = dist_fb.copy()
            else:
                ctrl.planner_settings.plan_for_gendist = True
                ctrl.planner_settings.gendist_torq = dist_fb.copy()
            t0 = time.perf_counter()
            horizon = min(TF - t_s, PLAN_HORIZON)      # receding horizon (replan every 500 s)
            traj = ctrl.calculate_trajectory(ct, horizon, x_hat_phys.copy(), os_k, gl)
            replan_solve_ms.append((time.perf_counter() - t0) * 1e3)
            ctrl.set_active_trajectory(traj)
            next_replan.pop(0)

        ag = gl.get_active_goal(ct, time_units="centuries")
        if GOAL_MODE == 'identity':
            # Fixed-attitude hold: error is the attitude angle between the true
            # quaternion and the reference (2*acos|<q,q_ref>|), not a boresight angle.
            q_ref = normalize(np.asarray(ag.to_ref(os_k)[0], float).reshape(4))
            err = 2.0 * np.degrees(np.arccos(min(1.0, abs(float(normalize(x[3:7]) @ q_ref)))))
        else:
            target = ag.to_ref(os_k)[0][1:4]
            err = _angle_deg(_boresight_eci(x[3:7], bu), target)   # TRUE pointing error
        u = ctrl.find_u(x_hat=x_hat_phys, sens=y, est_sat=ctrl_sat,
                        os_hat=os_k, goal=ag)

        rec["t"].append(t_s); rec["err"].append(err)
        rec["dist_est"].append(dist_est.copy())
        rec["dist_obs"].append(dist_obs.copy())
        rec["dist_true"].append((M * LEAK_AXIS).copy() if leak_active else np.zeros(3))
        rec["rw_h"].append(x[7:7 + n_rw].copy())
        rec["u"].append(np.asarray(u, float).copy())

        os_next = os_seq[k + 1]
        out = solve_ivp(fun=true_sat.dynamics_for_solver, t_span=(0, DT), y0=x,
                        method="RK45", args=(u, os_k, os_next),
                        rtol=1e-7, atol=1e-7)
        x = out.y[:, -1]; x[3:7] = normalize(x[3:7])

    for kk in rec:
        rec[kk] = np.asarray(rec[kk])
    rec["replan_solve_ms"] = replan_solve_ms
    rec["n_rw"] = n_rw
    return rec


def settle_after(t, v, t0, thr):
    m = t >= t0
    tt, vv = t[m], v[m]
    above = vv > thr
    if not above.any():
        return 0.0
    if above[-1]:
        return float("nan")
    return float(tt[int(np.flatnonzero(above)[-1]) + 1] - t0)


def figure(rec, M, tag, path):
    t = rec["t"]
    fig, axs = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    axs[0].plot(t, rec["err"], color="C0", lw=1.3)
    axs[0].axhline(5.0, ls=":", c="k", lw=0.7)
    axs[0].set_yscale("log"); axs[0].set_ylabel("pointing err [deg]")
    axs[0].set_title(f"P2.6 cold-gas leak ({tag}): M={M:g} N.m, onset t={LEAK_ONSET:.0f}s, "
                     f"momentum-rate disturbance observer")
    has_obs = "dist_obs" in rec and np.asarray(rec["dist_obs"]).size
    for j, c in enumerate("xyz"):
        axs[1].plot(t, rec["dist_true"][:, j], color=f"C{j}", lw=1.2, ls="--",
                    label=f"true {c}")
        if has_obs:
            axs[1].plot(t, rec["dist_obs"][:, j], color=f"C{j}", lw=1.4,
                        label=f"obs {c}")
        axs[1].plot(t, rec["dist_est"][:, j], color=f"C{j}", lw=0.7, alpha=0.4,
                    label=f"ukf {c}")
    axs[1].set_ylabel("disturbance torque [N.m]"); axs[1].legend(fontsize=6, ncol=3)
    n_rw = rec["n_rw"]; u = rec["u"]; n_mtq = u.shape[1] - n_rw
    for j in range(n_rw):
        axs[2].plot(t, rec["rw_h"][:, j], lw=1.1, label=f"RW{j}")
    axs[2].axhline(RW_HMAX, ls=":", c="k", lw=0.7); axs[2].axhline(-RW_HMAX, ls=":", c="k", lw=0.7)
    axs[2].set_ylabel("wheel momentum [Nms]"); axs[2].legend(fontsize=7, ncol=3)
    # MTQ usage: dipole command vs saturation (shows whether dumping is authority-
    # limited (rails at +/-MTQ_DIPOLE) or the planner is leaving headroom).
    for j in range(n_mtq):
        axs[3].plot(t, u[:, j], lw=1.0, label=f"MTQ{j}")
    axs[3].axhline(MTQ_DIPOLE, ls=":", c="k", lw=0.7); axs[3].axhline(-MTQ_DIPOLE, ls=":", c="k", lw=0.7)
    axs[3].set_ylabel("MTQ cmd [A.m2]"); axs[3].legend(fontsize=7, ncol=3)
    for j in range(n_rw):
        axs[4].plot(t, u[:, n_mtq + j], lw=1.0, label=f"RW{j}")
    axs[4].set_ylabel("wheel cmd [Nm]"); axs[4].set_xlabel("time [s]"); axs[4].legend(fontsize=7, ncol=3)
    for ax in axs:
        for tr in REPLAN_TIMES[1:]:
            ax.axvline(tr, color="gray", ls="--", lw=0.7)
        ax.axvline(LEAK_ONSET, color="r", ls=":", lw=1.0)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf"); plt.close(fig)


def summarize(rec, M):
    t, err = rec["t"], rec["err"]
    post = err[t >= LEAK_ONSET]; pre = err[t < LEAK_ONSET]
    de = np.linalg.norm(rec["dist_est"] - rec["dist_true"], axis=1)
    thr = 0.2 * (M * np.linalg.norm(LEAK_AXIS))
    mtq = rec["u"][:, :rec["u"].shape[1] - rec["n_rw"]]
    leak_norm = M * np.linalg.norm(LEAK_AXIS)
    # disturbance-estimate relative error over the settled window (2nd half of
    # the post-onset segment, after the estimator has converged)
    post_m = t >= LEAK_ONSET
    tp = t[post_m]
    settled_m = post_m & (t >= (LEAK_ONSET + 0.5 * (tp[-1] - LEAK_ONSET))) if tp.size else post_m
    dist_err_pct_settled = float(100.0 * np.mean(de[settled_m]) / leak_norm) if settled_m.any() else float("nan")
    # momentum-rate observer error over the same settled window (the source
    # actually fed to the planner when P26_DIST_SRC=obs)
    obs_err_pct_settled = float("nan")
    if "dist_obs" in rec and np.asarray(rec["dist_obs"]).size:
        do = np.linalg.norm(rec["dist_obs"] - rec["dist_true"], axis=1)
        if settled_m.any():
            obs_err_pct_settled = float(100.0 * np.mean(do[settled_m]) / leak_norm)
    return {
        "M": M, "max_err_post_deg": float(np.max(post)),
        "max_err_pre_deg": float(np.max(pre)) if pre.size else float("nan"),
        "final_err_deg": float(err[-1]),
        "max_wheel_momentum": float(np.max(np.abs(rec["rw_h"]))),
        "wheel_saturation_frac": float(np.max(np.abs(rec["rw_h"])) / RW_HMAX),
        "estimator_settle_s_after_onset": settle_after(t, de, LEAK_ONSET, thr),
        "dist_err_pct_settled": dist_err_pct_settled,
        "dist_src": DIST_SRC,
        "obs_err_pct_settled": obs_err_pct_settled,
        "dist_obs_final": rec["dist_obs"][-1].tolist() if "dist_obs" in rec and np.asarray(rec["dist_obs"]).size else None,
        "dist_est_final": rec["dist_est"][-1].tolist(),
        "dist_true_final": rec["dist_true"][-1].tolist(),
        "mtq_duty_cycle": float(np.mean(np.any(np.abs(mtq) > 1e-9, axis=1))),
        "replan_solve_ms": rec["replan_solve_ms"],
    }


def main():
    ts = os.environ.get("P26_TS", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    only = os.environ.get("P26_ONLY")
    mags = {only: MAGS[only]} if only in MAGS else MAGS
    print(f"[P2.6] cold-gas leak {CONFIG} tf={TF} onset={LEAK_ONSET} mags={list(mags)} ts={ts}")
    out = {}
    for tag, M in mags.items():
        print(f"[P2.6] magnitude {tag}: M={M:g} ...", flush=True)
        rec = run_leak(M)
        figure(rec, M, tag, os.path.join(OUT, f"fig_replan_{tag}"))
        s = summarize(rec, M); out[tag] = s
        series = {k: rec[k].tolist() for k in ("t", "err")}
        for kk in ("dist_est", "dist_obs", "dist_true", "rw_h", "u"):
            series[kk] = rec[kk].tolist()
        json.dump({"task": "P2.6_coldgas", "tag": tag, "M": M, "timestamp": ts,
                   "config": CONFIG, "tf": TF, "leak_onset": LEAK_ONSET,
                   "leak_axis": LEAK_AXIS.tolist(), "replan_times": REPLAN_TIMES,
                   "summary": s, "series": series},
                  open(os.path.join(OUT, f"P2.6_replan_{tag}_{ts}.json"), "w"), indent=2)
        print(f"  {tag}: max_post {s['max_err_post_deg']:.3f}deg final {s['final_err_deg']:.3f} "
              f"hmax {s['max_wheel_momentum']:.4f} | dist_src={s['dist_src']} "
              f"obs_err {s['obs_err_pct_settled']:.0f}% ukf_err {s['dist_err_pct_settled']:.0f}% "
              f"obs_final {np.round(s['dist_obs_final'],6) if s['dist_obs_final'] else None} "
              f"true {np.round(s['dist_true_final'],6)}", flush=True)
    print("[P2.6] done")
    return out


if __name__ == "__main__":
    main()
