"""SSC26 poster snippet verification.

Every code block printed on the poster is reproduced here VERBATIM between
the ``--- SNIPPET x ---`` markers and then executed. If this script exits 0,
the poster's code is known to run against the shipped API.

Run:  python papers/SSC26_poster/verify_snippets.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ADCS.pipeline import PipelineController
from ADCS.pipeline.control_law import ControlLaw, LawInterface, PD_Law
from ADCS.pipeline.data import AllocationConfig, CompensationConfig
from ADCS.CONOPS.goals import ECI_Goal, Nadir_Goal, No_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbit import Orbit
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.satellite.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import normalize


# ---------------------------------------------------------------------------
# Bus + orbit fixture (NOT printed on the poster -- harness only)
# ---------------------------------------------------------------------------

def make_bus():
    mtqs = [MTQ(axis=j, max_torque=1.0) for j in MathConstants.unitvecs]
    rws = [RW(axis=j, max_torque=0.007, J=0.001, h=0.005, h_max=0.0162)
           for j in MathConstants.unitvecs]
    mtms = [MTM(axis=j) for j in MathConstants.unitvecs]
    return Satellite(mass=4.0, J_0=np.diagflat([3.4, 2.9, 1.3]),
                     actuators=mtqs + rws, sensors=mtms,
                     boresight=np.array([0, 0, 1]))


def make_orbit():
    ephem = Ephemeris()
    os0 = Orbital_State(ephem=ephem, J2000=0.22 - TimeConstants.sec2cent,
                        R=7000 * np.array([0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2]),
                        V=np.array([8.0, 0.0, 0.0]))
    return Orbit(os0=os0, end_time=0.22 + 500.0 * TimeConstants.sec2cent,
                 dt=50.0, zonal_J=2, fast=False, verbose=False)


sat = make_bus()
orbit = make_orbit()
os_now = orbit.get_os(J2000=0.22 + 100.0 * TimeConstants.sec2cent)
x = np.concatenate([np.array([0.01, -0.005, 0.008]),
                    normalize(np.array([0.7, 0.3, -0.5, 0.2])),
                    0.005 * np.ones(3)])
sens = sat.sensor_readings(x=x, os=os_now)
goal = ECI_Goal(normalize(np.array([-0.139, -0.370, -0.919])))

RW_IDX = [i for i, a in enumerate(sat.actuators) if isinstance(a, RW)]
failures = []


def check(ok, label):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


# ===========================================================================
print("\n" + "=" * 62)
print("SNIPPET A - bring your own control law (Stage 2)")
print("=" * 62)
# --- SNIPPET A ---
class MyLaw(ControlLaw):
    interface = LawInterface()   # full attitude + rate
    kp, kd = 2e-5, 2e-2

    def compute(self, q_err, w_err=None, **kw):
        return -(self.kp * q_err + self.kd * w_err)
# --- END SNIPPET A ---

ctrl_a = PipelineController(sat, MyLaw())
u_a = ctrl_a.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
check(u_a.shape == (len(sat.actuators),), f"MyLaw drops in, u shape {u_a.shape}")
check(np.all(np.isfinite(u_a)), "MyLaw output is finite")


# ===========================================================================
print("\n" + "=" * 62)
print("SNIPPET B - a magnetic-only law gets a wheel (HEADLINE)")
print("=" * 62)
# --- SNIPPET B ---
law = PD_Law(kp=2e-5, kd=2e-2, eps=1.0)   # unmodified
mtq_only = PipelineController(sat, law)   # MTQ only
with_wheel = PipelineController(sat, law, # Stage 5 swap
    alloc_config=AllocationConfig(method='lp'))
# --- END SNIPPET B ---

u_mtq = mtq_only.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
u_rw = with_wheel.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
check(np.allclose(u_mtq[RW_IDX], 0.0),
      f"default: wheels idle (|u_rw| = {np.abs(u_mtq[RW_IDX]).max():.2e})")
check(np.abs(u_rw[RW_IDX]).max() > 1e-9,
      f"allocator='lp': wheels commanded (|u_rw| = {np.abs(u_rw[RW_IDX]).max():.2e})")

# The double-counting guard: a law that does its own gyro term makes the
# pipeline skip Stage 4's gyroscopic compensation.
class SelfGyroLaw(PD_Law):
    interface = LawInterface(includes_gyroscopic=True)

auto = CompensationConfig.from_law_interface(SelfGyroLaw(2e-5, 2e-2, 1.0).interface)
plain = CompensationConfig.from_law_interface(PD_Law(2e-5, 2e-2, 1.0).interface)
check(auto.enable_gyroscopic is False and plain.enable_gyroscopic is True,
      "includes_gyroscopic=True -> Stage 4 skips gyro (no double-count)")


# ===========================================================================
print("\n" + "=" * 62)
print("SNIPPET C - goal type as a design lever (Stage 1)")
print("=" * 62)
# --- SNIPPET C ---
ctrl = PipelineController(sat, law,
    alloc_config=AllocationConfig(method='lp'))
star = ECI_Goal(np.array([-0.14, -0.37, -0.92]))
u_eci = ctrl.find_u(x, sens, sat, os_now, star)
u_nadir = ctrl.find_u(x, sens, sat, os_now, Nadir_Goal())
# same law, same bus - Stage 1 converts each goal
# --- END SNIPPET C ---

check(not np.allclose(u_eci, u_nadir), "goal type changes the command")
u_none = ctrl.find_u(x, sens, sat, os_now, No_Goal())
check(np.all(np.isfinite(u_none)), "No_Goal also runs")


# ===========================================================================
print("\n" + "=" * 62)
print("SNIPPET D - allocator swap (Stage 5)")
print("=" * 62)
# --- SNIPPET D ---
AllocationConfig(method='lp')   # direction kept
AllocationConfig(method='qp')   # size kept, tilts
AllocationConfig(method='qpw')  # perp error x100
AllocationConfig(method='magnetic_cross')  # MTQ only
# --- END SNIPPET D ---

for m in ('lp', 'qp', 'qpw', 'pseudoinverse', 'magnetic_cross'):
    c = PipelineController(sat, law, alloc_config=AllocationConfig(method=m))
    u_m = c.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
    check(np.all(np.isfinite(u_m)) and np.abs(u_m).max() > 0.0,
          f"method='{m}' runs, |u|max={np.abs(u_m).max():.3e}")

# 'qpc' is deliberately NOT on the poster: its Lyapunov power gate returns an
# all-zero command on roughly half of sampled states, and the constraint it
# actually executes is not the one its docstring documents (a `if False:`
# block in ADCS/controller/mtq_w_rw_QPC.py disables the advertised branch).
c_qpc = PipelineController(sat, law, alloc_config=AllocationConfig(method='qpc'))
u_qpc = c_qpc.find_u(x_hat=x, sens=sens, est_sat=sat, os_hat=os_now, goal=goal)
check(np.all(np.isfinite(u_qpc)), f"method='qpc' runs (off-poster), |u|max={np.abs(u_qpc).max():.3e}")


# ===========================================================================
print("\n" + "=" * 62)
if failures:
    print(f"{len(failures)} SNIPPET CHECK(S) FAILED:")
    for f in failures:
        print(f"  - {f}")
    print("=" * 62)
    sys.exit(1)
print("ALL SNIPPET CHECKS PASSED - poster code runs verbatim")
print("=" * 62)
sys.exit(0)
