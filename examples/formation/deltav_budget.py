"""Delta-V budget sketch for formation maintenance across architectures.

Estimates per-satellite station-keeping delta-V from the secular drifts we've
characterized, under a set of TOGGLEABLE design policies. The point is to flag
design choices on/off and see what each costs -- not to be a high-fidelity
propagator.

WHAT IS GROUNDED (first-order, secular):
  * differential drag  -> in-plane (along-track) maintenance. Budget = the
    DIFFERENTIAL drag acceleration (vs a chosen reference) integrated over the
    mission. Choosing to fight only the differential (let the cluster decay
    together) vs. holding absolute altitude is a toggle.
  * differential nodal precession (delta-i_x driven) -> out-of-plane "SSO shear"
    maintenance. Budget = v * |d(delta-i)/dt| * T. This is the expensive one and
    is zero for same-inclination (node-based / dense) designs.
  * J2 delta-e rotation -> COMMON-MODE for equal a,i (rotates the whole pattern
    rigidly => free for formation keeping). Only costs delta-V if you insist on
    holding delta-e inertially fixed (rarely needed); exposed as a toggle.

WHAT IS STUBBED -- NEEDS LITERATURE REVIEW (do not trust until modeled):
  * drag shadowing: trailing sats in the wake of leaders see reduced drag. At
    ~650 km the flow is free-molecular (mean free path ~ km), so this is NOT the
    continuum-wake intuition; magnitude vs along-track spacing/attitude needs
    references. Hook: shadowing_factors() -- currently identity.
  * thruster plume impingement: firing toward a neighbour imparts force/torque/
    contamination, constraining thrust geometry/timing and possibly adding
    avoidance delta-V. Hook: plume_impingement_deltav() -- currently zero.

ASSUMPTIONS / LIMITATIONS: circular chief, constant density (solar activity
swings this by >10x -- pass your own), secular drifts only (deadbands change
maneuver cadence, not the secular delta-V total), impulsive-equivalent burns,
no coupling between sources. Good for relative comparison, not absolute truth.
"""

import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import numpy as np

from formation_lib import (MU, RE, J2, ALT_KM, INCLINATION, compute_elements,
                           center_index, relative_roe, sso_drift)

SEC_PER_YEAR = 365.25 * 86400.0


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
@dataclass
class SatSpec:
    name: str = ""
    mass_kg: float = 575.0     # Starlink V2 Mini dry
    area_m2: float = 4.0       # NOMINAL operating (sun-pointing feathered) cross-section [m^2];
                               # NOT the 105 m^2 broadside (that's the drag-modulation ceiling,
                               # see aero_authority). ** DISCUSS: right feathered area? **
    Cd: float = 2.4            # = Cn (standard 1/2 rho V^2 convention)

    @property
    def ballistic_coeff(self):           # kg/m^2 ; higher B => less drag
        return self.mass_kg / (self.Cd * self.area_m2)


@dataclass
class Environment:
    alt_km: float = ALT_KM
    inclination: float = INCLINATION
    density_kg_m3: float = 5e-14         # ~650 km, moderate solar activity (PASS YOUR OWN)

    @property
    def a_km(self):
        return RE + self.alt_km

    @property
    def v_ms(self):
        return np.sqrt(MU / self.a_km) * 1e3


@dataclass
class Policy:
    # in-plane / drag
    compensate_drag: bool = True         # null differential drag (else geometry drifts along-track)
    drag_reference: str = "centroid"     # "centroid" | "chief" | "none" : what we hold to
    maintain_absolute_altitude: bool = False  # reboost whole cluster vs let it decay together
    # out-of-plane / J2 / SSO
    compensate_sso_shear: bool = True    # null differential nodal precession (out-of-plane burns)
    hold_de_inertial: bool = False       # fight J2 delta-e rotation (usually unneeded: it's common-mode)
    # --- pending literature review: keep False until modeled ---
    drag_shadowing: bool = False         # wake shadowing in close formation -- MODEL TBD
    plume_impingement: bool = False      # thruster plume on neighbours -- MODEL TBD


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
def drag_acceleration(sat, env):
    """Free-stream drag deceleration [m/s^2]."""
    return 0.5 * env.density_kg_m3 * env.v_ms**2 / sat.ballistic_coeff


def de_rotation_rate(env):
    """J2 rotation rate of the relative eccentricity vector [rad/s] (eq. for omega-dot)."""
    a = env.a_km
    kappa = 0.75 * J2 * np.sqrt(MU) * RE**2 / a**3.5          # eta=1 (circular)
    return kappa * (5.0 * np.cos(env.inclination)**2 - 1.0)


def shadowing_factors(droe, sats, env, policy):
    """Per-sat drag multiplier from wake shadowing. PLACEHOLDER -> all ones.

    LIT REVIEW REQUIRED: free-molecular wake at ~650 km, dependence on
    along-track spacing / attitude / rarefaction. Do not enable until modeled.
    """
    if policy.drag_shadowing:
        raise NotImplementedError(
            "drag shadowing model pending literature review; keep policy.drag_shadowing=False")
    return np.ones(len(sats))


def plume_impingement_deltav(droe, sats, env, policy, years):
    """Extra delta-V from plume-impingement avoidance. PLACEHOLDER -> zeros.

    LIT REVIEW REQUIRED: plume models, standoff distance, attitude/timing
    constraints, contamination. Do not enable until modeled.
    """
    if policy.plume_impingement:
        raise NotImplementedError(
            "plume impingement model pending literature review; keep policy.plume_impingement=False")
    return np.zeros(len(sats))


def make_sats(n, b_spread=0.0, base=SatSpec()):
    """N satellites; b_spread fans the cross-sectional area +/- (=> differential drag)."""
    sats = []
    for i in range(n):
        f = 1.0 + b_spread * (2.0 * i / max(n - 1, 1) - 1.0)   # linear ramp across the set
        sats.append(SatSpec(name=f"sat{i}", mass_kg=base.mass_kg,
                            area_m2=base.area_m2 * f, Cd=base.Cd))
    return sats


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def maintenance_budget(R0, V0, meta, env, policy, years, sats=None, b_spread=0.0):
    """Per-satellite maintenance delta-V [m/s] broken down by source.

    Returns dict with per-source arrays (drag, absolute, j2, sso, total) plus
    formation summary (max/mean/sum of total). All m/s over `years`.
    """
    N = R0.shape[0]
    oes = compute_elements(R0, V0)
    chief = center_index(meta)
    droe = relative_roe(oes, chief)
    if sats is None:
        sats = make_sats(N, b_spread)
    T = years * SEC_PER_YEAR
    v = env.v_ms

    # --- in-plane: differential drag -------------------------------------- #
    shadow = shadowing_factors(droe, sats, env, policy)
    a_drag = np.array([drag_acceleration(s, env) for s in sats]) * shadow
    if policy.drag_reference == "centroid":
        ref = a_drag.mean()
    elif policy.drag_reference == "chief":
        ref = a_drag[chief]
    else:
        ref = 0.0
    dv_drag = np.abs(a_drag - ref) * T if policy.compensate_drag else np.zeros(N)
    dv_abs = a_drag * T if policy.maintain_absolute_altitude else np.zeros(N)

    # --- in-plane: J2 delta-e rotation (common-mode unless held inertial) -- #
    dv_j2 = np.zeros(N)
    if policy.hold_de_inertial:
        de_mag = np.hypot(droe[:, 2], droe[:, 3])
        dv_j2 = v * abs(de_rotation_rate(env)) * de_mag * T

    # --- out-of-plane: differential nodal precession (SSO shear) ----------- #
    dv_sso = np.zeros(N)
    if policy.compensate_sso_shear:
        _, rel = sso_drift(oes, chief)                  # relative nodal rate [rad/s]
        ddi_dt = np.abs(np.sin(env.inclination) * rel)  # d(delta-i_y)/dt
        dv_sso = v * ddi_dt * T

    dv_plume = plume_impingement_deltav(droe, sats, env, policy, years)
    total = dv_drag + dv_abs + dv_j2 + dv_sso + dv_plume
    return dict(drag=dv_drag, absolute=dv_abs, j2=dv_j2, sso=dv_sso, plume=dv_plume,
                total=total, sats=sats,
                total_max=float(total.max()), total_mean=float(total.mean()),
                total_sum=float(total.sum()))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _summary(label, b):
    print(f"  {label:<34} "
          f"drag {b['drag'].max():6.2f} | sso {b['sso'].max():7.2f} | "
          f"j2 {b['j2'].max():5.2f} | abs {b['absolute'].max():5.2f} | "
          f"TOTAL max {b['total_max']:7.2f}  mean {b['total_mean']:6.2f}  [m/s]")


def main():
    import suncatcher_hex_oes as hexd
    import suncatcher_ei_oes as einode
    import suncatcher_ei_incl_oes as eiincl

    env = Environment()
    years = 3.0
    print(f"Maintenance delta-V over {years:.0f} yr  (rho={env.density_kg_m3:.0e} kg/m^3, "
          f"v={env.v_ms/1e3:.2f} km/s);  per-satellite WORST (max) by source:\n")

    cases = [
        ("dense hex", hexd.build_cluster, 0.0),
        ("dense hex, B-spread 10%", hexd.build_cluster, 0.10),
        ("e/i node-based", einode.build_cluster, 0.10),
        ("e/i inclination-based", eiincl.build_cluster, 0.10),
    ]
    base_policy = Policy()
    for label, build, spread in cases:
        R0, V0, meta = build()
        b = maintenance_budget(R0, V0, meta, env, base_policy, years, b_spread=spread)
        _summary(label, b)

    print("\n  toggle effect on the inclination-based config:")
    R0, V0, meta = eiincl.build_cluster()
    _summary("  default (compensate SSO shear)",
             maintenance_budget(R0, V0, meta, env, Policy(), years, b_spread=0.10))
    _summary("  let it shear (no SSO comp)",
             maintenance_budget(R0, V0, meta, env, Policy(compensate_sso_shear=False), years, b_spread=0.10))
    _summary("  + reboost absolute altitude",
             maintenance_budget(R0, V0, meta, env, Policy(maintain_absolute_altitude=True), years, b_spread=0.10))


if __name__ == "__main__":
    main()
