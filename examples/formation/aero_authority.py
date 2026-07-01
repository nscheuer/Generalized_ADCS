"""Differential-drag / differential-lift control authority for the suncatcher
formation, wired to the real density (DensityModel) and panel-aero (AeroModel)
models, and mapped to ROE correction rates.

Run directly: prints METHODOLOGY, CONFIGURATION, AUTHORITY, NEEDS, FEASIBILITY.
All inputs are tunable constants at the top of the file.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

import numpy as np

from formation_lib import MU, RE, ALT_KM, INCLINATION
from ADCS.orbits.density_model import DensityModel
from ADCS.satellite_hardware.aero.aero_force import AeroModel

# ===================== TUNABLE CONFIGURATION ============================== #
MASS_KG       = 800.0      # satellite mass
AREA_FRONT_M2 = 105.0      # broadside ("front-on") collector area  -> max drag / lift
AREA_END_M2   = 2.0        # edge ("end-on") area                   -> min drag floor
CN            = 2.6        # panel normal (pressure) coefficient
CT            = 0.3        # panel tangential (shear) coefficient   (lift <- Cn != Ct)
SOLAR_LEVEL   = 0.5        # 0=solar min, 1=solar max  (mid = 0.5)
ALT           = ALT_KM     # km
INC           = INCLINATION

# maintenance NEEDS to screen against (from the formation characterization)
NEED_INPLANE_BSPREAD = 0.10        # ballistic-coefficient spread for differential-drag need
DRAG_MOD_RATIO       = 5.0         # achievable high/low drag ratio actually commanded (cap on modulation)
SHEAR_NODAL_MDEG_DAY = 1.0         # worst inclination-separated sat relative nodal drift [mdeg/day]
SWEEP_NTHETA, SWEEP_NPHI = 49, 49  # attitude-sweep resolution
# ========================================================================= #


def orbital_speed(alt_km=ALT):
    return np.sqrt(MU / (RE + alt_km)) * 1e3          # m/s


def build_aero():
    """Flat-panel 'suncatcher': big +/-x collector faces, small edge faces."""
    faces = np.array([[1, 0, 0.], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.]])
    areas = np.array([AREA_FRONT_M2, AREA_FRONT_M2,
                      AREA_END_M2, AREA_END_M2, AREA_END_M2, AREA_END_M2])
    return AeroModel(faces, areas, Cn=CN, Ct=CT)


def drag_lift_authority(aero, rho, v_ms, mass_kg, n_theta=SWEEP_NTHETA, n_phi=SWEEP_NPHI):
    """Sweep wind direction in the body frame through the panel model; return the
    achievable specific-force envelope [m/s^2]: max/min drag (-> along-track
    modulation authority) and peak lift (-> cross-track authority), plus L/D."""
    best = dict(max_drag=0.0, min_drag=np.inf, max_lift=0.0, LD=0.0)
    for th in np.linspace(1e-3, np.pi - 1e-3, n_theta):
        for ph in np.linspace(0.0, 2 * np.pi, n_phi):
            vb = v_ms * np.array([np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)])
            F = aero.force_body(vb, rho)
            vh = vb / np.linalg.norm(vb)
            drag = -float(F @ vh) / mass_kg
            lift = float(np.linalg.norm(F - (F @ vh) * vh)) / mass_kg
            if drag > 1e-30:
                best["max_drag"] = max(best["max_drag"], drag)
                best["min_drag"] = min(best["min_drag"], drag)
                if lift > best["max_lift"]:
                    best["max_lift"], best["LD"] = lift, lift / drag
    best["min_drag"] = best["min_drag"] if np.isfinite(best["min_drag"]) else 0.0
    # along-track authority = commandable drag modulation, capped by DRAG_MOD_RATIO
    lo = max(best["min_drag"], best["max_drag"] / DRAG_MOD_RATIO)
    best["drag_modulation"] = best["max_drag"] - lo
    return best


def methodology():
    return f"""\
METHODOLOGY
-----------
Free-molecular force from the panel model (AeroModel.force_body), per face with
outward normal n_hat, area A, incidence cosine c = max(0, n_hat . v_hat):
    F = -rho |V|^2  sum  A c [ Cn c n_hat + Ct (v_hat - c n_hat) ].
Lift (force perpendicular to the wind) exists only because Cn != Ct. The net
specific force a = F/m is decomposed at every attitude into:
    a_drag = -(a . v_hat)                      (along -velocity; in-plane)
    a_lift =  |a - (a . v_hat) v_hat|          (perpendicular; steerable R or N)
We sweep the wind direction over the body sphere ({SWEEP_NTHETA}x{SWEEP_NPHI}) to
find the achievable envelope: peak drag (broadside), floor drag (edge-on), and
peak lift. Commandable drag MODULATION is capped at a {DRAG_MOD_RATIO:.0f}:1 high/low ratio.

Authority -> ROE rates (Gauss variational, near-circular; v = n*a_c = speed):
    in-plane:  differential drag gives tangential a_T -> controls delta_a,
               delta_lambda (along-track) and delta_e.
    out-of-plane: cross-track a_N gives di/dt = cos(u)/(n a_c) a_N,
               dOmega/dt = sin(u)/(n a_c sin i) a_N. A SECULAR delta_i needs a_N
               phase-modulated over the orbit (bang-bang), netting
               d|delta_i|/dt ~ (2/pi) a_lift / v   ->   a_c * d|delta_i|/dt [km/yr].
The out-of-plane NEED is the cross-track rate that cancels the inclination-
separated worst-sat nodal shear ({SHEAR_NODAL_MDEG_DAY:.1f} mdeg/day).

ASSUMPTIONS / CAVEATS: first-order, secular-only; attitude sweep is a coarse
proxy for the real drag/lift envelope; constant density over the orbit; NO drag
shadowing or thruster-plume effects (pending DSMC literature review). Screening
tool -- verify magnitudes in the full sim."""


def main():
    v = orbital_speed()
    rho = DensityModel(solar_level=SOLAR_LEVEL).interpolate(ALT)
    a_c = RE + ALT
    aero = build_aero()
    auth = drag_lift_authority(aero, rho, v, MASS_KG)

    # ROE-rate translation of the authority
    ddi_dt_max = (2.0 / np.pi) * auth["max_lift"] / v          # rad/s
    shear_correctable_km_yr = ddi_dt_max * a_c * 365.25 * 86400.0

    # needs
    base_drag = auth["max_drag"]                                # broadside, full
    need_inplane = NEED_INPLANE_BSPREAD * base_drag             # differential drag from B-spread
    rel_nodal = (SHEAR_NODAL_MDEG_DAY * 1e-3 * np.pi / 180.0) / 86400.0   # rad/s
    ddi_need = np.sin(INC) * rel_nodal                          # d|delta_i|/dt to cancel
    need_oop_peak = (np.pi / 2.0) * v * ddi_need               # peak lift needed
    shear_km_yr = rel_nodal * a_c * 365.25 * 86400.0           # the shear itself

    print("=" * 72)
    print("DIFFERENTIAL DRAG / LIFT CONTROL AUTHORITY  (suncatcher formation)")
    print("=" * 72)
    print(methodology())

    print(f"""
CONFIGURATION (tunable at top of file)
--------------------------------------
mass            : {MASS_KG:.0f} kg
broadside area  : {AREA_FRONT_M2:.1f} m^2   (A/m = {AREA_FRONT_M2/MASS_KG:.4f} m^2/kg)
edge area       : {AREA_END_M2:.1f} m^2
Cn / Ct         : {CN} / {CT}
altitude        : {ALT:.0f} km     inclination : {np.degrees(INC):.1f} deg
solar level     : {SOLAR_LEVEL}  ->  rho = {rho:.3e} kg/m^3   v = {v/1e3:.2f} km/s
drag modulation : {DRAG_MOD_RATIO:.0f}:1 commanded""")

    print(f"""
CONTROL AUTHORITY (from attitude sweep of your panel model)
-----------------------------------------------------------
peak drag accel        : {auth['max_drag']:.3e} m/s^2   (broadside)
floor drag accel       : {auth['min_drag']:.3e} m/s^2   (edge-on)
drag modulation (a_T)  : {auth['drag_modulation']:.3e} m/s^2   -> in-plane authority
peak lift accel (a_N)  : {auth['max_lift']:.3e} m/s^2   -> out-of-plane authority
effective L/D          : {auth['LD']:.2f}
secular nodal authority: {shear_correctable_km_yr:.1f} km/yr   (max shear lift can null)""")

    print(f"""
MAINTENANCE NEEDS
-----------------
in-plane (differential drag, {NEED_INPLANE_BSPREAD*100:.0f}% B-spread) : {need_inplane:.3e} m/s^2
out-of-plane SSO shear ({SHEAR_NODAL_MDEG_DAY:.1f} mdeg/day = {shear_km_yr:.1f} km/yr) : peak lift {need_oop_peak:.3e} m/s^2""")

    ip = auth["drag_modulation"] / need_inplane
    oop = auth["max_lift"] / need_oop_peak
    verdict = lambda r: (f"FEASIBLE  (x{r:.1f} margin)" if r >= 1 else f"INFEASIBLE (short /{1/r:.1f})")
    print(f"""
FEASIBILITY (authority / need)
------------------------------
in-plane   (differential drag) : {verdict(ip)}
out-of-plane (differential lift): {verdict(oop)}
NOTE: out-of-plane is for the inclination-separated config only; the node-based
config has no shear, so it needs zero out-of-plane authority. Authority scales
linearly with A/m and with density (~33x across the solar cycle).""")
    print("=" * 72)


if __name__ == "__main__":
    main()
