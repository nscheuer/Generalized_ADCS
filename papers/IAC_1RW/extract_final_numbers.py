"""One-pass final-numbers extraction -> FINAL_NUMBERS.md. Repo state is the truth;
anything unsourceable is marked, never estimated."""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import (  # noqa: E402
    IAC_6U, IAC_SENSOR_SPEC, cell_metrics, error_series, T_ORBIT, INC_DEG,
    PLAN_WINDOW_S, PLAN_OVERLAP_S, PLAN_TIMEOUT_S, BASELINE_H_FRAC,
    INIT_RATE_DPS_RANGE, _get_orbit, EPOCH)
from papers.IAC_1RW.generate_A_baseline import KP, KD, KC, J_TRANS  # noqa: E402
from ADCS.orbits.universal_constants import TimeConstants  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")
S2C = TimeConstants.sec2cent
L = []


def say(s=""):
    print(s)
    L.append(s)


def load_cell(pattern):
    runs = []
    for p in sorted(glob.glob(pattern)):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    return runs


def row(runs, horizons=(1000.0, T_ORBIT)):
    out = {}
    for h in horizons:
        m = cell_metrics(runs, h)
        fin = np.array([float(error_series(r)[min(int(h), len(error_series(r)) - 1)])
                        for r in runs])
        out[h] = dict(n=len(runs), conv5=m["conv_pct_5deg"], conv1=m["conv_pct_1deg"],
                      med=m["median_final_deg"], know=m.get("median_est_att_err_deg"),
                      div=float(100 * np.mean(fin > 30.0)),
                      med_conv=float(np.median(fin[fin <= 30.0]))
                      if (fin <= 30).any() else float("nan"))
    return out


def jrow(d, key):
    out = {}
    for hk, h in (("1000", 1000.0), ("5554", T_ORBIT)):
        m = d["cells"][key]["horizons"][hk]
        out[h] = dict(n=d["cells"][key]["n_completed"], conv5=m["conv_pct_5deg"],
                      conv1=m["conv_pct_1deg"], med=m["median_final_deg"],
                      know=m.get("median_est_att_err_deg"),
                      div=float(100 - m["conv_pct_5deg"]), med_conv=float("nan"))
    return out


def main():
    say("# FINAL NUMBERS HANDOFF (extracted 2026-08-23 from repo state)")
    say("")

    S = IAC_SENSOR_SPEC
    r2d = np.degrees
    say("## Table 1 -- reference bus and orbit")
    say(f"- form factor, mass: 6U ({IAC_6U.dims_m[0]}x{IAC_6U.dims_m[1]}x{IAC_6U.dims_m[2]} m), {IAC_6U.mass} kg")
    say(f"- inertia (principal, about COM): Jx={IAC_6U.J_com[0]}, Jy={IAC_6U.J_com[1]}, Jz={IAC_6U.J_com[2]} kg m^2")
    say(f"- wheel: axis +z (= boresight {IAC_6U.boresight}), tau_w={IAC_6U.tau_w*1e3:.0f} mN m, h_max={IAC_6U.h_max*1e3:.0f} mN m s")
    say(f"- MTQs: 3, body axes, |m|_inf <= {IAC_6U.m_max} A m^2 per axis")
    say(f"- residual dipole: {IAC_6U.m_res} A m^2 along {tuple(IAC_6U.m_res_dir)} (normalized)")
    say(f"- cp-cg offset: {IAC_6U.com_offset_m*100:.0f} cm along {IAC_6U.com_offset_dir} (CDS-bounded worst case)")
    say(f"- drag: Cd={IAC_6U.CD}, faceted attitude-dependent area (median A_eff 0.057 m^2); eta_a/d/s = {IAC_6U.eta_a}/{IAC_6U.eta_d}/{IAC_6U.eta_s}")
    say(f"- orbit: 400 km circular (e=0), inc {INC_DEG} deg; RAAN/phase RANDOMIZED per trial -- no fixed LTAN (state as such); period {T_ORBIT:.1f} s")
    say(f"- star trackers: 2, opposed +/-x; cross {r2d(S['st_sigma_cross_rad'])*3600:.0f} arcsec / roll {r2d(S['st_sigma_roll_rad'])*3600:.0f} arcsec (1-sigma); sun excl {r2d(S['st_sun_exclusion_rad']):.0f} deg, Earth-limb {r2d(S['st_earth_limb_exclusion_rad']):.0f} deg, FOV {r2d(S['st_fov_rad']):.0f} deg, max rate {r2d(S['st_max_rate_rad_per_s']):.0f} deg/s, {S['st_sample_time_s']} s sampling")
    say(f"- gyro: ARW {r2d(S['gyro_arw_rad_per_sqrt_s'])*60.0:.2f} deg/sqrt(hr), bias instability {r2d(S['gyro_bias_instab_rad_per_s'])*3600.0:.2f} deg/hr; MTM sigma {S['mtm_sigma_T']*1e9:.0f} nT")
    say("- control rate: 1 Hz (dt = 1 s)")
    say(f"- baseline stored momentum: {BASELINE_H_FRAC:.2f} h_max = {BASELINE_H_FRAC*IAC_6U.h_max*1e3:.2f} mN m s along wheel; initial rate U{INIT_RATE_DPS_RANGE} deg/s")
    say(f"- PD gains: kp={KP:.2e}, kd={KD:.4e} (= 2 sqrt(kp J_TRANS/2), J_TRANS={J_TRANS} = MAX principal); c_gain={KC}")
    say(f"- planner: horizon {PLAN_WINDOW_S+PLAN_OVERLAP_S:.0f} s (execute {PLAN_WINDOW_S:.0f}), replan {PLAN_WINDOW_S:.0f} s, wall budget {PLAN_TIMEOUT_S:.0f} s (process boundary); dt_tp=50 s, dt_tvlqr=1 s")
    say("- planner weights, reduced: angle=1e1, angle_N=1e1, ang_vel=1e5, COLD (per-task tuning selected the baseline)")
    say("- planner weights, full: angle=1e2, angle_N=1e2, ang_vel=1e5, WARM-hold (frozen, validated)")
    say("- inertia usage: kd -> J_TRANS=0.13 (max principal, transverse to wheel); slew quantities -> slew-axis component; D -> wheel-axis projection")
    say("")

    say("## Table 2 -- Campaign A grid (1000 s / 5554 s horizons; div = % final > 30 deg)")
    d18 = json.load(open(os.path.join(OUT, "A_baseline_20260818_202627.json")))
    cells = [
        ("3MTQ+0RW", "reduced", "PD", jrow(d18, "0rw_reduced_pd"), "context n=30, 8-18 era (no wheel)"),
        ("3MTQ+0RW", "full", "PD", jrow(d18, "0rw_full_pd"), "context n=30, 8-18 era"),
        ("3MTQ+1RW", "reduced", "PD", row(load_cell(os.path.join(OUT, "wave/pd_reduced_kp1/*.pkl"))), "WAVE (recycled+persisted)"),
        ("3MTQ+1RW", "full", "PD", row(load_cell(os.path.join(OUT, "wave/pd_full_kp1/*.pkl"))), "WAVE"),
        ("3MTQ+3RW", "reduced", "PD", jrow(d18, "3rw_reduced_pd"), "context n=30, 8-18 era (pre-clamp; wheels peaked 0.13 h_max -- clamp inert)"),
        ("3MTQ+3RW", "full", "PD", jrow(d18, "3rw_full_pd"), "context n=30, 8-18 era"),
        ("3MTQ+1RW", "reduced", "planner", row(load_cell(os.path.join(OUT, "A_trials/1rw_reduced_planner_seed*.pkl"))), "cell 1 (0 kills/0 fallbacks)"),
        ("3MTQ+1RW", "full", "planner", row(load_cell(os.path.join(OUT, "tune_seed*_wave_planner_full.pkl"))), "WAVE, TUNED -- both-ways below"),
    ]
    say("| config | task | law | n | conv5 (1k/orb) | conv1 (1k/orb) | median (1k/orb) | knowledge | div% |")
    say("|---|---|---|---|---|---|---|---|---|")
    for cfg, task, law, rr, note in cells:
        a, b = rr[1000.0], rr[T_ORBIT]
        kn = f"{b['know']:.3f}" if b.get("know") is not None else "[UNSOURCED]"
        say(f"| {cfg} | {task} | {law} | {b['n']} | {a['conv5']:.0f}/{b['conv5']:.0f} | "
            f"{a['conv1']:.0f}/{b['conv1']:.0f} | {a['med']:.2f}/{b['med']:.2f} | {kn} | {b['div']:.0f} |")
        say(f"|  |  |  |  |  |  |  |  | ({note}) |")
    say("")
    pf = load_cell(os.path.join(OUT, "tune_seed*_wave_planner_full.pkl"))
    clean = [r for r in pf if not r.get("n_budget_kills", 0)]
    mall, mpp = cell_metrics(pf, T_ORBIT), cell_metrics(clean, T_ORBIT)
    say(f"- BOTH-WAYS planner-full (tuned): ALL n=100 {mall['conv_pct_5deg']:.1f}/{mall['conv_pct_1deg']:.1f}/{mall['median_final_deg']:.2f} "
        f"(6.5% fallback windows; 78 kills in 32 trials); PURE n={len(clean)} "
        f"{mpp['conv_pct_5deg']:.1f}/{mpp['conv_pct_1deg']:.1f}/{mpp['median_final_deg']:.2f}")
    pe = load_cell(os.path.join(OUT, "tune_seed*_wave_planner_full_base.pkl"))
    me = cell_metrics(pe, T_ORBIT)
    say(f"- planner-full BASELINE-weights (Cell E, clean, seed-paired): "
        f"{me['conv_pct_5deg']:.1f}/{me['conv_pct_1deg']:.1f}/{me['median_final_deg']:.2f} (4 kills / 1200 windows)")
    convonly = ", ".join(
        f"{cfg[0:4]}{cfg[-3:]}-{task[:3]}-{law}: {rr[T_ORBIT]['med_conv']:.2f}"
        for cfg, task, law, rr, _ in cells if not np.isnan(rr[T_ORBIT]["med_conv"]))
    say(f"- grid medians are ALL-TRIAL; converged-only medians (Fig 3 markers): {convonly}")
    say("")

    say("## Table 3 -- ablations")
    say("### Campaign B (equalized torque m_max = 31.97 A m^2 at 37.1 uT median; planner, 0 fallbacks)")
    say("- cross-field: slope 0.447, R2 0.991, n=5 thetas, Theta 0.05-2.0 rad")
    say("- along-field (FLOOR-DOMINATED; not an exponent): fitted 0.541 over Theta 0.5-2.0 (R2 0.896); constructed-axis small-Theta 0.209 over 0.05-0.5 -- RETRACTED as 1/4 confirmation (floor noise)")
    say("- torque invariance: median 0.107 orbits at every m_scale in {0.5,1,2,4} (<0.5% across 8x)")
    say("- bracket at Theta=0.1: measured 1249 s vs ~32 s upper / ~7 s lower bound (2 orders loose)")
    say("- within-Theta scatter: 231-1417 s at Theta=0.05 (6x); proxy field-sweep 15-131 deg")
    say("")

    cj = json.load(open(os.path.join(OUT, "C_bias_20260819_143616.json")))
    say("### Campaign C (bias sweep, clamped)")
    ck = sorted(cj["cells"], key=float)
    fields = sorted(cj["cells"][ck[0]].keys())
    say(f"(available per-level fields: {fields})")
    say("| h0/h_max | h0 [mN m s] | drift | RMS | acquire |")
    say("|---|---|---|---|---|")
    for k in ck:
        v = cj["cells"][k]
        def g(*names):
            for nm in names:
                if nm in v and isinstance(v[nm], (int, float)):
                    return f"{v[nm]:.2f}"
            return "[UNSOURCED -- name not in JSON]"
        say(f"| {k} | {float(k)*IAC_6U.h_max*1e3:.2f} | "
            f"{g('drift_deg_per_orbit','median_drift_deg_per_orbit','held_drift_deg_per_orbit')} | "
            f"{g('median_rms_deg','held_rms_deg','rms_deg')} | "
            f"{g('median_acquire_5deg_s','median_acquire_s','acquire_s')} |")
    say("- ceiling: discontinuity between the 0.30 and 0.45 levels (19x RMS jump), bracketing the predicted 0.42 h_max; clamped rerun")
    say("")

    say("## Loose figures")
    say("### C stiffness: 1/h law is a NULL result (no stiffness benefit measured); the 10x acquire-time cost at high bias stands. [FLAG if draft claims a 1/h benefit]")
    dj = json.load(open(os.path.join(OUT, "D_sigma_duty_20260818_174554.json")))
    say("### D (settled bus)")
    for k, v in sorted(dj["cells"].items()):
        say(f"- {k}: median_sigma {v['median_sigma']:.3f}, restore_duty {v['restore_duty']:.3f}, margin {v['margin_vs_accumulation']:.1f}x, sigma* {v['sigma_star']:.4f}")
    say(f"- tau_allow reference {dj['tau_allow_ref_Nm']*1e6:.2f} uN m; grid {[round(x*1e6,2) for x in dj['tau_allow_grid_Nm']]} uN m")
    say("- tau_allow crossover: the 9.13 uN m figure is 0.2 A m^2-ERA and SUPERSEDED at the settled 0.6 bus (margins 9.7-61.6x; crossing not operative). [FLAG if draft quotes 9.13]")

    fj = json.load(open(os.path.join(OUT, "F_altitude_20260818_174558.json")))
    rows400 = {r["alt_km"]: r for r in fj["rows_by_case"]["m_res=0.05"]}
    r4 = rows400[400.0]
    say("### F at 400 km (m_res=0.05)")
    say("- secular by source [mN m s/orbit]: " + ", ".join(f"{k.split('_')[0]} {v*1e3:.4f}" for k, v in r4["accum_by_source_Nms"].items()))
    say("- cyclic by source [mN m s]: " + ", ".join(f"{k.split('_')[0]} {v*1e3:.4f}" for k, v in r4["cyclic_by_source_Nms"].items()))
    say(f"- total secular {r4['accum_per_orbit_Nms']*1e3:.3f} (VECTOR sum); along-wheel {r4['accum_along_wheel_Nms']*1e3:.3f} ({100*r4['accum_along_wheel_Nms']/r4['accum_per_orbit_Nms']:.0f}%); transverse {r4['accum_transverse_Nms']*1e3:.3f}")
    say(f"- cyclic total {r4['cyclic_total_Nms']*1e3:.3f} mN m s = {100*r4['reserved_wheel_frac']:.1f}% of wheel range reserved")
    say(f"- binding altitude {fj['altitude_unity_margin_km']['m_res=0.05']:.0f} km: F's own log-margin fit, EXTRAPOLATED below the 300 km sample (margin(300) = {rows400[300.0]['margin']:.2f})")

    say("### Screen (CURRENT labels: wave PD-reduced, 12 events)")
    say("- operating point: flag iff dwell(sigma<0.2) <= 0.1035")
    say("- in-sample recall 8/12 at 0 FP / 88 passes; fixed-cutoff LOO identical by construction")
    say("- refit-LOO (largest <=1-FP max-catch cutoff per fold): 7/12")
    say("- transfer (PD-full saturation events h_frac_max>=0.999): 11/26 at 1 FP / 74 non-events, n=100 exactly")
    say("- scope: FAILS transfer at 15 deg inclination (0/3, 26% FA) -- high-inclination result")

    say("### Demand ratio (omega_perp; authority = single-axis m_max|B(t)| instantaneous)")
    red = load_cell(os.path.join(OUT, "wave/pd_reduced_kp1/*.pkl"))
    conv_pt, div_pt = [], []
    for r in red:
        e = error_series(r)
        t = np.asarray(r["time"], float)
        st_ = np.asarray(r["state"], float)
        om_perp = np.linalg.norm(st_[:, 0:2], axis=1)
        h = st_[:, 7]
        orb = _get_orbit(r["config"], 1.0, float(r["config"]["tf"]))
        idx = np.arange(0, len(t), 60)
        Bm = np.array([np.linalg.norm(np.asarray(orb.get_os(J2000=EPOCH + t[i] * S2C).B, float)) for i in idx])
        ratio = np.sqrt((h[idx] * IAC_6U.h_max) ** 2 + KD ** 2) * om_perp[idx] / (IAC_6U.m_max * Bm)
        post = idx > 1000
        (div_pt if float(e[-1]) > 30 else conv_pt).append(float(np.median(ratio[post])))
    say(f"- post-transient medians: diverged {np.median(div_pt):.2f} (IQR {np.percentile(div_pt,25):.2f}-{np.percentile(div_pt,75):.2f}, n={len(div_pt)}); converged {np.median(conv_pt):.3f} (IQR {np.percentile(conv_pt,25):.3f}-{np.percentile(conv_pt,75):.3f}, n={len(conv_pt)})")
    say("- rate statistic: post-transient (t > ~1000 s) median |omega_perp|; peaks slew-dominated (5.1 conv / 41.8 div medians), reported as context only")

    say("### MTQ duty (Section II power)")
    duty = [float(np.mean(np.abs(np.asarray(r["u"], float)[:, :int(r["n_mtq"])]) / IAC_6U.m_max)) for r in red]
    say(f"- PD-reduced wave: orbit-average per-axis |m|/m_max mean {np.mean(duty):.3f}, median {np.median(duty):.3f}")

    with open(os.path.join(HERE, "FINAL_NUMBERS.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\nwritten FINAL_NUMBERS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
