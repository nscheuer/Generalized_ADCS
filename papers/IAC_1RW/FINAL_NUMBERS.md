# FINAL NUMBERS HANDOFF (extracted 2026-08-23 from repo state)

## Table 1 -- reference bus and orbit
- form factor, mass: 6U (0.1x0.2x0.3 m), 12.0 kg
- inertia (principal, about COM): Jx=0.13, Jy=0.1, Jz=0.05 kg m^2
- wheel: axis +z (= boresight (0.0, 0.0, 1.0)), tau_w=2 mN m, h_max=15 mN m s
- MTQs: 3, body axes, |m|_inf <= 0.6 A m^2 per axis
- residual dipole: 0.05 A m^2 along (1.0, 1.0, 1.0) (normalized)
- cp-cg offset: 2 cm along (0.0, 0.0, 1.0) (CDS-bounded worst case)
- drag: Cd=2.2, faceted attitude-dependent area (median A_eff 0.057 m^2); eta_a/d/s = 0.3/0.2/0.5
- orbit: 400 km circular (e=0), inc 97.0 deg; RAAN/phase RANDOMIZED per trial -- no fixed LTAN (state as such); period 5553.6 s
- star trackers: 2, opposed +/-x; cross 5 arcsec / roll 30 arcsec (1-sigma); sun excl 30 deg, Earth-limb 25 deg, FOV 20 deg, max rate 2 deg/s, 0.25 s sampling
- gyro: ARW 0.15 deg/sqrt(hr), bias instability 0.30 deg/hr; MTM sigma 100 nT
- control rate: 1 Hz (dt = 1 s)
- baseline stored momentum: 0.05 h_max = 0.75 mN m s along wheel; initial rate U(0.05, 0.3) deg/s
- PD gains: kp=2.90e-04, kd=8.6833e-03 (= 2 sqrt(kp J_TRANS/2), J_TRANS=0.13 = MAX principal); c_gain=0.001
- planner: horizon 1000 s (execute 500), replan 500 s, wall budget 300 s (process boundary); dt_tp=50 s, dt_tvlqr=1 s
- planner weights, reduced: angle=1e1, angle_N=1e1, ang_vel=1e5, COLD (per-task tuning selected the baseline)
- planner weights, full: angle=1e2, angle_N=1e2, ang_vel=1e5, WARM-hold (frozen, validated)
- inertia usage: kd -> J_TRANS=0.13 (max principal, transverse to wheel); slew quantities -> slew-axis component; D -> wheel-axis projection

## Table 2 -- Campaign A grid (1000 s / 5554 s horizons; div = % final > 30 deg)
| config | task | law | n | conv5 (1k/orb) | conv1 (1k/orb) | median (1k/orb) | knowledge | div% |
|---|---|---|---|---|---|---|---|---|
| 3MTQ+0RW | reduced | PD | 30 | 0/0 | 0/0 | 98.98/74.00 | 0.017 | 100 |
|  |  |  |  |  |  |  |  | (context n=30, 8-18 era (no wheel)) |
| 3MTQ+0RW | full | PD | 30 | 0/0 | 0/0 | 136.57/125.41 | 0.019 | 100 |
|  |  |  |  |  |  |  |  | (context n=30, 8-18 era) |
| 3MTQ+1RW | reduced | PD | 100 | 83/88 | 73/82 | 0.39/0.23 | 0.005 | 12 |
|  |  |  |  |  |  |  |  | (WAVE (recycled+persisted)) |
| 3MTQ+1RW | full | PD | 100 | 64/78 | 55/72 | 0.67/0.26 | 0.007 | 20 |
|  |  |  |  |  |  |  |  | (WAVE) |
| 3MTQ+3RW | reduced | PD | 30 | 97/100 | 87/97 | 0.11/0.08 | 0.004 | 0 |
|  |  |  |  |  |  |  |  | (context n=30, 8-18 era (pre-clamp; wheels peaked 0.13 h_max -- clamp inert)) |
| 3MTQ+3RW | full | PD | 30 | 97/100 | 90/93 | 0.13/0.10 | 0.005 | 0 |
|  |  |  |  |  |  |  |  | (context n=30, 8-18 era) |
| 3MTQ+1RW | reduced | planner | 100 | 79/96 | 58/53 | 0.83/0.99 | 0.005 | 1 |
|  |  |  |  |  |  |  |  | (cell 1 (0 kills/0 fallbacks)) |
| 3MTQ+1RW | full | planner | 100 | 53/94 | 18/23 | 4.55/1.42 | 0.005 | 0 |
|  |  |  |  |  |  |  |  | (WAVE, TUNED -- both-ways below) |

- BOTH-WAYS planner-full (tuned): ALL n=100 94.0/23.0/1.42 (6.5% fallback windows; 78 kills in 32 trials); PURE n=68 92.6/19.1/1.47
- planner-full BASELINE-weights (Cell E, clean, seed-paired): 72.0/19.0/2.24 (4 kills / 1200 windows)
- grid medians are ALL-TRIAL; converged-only medians (Fig 3 markers): 3MTQ1RW-red-PD: 0.21, 3MTQ1RW-ful-PD: 0.19, 3MTQ1RW-red-planner: 0.99, 3MTQ1RW-ful-planner: 1.42

## Table 3 -- ablations
### Campaign B (equalized torque m_max = 31.97 A m^2 at 37.1 uT median; planner, 0 fallbacks)
- cross-field: slope 0.447, R2 0.991, n=5 thetas, Theta 0.05-2.0 rad
- along-field (FLOOR-DOMINATED; not an exponent): fitted 0.541 over Theta 0.5-2.0 (R2 0.896); constructed-axis small-Theta 0.209 over 0.05-0.5 -- RETRACTED as 1/4 confirmation (floor noise)
- torque invariance: median 0.107 orbits at every m_scale in {0.5,1,2,4} (<0.5% across 8x)
- bracket at Theta=0.1: measured 1249 s vs ~32 s upper / ~7 s lower bound (2 orders loose)
- within-Theta scatter: 231-1417 s at Theta=0.05 (6x); proxy field-sweep 15-131 deg

### Campaign C (bias sweep, clamped)
(available per-level fields: ['acquired_frac', 'h0_Nms', 'h_frac', 'median_acquire_5deg_s', 'median_drift_deg_per_orbit', 'median_final_deg', 'median_rms_deg', 'n'])
| h0/h_max | h0 [mN m s] | drift | RMS | acquire |
|---|---|---|---|---|
| 0.00 | 0.00 | 0.02 | 0.14 | 274.00 |
| 0.05 | 0.75 | 0.01 | 0.22 | 336.00 |
| 0.15 | 2.25 | -0.14 | 0.45 | 525.00 |
| 0.30 | 4.50 | -0.52 | 1.13 | 1600.00 |
| 0.45 | 6.75 | -21.41 | 21.27 | 2827.00 |
| 0.60 | 9.00 | -23.05 | 43.91 | 3094.00 |
- ceiling: discontinuity between the 0.30 and 0.45 levels (19x RMS jump), bracketing the predicted 0.42 h_max; clamped rerun

## Loose figures
### C stiffness: 1/h law is a NULL result (no stiffness benefit measured); the 10x acquire-time cost at high bias stands. [FLAG if draft claims a 1/h benefit]
### D (settled bus)
- 45deg|45|inertial: median_sigma 0.609, restore_duty 0.772, margin 28.2x, sigma* 0.0604
- 45deg|45|nadir: median_sigma 0.478, restore_duty 0.553, margin 33.1x, sigma* 0.0778
- 45deg|5|inertial: median_sigma 0.637, restore_duty 0.900, margin 17.9x, sigma* 0.0877
- 45deg|5|nadir: median_sigma 0.849, restore_duty 1.000, margin 12.7x, sigma* 0.0729
- 45deg|97|inertial: median_sigma 0.303, restore_duty 0.506, margin 43.9x, sigma* 0.0693
- 45deg|97|nadir: median_sigma 0.411, restore_duty 0.828, margin 26.0x, sigma* 0.0779
- boresight|45|inertial: median_sigma 0.352, restore_duty 0.547, margin 39.4x, sigma* 0.0789
- boresight|45|nadir: median_sigma 0.811, restore_duty 0.892, margin 18.2x, sigma* 0.0517
- boresight|5|inertial: median_sigma 0.946, restore_duty 1.000, margin 9.8x, sigma* 0.0390
- boresight|5|nadir: median_sigma 0.274, restore_duty 0.422, margin 37.7x, sigma* 0.1311
- boresight|97|inertial: median_sigma 0.674, restore_duty 0.803, margin 23.8x, sigma* 0.0659
- boresight|97|nadir: median_sigma 0.898, restore_duty 0.917, margin 15.5x, sigma* 0.0382
- orbit_normal|45|inertial: median_sigma 0.548, restore_duty 0.661, margin 33.6x, sigma* 0.0778
- orbit_normal|45|nadir: median_sigma 0.506, restore_duty 0.817, margin 21.1x, sigma* 0.0773
- orbit_normal|5|inertial: median_sigma 0.172, restore_duty 0.278, margin 51.3x, sigma* 0.1273
- orbit_normal|5|nadir: median_sigma 0.951, restore_duty 1.000, margin 9.7x, sigma* 0.0396
- orbit_normal|97|inertial: median_sigma 0.180, restore_duty 0.317, margin 53.8x, sigma* 0.0827
- orbit_normal|97|nadir: median_sigma 0.197, restore_duty 0.356, margin 61.6x, sigma* 0.0853
- tau_allow reference 2.50 uN m; grid [0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.07, 0.08, 0.11, 0.13, 0.17, 0.22, 0.27, 0.35, 0.44, 0.55, 0.7, 0.89, 1.13, 1.43, 1.8, 2.29, 2.89, 3.67, 4.64, 5.88, 7.44, 9.43, 11.94, 15.12, 19.14, 24.24, 30.7, 38.88, 49.24, 62.36, 78.97, 100.0] uN m
- tau_allow crossover: the 9.13 uN m figure is 0.2 A m^2-ERA and SUPERSEDED at the settled 0.6 bus (margins 9.7-61.6x; crossing not operative). [FLAG if draft quotes 9.13]
### F at 400 km (m_res=0.05)
- secular by source [mN m s/orbit]: GG 0.0000, Drag 1.4183, SRP 0.0261, Dipole 1.2901
- cyclic by source [mN m s]: GG 0.0000, Drag 0.0000, SRP 0.0014, Dipole 2.3639
- total secular 1.755 (VECTOR sum); along-wheel 0.994 (57%); transverse 1.446
- cyclic total 2.365 mN m s = 15.8% of wheel range reserved
- binding altitude 262 km: F's own log-margin fit, EXTRAPOLATED below the 300 km sample (margin(300) = 2.39)
### Screen (CURRENT labels: wave PD-reduced, 12 events)
- operating point: flag iff dwell(sigma<0.2) <= 0.1035
- in-sample recall 8/12 at 0 FP / 88 passes; fixed-cutoff LOO identical by construction
- refit-LOO (largest <=1-FP max-catch cutoff per fold): 7/12
- transfer (PD-full saturation events h_frac_max>=0.999): 11/26 at 1 FP / 74 non-events, n=100 exactly
- scope: FAILS transfer at 15 deg inclination (0/3, 26% FA) -- high-inclination result
### Demand ratio (omega_perp; authority = single-axis m_max|B(t)| instantaneous)
- post-transient medians: diverged 0.89 (IQR 0.72-1.58, n=12); converged 0.004 (IQR 0.003-0.009, n=88)
- rate statistic: post-transient (t > ~1000 s) median |omega_perp|; peaks slew-dominated (5.1 conv / 41.8 div medians), reported as context only
### MTQ duty (Section II power)
- PD-reduced wave: orbit-average per-axis |m|/m_max mean 0.190, median 0.111
