# Figure 3 data-source map (amendment 1 to the plotting spec)

All paths relative to papers/IAC_1RW/output_data/.

| figure input | source | field / recipe |
|---|---|---|
| A cell grid, PD (both panels) | A_baseline_20260819_110359.json (1rw cells, clamp-era) + A_baseline_20260818_202627.json (0rw/3rw context cells) | per cell: median_final_deg (5554 horizon), conv_pct_5deg -> divergence fraction = 100-conv5 for the open-marker annotation |
| A cell grid, planner | A_baseline_20260820_131727.json (reduced+full planner) + WAVE planner-full cell when landed (wave/ + WAVE_RESULTS.txt) | same fields; reduced-planner cell 1 stands as-run |
| per-cell D (x-coordinate) | same JSONs | median per_trial_h_frac_end = realized along-wheel secular per orbit / h_max (one-orbit footing, matches T_corr) |
| divergent-cell median | RECOMPUTE over converged trials only (pkls for planner cells; for PD cells use conv-only median from per-trial JSON columns where present, else from the wave PD-reduced pkls) | marker rule: median over CONVERGED trials; divergence % annotated separately |
| C bias ceiling (B3 verification marker) | C_bias_20260819_143616.json | breakpoint between 0.30 and 0.45 h_max levels (19x RMS jump); place on the B3 cut |
| F altitude wall (262 km arrow + 6U-at-600 overlay) | F_altitude JSON (F_*.json) | per-altitude tau_sec components; D(400)=0.07 reference, D(600) from the same table; boundary altitude ~262 km |
| dwell scope note | SCREEN_PROVENANCE.md + LOWINC_PREDICTION.md | text only, pointer to VI-E; inclination-scoped wording |
| wheel-class ticks (top axis) | Table 2 (manuscript) families: ~3 / 15 / 50 mN m s | ticks at D scaled by h_max ratio from the reference 15 |

## Choices confirmed
- **T_corr = one orbit** -- the campaign's own footing (C drift is per-orbit, D/F
  capacities per-orbit); B2 drawn at T_orbit with the fainter family line at T_orbit/4.
- Marker placement for divergent cells: CONVERGED-ONLY median + annotated fraction
  (never the mixed median).
- One envelope per panel, planner as markers (per spec section 7).

## Dimensional overlays -- sourcing verdict (amendment 2)
- **6U at 400 and 600 km: COMPUTABLE, fully in-repo** (factory bus + F's altitude
  table). These two markers are safe.
- **1U and 3U: NOT defensibly computable from the repo.** Mass/area are standard
  published values (1.33 kg / 4 kg etc.) but the disturbance model also needs a
  RESIDUAL DIPOLE and a cp-cg OFFSET per class, and neither is sourced anywhere in
  this repo -- the 6U used 0.05 A m^2 and 2 cm as SPEC'D values, not scalings. Per
  the registered rule: **markers drop unless Table 2's catalogue provides per-class
  dipole + offset figures** (Patrick's side); the caption then says "6U shown at two
  altitudes; smaller classes omitted for lack of sourced disturbance figures" rather
  than estimating.
