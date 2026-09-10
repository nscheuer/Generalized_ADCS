# Task 6 -- 'It Works': planner vs PD headline figure

Grouped-bar restatement of the 2x2x2 matrix (`fig_p2_matrix` histograms), built from `p2p1_matrix_stats.json` (100-trial MC, 1000 s, final error < 5 deg).

| Config / goal | role | controller | converged % | mean final err | median | p95 |
|---|---|---|---|---|---|---|
| 3+0 / reduced (diag) | planner | ALTRO | 84% | 3.85° | 0.04° | 18.34° |
| 3+0 / reduced | pd | Lovera | 27% | 21.54° | 12.21° | 91.74° |
| 3+1 / full (diag) | planner | ALTRO | 94% | 1.19° | 0.06° | 10.10° |
| 3+1 / full | pd | LP | 90% | 9.06° | 1.08° | 74.79° |
| 3+1 / reduced | planner | ALTRO | 99% | 0.26° | 0.03° | 0.55° |
| 3+1 / reduced | pd | LP | 97% | 1.58° | 1.06° | 1.49° |
| 3+0 / full | planner | ALTRO | 18% | 18.31° | 14.26° | 46.92° |
| 3+0 / full | pd | Lovera | 0% | 85.17° | 83.45° | 156.43° |

**Headline:** on the controllability-appropriate diagonal the planner converges where the PD baseline does not — MTQ-only reduced pointing 84% vs 27% (mean 3.85° vs 21.5°), and 3+1 full attitude 94% vs 90% but mean 1.19° vs 9.06° (the PD tail is much heavier: p95 10.1° vs 74.8°). On the easy case (3+1 reduced) both succeed (99/97%). On the over-asked limit case (3+0 full = full 3-DOF on a 2-DOF-controllable plant) both fail, but the planner degrades less (18% vs 0%, mean 18.3° vs 85.2°).
