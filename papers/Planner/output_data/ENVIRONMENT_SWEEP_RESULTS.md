# B2 -- disturbance environment sweep (3+1 reduced)

Paired N=20, tf=1000s. Planner plans on a clean est_sat (no disturbances). Plant carries the listed environment.

| Environment | converged % | mean final | median final |
|---|---|---|---|
| none | 100% | 0.06° | 0.02° |
| dipole-0.05 | 100% | 0.36° | 0.34° |
| dipole-0.10 | 100% | 0.80° | 0.70° |
| GG+drag+SRP | 100% | 0.06° | 0.04° |
| full-0.05 | 100% | 0.38° | 0.35° |
| full-0.10 | 100% | 0.73° | 0.65° |

Residual-dipole values (0.05, 0.10 A·m²) bracket a typical uncompensated 3U residual. Convergence is preserved across all environments, including the largest dipole + full aero/SRP, none of which are in the planner's model.
