# Task 2 -- J-scaling validation (3MTQ+3RW, full attitude)

Same law (framework LP-PD), same scenarios; gains scaled by trace(J)/trace(J1).

| Inertia | tr J | gain x | conv% scaled | mean° scaled | conv% UNSCALED | mean° UNSCALED |
|---|---|---|---|---|---|---|
| J1 (ref) | 0.0480 | 1.00 | 100 | 0.01 | 100 | 0.01 |
| J2 (BeaverCube) | 0.0755 | 1.57 | 100 | 0.01 | 100 | 0.01 |
| J3 (20x J1) | 0.9600 | 20.00 | 100 | 0.00 | 1 | 27.32 |

Scaled-gain convergence spread across inertias: 0 points (INVARIANT -> validates the claim). Unscaled gains DEGRADE on the larger inertia (canonical gains too small) -> the scaling is load-bearing.
