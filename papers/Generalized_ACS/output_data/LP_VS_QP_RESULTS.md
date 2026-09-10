# Task 5 -- LP vs QP allocation direction error

Representative body field B = [26.83, 10.73, 8.05] uT. Fibonacci-sphere of 700 request directions x 3 magnitude scales (0.5/1.0/1.5 x max achievable) per config.

| Config | LP mean | LP p95 | LP max | QP mean (bdry) | QP p95 (bdry) | QP max (bdry) |
|---|---|---|---|---|---|---|
| 3MTQ+0RW | 0.0010 | 0.0000 | 0.1699 | 32.73 | 71.77 | 87.73 |
| 3MTQ+1RW | 0.0015 | 0.0000 | 0.4150 | 11.55 | 25.55 | 38.68 |

LP direction error is ~0 by construction (achieved torque is parallel to the request, or zero when the direction is unachievable). QP minimizes Euclidean torque error, so when the request points partly along the field-unachievable direction it delivers a tilted torque -- the tilt is ~0 for interior requests and grows toward 90 deg as the request approaches the unachievable direction.
