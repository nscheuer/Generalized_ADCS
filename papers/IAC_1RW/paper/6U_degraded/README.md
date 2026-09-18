# 6U degraded-actuator synthesis

This folder provides a compact concluding comparison for progressive reaction-wheel loss on the common disturbed 6U spacecraft:

| Architecture | Allocation | Wheels retained |
|---|---|---|
| 3+3 | LP | body x, y, z |
| 3+2 | LP | body x, y |
| 3+1 | LP | body x |
| 3+0 | QP | none |

All four architecture points use 3 orthogonal 0.6 A m2 magnetorquers, the same 12 kg 6U inertia and geometry, the same environmental models, random initial and target attitudes, a 400 km orbit, and 10 runs over one orbit. The 3+3, 3+1, and 3+0 archives are reused from `6U_noestimator_disturbance`; `run_3p2_mc.py` supplies the previously missing matched 3+2 campaign.

## Main result

Wheel loss is gradual until the final wheel is removed. Mean time to remain within 5 deg for at least 120 s increases from 2.68 min (3+3), to 3.65 min (3+2), to 5.29 min (3+1). These architectures retain 97.1%, 96.1%, and 94.3% one-orbit useful pointing, respectively, and all maintain 100% compliance during the final 20% of the orbit. Their corresponding tail RMS errors are 0.609, 0.621, and 0.700 deg.

The 3+0 QP case is the capability cliff: none of the 10 runs reaches the sustained 5 deg requirement, its tail RMS error is 96.9 deg, and useful pointing is zero. For this slew-and-hold mission, one well-managed wheel therefore recovers nearly all of the nominal three-wheel pointing capability; removing that wheel changes the mission from degraded to infeasible.

For repeated operations in the 3+1 state, the recovery plot combines the calibrated operational-envelope model with the prior reactive, gamma=0.5 co-allocation, and feasibility-aware planner archives. Their empirical peak wheel-momentum excursions are 6.98, 5.05, and 0.54 mN m s per representative slew. The planner consequently preserves roughly 80--93% of the uptime lost by reactive allocation over the plotted 0--1 slew/h range. This is the strongest compact argument for the planner: it adds little value to the already-successful one-off slew, but materially extends useful operations when slews repeat and storage is scarce.

## Figures

- `outputs/6u_wheel_loss_convergence.png`: mean attitude convergence with 1-sigma spread for progressive wheel loss.
- `outputs/6u_degradation_metric_summary.png`: slewing capability, disturbance rejection, final pointing RMS, and one-orbit uptime.
- `outputs/6u_momentum_headroom_by_architecture.png`: minimum wheel-storage headroom over the one-orbit campaigns.
- `outputs/6u_3p1_operational_recovery.png`: repeated-slew uptime, momentum deposited by the three management methods, and recovered mission capability.
- `outputs/6u_degradation_metrics.csv`: numerical architecture summary.

## Metric definitions and scope

- **Slewing capability:** first entry below 5 deg that remains there for 120 s. Failed runs are right-censored at the 92.55 min simulation duration and labelled by their success count.
- **Disturbance rejection:** fraction of samples below 5 deg during the final 20% of the orbit.
- **Final pointing error:** RMS attitude error during the final 20% of the orbit, evaluated per run and shown as mean plus/minus one standard deviation.
- **Uptime:** fraction of the complete orbit below 5 deg, including initial acquisition.
- **Momentum headroom:** `1 - max(|h_i|/h_max)` across all retained wheels and the complete orbit.
- **Recovered capability:** `(U_method - U_reactive)/(100 - U_reactive)`, clipped to 0--100%.

The first three plots are direct matched Monte Carlo evidence. The repeated-operations plot is a reduced-order extrapolation calibrated from earlier saved simulations; it is intended as a synthesis/design-space result, not a new Monte Carlo campaign. The planner archive tracks boresight whereas the reactive and co-allocation archives use full-attitude error, so absolute cross-method accuracy should not be inferred from this figure. Its defensible comparison is operational burden: observed settling time, wheel-momentum excursion, desaturation burden, and resulting predicted uptime.

## Reproduce

From the repository root:

```bash
venv/bin/python papers/IAC_1RW/paper/6U_degraded/run_3p2_mc.py
venv/bin/python papers/IAC_1RW/paper/6U_degraded/plot_degradation.py
```

The plotting script automatically selects the latest matching `.sim` archive for each architecture.
