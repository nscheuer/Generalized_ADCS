# Case II — 3+1 Operational Envelope

This folder evaluates when a three-magnetorquer, one-reaction-wheel
architecture remains attractive under repeated mission slews.  It does not run
a new Monte Carlo campaign.  `operations_envelope.py` calibrates a deterministic
24-hour operations model from the existing reactive mode-switching,
gamma=0.5 co-allocation, and feasibility-aware planner archives.

## Model

For each method, the script measures the mean time required to remain below a
5 degree pointing error for 120 seconds and the mean peak wheel-momentum
excursion before that point.  These empirical slew costs are chained at the
requested slew frequency.  The wheel begins at the 25% restart threshold and
an emergency desaturation is charged whenever the accumulated load consumes
the 25%–75% momentum band.

The 75%–25% desaturation duration is measured from complete events in the
existing 1000-minute reactive archive.  It is scaled with wheel capacity and
inverse magnetic authority.  Altitude scales the per-slew load using the
disturbance-to-magnetic-authority ratio from the Case-I first-principles model.
RW torque scales acquisition time; MTQ dipole scales desaturation time.

Useful pointing uptime is the fraction of the 24-hour horizon left after slew
acquisition and explicit desaturation downtime, multiplied by the measured
tail fraction satisfying the 5 degree requirement.

The generated figures separately sweep slew demand against wheel capacity,
altitude, and RW/MTQ authority.  A compact representative plot at 635 km shows
both useful uptime and expected emergency dumps per day, making clear whether
lost science time is driven by acquisition or momentum management.

The gamma=0.5 implementation in the saved campaign is a continuous convex
blend.  It does not contain its own 75%/25% state machine.  In this screening
model it receives the same emergency hysteretic fallback as the reactive
controller, but reaches it less frequently because its measured momentum
excursion per slew is smaller.

## Calibrated values and result

| Method | Mean acquisition | Mean peak wheel excursion | Tail time below 5 deg |
|---|---:|---:|---:|
| Reactive 75/25 | 128.5 min | 6.980 mN m s | 81.8% |
| Co-allocation, gamma=0.5 | 31.9 min | 5.046 mN m s | 100.0% |
| Feasibility-aware planner | 11.2 min | 0.541 mN m s | 98.0% |

The measured mean duration of 16 complete reactive 75%–25% desaturation
episodes is 169.4 minutes.  At 635 km with the reference 15 mN m s wheel and
0.6 A m² MTQs, the model predicts the following 24-hour useful-pointing uptime:

| Slew demand | Reactive | Co-allocation | Planner |
|---:|---:|---:|---:|
| 0.10 / h | 61.4% | 94.7% | 96.1% |
| 0.25 / h | 14.7% | 69.8% | 93.4% |
| 0.50 / h | 0.0% | 26.5% | 88.8% |
| 1.00 / h | 0.0% | 0.0% | 79.7% |

Values clipped to zero mean that the reduced model assigns at least the full
24-hour horizon to acquisition and desaturation, not that the spacecraft
literally never points successfully.

Generated figures:

- `uptime_vs_slew_demand_and_wheel_capacity.png`
- `uptime_vs_slew_demand_and_altitude.png`
- `uptime_vs_rw_and_mtq_authority.png`
- `representative_uptime_and_desaturation_burden.png`
- `operations_calibration.csv`

## Interpretation limits

This is an operations-level surrogate, not a repeated-slew dynamics
simulation.  It conservatively chains the mean absolute momentum excursion in
one direction; truly random signed increments would behave more like a random
walk and generally saturate later.  Slew-angle distributions differ between
the saved full-attitude reactive/co-allocation campaigns and the reduced-
attitude planner campaign.  The figures are therefore suitable for identifying
operational regimes and motivating a matched repeated-slew experiment, not for
flight acceptance.

Run from the repository root:

```bash
venv/bin/python papers/IAC_1RW/paper/BC2_operations/operations_envelope.py
```
