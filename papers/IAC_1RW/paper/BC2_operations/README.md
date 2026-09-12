# 6U Pointing-Time Comparison

This folder compares the saved 6U reactive, co-allocation, and planner Monte
Carlo campaigns using direct simulation measurements. It does not extrapolate
to a 24-hour operations model.

For every run, `operations_envelope.py` integrates the recorded simulation
intervals whose pointing error is within a stated bound. The figures show the
mean fraction of each run spent in:

- ≤ 5° pointing
- ≤ 1° pointing

Each bar is the mean across the saved Monte Carlo runs and its whisker is one
standard deviation. The values are percentages of the recorded run duration,
so campaigns of different lengths remain directly comparable.

The script automatically loads the latest matching `.sim` archive from each
6U campaign and writes:

- `time_in_5deg_pointing.png`
- `time_in_1deg_pointing.png`
- `pointing_time_summary.csv`

Run from the repository root:

```bash
venv/bin/python papers/IAC_1RW/paper/BC2_operations/operations_envelope.py
```

## Analytical repeated-slew envelopes

`operations_envelope_analytical.py` restores the four analytical envelope
plots. It uses the same saved 6U campaigns for calibration, but models
repeated slews, wheel storage, altitude, and actuator authority.

Slew demand is expressed in degrees per hour. The model converts it to a slew
count using the measured mean rotation of 109.5° per slew. Its colour scale and
ordinate are **estimated time in ≤5° pointing [% of day]**: the directly
measured ≤5° fraction from each saved campaign, reduced by the modeled time
spent acquiring slews and in desaturation mode.

It also produces `momentum_deposited_per_slew.png`, a direct calibration plot
of the peak wheel-momentum excursion accumulated before each method reaches
sustained ≤5° pointing. This is the per-slew storage cost used by the
analytical envelope; it is not an integrated momentum total over a full run.

Run it with:

```bash
venv/bin/python papers/IAC_1RW/paper/BC2_operations/operations_envelope_analytical.py
```
