# Case I — BeaverCube II

This folder supports the BeaverCube II case study in Section 6.1.

The case study makes two deliberately separate claims:

1. `run_ground_tracking_mc.py` performs a 10-run, disturbed 3-MTQ + 1-RW
   BeaverCube II ground-contact Monte Carlo.  It uses a Boston coordinate
   target, a 635 km circular pass with a 15 degree cross-track miss, and
   scheduled pre-pointing before acquisition of signal (AOS), with large
   30–170 degree randomized acquisition errors and 0.1–1 degree/s initial
   rates.  The purely geometric horizon-to-horizon window is 680 s.  A usable connection requires
   both geometric visibility and less than or equal to 5 degree boresight
   error; atmospheric/refraction losses are deliberately excluded.
   `plot_ground_tracking_mc.py` produces a mean ±1σ pointing-error plot and
   reports minimum, mean, maximum, and 1σ usable-link time in both the figure
   and `bc2_ground_contact_times_mc_10.csv`.
2. `sizing_map.py` makes a first-principles screening plot; it is not a
   simulation or an instantaneous all-direction feasibility proof.  Panel A
   calculates the MTQ dipole needed to balance a conservative sum of gravity-
   gradient, drag, SRP, and residual-dipole torque.  Panel B calculates the
   wheel momentum storage accumulated over a 10 minute contact if the full
   disturbance is carried by the wheel and no unloading occurs.  In both
   panels, the horizontal axis is the bus long dimension and each colored
   curve is a different altitude; horizontal actuator-capability lines show
   the corresponding size boundary.  This is a conservative capacity screen,
   deliberately separate from the later desaturation/science-uptime analysis.
   A 3+1 MTQ/RW system also has singular instantaneous magnetic geometries, so
   neither panel guarantees control for every disturbance and field direction.

The screening map therefore identifies candidates for simulation rather than
claiming mission acceptance.  The markers use BC2 factory limits (0.2 A m²
per MTQ, 3.6 mN m s wheel storage) and CanX-2 reference values (635 km,
10 x 10 x 34 cm, 0.1 A m² coils, 30 mN m s storage).  The CanX-2 coil value
is a literature reference value, not a flight-certified limit, and should be
replaced if a primary hardware specification becomes available.  The map’s
key result is still a broad altitude plateau: drag falls rapidly through lower
LEO, then residual-dipole torque and declining magnetic field dominate.

The saved horizon-to-horizon Monte Carlo has usable-link min / mean / max of
436 / 591.6 / 680 s, with 1σ = 71.0 s at the 2 s simulation resolution.  The
geometric window is 680 s; the loss of usable time comes from the large random
acquisition errors and initial rates.  It is a controller-only pre-pointed
acquisition demonstration with biases disabled, not a flight acceptance test.
An estimator-in-the-loop result and an explicit mission error requirement are
still needed before making that claim.

Run from the repository root:

```bash
venv/bin/python papers/IAC_1RW/paper/BC2/run_ground_tracking_mc.py
venv/bin/python papers/IAC_1RW/paper/BC2/plot_ground_tracking_mc.py
venv/bin/python papers/IAC_1RW/paper/BC2/sizing_map.py
```
