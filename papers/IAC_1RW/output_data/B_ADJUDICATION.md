# Campaign B final adjudication (2026-08-22) -- with a retraction on the record

## RETRACTED: "small-theta along-field slope 0.209 confirms IV-B's 1/4"

Patrick's arithmetic check kills it, and the scatter confirms the kill:
- Measured times are ~100x the four-leg bracket bound (1249 s vs ~32 s upper bound at
  theta=0.1 with the equalized torque). Bracket-scale these are not.
- Torque-invariance excludes both route exponents: route A predicts t ~ tau^(-1/2)
  (8x torque -> -65%), route B tau^(-1/3) (-50%); measured flat to <0.5%.
- Within-theta scatter DWARFS the between-theta trend: at theta=0.05 completion spans
  231-1417 s (proxy sweep 15-92 deg); at 0.1, 477-1469 s; at 0.2, 566-1753 s. A 6x
  spread within each theta makes the 0.209 fit-of-medians a fit to floor noise. The
  proximity to 0.25 was coincidence.

## What stands (the stronger result, Patrick's framing)

- **The reachability bound is loose in practice by two orders of magnitude**, and
  along-field slewing is ORBIT-KINEMATICS-LIMITED and TORQUE-INDEPENDENT: you cannot
  buy your way out with a bigger actuator. Measured, not argued: <0.5% change across
  8x torque. The wheel's case: it removes an orbit-timescale floor no torque touches.
- **Check 1: planner throughout** -- 47 plans, 0 fallbacks in the top-up (261/0 in
  the main sweep). The floor is planner-scheduling / orbit kinematics, NOT
  LP-allocator refusal.
- **Check 2: the fixed-sweep-angle version does NOT hold** -- completion is not at a
  constant field-sweep angle (proxy sweep spans 15-131 deg within small theta). The
  mechanism is START-PHASE-DEPENDENT WAITING for favorable geometry: lucky starts
  finish at ~15-40 deg of sweep, unlucky ones wait up to ~130 deg (~a third of the
  field cycle) at small theta, growing past 150-210 deg at theta >= 1 (multiple
  windows needed). The scatter CEILING, not a fixed angle, is the invariant
  candidate. (Proxy = uniform-rotation approximation; main-sweep raw rows lack orbit
  fields, but top-up configs regenerate deterministically from axis_seed if the
  figure needs exact sweep angles.)
- **Cross-field 1/2 stands clean** (0.447, R2 0.991) -- directly-actuated axes, no
  wait.
- Floor table stands: medians flat at 0.107 orbits across m_scale 0.5-4.0.
