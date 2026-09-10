# Reaction-wheel failure during LVLH (nadir) tracking

Full-LVLH (nadir-locked) tracking on a real 600 km circular orbit (IGRF-13 field, gravity-gradient ~6e-08 N·m; NO artificial body torque). One wheel killed at t=500 s, run one full orbit (tf=6000 s). 3MTQ+3RW (->3+2) keeps the full 3-axis LVLH goal; 3MTQ+1RW (->3+0) becomes underactuated and tracks a reduced-attitude goal (boresight->nadir, roll released).

| Config | goal | boresight→nadir (max / final) | transverse rate (max) |
|---|---|---|---|
| 3MTQ+3RW → 3+2 | full LVLH | 0.0° / 0.0° | 0.000°/s |
| 3MTQ+1RW → 3+0 | reduced (boresight) | 13.5° / 3.0° | 0.068°/s |

3+3→3+2 stays fully actuated (2 wheels + 3 MTQ span R^3), so it holds full LVLH through the failure with no transient. 3+1→3+0 drops to magnetorquer-only: full 3-axis LVLH is no longer controllable (no torque about the instantaneous field), so the framework relaxes to a reduced-attitude goal and holds the mission-critical **boresight on nadir** (bounded, recovering to a few degrees as the field sweeps over the orbit) while **releasing roll** about the boresight (full-LVLH error grows to ~110°, irrelevant for nadir pointing). The transverse body rate stays below 0.1°/s -- bounded, not tumbling.

Why reduced-attitude is required (not optional): a full-attitude controller on the underactuated bus feeds the direction-preserving LP a desired torque with an unachievable along-field component, so the LP returns zero (α=0) and *all* axes lose control -> tumble. Projecting out the uncontrollable roll keeps the allocation feasible and the boresight held -- a direct demonstration of the framework's reduced-attitude goal handling.
