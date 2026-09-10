# C1 -- boresight-roll foresight (3+1, vector A→B)

A→B separation 90°, slew axis (ECI) [1.0, 0.0, 0.0]. Single RW on +z, boresight +y; coast [700,700] s is roll-free.

- wheel-axis · slew-axis alignment: end-of-A 0.03 → end-of-coast 0.55
- |h| at slew start: 0.15 mN·m·s
- peak roll rate during coast: 0.14 deg/s
- B acquired: +144 s after slew start

**Foresight pre-positioning OBSERVED**: during the roll-free coast the planner rotated the wheel axis toward the upcoming slew axis and/or pre-loaded the wheel, then executed the slew.
