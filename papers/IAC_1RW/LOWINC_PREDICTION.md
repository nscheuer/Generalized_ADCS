# Pre-registered: low-inclination spot-check (Section VII's largest untested claim)

Committed BEFORE the run. Setup: PD, 3+1, reduced attitude, boresight mounting,
n=30 (seeds 0-29), inclination 15 deg, everything else the settled bus and standard
protocol (campaign gains kp=2.9e-4, kd=8.68e-3); full covariate set; per-trial pkls.

Baselines (97-deg money cell, from the clean campaign JSON): dwell(sigma<0.2) median
0.150 (converged-only 0.153), screen flag rate 9%, divergence fraction 11%.

## Predictions (direction, stated pre-run)

1. **Sigma duty degrades:** median dwell(sigma<0.2) DECREASES vs 0.150 -- the
   near-equatorial field rotates less, sigma diversity shrinks, dump-capable windows
   become rarer for a fixed boresight mounting.
2. **Divergence fraction RISES** above the 11% baseline (more dump-starved draws).
3. **Screen transfer:** flag rate rises accordingly. TRANSFER FAILURE is declared if
   the false-alarm rate among CONVERGED trials exceeds 10% at the frozen threshold
   0.1035 -- the live risk, since the threshold sits AT the nearest converged trial's
   dwell in the fit population (zero permissive-side margin; see the sensitivity
   grid).
4. **Stated falsification risk:** if low-dwell draws become common AND the divergence
   fraction does NOT rise, the dwell -> divergence link itself weakens materially and
   is reported as such. (n=30 resolves only large effects; binomial sigma ~ 6% at
   p~0.11 -- quote intervals, not bare percentages.)

Adjudicated exactly as written when the cell lands. Queued behind the validation run
(one-job rule); ~1 h of machine time.

## ADJUDICATED (2026-08-21, exactly as registered)

Result: n=30, divergence 3/30 (10%), dwell median 0.257, flag rate 23%,
catches 0/3, FA among converged 7/27 (26%). Finals: 27 converged trials median
~0.2 deg (many at 0.0-0.1); diverged = {36.5, 61.9, 88.8 deg}.

1. **P1 (dwell decreases): REFUTED -- INVERTED.** Median dwell ROSE 0.150 -> 0.257.
   The geometric intuition was wrong: near the equator B_ECI is nearly FIXED in
   direction, so for an inertial-hold draw sigma is nearly CONSTANT per trial -- the
   dwell distribution goes extreme/bimodal (per-draw all-or-nothing) rather than
   shifting down.
2. **P2 (divergence rises): NOT CONFIRMED.** 10% vs 11% baseline -- flat (n=30,
   binomial sigma ~6%; quote intervals). Section VII's "near-equatorial is the
   difficult case" is NOT supported by rate; what changes is the ROUTE.
3. **P3: TRANSFER FAILURE DECLARED** at the registered line (FA 26% > 10%): flag rate
   23%, recall 0/3 -- all three divergences occurred at dwell ABOVE threshold while
   seven clean trials sat below it. The screen does not transfer to low inclination.
4. **P4 (stated falsification risk): FIRED in its sharpest form.** Low-dwell draws
   became common AND divergence did not rise AND the divergences happened at high
   dwell: at 15 deg inclination the dwell -> divergence link is BROKEN. The link is
   an inclination-conditional (high-inc) result, not a universal one.

PAPER CONSEQUENCE (Section VII + VI-E): the screen is explicitly INCLINATION-SCOPED
-- fit and validated at 97 deg, fails to transfer at 15 deg, printed as a scope
limit ("the screen must be refit per orbit class; its mechanism variable loses
discriminating power where the field direction is quasi-static"). Section VII's
difficulty claim softens to: the low-inclination regime changes the failure
geometry (quasi-static B, per-draw all-or-nothing sigma) rather than raising the
failure rate in this spot-check; converged-draw precision is actually BETTER
(median ~0.2 deg). One honest paragraph, three refuted predictions, all registered.
