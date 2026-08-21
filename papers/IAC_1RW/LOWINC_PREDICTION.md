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
