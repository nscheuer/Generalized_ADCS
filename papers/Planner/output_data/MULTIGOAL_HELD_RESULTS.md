# B1 -- multi-goal redefined 'held' metric (recomputed from saved 100-trial .sim)

**held (NEW)** = longest continuous interval with pointing error < 5° is ≥ 100 s, within each 500 s active window. Also: time-to-acquire (first <5°) and steady-state error (mean over the longest held interval).

| Config | Goal | N | acquired% | **held% (NEW)** | old 'held'% (final<5°) | median t_acquire [s] | median steady-state [°] |
|---|---|---|---|---|---|---|---|
| 3+1 | A | 100 | 96% | **77%** | 10% | 232 | 1.28 |
| 3+1 | B | 100 | 99% | **99%** | 14% | 48 | 0.52 |
| 3+1 | C | 100 | 100% | **100%** | 98% | 37 | 0.29 |
| 3+3 | A | 100 | 100% | **100%** | 22% | 156 | 0.43 |
| 3+3 | B | 100 | 100% | **100%** | 25% | 19 | 0.31 |
| 3+3 | C | 100 | 100% | **100%** | 100% | 23 | 0.17 |

**Takeaway:** under the redefined 'held' (≥100 s continuous <5°), goals A/B jump from the old 10–25% to the values above, correctly crediting the acquire-and-hold-then-anticipate behavior. Time-to-acquire and steady-state error show the hold is fast and tight; the old final-in-window metric only looked bad because the planner had already begun the next slew.
