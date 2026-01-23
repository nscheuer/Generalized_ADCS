# Paper TODO Tests

This directory contains test infrastructure and validation tests organized by paper TODO items.
These tests verify that the codebase can generate data needed for paper figures and tables.

**Total Tests: 90**
**TODO Coverage: 42 of ~47 testable TODOs (89%)**

## Quick Start

```bash
# Run all paper TODO tests
pytest testing/paper_todo_tests/ -v

# Run with pretty output (tables, formatted results)
pytest testing/paper_todo_tests/ -v -s

# Quick summary
pytest testing/paper_todo_tests/ -q
```

## Test Files Overview

| File | Tests | Description |
|------|-------|-------------|
| `test_todo_data_computational.py` | 4 | Timing, memory, computational requirements |
| `test_todo_data_desaturation.py` | 18 | Momentum tracking, desaturation analysis |
| `test_todo_data_generation.py` | 7 | CubeSat configs, practitioner metrics |
| `test_todo_data_lp_qp_comparison.py` | 14 | LP vs QP allocation comparison |
| `test_todo_data_sensitivity.py` | 3 | Parameter sensitivity analysis |
| `test_todo_data_thruster.py` | 17 | Thruster model validation |
| `test_todo_fig_generation.py` | 9 | Figure data: polytopes, envelopes, spheres |
| `test_todo_sim_controller_comparison.py` | 6 | Controller comparison tables |
| `test_todo_sim_monte_carlo.py` | 4 | Monte Carlo infrastructure |
| `test_todo_sim_scenarios.py` | 8 | Pointing, tracking, failure response |

## TODO Coverage by Category

| Category | Covered | Total | Coverage |
|----------|---------|-------|----------|
| DATA | 9 | 9 | 100% |
| SIM | 7 | 7 | 100% |
| FIG | 6 | 6 | 100% |
| DAA | 4 | 4 | 100% |
| DESAT | 5 | 6 | 83% |
| SMALLSAT | 4 | 5 | 80% |
| Other | 7 | 10 | 70% |

**Not testable:** ~15 items (proofs, writing tasks)

## Adjustable Parameters

Each test file has parameters at the top for easy modification:

```python
# Example from test_todo_data_sensitivity.py
INERTIA_ERROR_RANGE = [-20, -10, -5, 0, 5, 10, 20]  # Percent
N_TRIALS_PER_CONDITION = 10
PRETTY_OUTPUT = True
```

## Pretty Output Examples

When run with `-s`, tests produce formatted tables:

```
══════════════════════════════════════════════════════════════════════
  TODO-FIG-7: Actuator Failure Response Data
══════════════════════════════════════════════════════════════════════

  ── Failure Response Summary ──
  ┌────────────────────┬────────────┬────────────┬─────────────┐
  │ Scenario           │ Mean α     │ Min α      │ Degradation │
  ├────────────────────┼────────────┼────────────┼─────────────┤
  │ No Failure         │     1.0000 │     1.0000 │        0.0% │
  │ RW-X Fails         │     0.9234 │     0.8567 │        7.7% │
  │ RW-Y Fails         │     0.9156 │     0.8423 │        8.4% │
  │ RW-Z Fails         │     0.9312 │     0.8734 │        6.9% │
  │ MTQ-X Fails        │     0.8945 │     0.7823 │       10.6% │
  └────────────────────┴────────────┴────────────┴─────────────┘
```

## Converting to Paper Figures

Tests generate data structures. To create publication figures:

1. Run test to generate data
2. Export data to CSV/pickle
3. Use matplotlib scripts in `research/paper_figures/`

Example workflow:
```python
# In a script or notebook
import sys
sys.path.append('testing/paper_todo_tests')
from test_todo_fig_generation import TestTorqueEnvelopeData

# Generate data
test = TestTorqueEnvelopeData()
# ... access internal data for plotting
```

## Related Documentation

- `research/TODO_TEST_COVERAGE.md` - Full coverage analysis
- `research/PAPER_TODO_FEASIBILITY_ANALYSIS.md` - TODO feasibility
- `research/THRUSTER_INTEGRATION_ANALYSIS.md` - Thruster integration plan

## Adding New Tests

1. Create file: `test_todo_<category>_<topic>.py`
2. Add TODO IDs in module docstring
3. Include adjustable parameters at top
4. Add `PrettyOutput` class for formatting
5. Update this README and coverage doc
