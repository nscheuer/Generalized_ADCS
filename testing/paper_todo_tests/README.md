# Paper TODO Tests

This directory contains test infrastructure and validation tests organized by paper TODO items.
These tests verify that the codebase capabilities match what's needed for paper figures and tables.

## Running Tests

```bash
# Run all paper TODO tests
pytest testing/paper_todo_tests/ -v

# Run with pretty output (visible tables and formatted results)
pytest testing/paper_todo_tests/ -v -s

# Run specific TODO category
pytest testing/paper_todo_tests/test_todo_data_computational.py -v -s
```

## Test Files Overview

| File | TODO IDs Covered | Tests | Description |
|------|------------------|-------|-------------|
| `test_todo_data_computational.py` | TODO-DATA-6, DATA-7, SMALLSAT-5, JGCD-4, JGCD-6 | 4 | Timing benchmarks, memory profiling, computational requirements |
| `test_todo_data_desaturation.py` | TODO-DATA-5, DESAT-1 to DESAT-6 | 18 | RW momentum tracking, desaturation gain sweeps |
| `test_todo_data_lp_qp_comparison.py` | TODO-DATA-4 | 14 | LP vs QP allocation comparison, timing |
| `test_todo_data_sensitivity.py` | TODO-DATA-5, DATA-9, JGCD-4 | 3 | Inertia/B-field error sensitivity analysis |
| `test_todo_data_thruster.py` | TODO-DATA-1 | 17 | Thruster model validation, MIB quantization |
| `test_todo_sim_controller_comparison.py` | TODO-SIM-9, DATA-2, BACKGROUND-2 | 6 | Controller comparison tables |
| `test_todo_sim_monte_carlo.py` | TODO-DATA-3, SIM-4, JGCD-10 | 4 | Monte Carlo infrastructure, bootstrap CI |

**Total: 66 tests**

## Adjustable Parameters

Each test file has adjustable parameters at the top for easy modification as paper requirements change:

```python
# Example from test_todo_data_sensitivity.py
INERTIA_ERROR_RANGE = [-20, -10, -5, 0, 5, 10, 20]  # Percent
N_TRIALS_PER_CONDITION = 10
```

Modify these to adjust sweep ranges, number of trials, or tolerances.

## Pretty Output

Tests include a `PrettyOutput` class that displays formatted tables and results during test runs.
Use `-s` flag with pytest to see this output:

```
══════════════════════════════════════════════════════════════════════
  TODO-DATA-7: Allocation Timing Benchmarks
══════════════════════════════════════════════════════════════════════

  Timing Benchmark Results
  ──────────────────────────────────────────────────────────────────────
  │       Component       │  Mean (ms)  │  Std (ms)   │  Min (ms)   │  Max (ms)   │
  ├───────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
  │    LP Allocation      │    0.234    │    0.045    │    0.198    │    0.412    │
  │    QP Allocation      │    0.567    │    0.089    │    0.489    │    0.834    │
  └───────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

## TODO ID Reference

See `research/PAPER_TODO_FEASIBILITY_ANALYSIS.md` for complete TODO list and feasibility analysis.

### Key TODO Categories

- **TODO-DATA-X**: Data generation for figures/tables
- **TODO-SIM-X**: Simulation capabilities
- **TODO-JGCD-X**: JGCD paper specific items
- **TODO-SMALLSAT-X**: SmallSat paper specific items
- **TODO-DESAT-X**: Desaturation analysis items

## Adding New Tests

1. Create a new file: `test_todo_<category>_<topic>.py`
2. Add TODO IDs in docstring
3. Include adjustable parameters at top
4. Add `PrettyOutput` for formatted display
5. Update this README

## Related Files

- `research/PAPER_TODO_FEASIBILITY_ANALYSIS.md` - Full TODO analysis
- `research/THRUSTER_INTEGRATION_ANALYSIS.md` - Thruster integration plan
- `ADCS/satellite_hardware/actuators/thruster.py` - Thruster actuator implementation
