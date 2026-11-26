from GenADCS.backends.python.propagation import (
    ReferenceAttitudePropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.solver import AttitudeSolver


def test_attitude_solver_stores_propagator_and_exposes_dt() -> None:
    rk4 = RK4AttitudePropagator(0.25)
    solver = AttitudeSolver(rk4)
    assert solver.propagator is rk4
    assert solver.dt == 0.25

    ref = ReferenceAttitudePropagator(0.5, "nadir")
    ref_solver = AttitudeSolver(ref)
    assert ref_solver.propagator is ref
    assert ref_solver.dt == 0.5
