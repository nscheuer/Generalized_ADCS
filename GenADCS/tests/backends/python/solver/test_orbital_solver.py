from GenADCS.backends.python.propagation import KeplerianOrbitalPropagator
from GenADCS.backends.python.solver import OrbitalSolver


def test_orbital_solver_stores_propagator_and_exposes_dt() -> None:
    propagator = KeplerianOrbitalPropagator(12.5)
    solver = OrbitalSolver(propagator)
    assert solver.propagator is propagator
    assert solver.dt == 12.5
