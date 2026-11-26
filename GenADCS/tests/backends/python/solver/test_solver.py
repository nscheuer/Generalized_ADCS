import math
import pytest

from GenADCS.backends.python.config import DynamicsConfig
from GenADCS.backends.python.propagation import (
    CoupledRK4Propagator,
    KeplerianOrbitalPropagator,
    RK4AttitudePropagator,
)
from GenADCS.backends.python.solver import (
    AttitudeSolver,
    CoupledSolver,
    OrbitalSolver,
    Solver,
)


def test_split_and_coupled_solver_construction() -> None:
    orbital = OrbitalSolver(KeplerianOrbitalPropagator(0.5))
    attitude = AttitudeSolver(RK4AttitudePropagator(0.5))
    split = Solver(dynamics=DynamicsConfig(), orbital=orbital, attitude=attitude, tf=10.0)
    assert split.orbital is orbital
    assert split.attitude is attitude
    assert split.coupled is None
    assert split.tf == 10.0

    coupled_inner = CoupledSolver(CoupledRK4Propagator(0.5), tf=10.0)
    coupled = Solver(dynamics=DynamicsConfig(), coupled=coupled_inner)
    assert coupled.coupled is coupled_inner
    assert coupled.orbital is None
    assert coupled.attitude is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"orbital": OrbitalSolver(KeplerianOrbitalPropagator(0.5)), "tf": 1.0},
        {"attitude": AttitudeSolver(RK4AttitudePropagator(0.5)), "tf": 1.0},
        {
            "orbital": OrbitalSolver(KeplerianOrbitalPropagator(0.5)),
            "attitude": AttitudeSolver(RK4AttitudePropagator(0.5)),
            "coupled": CoupledSolver(CoupledRK4Propagator(0.5), tf=1.0),
        },
        {
            "orbital": OrbitalSolver(KeplerianOrbitalPropagator(0.5)),
            "coupled": CoupledSolver(CoupledRK4Propagator(0.5), tf=1.0),
        },
        {
            "attitude": AttitudeSolver(RK4AttitudePropagator(0.5)),
            "coupled": CoupledSolver(CoupledRK4Propagator(0.5), tf=1.0),
        },
    ],
)
def test_solver_rejects_ambiguous_or_incomplete_composition(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="coupled solver or both"):
        Solver(**kwargs)


def test_split_solver_validates_tf_dt_and_chunk_steps() -> None:
    orbital = OrbitalSolver(KeplerianOrbitalPropagator(0.5))
    attitude = AttitudeSolver(RK4AttitudePropagator(0.5))

    with pytest.raises(ValueError, match="nonnegative tf"):
        Solver(orbital=orbital, attitude=attitude, tf=-1.0)
    with pytest.raises(ValueError, match="nonnegative tf"):
        Solver(orbital=orbital, attitude=attitude, tf=math.inf)
    with pytest.raises(ValueError, match="nonnegative tf"):
        Solver(orbital=orbital, attitude=attitude, tf=None)
    with pytest.raises(ValueError, match="matching dt"):
        Solver(
            orbital=OrbitalSolver(KeplerianOrbitalPropagator(1.0)),
            attitude=attitude,
            tf=2.0,
        )
    with pytest.raises(ValueError, match="positive integer"):
        Solver(orbital=orbital, attitude=attitude, tf=2.0, progress_chunk_steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        Solver(orbital=orbital, attitude=attitude, tf=2.0, progress_chunk_steps=True)
