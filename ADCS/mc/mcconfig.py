from dataclasses import dataclass
from typing import Optional
import numpy as np

from ADCS.orbits.orbit import Orbit
from ADCS.CONOPS.goals import Goal


@dataclass
class MCConfig:
    r"""
    Configuration container defining run-to-run sampling rules for Monte Carlo ADCS
    simulations.

    Each attribute may be either a fixed value or a callable sampler. If a callable
    is provided, it is invoked once per Monte Carlo run to generate the value applied
    to that run. This class is consumed by :func:`~ADCS.mc.simulate_mc.simulate_mc` to
    perturb initial conditions, goals, timing parameters, and orbit definitions.

    All attributes are optional; unspecified fields fall back to the base values
    passed directly to the Monte Carlo simulation function.

    :param w:
        Initial angular velocity override. If provided, replaces ``State.w``.
    :type w:
        numpy.ndarray or None

    :param q:
        Initial attitude quaternion override. If provided, replaces ``State.q``.
    :type q:
        numpy.ndarray or None

    :param h:
        Initial reaction wheel momentum override. If provided, replaces ``State.h``.
    :type h:
        numpy.ndarray or None

    :param orbit:
        Orbit override for the run. This may be an
        :class:`~ADCS.orbits.orbit.Orbit` instance, an
        :class:`~ADCS.orbits.orbital_state.Orbital_State`, or an equivalent
        dictionary representation, depending on how it is sampled and applied.
    :type orbit:
        :class:`~ADCS.orbits.orbit.Orbit` or None

    :param goal:
        Goal override for the run. If provided, replaces the base goal passed to
        the Monte Carlo simulation.
    :type goal:
        :class:`~ADCS.CONOPS.goals.Goal` or None

    :param dt:
        Simulation time step override in seconds.
    :type dt:
        float or None

    :param tf:
        Simulation duration override in seconds.
    :type tf:
        float or None

    """
    w: Optional[np.ndarray] = None         
    q: Optional[np.ndarray] = None          
    h: Optional[np.ndarray] = None          

    orbit: Optional[Orbit] = None          

    goal: Optional[Goal] = None

    dt: Optional[float] = None
    tf: Optional[float] = None
