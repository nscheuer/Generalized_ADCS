"""The implicit default environment is lazy and shared safely across threads."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from threading import Barrier, Lock

import numpy as np
import pytest

import ADCS.environment.environment as environment_module
from ADCS.environment.environment import Environment
from ADCS.orbits.orbital_state import Orbital_State


@pytest.fixture
def isolated_default_environment():
    previous = environment_module._DEFAULT_ENVIRONMENT
    environment_module._DEFAULT_ENVIRONMENT = None
    yield
    current = environment_module._DEFAULT_ENVIRONMENT
    if current is not None and current is not previous:
        current.ephemeris.close()
    environment_module._DEFAULT_ENVIRONMENT = previous


def test_default_environment_is_lazy_and_returns_same_instance(isolated_default_environment):
    assert environment_module._DEFAULT_ENVIRONMENT is None

    first = environment_module.default_environment()
    second = environment_module.default_environment()

    assert isinstance(first, Environment)
    assert second is first


def test_concurrent_first_access_constructs_one_default_environment(
    isolated_default_environment, monkeypatch
):
    real_environment = Environment
    creation_count = 0
    count_lock = Lock()

    def counted_environment():
        nonlocal creation_count
        with count_lock:
            creation_count += 1
        return real_environment()

    monkeypatch.setattr(environment_module, "Environment", counted_environment)
    workers = 24
    start = Barrier(workers)

    def get_default():
        start.wait()
        return environment_module.default_environment()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        environments = list(pool.map(lambda _: get_default(), range(workers)))

    assert creation_count == 1
    assert all(environment is environments[0] for environment in environments)


def test_orbital_states_reuse_default_environment_and_its_models(isolated_default_environment):
    state_args = {
        "J2000": 0.26,
        "R": np.array([6878.0, 0.0, 0.0]),
        "V": np.array([0.0, 7.6, 0.0]),
        "S": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1e-5, 0.0]),
        "rho": 1e-12,
    }

    states = [Orbital_State(**state_args) for _ in range(8)]
    environment = states[0].environment

    assert environment is environment_module._DEFAULT_ENVIRONMENT
    assert all(state.environment is environment for state in states)
    assert all(state.ephem is environment.ephemeris for state in states)
    assert all(state.atmosphere is environment.atmosphere for state in states)


def test_explicit_environment_is_used_without_default_singleton(isolated_default_environment, monkeypatch):
    explicit = Environment()

    def unexpected_default():
        raise AssertionError("explicit environment should bypass the default singleton")

    monkeypatch.setattr(environment_module, "default_environment", unexpected_default)
    state = Orbital_State(
        J2000=0.26,
        R=np.array([6878.0, 0.0, 0.0]),
        V=np.array([0.0, 7.6, 0.0]),
        S=np.array([1.0, 0.0, 0.0]),
        B=np.array([0.0, 1e-5, 0.0]),
        rho=1e-12,
        environment=explicit,
    )

    assert state.environment is explicit
    assert environment_module._DEFAULT_ENVIRONMENT is None
    explicit.ephemeris.close()


def _child_uses_fresh_default_environment(parent_environment, connection):
    child_environment = environment_module.default_environment()
    connection.send(child_environment is not parent_environment)
    child_environment.ephemeris.close()
    connection.close()


def test_forked_process_recreates_its_default_environment(isolated_default_environment):
    parent_environment = environment_module.default_environment()
    context = multiprocessing.get_context("fork")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_uses_fresh_default_environment,
        args=(parent_environment, child_connection),
    )

    process.start()
    child_connection.close()
    assert parent_connection.recv() is True
    process.join(timeout=10)

    assert process.exitcode == 0
