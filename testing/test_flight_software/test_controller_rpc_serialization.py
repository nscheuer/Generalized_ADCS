"""
These tests round-trip the encode/decode pairs (goal / orbital-state /
estimated-orbital-state) and assert (a) semantic recovery and (b) payloads
are strictly XML-RPC-safe (only bool/int/float/str/None/list/dict, no
numpy). Round-trips verified sound on origin/main -> PR #37 model
(test-only, locks the HIL wire contract).
"""

import numpy as np
import pytest

import ADCS.remote.controller_rpc as rpc
from ADCS.CONOPS.goals import ECI_Goal, No_Goal
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.estimators.estimator_helpers import EstimatedOrbital_State
from ADCS.state import EstimatedState, State


@pytest.fixture(scope="module")
def os0():
    return Orbital_State(ephem=Ephemeris(), J2000=0.22,
                         R=np.array([7000.0, 12.0, -34.0]),
                         V=np.array([1.1, 7.4, 2.0]))


def _assert_xmlrpc_safe(obj, where=""):
    """xmlrpc.client only marshals bool/int/float/str/None/list/tuple/dict
    (and datetime/bytes). Any numpy type would raise at wire time."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"{where}: non-str dict key {k!r}"
            _assert_xmlrpc_safe(v, where + f".{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_xmlrpc_safe(v, where + f"[{i}]")
    else:
        assert isinstance(obj, (bool, int, float, str, type(None))), (
            f"{where}: payload value {obj!r} of type {type(obj).__name__} "
            f"is not XML-RPC-safe (numpy types break xmlrpc.client)")
        assert type(obj).__module__ == "builtins", (
            f"{where}: {type(obj)} is not a builtin (numpy scalar?)")


def test_goal_payload_roundtrip_and_safe(os0):
    g = ECI_Goal(np.array([1.0, 0.0, 0.0]))
    p = rpc._goal_to_payload(g)
    _assert_xmlrpc_safe(p, "goal")
    g2 = rpc._goal_from_payload(p)
    np.testing.assert_allclose(np.asarray(g.to_ref(os0)[0])[1:4],
                               np.asarray(g2.to_ref(os0)[0])[1:4], atol=1e-12)

    ng = rpc._goal_from_payload(rpc._goal_to_payload(No_Goal()))
    assert isinstance(ng, No_Goal)
    _assert_xmlrpc_safe(rpc._goal_to_payload(None), "goal_none")


def test_orbital_state_payload_roundtrip_and_safe(os0):
    p = rpc._os_to_payload(os0)
    _assert_xmlrpc_safe(p, "os")
    os2 = rpc._os_from_payload(p)
    np.testing.assert_allclose(os2.R, os0.R, atol=1e-9)
    np.testing.assert_allclose(os2.V, os0.V, atol=1e-9)
    assert rpc._os_to_payload(None) is None
    assert rpc._os_from_payload(None) is None


def test_estimated_orbital_state_payload_roundtrip_and_safe(os0):
    P = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Q = np.eye(6) * 1e-3
    eos = EstimatedOrbital_State(os=os0, P=P, Q=Q)
    p = rpc._estimated_orbital_state_to_payload(eos)
    _assert_xmlrpc_safe(p, "eos")
    eos2 = rpc._estimated_orbital_state_from_payload(p)
    np.testing.assert_allclose(eos2.os.R, os0.R, atol=1e-9)
    np.testing.assert_allclose(np.asarray(eos2.P), np.asarray(eos.P), atol=1e-12)
    np.testing.assert_allclose(np.asarray(eos2.Q), np.asarray(eos.Q), atol=1e-12)
    assert rpc._estimated_orbital_state_to_payload(None) is None


@pytest.mark.parametrize(
    "state",
    [
        State(w=[0.1, -0.2, 0.3], q=[1.0, 0.0, 0.0, 0.0], h=[0.4]),
        EstimatedState(
            w=[0.1, -0.2, 0.3],
            q=[1.0, 0.0, 0.0, 0.0],
            h=[0.4],
            act_bias=[0.01],
            sens_bias=[0.02, 0.03],
            dist_param=[0.04],
        ),
    ],
)
def test_state_payload_roundtrip_is_versioned_and_safe(state):
    payload = rpc._state_to_payload(state)
    _assert_xmlrpc_safe(payload, "state")
    assert payload["schema_version"] == 1
    restored = rpc._state_from_payload(payload)
    assert type(restored) is type(state)
    np.testing.assert_allclose(restored.as_array(), state.as_array())
    if isinstance(state, EstimatedState):
        np.testing.assert_allclose(restored.as_estimator_array(), state.as_estimator_array())
