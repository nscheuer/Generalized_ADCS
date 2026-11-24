from ._satellite import Satellite as _Satellite
from .ADCS_subsystem import ADCSSubsystem
from GenADCS.backends.python.hardware.state_machine import StateMachine


class Satellite(_Satellite):
    """Python-compatible satellite wrapper around the native C++ value type."""

    def __init__(self, geometry, adcs=None, state_machine=None):
        if adcs is None:
            adcs = ADCSSubsystem()
        if state_machine is None:
            state_machine = StateMachine()
        super().__init__(geometry, state_machine.mode)
        self._geometry = geometry
        self.adcs = adcs
        self.state_machine = state_machine

    @property
    def geometry(self):
        return self._geometry


__all__ = ["Satellite"]
