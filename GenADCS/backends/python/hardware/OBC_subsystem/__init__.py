"""On-Board Computer (OBC) subsystem, state machine, and FDIR."""

from .fdir import AngularRateMonitor, FDIR, FDIRMonitor
from .obc_subsystem import OBCSubsystem
from .state_machine import (
    After,
    Completion,
    FDIRTriggered,
    InFOV,
    Mode,
    StateMachine,
    Telecommand,
    Transition,
    Trigger,
)

__all__ = [
    "OBCSubsystem",
    "StateMachine",
    "Mode",
    "Transition",
    "Trigger",
    "Telecommand",
    "InFOV",
    "Completion",
    "After",
    "FDIRTriggered",
    "FDIR",
    "FDIRMonitor",
    "AngularRateMonitor",
]
