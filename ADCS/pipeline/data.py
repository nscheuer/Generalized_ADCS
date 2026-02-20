"""
Core data structures for the generalized ADCS pipeline.

These dataclasses map directly to the spec documents
(goal_formulation_spec.md, pipeline_spec.md, allocation_spec.md)
for paper traceability.
"""

__all__ = [
    "GoalSpec",
    "WorldVectorSpec",
    "LawInterface",
    "GoalFormulationOutput",
    "CompensationConfig",
    "CompensationInputs",
    "AllocationConfig",
    "AllocationResult",
    "ActuatorGroup",
]

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Goal Formulation
# ---------------------------------------------------------------------------

@dataclass
class WorldVectorSpec:
    """Specification for a world-frame target direction.

    Supports named directions (nadir, sun, etc.), explicit ECI vectors,
    and coordinate targets (lat/lon/alt).
    """
    type: str                                # 'named', 'vector', 'coordinate'
    name: Optional[str] = None               # 'nadir', 'sun', 'bfield', 'ram', ...
    vector: Optional[np.ndarray] = None      # explicit ECI unit vector
    coordinate: Optional[dict] = None        # {lat, lon, alt} or {x, y, z, frame}


@dataclass
class GoalSpec:
    """Pipeline-internal goal representation.

    Populated from legacy Goal objects via an adapter in PipelineController.
    Supports full-attitude (quaternion) and reduced-attitude (body vector +
    world target) goals.
    """
    goal_type: str                                  # 'full', 'reduced', 'none'
    # Full attitude (one populated):
    q_goal: Optional[np.ndarray] = None             # Hamilton, scalar-first
    dcm_goal: Optional[np.ndarray] = None
    # Reduced attitude:
    b_hat: Optional[np.ndarray] = None              # body-frame direction to align
    u_spec: Optional[WorldVectorSpec] = None        # world-frame target spec
    u_hat_eci: Optional[np.ndarray] = None          # resolved world-frame unit vector
    # Angular velocity:
    omega_ref_eci: Optional[np.ndarray] = None      # reference omega in ECI
    # Time-varying body vector (for finite differencing):
    b_hat_next: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Control Law Interface (from pipeline_spec.md)
# ---------------------------------------------------------------------------

@dataclass
class LawInterface:
    """Declares what a control law expects as input and produces as output.

    This struct lets the goal formulation and compensation blocks adapt
    automatically to any control law.
    """
    attitude_type: str = 'full'             # 'full', 'reduced'
    omega_type: str = 'omega_error'         # 'omega_error', 'omega_raw', 'no_omega'
    world_vector_frame: str = 'body'        # 'body', 'world' (reduced laws only)
    quat_convention: str = 'hamilton_scalar_first'
    error_convention: str = 'current_inv_times_goal'
    output_type: str = 'torque'             # 'torque', 'actuator_commands'
    includes_gyroscopic: bool = False
    includes_frame_rotation: bool = False
    includes_disturbance_ff: bool = False
    includes_damping: bool = True


# ---------------------------------------------------------------------------
# Goal Formulation Output
# ---------------------------------------------------------------------------

@dataclass
class GoalFormulationOutput:
    """Output of the goal formulation stage.

    Carries attitude error, omega error/raw, projection matrix, and
    reference angular velocity for use by the control law and
    compensation blocks.
    """
    attitude_output: np.ndarray             # q_err (3-vec) or (b_hat, r_target) tuple
    omega_output: Optional[np.ndarray]      # projected omega error, raw omega, or None
    P: np.ndarray                           # projection matrix [3x3]
    omega_ref_body: np.ndarray              # reference angular velocity in body frame
    goal_type: str                          # 'full', 'reduced', 'none'
    inject_damping: bool = False            # True when law has no omega input


# ---------------------------------------------------------------------------
# Compensation (from pipeline_spec.md)
# ---------------------------------------------------------------------------

@dataclass
class CompensationConfig:
    """Toggle flags for compensation terms.

    These auto-configure from LawInterface flags: if the law already
    includes gyroscopic compensation internally, the pipeline skips it.
    """
    enable_gyroscopic: bool = True
    enable_frame_rotation: bool = False
    enable_disturbance_ff: bool = False
    enable_damping_injection: bool = False
    damping_gain: float = 0.0               # k_d for damping injection

    @classmethod
    def from_law_interface(cls, li: LawInterface) -> 'CompensationConfig':
        """Auto-configure compensation from what the law already includes."""
        return cls(
            enable_gyroscopic=not li.includes_gyroscopic,
            enable_frame_rotation=not li.includes_frame_rotation,
            enable_disturbance_ff=not li.includes_disturbance_ff,
            enable_damping_injection=(li.omega_type == 'no_omega' and li.includes_damping),
        )


@dataclass
class CompensationInputs:
    """Data passed from goal formulation to the compensation block."""
    P: np.ndarray                           # projection matrix
    omega_ref_body: np.ndarray              # reference angular velocity, body frame
    goal_type: str                          # 'full', 'reduced', 'none'
    inject_damping: bool = False


# ---------------------------------------------------------------------------
# Allocation (from allocation_spec.md)
# ---------------------------------------------------------------------------

@dataclass
class ActuatorGroup:
    """A group of actuators of the same type.

    Attributes:
        group_type: 'rw', 'mtq', 'thruster', 'custom'
        axes: [3 x n] matrix of actuator torque axes in body frame
        u_max: [n] vector of maximum command magnitudes
        u_min: [n] vector of minimum command magnitudes (default -u_max)
        indices: indices into the full actuator command vector
    """
    group_type: str
    axes: np.ndarray                        # [3 x n]
    u_max: np.ndarray                       # [n]
    u_min: Optional[np.ndarray] = None      # [n], defaults to -u_max
    indices: Optional[np.ndarray] = None    # indices into full u vector

    def __post_init__(self):
        if self.u_min is None:
            self.u_min = -self.u_max


@dataclass
class AllocationConfig:
    """Configuration for the allocation stage."""
    method: str = 'magnetic_cross'          # 'magnetic_cross', 'lp', 'qp', 'pseudoinverse'
    enable_desaturation: bool = False
    desaturation_strategy: str = 'nullspace'  # 'nullspace', 'weighted', 'scheduled'


@dataclass
class AllocationResult:
    """Output of the allocation stage."""
    u: np.ndarray                           # actuator command vector
    tau_achieved: Optional[np.ndarray] = None  # achieved torque (if computable)
    alpha: float = 1.0                      # scaling factor (1.0 = no saturation)
    direction_error: float = 0.0            # angle between desired and achieved torque
