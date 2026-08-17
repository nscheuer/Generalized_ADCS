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
    "DesaturationConfig",
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
    attitude_representation: str = 'quaternion_vector'
        # 'quaternion_vector' : q_e[1:4] vector part (3-vec, default)
        # 'quaternion_full'   : full error quaternion (4-vec)
        # 'mrp'               : Modified Rodrigues Parameters (3-vec)
        # 'cayley'            : Cayley / classical Rodrigues (3-vec)
        # 'dcm'               : Direction Cosine Matrix (3x3)
        # 'euler_321'         : Euler angles 3-2-1 in degrees (3-vec)
        # '2mrp'              : 2x MRP (3-vec)
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
        """Auto-configure compensation from what the law already includes.

        Gyroscopic and frame rotation are auto-enabled (unless the law
        handles them internally).  Disturbance FF is opt-in only because
        it requires model parameters.
        """
        return cls(
            enable_gyroscopic=not li.includes_gyroscopic,
            enable_frame_rotation=not li.includes_frame_rotation,
            enable_disturbance_ff=False,  # opt-in: requires model params
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

    Fields are documented inline below rather than in an ``Attributes:``
    block: autodoc already documents annotated dataclass fields, and doing
    both makes Sphinx emit a duplicate-object warning (fatal under ``-W``).
    """

    #: ``'rw'``, ``'mtq'``, ``'thruster'`` or ``'custom'``
    group_type: str
    #: ``[3 x n]`` matrix of actuator torque axes in the body frame
    axes: np.ndarray
    #: ``[n]`` vector of maximum command magnitudes
    u_max: np.ndarray
    #: ``[n]`` vector of minimum command magnitudes (defaults to ``-u_max``)
    u_min: Optional[np.ndarray] = None
    #: indices into the full actuator command vector
    indices: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.u_min is None:
            self.u_min = -self.u_max


@dataclass
class AllocationConfig:
    """Configuration for the allocation stage.

    Supported methods:
        'magnetic_cross' : Cross-product inversion for MTQ-only (Phase 1)
        'lp'             : Direction-preserving LP (max alpha along tau_hat)
        'qp'             : Bounded least-squares (min ||B_tau u - tau_des||^2)
        'qpw'            : Direction-weighted QP (penalize perp error more)
        'qpc'            : Energy-constrained QP (Lyapunov power gate)
        'pseudoinverse'  : Moore-Penrose pinv + clip
    """
    method: str = 'lp'

    # QP weighting matrix (for 'qp' with regularization)
    W: Optional[np.ndarray] = None          # [3x3] torque error weighting
    lambda_reg: float = 0.0                 # regularization weight on ||u||^2

    # QPW direction-weighting params
    w_parallel: float = 1.0                 # weight on parallel error
    w_perpendicular: float = 100.0          # weight on perpendicular error

    # LP projection fallback
    lp_project_when_infeasible: bool = True

    # Momentum management (Phase 5)
    enable_desaturation: bool = False
    desat_config: Optional['DesaturationConfig'] = None


@dataclass
class DesaturationConfig:
    """Configuration for momentum management / wheel desaturation.

    Fields are documented inline below rather than in an ``Attributes:``
    block, for the same reason as :class:`ActuatorGroup`.
    """

    #: ``'nullspace'`` (zero torque impact, overactuated only),
    #: ``'weighted'`` (augmented QP cost, trades pointing for desat), or
    #: ``'scheduled'`` (add desat torque when MTQ authority is high)
    strategy: str = 'nullspace'
    #: desaturation gain, scaling ``h_rw`` error into a desired dump torque
    k_desat: float = 0.01
    #: target RW momentum in the body frame ``[3]``; defaults to zeros
    h_rw_target: Optional[np.ndarray] = None
    #: weight on the desaturation term in the ``'weighted'`` strategy
    w_desat: float = 1.0
    #: minimum MTQ authority fraction ``[0, 1]`` for ``'scheduled'`` to fire
    authority_threshold: float = 0.3
    #: minimum RW momentum error norm to trigger desaturation
    h_rw_threshold: float = 0.0

    def __post_init__(self):
        if self.h_rw_target is None:
            self.h_rw_target = np.zeros(3)


@dataclass
class AllocationResult:
    """Output of the allocation stage."""
    u: np.ndarray                           # actuator command vector
    tau_achieved: Optional[np.ndarray] = None  # achieved torque (if computable)
    alpha: float = 1.0                      # fraction of desired torque achieved
    direction_error: float = 0.0            # angle between desired and achieved (rad)
    feasible: bool = True                   # True if full desired torque achievable
