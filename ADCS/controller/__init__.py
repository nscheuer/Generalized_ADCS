from .controller import Controller
from .bdot import BDot
from .mtq_w_rw import MTQ_w_RW
from .mtq_lovera import MTQ_Lovera
from .mtq_wisniewski import MTQ_Wisniewski
from .mtq_w_rw_LP import MTQ_w_RW_LP
from .mtq_w_rw_QP import MTQ_w_RW_QP
from .mtq_w_rw_QPW import MTQ_w_RW_QPW
from .mtq_w_rw_QPG import MTQ_w_RW_QPG
from .mtq_w_rw_QPC import MTQ_w_RW_QPC
from .plan_and_track_base import PlanAndTrackBase
from .plan_and_track_exact import Plan_and_Track_Exact
from .plan_and_track_lqr import Plan_and_Track_LQR
from .plan_and_track_lqr_disturbed import Plan_and_Track_LQR_Disturbed
from .plan_and_track_mpc import (
    Plan_and_Track_ComputedTorque,
    Plan_and_Track_ComputedTorque_Python,
    Plan_and_Track_MPC,
    Plan_and_Track_MPC_Python,
    MPCParams
)
from .plan_and_track_actualb import (
    Plan_and_Track_ActualB,
    Plan_and_Track_ActualB_Python
)
from .plan_and_track_computed_torque import Plan_and_Track_ComputedTorque2
from .plan_and_track_python_alilqr import Plan_and_Track_PythonALILQR

# C++ TinyMPC controllers (optional - may not be built)
try:
    from .plan_and_track_tinympc_cpp import (
        Plan_and_Track_TinyMPC_Cpp,
        Plan_and_Track_TinyMPC_Cpp_Python
    )
    _HAS_TINYMPC = True
except ImportError:
    _HAS_TINYMPC = False

__all__ = [
    "Controller", "BDot", "MTQ_w_RW", "MTQ_Lovera", "MTQ_Wisniewski", 
    "MTQ_w_RW_LP", "MTQ_w_RW_QP", "MTQ_w_RW_QPW", "MTQ_w_RW_QPG", "MTQ_w_RW_QPC", 
    "PlanAndTrackBase", "Plan_and_Track_Exact", "Plan_and_Track_LQR", 
    "Plan_and_Track_LQR_Disturbed", 
    "Plan_and_Track_ComputedTorque", "Plan_and_Track_ComputedTorque_Python",
    "Plan_and_Track_MPC", "Plan_and_Track_MPC_Python",
    "Plan_and_Track_ActualB", "Plan_and_Track_ActualB_Python",
    "Plan_and_Track_PythonALILQR", "MPCParams"
]

if _HAS_TINYMPC:
    __all__.extend(["Plan_and_Track_TinyMPC_Cpp", "Plan_and_Track_TinyMPC_Cpp_Python"])
