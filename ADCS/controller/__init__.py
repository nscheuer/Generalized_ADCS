import importlib
from typing import Any

__all__ = [
    "Controller",
    "BDot",
    "MTQ_w_RW",
    "MTQ_Lovera",
    "MTQ_Wisniewski",
    "MTQ_w_RW_LP",
    "MTQ_w_RW_QP",
    "MTQ_w_RW_QPW",
    "MTQ_w_RW_QPG",
    "MTQ_w_RW_QPC",
    "PlanAndTrackBase",
    "Plan_and_Track_Exact",
    "Plan_and_Track_LQR",
    "Plan_and_Track_LQR_Disturbed",
    "SALTRO",
    "helpers",
]

_SYMBOL_TO_MODULE = {
    "Controller": ".controller",
    "BDot": ".bdot",
    "MTQ_w_RW": ".mtq_w_rw",
    "MTQ_Lovera": ".mtq_lovera",
    "MTQ_Wisniewski": ".mtq_wisniewski",
    "MTQ_w_RW_LP": ".mtq_w_rw_LP",
    "MTQ_w_RW_QP": ".mtq_w_rw_QP",
    "MTQ_w_RW_QPW": ".mtq_w_rw_QPW",
    "MTQ_w_RW_QPG": ".mtq_w_rw_QPG",
    "MTQ_w_RW_QPC": ".mtq_w_rw_QPC",
    "PlanAndTrackBase": ".plan_and_track_base",
    "Plan_and_Track_Exact": ".plan_and_track_exact",
    "Plan_and_Track_LQR": ".plan_and_track_lqr",
    "Plan_and_Track_LQR_Disturbed": ".plan_and_track_lqr_disturbed",
    "SALTRO": ".saltro",
    "helpers": ".helpers",
}


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_SYMBOL_TO_MODULE[name], __name__)
    value = module if name == "helpers" else getattr(module, name)
    globals()[name] = value
    return value
