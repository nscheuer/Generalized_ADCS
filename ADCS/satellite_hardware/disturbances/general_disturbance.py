__all__ = ["General_Disturbance"]

from .disturbance import Disturbance

class General_Disturbance(Disturbance):
    """Marker base class for user-defined disturbance models.

    Subclass it and implement ``torque`` (and ``main_param`` plus the
    parameter Jacobians if the disturbance is to be estimated). The class
    itself carries no model: instantiating it directly used to succeed and
    then fail deep inside the dynamics (``torque``) or the augmented filters
    (``main_param``), so it now refuses up front.
    """

    def __init__(self, *args, **kwargs) -> None:
        if type(self) is General_Disturbance:
            raise TypeError(
                "General_Disturbance is an abstract marker base with no torque "
                "model; subclass it and implement torque() (and main_param if "
                "estimate_dist=True) instead of instantiating it directly"
            )
        super().__init__(*args, **kwargs)
