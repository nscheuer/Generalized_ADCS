"""Composable plots for simulation results."""

from .plot import (plot, Subplot, AttitudePlot, RatePlot, PositionPlot,
                   VelocityPlot, WheelMomentumPlot, WheelSpeedPlot,
                   SatelliteAttitudeQuaternionPlot, SatelliteAttitudeQXPlot,
                   SatelliteAttitudeQYPlot, SatelliteAttitudeQZPlot,
                   SatelliteAttitudeQWPlot, EulerAnglePlot, MRPPlot,
                   RodriguezPlot, BodyAngularRatePlot,
                   InertialAngularRatePlot, AngularMagnitudePlot,
                   AltitudePlot, LatLonPlot, SunVectorPlot, EclipseStatePlot,
                   EarthVectorPlot)
from .plot import (GyroscopePlot, MagnetometerPlot, StarTrackerPlot,
                   CoarseSunSensorPlot, FineSunSensorPlot, FineEarthSensorPlot,
                   GNSSPositionPlot, GNSSVelocityPlot)
from .plot import AccelerometerPlot
from .plot import (EstimatedAttitudePlot, EstimatedRatePlot,
                   EstimatedAttitudeCovariancePlot, EstimatedRateCovariancePlot)
from .plot import AbsoluteKnowledgeErrorQuaternionPlot, AbsoluteKnowledgeErrorRatePlot
from .plot import AbsolutePointingErrorPlot, AttitudeControlErrorPlot
from .plot import MeasurementInnovationPlot, MeasurementInnovationCovariancePlot
from .plot import NormalizedInnovationSquaredPlot, NormalizedEstimationErrorSquaredPlot
from .plot import MeasurementResidualPlot, KalmanGainPlot
from .plot import (GoalAttitudePlot, GoalRatePlot, FeedforwardTorquePlot,
                   FeedforwardAttitudePlot, FeedforwardRatePlot,
                   ReferenceTorquePlot, CommandedWheelTorquePlot,
                   CommandedMagneticDipolePlot, OBCModePlot, ADCSModePlot)

__all__ = ["plot", "Subplot", "AttitudePlot", "RatePlot", "PositionPlot",
           "VelocityPlot", "WheelMomentumPlot", "WheelSpeedPlot",
           "SatelliteAttitudeQuaternionPlot", "SatelliteAttitudeQXPlot",
           "SatelliteAttitudeQYPlot", "SatelliteAttitudeQZPlot",
           "SatelliteAttitudeQWPlot"]
__all__ += ["EulerAnglePlot", "MRPPlot", "RodriguezPlot"]
__all__ += ["BodyAngularRatePlot", "InertialAngularRatePlot", "AngularMagnitudePlot"]
__all__ += ["AltitudePlot", "LatLonPlot"]
__all__ += ["SunVectorPlot", "EclipseStatePlot", "EarthVectorPlot"]
from .plot import (AtmosphericDensityPlot, RelativeAtmosphericVelocityPlot,
                   MagneticFieldPlot, MagneticFieldMagnitudePlot)
__all__ += ["AtmosphericDensityPlot", "RelativeAtmosphericVelocityPlot",
            "MagneticFieldPlot", "MagneticFieldMagnitudePlot"]
from .plot import (AerodynamicForcePlot, AerodynamicTorquePlot, SRPForcePlot,
                   SRPTorquePlot, LunisolarForcePlot, GravityGradientTorquePlot,
                   JForcePlot, MagneticTorquePlot, ForcePlot, TorquePlot)
__all__ += ["AerodynamicForcePlot", "AerodynamicTorquePlot", "SRPForcePlot",
            "SRPTorquePlot", "LunisolarForcePlot"]
__all__ += ["GravityGradientTorquePlot", "JForcePlot"]
__all__ += ["MagneticTorquePlot"]
__all__ += ["GyroscopePlot", "MagnetometerPlot", "StarTrackerPlot",
            "CoarseSunSensorPlot", "FineSunSensorPlot", "FineEarthSensorPlot",
            "GNSSPositionPlot", "GNSSVelocityPlot"]
__all__ += ["AccelerometerPlot"]
__all__ += ["ForcePlot", "TorquePlot"]
__all__ += ["EstimatedAttitudePlot", "EstimatedRatePlot",
            "EstimatedAttitudeCovariancePlot", "EstimatedRateCovariancePlot"]
__all__ += ["AbsoluteKnowledgeErrorQuaternionPlot", "AbsoluteKnowledgeErrorRatePlot"]
__all__ += ["AbsolutePointingErrorPlot", "AttitudeControlErrorPlot"]
__all__ += ["MeasurementInnovationPlot", "MeasurementInnovationCovariancePlot"]
__all__ += ["NormalizedInnovationSquaredPlot", "NormalizedEstimationErrorSquaredPlot"]
__all__ += ["MeasurementResidualPlot", "KalmanGainPlot"]
__all__ += ["GoalAttitudePlot", "GoalRatePlot", "FeedforwardTorquePlot",
            "FeedforwardAttitudePlot", "FeedforwardRatePlot",
            "ReferenceTorquePlot", "CommandedWheelTorquePlot",
            "CommandedMagneticDipolePlot", "OBCModePlot", "ADCSModePlot"]
