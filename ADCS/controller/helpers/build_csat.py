__all__ = ["build_cpp_satellite"]

from ADCS.controller.helpers import PlannerSettings
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, MTQ, RW

# import trajectory_planner.build.tplaunch as tplaunch
# import trajectory_planner.build.pysat as pysat

def build_cpp_satellite(est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> pysat.Satellite:
    csat = pysat.Satellite()
    csat.change_Jcom(est_sat.J_0)

    for act in est_sat.actuators:
        add_actuator(act=act, csat=csat, planner_settings=planner_settings)

    if planner_settings.wmax > 0:
        csat.set_AV_constraint(planner_settings.wmax)
    if planner_settings.sun_limit_angle > 0:
        csat.add_sunpoint_constraint(planner_settings.camera_axis, planner_settings.sun_limit_angle, 0)
    if planner_settings.plan_for_gg:
        csat.add_gg_torq()
    if planner_settings.plan_for_aero:
        csat.add_aero_torq(planner_settings.drag_coeff,planner_settings.coeff_N)
    if planner_settings.plan_for_srp:
        csat.add_srp_torq(planner_settings.srp_coeff,planner_settings.coeff_N)
    if planner_settings.plan_for_resdipole:
        csat.add_resdipole_torq(planner_settings.res_dipole.reshape((3,1)))
    if planner_settings.plan_for_prop:
        csat.add_prop_torq(planner_settings.prop_torque.reshape((3,1)))
    if planner_settings.plan_for_gendist:
        csat.add_gendist_torq(planner_settings.gendist_torq.reshape((3,1)))

    return csat


def add_actuator(act: Actuator, csat: pysat.Satellite, planner_settings: PlannerSettings) -> None:
    if isinstance(act, MTQ):
        csat.add_MTQ(act.axis, act.u_max, planner_settings.mtq_control_weight)
    elif isinstance(act, RW):
        mult = getattr(planner_settings, 'RWh_max_mult', 0.8)
        cost_threshold = act.h_max * mult

        csat.add_RW(
            act.axis, 
            act.J, 
            act.u_max, 
            act.h_max * 0.8,                     # Hard constraint limit (keep as is or adjust)
            planner_settings.rw_control_weight, 
            planner_settings.rw_AM_weight, 
            cost_threshold,                      # <--- FIXED: Now dynamic
            0.0, # planner_settings.rw_stic_weight? 
            act.h_max * 0.9 # Stiction threshold
        )
    else:
        raise ValueError(f"Unknown actuator received: {act.__name__}")