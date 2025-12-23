__all__ = ["PlannerSettings"]

import numpy as np

from ADCS.controller.helpers.planner_subsettings import SolverPassConfig, CostWeights, InitTrajConfig
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance, Prop_Disturbance, General_Disturbance

class PlannerSettings:
    def __init__(
            self, 
            est_sat: EstimatedSatellite, 
            dt_control: float = 1.0,
            pass1_config: SolverPassConfig = None,
            pass2_config: SolverPassConfig = None,
            cost_main: CostWeights = None,
            cost_second: CostWeights = None,
            cost_tvlqr: CostWeights = None,
            init_traj: InitTrajConfig = None,
            dt_tvlqr: float = 1,
            tvlqr_len: float = 1000,
            tvlqr_overlap: float = 1,
            dt_tp: float = None,
            precalculation_time: float = 100,
            traj_overlap: float = 100,
            bdot_on: int = 1,
            debug_plot_on: bool = False,
            include_gg: bool = False,
            include_resdipole: bool = False,
            include_prop: bool = False, 
            include_drag: bool = False, 
            include_srp: bool = False, 
            include_gendist: bool = False
    ) -> None:
        
        self.est_sat = est_sat

        # Physics and Timing
        self.dt_tvlqr = dt_tvlqr
        self.tvlqr_len = tvlqr_len
        self.tvlqr_overlap = tvlqr_overlap
        self.dt_tp = 10*dt_tvlqr if dt_tp is None else dt_tp
        self.precalculation_time = precalculation_time
        self.default_traj_length = tvlqr_len
        self.traj_overlap = traj_overlap
        self.debug_plot_on = debug_plot_on
        self.bdot_on = bdot_on
        self.verbosity = False
        self.eps = 2.22044604925031e-16

        # Solver Configurations
        self.pass1 = pass1_config if pass1_config else SolverPassConfig()

        if pass2_config:
            self.pass2 = pass2_config
        else:
            self.pass2 = SolverPassConfig()
            self.pass2.convergence.max_outer_iter = 14
            self.pass2.convergence.max_inner_iter = 200
            self.pass2.regularization.reg_max = 1e12

        # Cost Configuration
        self.cost_main = cost_main if cost_main else CostWeights()
        self.cost_second = cost_second if cost_second else CostWeights(angle_N=1000.0, ang_vel_N=1.0)
        self.cost_tvlqr = cost_tvlqr if cost_tvlqr else CostWeights(angle=10, ang_vel=10.0, angle_N=100, ang_vel_N=100)

        # Initilization
        self.init_traj = init_traj if init_traj else InitTrajConfig()

        # Hardware Constraints
        self.control_limit_scale = 0.75
        self.umax = self.control_limit_scale * np.array([act.u_max for act in self.est_sat.actuators])
        self.wmax = 0.02
        self.sun_limit_angle = 20 * np.pi/180.0
        self.camera_axis = np.array([[1, 0, 0]]).T

        # Actuator Weights for the C++ model construction
        self.mtq_control_weight = 0.0001
        self.rw_control_weight = 0.001
        self.magic_control_weight = 0.0001
        self.rw_AM_weight = 0.1
        self.rw_stic_weight = 0.01
        self.RWh_max_mult = 2.0
        self.RWh_stiction_mult = 0.05
        self.RWh_ok_mult = 0.4

        # Disturbance Settings
        self.plan_for_aero = include_drag
        self.plan_for_prop = include_prop
        self.plan_for_srp = include_srp
        self.plan_for_gg = include_gg
        self.plan_for_gendist = include_gendist
        self.plan_for_resdipole = include_resdipole

        self.srp_coeff = np.zeros((3,))
        self.drag_coeff = np.zeros((3,))
        self.coeff_N = 0
        self.res_dipole = sum([j.current_torque if isinstance(j, Dipole_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))
        self.prop_torque = sum([j.current_torque if isinstance(j, Prop_Disturbance) else np.zeros(3) for j in est_sat.disturbances], start=np.zeros(3)).reshape((3,))
        self.gendist_torq = np.array([0, 0, 0])
        self.J_est = est_sat.J_0

    def systemSettings(self):
        return (self.J_est, self.dt_tp, self.dt_tvlqr, self.eps, 
                self.tvlqr_len, self.tvlqr_overlap)
    
    def mainAlilqrSettings(self):
        return (
            self.pass1.line_search.to_tuple(),
            self.pass1.aug_lag.to_tuple(),
            self.pass1.convergence.to_tuple(state_len=self.est_sat.state_len),
            self.pass1.regularization.to_tuple()
        )
    
    def secondAlilqrSettings(self):
        return (
            self.pass2.line_search.to_tuple(),
            self.pass2.aug_lag.to_tuple(),
            self.pass2.convergence.to_tuple(state_len=self.est_sat.state_len),
            self.pass2.regularization.to_tuple()
        )

    def initTrajSettings(self):
        return self.init_traj.to_tuple()

    def optMainCostSettings(self):
        return self.cost_main.to_tuple()

    def optSecondCostSettings(self):
        return self.cost_second.to_tuple()

    def optTVLQRCostSettings(self, tracking_LQR_formulation):
        # Note: tracking_LQR_formulation is passed in from ADCS at runtime
        return self.cost_tvlqr.to_tuple(tracking_LQR_formulation)

    def planner_disturbance_settings(self):
        return (
            (self.plan_for_aero, self.plan_for_prop, self.plan_for_srp, 
             self.plan_for_gg, self.plan_for_resdipole, self.plan_for_gendist),
            self.srp_coeff, self.drag_coeff, self.coeff_N, 
            self.prop_torque, self.gendist_torq, self.res_dipole
        )