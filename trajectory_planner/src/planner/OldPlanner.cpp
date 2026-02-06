/**
 * @file OldPlanner.cpp
 * @brief ALTRO (Augmented Lagrangian TRajectory Optimizer) for spacecraft attitude control
 *
 * This file implements an Augmented Lagrangian iLQR (AL-iLQR) trajectory optimizer for
 * spacecraft attitude maneuvers. The algorithm solves constrained optimal control problems
 * by combining iterative Linear Quadratic Regulator (iLQR) with augmented Lagrangian
 * constraint handling.
 *
 * Algorithm Overview:
 * -------------------
 * 1. OUTER LOOP (Augmented Lagrangian): Updates constraint penalties and Lagrange multipliers
 *    - Penalty parameter mu scales quadratic constraint violation costs
 *    - Lagrange multipliers lambda provide first-order constraint gradient information
 *    - Penalty increases when constraints not satisfied, driving solution toward feasibility
 *
 * 2. INNER LOOP (iLQR): Solves the unconstrained subproblem via dynamic programming
 *    - BACKWARD PASS: Computes optimal feedback gains K and feedforward terms d
 *      by solving Riccati-like equations from final time to initial time
 *    - FORWARD PASS: Rolls out new trajectory using computed gains with line search
 *      to ensure cost decrease
 *
 * Key Components:
 * ---------------
 * - Satellite dynamics: Quaternion attitude + angular velocity with actuator models
 *   (magnetorquers, reaction wheels, magic actuators)
 * - Cost function: Quadratic costs on attitude error, angular velocity, and control effort
 * - Constraints: Control limits, angular velocity limits, sun-pointing keepout zones
 * - Regularization: Levenberg-Marquardt style regularization for numerical stability
 *
 * References:
 * -----------
 * - Howell & Jackson, "ALTRO: A Fast Solver for Constrained Trajectory Optimization"
 * - Tassa et al., "Synthesis and Stabilization of Complex Behaviors through Online
 *   Trajectory Optimization"
 */

// #include "PlannerUtil.hpp"
#include "OldPlanner.hpp"
// #include "PlannerUtil.hpp"
// #include "PlannerPython.hpp"
// #include "PlannerUtilPy.hpp"
#include "../ArmaNumpy.hpp"
// #include <stdexcept>
// #include <typeinfo>
// #include <pybind11/numpy.h>
// #define earth_mu 3.986e14
#define EIGS_NORM "inf"
#define EIGS_MULT 1.0 //(1.0/3.0)
#define EIGS_POW  0.0
#define RAND_MAX_INIT 1000.0



namespace py = pybind11;
using namespace arma;
using namespace std;

OldPlanner::OldPlanner(){
}


OldPlanner::OldPlanner(Satellite sat_in,ALL_SETTINGS_FORM allSettings){
  sat = sat_in;
  OldPlanner::updateParameters_notsat(get<0>(allSettings),get<1>(allSettings),get<2>(allSettings),get<3>(allSettings),get<4>(allSettings),get<5>(allSettings),get<6>(allSettings));

}
 ALL_SETTINGS_FORM OldPlanner::readParameters() {
  return std::make_tuple( systemSettings,  alilqrSettings,  alilqrSettings2,  initialTrajSettings,  costSettings, costSettings2, costSettings_tvlqr_full);
}
void OldPlanner::setVerbosity(bool verbosity) {
  //REMOVE BEFORE FLIGHT
  verbose = verbosity;
}
void OldPlanner::updateParameters_notsat(SYSTEM_SETTINGS_FORM systemSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings2_tmp, INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp) {
    // verbose flag is controlled by setVerbosity(), don't override here

    costSettings = costSettings_tmp;
    costSettings2 = costSettings2_tmp;
    // costSettings_tvlqr = costSettings_tvlqr_tmp;
    costSettings_tvlqr_full = costSettings_tvlqr_tmp;

    initialTrajSettings = initialTrajSettings_tmp;
    bdotgain = get<0>(initialTrajSettings); // gain used in generation of trajectories for bdot
    HLangleLimit = get<1>(initialTrajSettings); //dividing line between using high settings or low settings in trajectory generation. in radians!
    highSettings = get<2>(initialTrajSettings);
    lowSettings = get<3>(initialTrajSettings);

    gyrogainH = get<0>(highSettings); //unused, to offset wxJw term in dynamics
    dampgainH = get<1>(highSettings); //used in initial trajectory generation to damp J*w
    velgainH = get<2>(highSettings); //used in initial trajectory generation to  counter J*(w-w_goal)
    quatgainH = get<3>(highSettings); //used in initial trajectory generation to  counter pointing error
    randvalH = get<4>(highSettings); //level of random noise to add to initial trajectory generation
    umaxmultH = get<5>(highSettings); //how much to reduce/increase maximum control levels for trajectory generation.

    gyrogainL = get<0>(lowSettings); //see above but for low angle settings
    dampgainL = get<1>(lowSettings);
    velgainL = get<2>(lowSettings);
    quatgainL = get<3>(lowSettings);
    randvalL = get<4>(lowSettings);
    umaxmultL = get<5>(lowSettings);

    angle_weight = get<0>(costSettings); //cost of angle error
    angvel_weight = get<1>(costSettings); //cost of av error
    u_weight = get<2>(costSettings); //cost of actuation.
    av_with_mag_weight = get<3>(costSettings); //cost of angvel alignment with magnetic field
    ang_av_weight = get<4>(costSettings); //cost of orientation error that aligns with av error.
    angle_weight_N = get<5>(costSettings); // cost of angle error in final timestep.
    angvel_weight_N = get<6>(costSettings);// cost of av error in final timestep.
    av_with_mag_weight_N = get<7>(costSettings); // cost of av aligned with B field error in final timestep.
    ang_av_weight_N = get<8>(costSettings); // cost of av/ang error alignment in final timestep.
    whichAngCostFunc = get<9>(costSettings); //determines from a variety of options how angle cost is calculated. 0-3 for vector angle, 0-4 for quaternion.
    useRawControlCost = get<10>(costSettings);//if true (1), control cost is 0.5*u.T@W@u. if false (0), control cost is 0.5*(u-u_prev).T@W@(u-u_prev)
    useFullCostHess = get<11>(costSettings); // 0 = Gauss-Newton (PSD), 1 = Full Newton

    angle_weight2 = get<0>(costSettings2);
    angvel_weight2 = get<1>(costSettings2);
    u_weight2 = get<2>(costSettings2);
    av_with_mag_weight2 = get<3>(costSettings2);
    ang_av_weight2 = get<4>(costSettings2);
    angle_weight_N2 = get<5>(costSettings2);
    angvel_weight_N2 = get<6>(costSettings2);
    av_with_mag_weight_N2 = get<7>(costSettings2);
    ang_av_weight_N2 = get<8>(costSettings2);
    whichAngCostFunc2 = get<9>(costSettings2);
    useRawControlCost2 = get<10>(costSettings2);
    useFullCostHess2 = get<11>(costSettings2);

    angle_weight_tvlqr = get<0>(costSettings_tvlqr_tmp);
    angvel_weight_tvlqr = get<1>(costSettings_tvlqr_tmp);
    u_weight_tvlqr = get<2>(costSettings_tvlqr_tmp);
    av_with_mag_weight_tvlqr = get<3>(costSettings_tvlqr_tmp);
    ang_av_weight_tvlqr = get<4>(costSettings_tvlqr_tmp);
    angle_weight_N_tvlqr = get<5>(costSettings_tvlqr_tmp);
    angvel_weight_N_tvlqr = get<6>(costSettings_tvlqr_tmp);
    av_with_mag_weight_N_tvlqr = get<7>(costSettings_tvlqr_tmp);
    ang_av_weight_N_tvlqr = get<8>(costSettings_tvlqr_tmp);
    whichAngCostFunc_tvlqr = get<9>(costSettings_tvlqr_tmp);
    useRawControlCost_tvlqr = get<10>(costSettings_tvlqr_tmp);
    tracking_LQR_formulation = get<11>(costSettings_tvlqr_tmp); //mode for LQR calculation.

    costSettings_tvlqr = make_tuple(angle_weight_tvlqr,angvel_weight_tvlqr,u_weight_tvlqr,
                    av_with_mag_weight_tvlqr,ang_av_weight_tvlqr,
                    angle_weight_N_tvlqr,angvel_weight_N_tvlqr,av_with_mag_weight_N_tvlqr,
                    ang_av_weight_N_tvlqr,whichAngCostFunc_tvlqr,useRawControlCost_tvlqr,0);

    // systemSettings = systemSettings_tmp;
    dt = get<1>(systemSettings_tmp);
    dt_tvlqr = get<2>(systemSettings_tmp);
    eps = get<3>(systemSettings_tmp);
    tvlqr_len = get<4>(systemSettings_tmp);//lenght of TVLQR segemnts
    tvlqr_overlap = get<5>(systemSettings_tmp); //overlap between TVLQR segments
    systemSettings = make_tuple(sat.Jcom,dt,dt_tvlqr,eps,tvlqr_len,tvlqr_overlap);

    alilqrSettings = alilqrSettings_tmp;
    lineSearchSettings = get<0>(alilqrSettings_tmp);
    auglagSettings = get<1>(alilqrSettings_tmp);
    breakSettings = get<2>(alilqrSettings_tmp);
    regSettings = get<3>(alilqrSettings_tmp);

    maxLsIter = get<0>(lineSearchSettings); //max iterations on linesearch inside ilqr
    beta1 = get<1>(lineSearchSettings); //limits on how much improvement to expect in ilqr.
    beta2 = get<2>(lineSearchSettings);

    lagMultInit = get<0>(auglagSettings); //starting value of lagrange multipliers on constraints
    lagMultMax = get<1>(auglagSettings);//max value of lagrange multipliers on constraints
    penInit = get<2>(auglagSettings); //initial penalty value
    penMax = get<3>(auglagSettings); //maximum penalty value
    penScale = get<4>(auglagSettings); //scaling rate of penalty

    maxOuterIter = get<0>(breakSettings);
    maxIlqrIter = get<1>(breakSettings);
    maxIter = get<2>(breakSettings);
    gradTol = get<3>(breakSettings);
    ilqrCostTol = get<4>(breakSettings);
    costTol = get<5>(breakSettings);
    zCountLim = get<6>(breakSettings);
    cmax = get<7>(breakSettings); //currently unused.
    maxCost = get<8>(breakSettings);
    xmax = (get<9>(breakSettings));
    // breakSettings = make_tuple(maxOuterIter,maxIlqrIter,maxIter,gradTol,ilqrCostTol,costTol,zCountLim,cmax,maxCost,xmax);

    regInit = get<0>(regSettings);
    regMin = get<1>(regSettings);
    regMax = get<2>(regSettings);
    regScale = get<3>(regSettings);
    regBump = get<4>(regSettings);
    regMinConds = get<5>(regSettings);
    regBumpRandAddRatio = get<6>(regSettings);
    useDynamicsHess = get<7>(regSettings);
    useConstraintHess = get<8>(regSettings);

    // alilqrSettings = make_tuple(lineSearchSettings,auglagSettings,breakSettings,regSettings);

    alilqrSettings2 = alilqrSettings2_tmp;
    lineSearchSettings2 = get<0>(alilqrSettings2_tmp);
    auglagSettings2 = get<1>(alilqrSettings2_tmp);
    breakSettings2 = get<2>(alilqrSettings2_tmp);
    regSettings2 = get<3>(alilqrSettings2_tmp);

    maxLsIter2 = get<0>(lineSearchSettings2);
    beta12 = get<1>(lineSearchSettings2);
    beta22 = get<2>(lineSearchSettings2);

    lagMultInit2 = get<0>(auglagSettings2);
    lagMultMax2 = get<1>(auglagSettings2);
    penInit2 = get<2>(auglagSettings2);
    penMax2 = get<3>(auglagSettings2);
    penScale2 = get<4>(auglagSettings2);

    maxOuterIter2 = get<0>(breakSettings2);
    maxIlqrIter2 = get<1>(breakSettings2);
    maxIter2 = get<2>(breakSettings2);
    gradTol2 = get<3>(breakSettings2);
    ilqrCostTol2 = get<4>(breakSettings2);
    costTol2 = get<5>(breakSettings2);
    zCountLim2 = get<6>(breakSettings2);
    cmax2 = get<7>(breakSettings2);
    maxCost2 = get<8>(breakSettings2);
    xmax2 = (get<9>(breakSettings2));
    // breakSettings2 = make_tuple(maxOuterIter2,maxIlqrIter2,maxIter2,gradTol2,ilqrCostTol2,costTol2,zCountLim2,cmax2,maxCost2,xmax2);

    regInit2 = get<0>(regSettings2);
    regMin2 = get<1>(regSettings2);
    regMax2 = get<2>(regSettings2);
    regScale2 = get<3>(regSettings2);
    regBump2 = get<4>(regSettings2);
    regBumpRandAddRatio2 = get<6>(regSettings2);
    useDynamicsHess2 = get<7>(regSettings2);
    useConstraintHess2 = get<8>(regSettings2);
}

BEFORE_OUTPUT_FORM OldPlanner::trajOptBefore(VECTOR_INFO_FORM vecs_w_time,double dt_use, TIME_FORM time_start, TIME_FORM time_end, vec x0, int bdotOn)
{
  x0 = sat.state_norm(x0);
  // x0.rows(3, 6) = normalise(x0.rows(3,6));
  //double dt = this->dt;//readJsonDouble(trajOptSettingsFile, "dt");
  // double dt_readin = (t(1)-t(0))*36525.0*24.0*3600.0;
  // vec t = get<0>(vecs_w_time);

  VECTOR_INFO_FORM vecs = findVecTimes(vecs_w_time,dt_use/(36525.0*24.0*3600.0),time_start,time_end);
  // vec times = get<0>(vecs);
  mat dt_timevec = get<0>(vecs);

  int traj_length = dt_timevec.n_elem;
  // VECTOR_INFO_FORM vecs = get<1>(time_tuple);
  COST_SETTINGS_FORM costSettings_tmp = this->costSettings;

//  double penInit = this->penInit;//readJsonDouble(trajOptSettingsFile, "penInit");
  //double penScale = this->penScale;//readJsonDouble(trajOptSettingsFile, "penScale");
  mat ECIvec = get<6>(vecs);
  // if(verbose) {
  // }
  mat satvec = get<5>(vecs);

  mat nBset = normalise(get<3>(vecs));
  //

  double mu0 = 0.0;//penInit*pow(penScale,1);

  mat U = mat(sat.control_N(),traj_length).zeros();
  U.fill(datum::nan);
  mat X = mat(sat.state_N(),traj_length).fill(datum::nan);
  mat TQ = mat(3,traj_length).fill(datum::nan);
  TRAJECTORY_FORM traj;

  if(verbose){cout<<"setting bdotOn="<<bdotOn<<endl;}
  if(bdotOn==0 || sat.number_MTQ<3 || bdotOn>5)
  {
    if(verbose)
    {
      cout<<"bdotOn is false, generating random initial trajectory!";
    }
    vec umax = join_cols(vec(sat.MTQ_max),0.1*vec(sat.RW_max_torq),0.1*vec(sat.magic_max_torq));
    if(verbose){cout << "UMAX" << umax << "\n";}
    U = diagmat(umax)*randn(size(U))/RAND_MAX_INIT;
    if(verbose)
    {
      cout<<U;
    }
     traj = OldPlanner::generateInitialTrajectory(dt_use,x0, U, vecs);
     assert(approx_equal(get<1>(traj),U,"abstol",1e-10));
     X = get<0>(traj);
  }
  else if(bdotOn==4)
  {
    // Mode 4: PD control for initial trajectory
    // Works well for fixed quaternion goals where smartbdot fails
    if(verbose){cout<<"PD control initialization\n";}
    
    mat Bset = get<3>(vecs);
    mat Rset = get<1>(vecs);
    mat Vset = get<2>(vecs);
    mat Sset = get<4>(vecs);
    mat ECIvec = get<6>(vecs);
    vec pset = get<7>(vecs);
    vec t = get<0>(vecs);
    
    mat Xset = mat(sat.state_N(), traj_length);
    mat Uset = mat(sat.control_N(), traj_length).zeros();
    mat TQset = mat(3, traj_length).zeros();
    
    vec xk = sat.state_norm(x0);
    Xset.col(0) = xk;
    
    // umax in optimizer units: MTQ physical, RW/magic scaled by 1/NONMTQ_TORQ_SCALE
    vec umax = join_cols(vec(sat.MTQ_max), 
                         vec(sat.RW_max_torq) / NONMTQ_TORQ_SCALE, 
                         vec(sat.magic_max_torq) / NONMTQ_TORQ_SCALE);
    
    // PD gains - scaled to MTQ torque capability
    // Max MTQ torque ≈ m_max * B ≈ 0.15 * 30e-6 = 4.5e-6 N*m
    // For 180° slew (qerr_vec norm ~1), we want tau ≈ Kp * J * 1
    // To get tau ≈ 30% of max: Kp ≈ 0.3 * tau_max / J_avg
    double J_trace = trace(sat.Jcom);
    double B_typical = 30e-6;  // 30 uT typical field
    double tau_max_approx = norm(vec(sat.MTQ_max)) * B_typical;
    double Kp = 0.3 * tau_max_approx / (J_trace / 3.0);  // ~30% saturation at 180°
    double Kd = 10.0 * Kp;  // Derivative gain for damping
    
    for(int k = 0; k < traj_length - 1; k++) {
      vec4 qk = normalise(xk.rows(3, 6));
      vec3 wk = xk.head(3);
      mat33 RmatT = rotMat(qk).t();
      vec3 Bk = Bset.col(k);
      vec3 Bbody = RmatT * Bk;
      double nB2 = dot(Bk, Bk);
      
      // Get goal quaternion
      vec ek = ECIvec.col(k);
      vec4 qgoal;
      if((ek.n_elem == 3) || ((ek.n_elem == 4) && (isnan(ek(0))))) {
        // Vector goal - skip PD, use small random
        Uset.col(k) = 0.01 * umax % (2*randu(sat.control_N()) - 1);
      } else {
        // Quaternion goal
        qgoal = normalise(ek);
        
        // Compute quaternion error: qerr = qgoal * qk^-1
        vec4 qerr = normquaterr(qgoal, qk);
        
        // Ensure shortest path rotation
        if(qerr(0) < 0) {
          qerr = -qerr;
        }
        
        // Desired torque: tau = Kp * J * qerr_vec - Kd * J * omega
        vec3 qerr_vec = qerr.rows(1, 3);
        vec3 tau_des = Kp * sat.Jcom * qerr_vec - Kd * sat.Jcom * wk;
        
        // Convert desired torque to MTQ dipole: m = (B x tau) / |B|^2
        vec3 m_des = cross(Bbody, tau_des) / nB2;
        
        // Map to MTQ axes and saturate
        vec uk = vec(sat.control_N()).zeros();
        uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t() * m_des;
        
        // ========================================
        // ADD RW CONTROL: Use RW for direct torque on RW axis
        // This enables the optimizer to see RW benefits from initial traj
        // ========================================
        if(sat.number_RW > 0) {
          // Project desired torque onto each RW axis
          for(int j = 0; j < sat.number_RW; j++) {
            vec3 rw_axis = sat.RW_axes.at(j);
            double tau_rw_des = dot(rw_axis, tau_des);
            // Use a fraction of what RW can provide (leave room for optimization)
            double rw_torq = 0.5 * std::max(-sat.RW_max_torq.at(j), std::min(tau_rw_des, sat.RW_max_torq.at(j)));
            // Store in optimizer units (scaled by NONMTQ_TORQ_SCALE)
            uk(sat.number_MTQ + j) = rw_torq / NONMTQ_TORQ_SCALE;
          }
        }
        
        // Saturate (recompute with RW included)
        double ur = max(abs(uk / umax));
        if(ur > 1.0) {
          uk = uk / ur;
        }
        
        Uset.col(k) = uk;
      }
      
      // Propagate dynamics
      DYNAMICS_INFO_FORM dyn_kn1 = make_tuple(Bset.col(k), Rset.col(k), pset(k), Vset.col(k), Sset.col(k), 0);
      DYNAMICS_INFO_FORM dyn_k = make_tuple(Bset.col(k+1), Rset.col(k+1), pset(k+1), Vset.col(k+1), Sset.col(k+1), 0);
      
      tuple<vec, vec> dynout = rk4z(dt_use, xk, Uset.col(k), sat, dyn_kn1, dyn_k);
      xk = sat.state_norm(get<0>(dynout));
      Xset.col(k+1) = xk;
    }
    
    traj = make_tuple(Xset, Uset, t, TQset);
    U.cols(0, traj_length-1) = Uset;
    X = Xset;
    TQ = TQset;
    
    if(verbose){
      cout<<"PD init complete, |U| = "<<norm(Uset, "fro")<<"\n";
    }
  }
  else if(bdotOn==5)
  {
    // Mode 5: PD control + small random noise for exploration
    // Noise is scaled by actuator strength, inertia, and timestep
    if(verbose){cout<<"PD + noise control initialization\n";}
    
    mat Bset = get<3>(vecs);
    mat Rset = get<1>(vecs);
    mat Vset = get<2>(vecs);
    mat Sset = get<4>(vecs);
    mat ECIvec = get<6>(vecs);
    vec pset = get<7>(vecs);
    vec t = get<0>(vecs);
    
    mat Xset = mat(sat.state_N(), traj_length);
    mat Uset = mat(sat.control_N(), traj_length).zeros();
    mat TQset = mat(3, traj_length).zeros();
    
    vec xk = sat.state_norm(x0);
    Xset.col(0) = xk;
    
    // umax in optimizer units: MTQ physical, RW/magic scaled by 1/NONMTQ_TORQ_SCALE
    vec umax = join_cols(vec(sat.MTQ_max), 
                         vec(sat.RW_max_torq) / NONMTQ_TORQ_SCALE, 
                         vec(sat.magic_max_torq) / NONMTQ_TORQ_SCALE);
    
    // PD gains - scaled to MTQ torque capability
    double J_trace = trace(sat.Jcom);
    double B_typical = 30e-6;
    double tau_max_approx = norm(vec(sat.MTQ_max)) * B_typical;
    double Kp = 0.3 * tau_max_approx / (J_trace / 3.0);
    double Kd = 10.0 * Kp;
    
    // Noise scaling:
    // - Base: 5% of actuator max
    // - Scale by sqrt(min_J / trace(J)) to account for inertia
    // - Scale by sqrt(10 / dt) to prevent crazy spins with large timesteps
    double J_min = min(eig_sym(sat.Jcom));
    double inertia_scale = sqrt(J_min / (J_trace / 3.0));
    double dt_scale = sqrt(std::min(10.0, 10.0 / dt_use));
    double noise_frac = 0.05 * inertia_scale * dt_scale;
    
    if(verbose){
      cout<<"Noise scaling: inertia="<<inertia_scale<<", dt="<<dt_scale<<", frac="<<noise_frac<<"\n";
    }
    
    for(int k = 0; k < traj_length - 1; k++) {
      vec4 qk = normalise(xk.rows(3, 6));
      vec3 wk = xk.head(3);
      mat33 RmatT = rotMat(qk).t();
      vec3 Bk = Bset.col(k);
      vec3 Bbody = RmatT * Bk;
      double nB2 = dot(Bk, Bk);
      
      vec ek = ECIvec.col(k);
      vec4 qgoal;
      vec uk = vec(sat.control_N()).zeros();
      
      if((ek.n_elem == 3) || ((ek.n_elem == 4) && (isnan(ek(0))))) {
        // Vector goal - use small random
        uk = noise_frac * umax % (2*randu(sat.control_N()) - 1);
      } else {
        // Quaternion goal - PD control
        qgoal = normalise(ek);
        vec4 qerr = normquaterr(qgoal, qk);
        if(qerr(0) < 0) {
          qerr = -qerr;
        }
        
        vec3 qerr_vec = qerr.rows(1, 3);
        vec3 tau_des = Kp * sat.Jcom * qerr_vec - Kd * sat.Jcom * wk;
        vec3 m_des = cross(Bbody, tau_des) / nB2;
        
        uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t() * m_des;
        
        // Add RW control for direct axis torque
        if(sat.number_RW > 0) {
          for(int j = 0; j < sat.number_RW; j++) {
            vec3 rw_axis = sat.RW_axes.at(j);
            double tau_rw_des = dot(rw_axis, tau_des);
            double rw_torq = 0.5 * std::max(-sat.RW_max_torq.at(j), std::min(tau_rw_des, sat.RW_max_torq.at(j)));
            // Store in optimizer units (scaled by NONMTQ_TORQ_SCALE)
            uk(sat.number_MTQ + j) = rw_torq / NONMTQ_TORQ_SCALE;
          }
        }
        
        // Add noise scaled to not overwhelm PD signal
        double pd_strength = norm(uk);
        double noise_scale = std::max(noise_frac, 0.1 * pd_strength / norm(umax));
        uk += noise_scale * umax % (2*randu(sat.control_N()) - 1);
      }
      
      // Saturate
      double ur = max(abs(uk / umax));
      if(ur > 1.0) {
        uk = uk / ur;
      }
      
      Uset.col(k) = uk;
      
      // Propagate dynamics
      DYNAMICS_INFO_FORM dyn_kn1 = make_tuple(Bset.col(k), Rset.col(k), pset(k), Vset.col(k), Sset.col(k), 0);
      DYNAMICS_INFO_FORM dyn_k = make_tuple(Bset.col(k+1), Rset.col(k+1), pset(k+1), Vset.col(k+1), Sset.col(k+1), 0);
      
      tuple<vec, vec> dynout = rk4z(dt_use, xk, Uset.col(k), sat, dyn_kn1, dyn_k);
      xk = sat.state_norm(get<0>(dynout));
      Xset.col(k+1) = xk;
    }
    
    traj = make_tuple(Xset, Uset, t, TQset);
    U.cols(0, traj_length-1) = Uset;
    X = Xset;
    TQ = TQset;
    
    if(verbose){
      cout<<"PD+noise init complete, |U| = "<<norm(Uset, "fro")<<"\n";
    }
  }
  else
  {
    std::tuple<TRAJECTORY_FORM,double> bdotout = OldPlanner::bdot(x0,dt_use,traj_length,vecs,costSettings_tmp,mu0);
    if(verbose)
    {
      cout<<"bdot attempted\n";
    }
    traj = std::get<0>(bdotout);
    if (bdotOn == 2)
    {
      std::tuple<TRAJECTORY_FORM,double> sbdotout = OldPlanner::smartbdot(x0,dt_use,traj_length,vecs,costSettings_tmp,mu0,false);
      if(verbose){cout<<"smart bdot complete\n";}
      traj = std::get<0>(sbdotout);
      mat u0 = get<1>(traj);
      mat u_bdot = get<1>(std::get<0>(bdotout));
      
      // Debug: show raw smartbdot output before fallback
      if(verbose){
        cout<<"smartbdot raw |U| = "<<norm(u0, "fro")<<"\n";
        cout<<"smartbdot raw U col norms: ";
        for(int k = 0; k < std::min(5, (int)u0.n_cols); k++) {
          cout<<norm(u0.col(k))<<" ";
        }
        cout<<"\n";
      }
      
      // Per-timestep fallback: use bdot controls where smartbdot produces near-zero
      int fallback_count = 0;
      for(int k = 0; k < (int)u0.n_cols; k++) {
        double col_norm = norm(u0.col(k));
        if(col_norm < 1e-10) {
          u0.col(k) = u_bdot.col(k);
          fallback_count++;
        }
      }
      if(verbose && fallback_count > 0){
        cout<<"smartbdot: "<<fallback_count<<" of "<<u0.n_cols<<" timesteps fell back to bdot\n";
      }
      
      if(verbose){cout<<size(get<0>(traj))<<" "<<size(get<1>(traj))<<"\n";}
      TRAJECTORY_FORM traj2 = OldPlanner::generateInitialTrajectory(dt_use,x0, u0,vecs);//+diagmat(mean(abs(u0),1)*randval)*(randu(size(u0))-0.5), vecs);
      traj = traj2;  // Use the regenerated trajectory with blended controls
      if(verbose){cout<<size(get<0>(traj2))<<" "<<size(get<1>(traj2))<<"\n";}
      if(verbose){cout<<"smartbdot traj generated\n";}
    }
    else if (bdotOn == 3)
    {
      std::tuple<TRAJECTORY_FORM,double> sbdotout = OldPlanner::smartbdot(x0,dt_use,traj_length,vecs,costSettings_tmp,mu0,false);
      traj = std::get<0>(sbdotout);
      mat u0 = get<1>(traj);
      mat u_bdot = get<1>(std::get<0>(bdotout));
      
      // Per-timestep fallback: use bdot controls where smartbdot produces near-zero
      int fallback_count = 0;
      for(int k = 0; k < (int)u0.n_cols; k++) {
        double col_norm = norm(u0.col(k));
        if(col_norm < 1e-10) {
          u0.col(k) = u_bdot.col(k);
          fallback_count++;
        }
      }
      if(verbose && fallback_count > 0){
        cout<<"smartbdot: "<<fallback_count<<" of "<<u0.n_cols<<" timesteps fell back to bdot\n";
      }
      
      SMARTBDOT_SETTINGS_FORM sbSettings = this->highSettings;
      mat ECIvec = get<6>(vecs);
      mat satvec = get<5>(vecs);

      vec ek = ECIvec.col(0);
      vec3 ak = normalise(satvec.col(0));
      double ang0;

      if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
        ek = ek.tail(3);
        ang0 = acos(norm_dot(ak,rotMat(x0.rows(3,6)).t()*ek));
      }else{
        ang0  = acos(2.0*pow(norm_dot(x0.rows(3,6),ek),2.0)-1.0);
      }
      if (ang0 >= HLangleLimit){
        sbSettings = this->lowSettings;
      }
      double randval = get<4>(sbSettings);
      traj = OldPlanner::generateInitialTrajectory(dt_use,x0, u0+diagmat(max(abs(u0),1)*randval)*(2*randu(size(u0))-0.5), vecs);
    }
    U.cols(0,traj_length-1) = get<1>(traj);
    X = get<0>(traj);
    TQ = get<3>(traj);
    if(verbose){cout<<x0<<dt_use<<traj_length<<mu0<<X.has_nan()<<U.has_nan()<<"\n";}
    if((X.has_nan() || U.has_nan()) && verbose){
        cout<<X<<"\n";
        cout<<U<<"\n";
      }
  }
  traj = make_tuple(X,U,dt_timevec,TQ);
  if(verbose){cout<<"initial traj done\n";}


  return make_tuple(traj, vecs, costSettings_tmp);
}
AFTER_OUTPUT_FORM OldPlanner::trajOptAfter(VECTOR_INFO_FORM vecs_w_time,double dt_prev, TIME_FORM time_start, TIME_FORM time_end, ALILQR_OUTPUT_FORM alilqrOut)
{
  OPT_FORM opt = get<0>(alilqrOut);
  double muOut = std::get<1>(alilqrOut);
  double gradOut = std::get<2>(alilqrOut);
  double gradOut2 = gradOut;

  mat Xset = std::get<0>(opt);
  mat Uset = std::get<1>(opt);
  mat TQset = std::get<2>(opt);
  mat Kset = std::get<3>(opt);
  mat lambdaSet = std::get<4>(opt);
  if(verbose) {
    cout << "trajOptAfter: Kset from opt: (" << Kset.n_rows << "," << Kset.n_cols << "), n_elem=" << Kset.n_elem << "\n";
  }
  vec time_vec = std::get<5>(opt);


  // int traj_length = floor((time_end-time_start)*36525.0*24.0*3600.0/dt_tvlqr);
  VECTOR_INFO_FORM vecs_tvlqr = findVecTimes(vecs_w_time,dt_tvlqr/(36525.0*24.0*3600.0),time_start,time_end);


  vec tvlqr_times = get<0>(vecs_tvlqr);
  int traj_length = tvlqr_times.n_elem;
  if(verbose){cout<<" refs "<<traj_length<<"\n";}
  if(verbose){cout<<"tvlqr times found\n";}
  // VECTOR_INFO_FORM vecs_tvlqr = get<1>(time_tuple_tvlqr);
  mat Rset_tvlqr = get<1>(vecs_tvlqr);
  //COST_SETTINGS_FORM costSettings2 = this->costSettings2;
  //ALILQR_SETTINGS_FORM alilqrSettings2 = this->alilqrSettings2;

  OPT_FORM opt2;
  TRAJECTORY_FORM trajLong = make_tuple(Xset,Uset,time_vec.head(Xset.n_cols),TQset);
  if(verbose) {
    cout<<"completed ALILQR successfully\n";
    cout<<dt_prev<<" "<<dt_tvlqr<<endl;
  }
  if(dt_prev/dt_tvlqr > 1){
    double colMissing = max(0,int(Rset_tvlqr.n_cols) - 1 - int((int(Uset.n_cols)-2)*dt_prev/dt_tvlqr));

    if(verbose) {
      cout<<"colMiss: "<<colMissing<<"\n";
      cout<<"pass2_warm_start_mode: "<<pass2_warm_start_mode<<", Kset.n_elem: "<<Kset.n_elem<<"\n";
    }

    // Warm-start mode: 0=ZOH, 1=K-gain feedback, 2=SLERP interpolation, 3=closed-loop inverse dynamics
    if (pass2_warm_start_mode == 3) {
      // Closed-loop inverse dynamics: at each fine timestep, compute controls
      // from the ACTUAL simulated state targeting the SLERP'd reference ω.
      // Prevents drift compounding that breaks open-loop warm-starts.
      double tf = (time_end - time_start) * 36525.0 * 24.0 * 3600.0;
      if(verbose) {
        cout << "Using closed-loop inverse dynamics warm-start for Pass2 (dt_coarse=" << dt_prev
             << ", dt_fine=" << dt_tvlqr << ", tf=" << tf << ")\n";
      }
      trajLong = closedLoopInvDynWarmStart(Xset, dt_prev, dt_tvlqr, tf, sat, vecs_tvlqr);

      // Check for NaN - fall back to ZOH if it failed
      mat X_check = get<0>(trajLong);
      if (X_check.has_nan()) {
        if(verbose) cout << "  WARNING: closed-loop warm-start produced NaN, falling back to ZOH\n";
        int interp_ratio = int(dt_prev / dt_tvlqr);
        int K_ctrl = Uset.n_cols;
        mat UsetLong = repelem(Uset.cols(0, K_ctrl - 2), 1, interp_ratio);
        if(colMissing > 0){
          UsetLong = join_rows(UsetLong, repelem(Uset.col(K_ctrl - 2), 1, int(colMissing)));
        }
        UsetLong = join_rows(UsetLong, Uset.col(K_ctrl - 1));
        trajLong = OldPlanner::generateInitialTrajectory(dt_tvlqr, Xset.col(0), UsetLong, vecs_tvlqr);
      } else if(verbose) {
        mat U_ws = get<1>(trajLong);
        cout << "  Closed-loop warm-start complete: X=(" << X_check.n_rows << "," << X_check.n_cols
             << "), U=(" << U_ws.n_rows << "," << U_ws.n_cols << ")\n";
      }
    } else if (pass2_warm_start_mode == 2) {
      // SLERP interpolation: directly interpolate Pass1 states and controls to fine grid
      // No dynamics re-simulation — preserves Pass1 topology exactly
      double tf = (time_end - time_start) * 36525.0 * 24.0 * 3600.0;
      int N_fine = int(tf / dt_tvlqr) + 1;
      if(verbose) {
        cout << "Using SLERP warm-start for Pass2 (dt_coarse=" << dt_prev
             << ", dt_fine=" << dt_tvlqr << ", N_fine=" << N_fine << ")\n";
      }

      // SLERP-interpolate states (handles quaternions properly)
      mat Xset_fine = slerpInterpolateTrajectory(Xset, dt_prev, dt_tvlqr, tf);

      // Ensure correct length
      if ((int)Xset_fine.n_cols < N_fine) {
        mat pad = repmat(Xset_fine.col(Xset_fine.n_cols - 1), 1, N_fine - Xset_fine.n_cols);
        Xset_fine = join_rows(Xset_fine, pad);
      } else if ((int)Xset_fine.n_cols > N_fine) {
        Xset_fine = Xset_fine.head_cols(N_fine);
      }

      // ZOH-interpolate controls (same as legacy but without re-simulation)
      int interp_ratio = int(dt_prev / dt_tvlqr);
      int K_ctrl = Uset.n_cols;
      mat UsetLong = repelem(Uset.cols(0, K_ctrl - 2), 1, interp_ratio);
      if(colMissing > 0){
        UsetLong = join_rows(UsetLong, repelem(Uset.col(K_ctrl - 2), 1, int(colMissing)));
      }
      UsetLong = join_rows(UsetLong, Uset.col(K_ctrl - 1));
      // Trim or pad to match N_fine
      if ((int)UsetLong.n_cols < N_fine) {
        mat pad = repmat(UsetLong.col(UsetLong.n_cols - 1), 1, N_fine - UsetLong.n_cols);
        UsetLong = join_rows(UsetLong, pad);
      } else if ((int)UsetLong.n_cols > N_fine) {
        UsetLong = UsetLong.head_cols(N_fine);
      }

      vec t_fine = linspace(0, tf, N_fine);
      mat TQset_fine(3, N_fine, fill::zeros);  // Will be recomputed by optimizer
      trajLong = make_tuple(Xset_fine, UsetLong, t_fine, TQset_fine);

      if(verbose) {
        cout << "  SLERP warm-start complete: X=(" << Xset_fine.n_rows << "," << Xset_fine.n_cols
             << "), U=(" << UsetLong.n_rows << "," << UsetLong.n_cols << ")\n";
      }
    } else if (pass2_warm_start_mode == 1 && Kset.n_elem > 0) {
      // K-gain warm-start: propagate with feedback from coarse K-gains
      double tf = (time_end - time_start) * 36525.0 * 24.0 * 3600.0;
      if(verbose) {
        cout << "Using K-gain warm-start for Pass2 (dt_coarse=" << dt_prev
             << ", dt_fine=" << dt_tvlqr << ", tf=" << tf << ")\n";
      }
      trajLong = kgainWarmStart(Xset, Uset, Kset, dt_prev, dt_tvlqr, tf,
                                 sat, vecs_tvlqr, quaternionTo3VecMode);
      if(verbose) {
        mat X_kgain = get<0>(trajLong);
        mat U_kgain = get<1>(trajLong);
        cout << "  K-gain warm-start complete: X=(" << X_kgain.n_rows << "," << X_kgain.n_cols
             << "), U=(" << U_kgain.n_rows << "," << U_kgain.n_cols << ")\n";
      }
    } else {
      // Legacy ZOH interpolation (mode 0): re-simulates dynamics
      int interp_ratio = int(dt_prev / dt_tvlqr);
      int K_ctrl = Uset.n_cols;
      mat UsetLong = repelem(Uset.cols(0, K_ctrl - 2), 1, interp_ratio);
      if(colMissing > 0){
        UsetLong = join_rows(UsetLong, repelem(Uset.col(K_ctrl - 2), 1, int(colMissing)));
      }
      UsetLong = join_rows(UsetLong, Uset.col(K_ctrl - 1));

      trajLong = OldPlanner::generateInitialTrajectory(dt_tvlqr,Xset.col(0), UsetLong, vecs_tvlqr);
    }

    // mat xtmp = get<0>(trajLong);
    // TRAJECTORY_FORM traj_tvlqr = trajLong;
    // TRAJECTORY_FORM opt2 = trajLong;
    auto t_start_pass2 = std::chrono::high_resolution_clock::now();

    if (skip_pass2_optimization) {
      // Skip alilqr: use ZOH forward-simulated trajectory directly for K-gains.
      // This avoids the optimizer creating wound trajectories at dt=1.
      // The trajectory is dynamically feasible (from generateInitialTrajectory)
      // but not locally optimal. K-gains correct tracking errors.
      
      // Target: N_fine states at dt_tvlqr
      int N_fine = int(round((time_end - time_start) * 36525.0 * 24.0 * 3600.0 / dt_tvlqr)) + 1;
      int N_ctrl = N_fine - 1;
      
      // ZOH-interpolate coarse controls to fine grid
      int interp_ratio_zoh = int(dt_prev / dt_tvlqr);
      int K_ctrl_zoh = Uset.n_cols;
      mat UsetLong_zoh = repelem(Uset.cols(0, K_ctrl_zoh - 2), 1, interp_ratio_zoh);
      // Pad with last control to reach N_ctrl
      while ((int)UsetLong_zoh.n_cols < N_ctrl) {
        UsetLong_zoh = join_rows(UsetLong_zoh, Uset.col(K_ctrl_zoh - 1));
      }
      // Trim to exact size
      UsetLong_zoh = UsetLong_zoh.head_cols(N_ctrl);
      
      trajLong = OldPlanner::generateInitialTrajectory(dt_tvlqr, Xset.col(0), UsetLong_zoh, vecs_tvlqr);
      
      // Convert TRAJECTORY_FORM to OPT_FORM (add empty K, S, times)
      mat X_zoh = get<0>(trajLong);
      mat U_zoh = get<1>(trajLong);
      mat TQ_zoh = get<3>(trajLong);
      mat K_empty;
      mat S_empty;
      int N_zoh = X_zoh.n_cols;
      vec times_zoh = linspace(0, (N_zoh-1)*dt_tvlqr, N_zoh) / (36525.0*24.0*3600.0) + time_start;
      opt2 = make_tuple(X_zoh, U_zoh, TQ_zoh, K_empty, S_empty, times_zoh);
      gradOut2 = 0.0;
      
      auto t_end_pass2 = std::chrono::high_resolution_clock::now();
      auto pass2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_pass2 - t_start_pass2).count();
      cout << "TIMING: Pass 2 SKIPPED (ZOH forward sim only, dt=" << dt_tvlqr << "s, N=" << get<0>(trajLong).n_cols << "): " << pass2_ms << " ms\n";
    } else {
      _useEuler = use_euler_pass2;
      ALILQR_OUTPUT_FORM alilqrOut2 = OldPlanner::alilqr(dt_tvlqr,trajLong, vecs_tvlqr, costSettings2,alilqrSettings2,false);
      _useEuler = false;
      auto t_end_pass2 = std::chrono::high_resolution_clock::now();
      auto pass2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_pass2 - t_start_pass2).count();
      cout << "TIMING: Pass 2 (dt=" << dt_tvlqr << "s, N=" << get<0>(trajLong).n_cols << "): " << pass2_ms << " ms\n";

      opt2 = get<0>(alilqrOut2);
      // double muOut2 = std::get<1>(alilqrOut2);
      gradOut2 = std::get<2>(alilqrOut2);
      if(verbose) {
        cout<<"completed full length ALILQR successfully\n";
      }
    }
  }else{
    // Same timestep (dt_tp == dt_tvlqr): no warm-start interpolation needed.
    // If Pass 1 used slacks, run Pass 2 without slacks to get feasible trajectory.
    if (!skip_pass2_optimization) {
      // Same-dt Pass 2: check if Pass 1 slacks are small enough to skip forward-sim
      double slack_max_val = (slack_Sset.n_elem > 0) ? abs(slack_Sset).max() : 0.0;
      double slack_tol = 1e-2;  // Accept Pass 1 trajectory if slacks this small
      
      if (slack_max_val < slack_tol && slack_max_val >= 0) {
        // Slacks are tiny — Pass 1 trajectory is practically feasible
        // Use it directly (avoids forward-sim destroying topology at 180° bifurcation)
        cout << "Pass 2: using Pass 1 trajectory directly (slack_max=" << slack_max_val << " < " << slack_tol << ")\n";
        mat K_empty, S_empty;
        vec times_p1 = linspace(0, (Xset.n_cols-1)*dt_tvlqr, Xset.n_cols);
        opt2 = make_tuple(Xset, Uset, get<2>(opt), K_empty, S_empty, times_p1);
        gradOut2 = 0;
      } else {
        // Slacks too large — try forward-sim to get feasible trajectory
        cout << "Pass 2: forward-simming Pass 1 controls (slack_max=" << slack_max_val << " >= " << slack_tol << ")\n";
        TRAJECTORY_FORM trajFwd = generateInitialTrajectory(dt_tvlqr, Xset.col(0), Uset, vecs_tvlqr);
        mat Xfwd = get<0>(trajFwd);
        mat K_empty, S_empty;
        vec times_fwd = linspace(0, (Xfwd.n_cols-1)*dt_tvlqr, Xfwd.n_cols);
        opt2 = make_tuple(Xfwd, get<1>(trajFwd), get<3>(trajFwd), K_empty, S_empty, times_fwd);
        gradOut2 = 0;
      }
    } else {
      opt2 = opt;
    }
    tvlqr_times = time_vec;
  }

  mat K_lqr;
  mat S_lqr;
  if(tracking_LQR_formulation==0){
    K_lqr = mat(sat.control_N()*sat.reduced_state_N(),0,fill::zeros);
  }else if(tracking_LQR_formulation==2){
    K_lqr = mat(sat.control_N()*(sat.reduced_state_N()+3),0,fill::zeros);
  }else{
    K_lqr = mat(sat.control_N()*sat.reduced_state_N(),0,fill::zeros);
  }
  // K_lqr.zeros();
  if(tracking_LQR_formulation==0){
  }else if(tracking_LQR_formulation==2){
    S_lqr = mat((sat.reduced_state_N()+3)*(sat.reduced_state_N()+3),0,fill::zeros);
  }else{
    S_lqr = mat(sat.reduced_state_N()*sat.reduced_state_N(),0,fill::zeros);
  }
  // S_lqr.zeros();
  double tvlqr_overlap_tmp = floor(tvlqr_overlap/dt_tvlqr)*dt_tvlqr;
  double tvlqr_len_tmp = floor(tvlqr_len/dt_tvlqr)*dt_tvlqr;

  if(verbose){cout<<"time to find K\n";}

  mat U_lqr = get<1>(opt2);
  mat X_lqr = get<0>(opt2);
  mat TQ_lqr = get<2>(opt2);
  vec dt_lqr = get<5>(opt2);
  
  // Variables for segment iteration (NSSR_3+1 style - forward iteration)
  // Initialize so first iteration starts at beginning of trajectory
  TIME_FORM time_start_tmp = time_start;
  TIME_FORM time_end_tmp = time_start + (tvlqr_overlap_tmp)/(36525.0*24.0*3600.0);
  double col0 = 0;
  double col1 = col0 + (tvlqr_overlap_tmp/dt_tvlqr);
  double col1u = 0;
  
  // Use original (unscaled) costs for TVLQR K-gain computation
  sat.use_original_costs(true);
  
  do {
    time_start_tmp = time_end_tmp - (tvlqr_overlap_tmp)/(36525.0*24.0*3600.0);
    time_end_tmp = min(time_start_tmp + (tvlqr_len_tmp)/(36525.0*24.0*3600.0), time_end);
    
    col0 = col1 - (tvlqr_overlap_tmp/dt_tvlqr);
    col1 = std::min((double)(X_lqr.n_cols)-1, col0 + tvlqr_len_tmp/dt_tvlqr);
    col1u = std::min((double)(U_lqr.n_cols)-1, col0 + tvlqr_len_tmp/dt_tvlqr);
    
    VECTOR_INFO_FORM vecs_tvlqr_tmp = findVecTimes(vecs_w_time, dt_tvlqr/(36525.0*24.0*3600.0), time_start_tmp, time_end_tmp);
    
    if(verbose) cout << col0 << " " << col1 << " " << col1u << " " << Rset_tvlqr.n_cols << "\n";
    
    vec dt_lqr_tmp = dt_lqr.subvec(col0, col1);
    mat X_lqr_tmp = X_lqr.cols(col0, col1);
    mat U_lqr_tmp = U_lqr.cols(col0, col1u);
    mat TQ_lqr_tmp = TQ_lqr.cols(col0, col1u);
    vec rd_offset = join_cols(sat.mtq_ax_mat.t()*sat.res_dipole*sat.plan_for_resdipole, vec(sat.control_N()-sat.number_MTQ).zeros());
    U_lqr_tmp.each_col() -= rd_offset;
    
    TRAJECTORY_FORM traj_tvlqr_tmp = make_tuple(X_lqr_tmp, U_lqr_tmp, dt_lqr_tmp, TQ_lqr_tmp);
    
    COST_SETTINGS_FORM costSettingsFindK = this->costSettings_tvlqr;
    cube Kcube;
    cube Scube;
    std::tuple<cube, cube> KS = make_tuple(Kcube, Scube);
    
    if(tracking_LQR_formulation == 0) {
      KS = OldPlanner::findK(dt_tvlqr, traj_tvlqr_tmp, vecs_tvlqr_tmp, costSettingsFindK);
    } else if(tracking_LQR_formulation == 2) {
      KS = OldPlanner::findKwDist(dt_tvlqr, traj_tvlqr_tmp, vecs_tvlqr_tmp, costSettingsFindK);
    } else {
      KS = OldPlanner::findK(dt_tvlqr, traj_tvlqr_tmp, vecs_tvlqr_tmp, costSettingsFindK);
    }
    
    mat K_lqr_tmp = packageK(get<0>(KS));
    mat S_lqr_tmp = packageS(get<1>(KS));
    
    // Concatenate using join_rows (NSSR_3+1 approach)
    double Klen = ((K_lqr.n_cols - tvlqr_overlap_tmp/dt_tvlqr) < 0) ? 0 : K_lqr.n_cols - tvlqr_overlap_tmp/dt_tvlqr;
    K_lqr = join_rows(K_lqr.head_cols((int)Klen), K_lqr_tmp);
    double Slen = ((S_lqr.n_cols - tvlqr_overlap_tmp/dt_tvlqr - 1) < 0) ? 0 : S_lqr.n_cols - tvlqr_overlap_tmp/dt_tvlqr - 1;
    S_lqr = join_rows(S_lqr.head_cols((int)Slen), S_lqr_tmp);
    
    if(verbose) cout << size(K_lqr) << "\n" << size(S_lqr) << "\n";
    
  } while((time_end_tmp < (time_end) - EPSVAR) && (time_start_tmp < (time_end - tvlqr_overlap_tmp/(36525.0*24.0*3600.0))));
  
  // Restore scaled costs for optimizer
  sat.use_original_costs(false);
  
  if(verbose){cout<<"K found\n";}

  // OPT_TIMES_FORM main_opt_times = (addOptTimes(opt));
  OPT_FORM lqr_opt = make_tuple(get<0>(opt2),get<1>(opt2),get<2>(opt2),K_lqr,S_lqr,tvlqr_times.head(get<0>(opt2).n_cols));
  
  // Scale RW/magic controls back to physical units
  // Optimizer uses scaled controls: u_scaled = u_physical / NONMTQ_TORQ_SCALE
  // Convert back: u_physical = u_scaled * NONMTQ_TORQ_SCALE
  if(NONMTQ_TORQ_SCALE != 1.0 && (sat.number_RW > 0 || sat.number_magic > 0)) {
    int rw_magic_start = sat.number_MTQ;
    int rw_magic_end = sat.number_MTQ + sat.number_RW + sat.number_magic - 1;
    
    // Scale Uset in opt2
    mat Uset_opt2 = get<1>(opt2);
    Uset_opt2.rows(rw_magic_start, rw_magic_end) *= NONMTQ_TORQ_SCALE;
    get<1>(opt2) = Uset_opt2;
    
    // Scale Uset in lqr_opt
    mat Uset_lqr = get<1>(lqr_opt);
    Uset_lqr.rows(rw_magic_start, rw_magic_end) *= NONMTQ_TORQ_SCALE;
    get<1>(lqr_opt) = Uset_lqr;
    
    // Scale Uset in trajLong
    mat Uset_traj = get<1>(trajLong);
    Uset_traj.rows(rw_magic_start, rw_magic_end) *= NONMTQ_TORQ_SCALE;
    get<1>(trajLong) = Uset_traj;
    
    // K-gains are left in optimizer units.
    // The Python tracking controller applies NONMTQ_TORQ_SCALE when computing du for RW.
    // This matches how warmstart uses K-gains with optimizer-unit controls.
    
    if(verbose) {
      cout << "Scaled RW/magic controls by NONMTQ_TORQ_SCALE=" << NONMTQ_TORQ_SCALE << "\n";
    }
  }
  
  //return success
  return std::make_tuple(1, gradOut2, opt2, lqr_opt, trajLong);
}

AFTER_OUTPUT_FORM OldPlanner::trajOpt(VECTOR_INFO_FORM &vecs,int N, TIME_FORM time_start, TIME_FORM time_end, vec x0, int bdotOn)
{
  if(verbose){
    const arma::vec& t   = std::get<0>(vecs);
    const arma::mat& r   = std::get<1>(vecs);
    const arma::mat& v   = std::get<2>(vecs);
    const arma::mat& b   = std::get<3>(vecs);
    const arma::mat& s   = std::get<4>(vecs);
    const arma::mat& a   = std::get<5>(vecs);
    const arma::mat& e   = std::get<6>(vecs);
    const arma::vec& p   = std::get<7>(vecs);
    const arma::vec& rho = std::get<8>(vecs);

    cout << "=== VECTOR_INFO_FORM DEBUG ===\n";
    cout << "t   : vec  n_elem = " << t.n_elem << "\n";
    cout << "r   : mat  " << r.n_rows << " x " << r.n_cols << "\n";
    cout << "v   : mat  " << v.n_rows << " x " << v.n_cols << "\n";
    cout << "b   : mat  " << b.n_rows << " x " << b.n_cols << "\n";
    cout << "s   : mat  " << s.n_rows << " x " << s.n_cols << "\n";
    cout << "a   : mat  " << a.n_rows << " x " << a.n_cols << "\n";
    cout << "e   : mat  " << e.n_rows << " x " << e.n_cols << "\n";
    cout << "p   : vec  n_elem = " << p.n_elem << "\n";
    cout << "rho : vec  n_elem = " << rho.n_elem << "\n";
    cout << "==============================\n";
    cout<<"x0:\n"<<x0<<"\n";
  }

  auto t_start_pass1 = std::chrono::high_resolution_clock::now();
  BEFORE_OUTPUT_FORM results = OldPlanner::trajOptBefore(vecs, dt, time_start, time_end, x0, bdotOn);
  if(verbose){cout<<"past trajOptBefore\n";}
  TRAJECTORY_FORM traj_init = get<0>(results);
  VECTOR_INFO_FORM vecs_dt = get<1>(results);
  COST_SETTINGS_FORM costSettings_tmp = get<2>(results);
  _useEuler = false;  // Pass 1 always uses RK4 (coarse dt, cheap anyway)

  // Infeasible start: optionally replace init trajectory with SLERP
  if (use_infeasible_start) {
    if (infeasible_ctrl_mode <= 2) {
      // Modes 0-2: SLERP states + specified controls (original ALTRO-style infeasible start)
      mat ECIvec_init = get<6>(vecs_dt);
      vec ek0 = ECIvec_init.col(0);
      if (ek0.n_elem == 4 && !isnan(ek0(0))) {
        vec4 q_goal = normalise(ek0);
        int N_init = get<0>(traj_init).n_cols;
        traj_init = generateSlerpTrajectory(dt, x0, q_goal, N_init, vecs_dt, infeasible_ctrl_mode);
        if (verbose) { cout << "Infeasible start: SLERP + ctrl_mode=" << infeasible_ctrl_mode << " for Pass 1\n"; }
      } else if (verbose) {
        cout << "Infeasible start: vector goal detected, using standard init with defects\n";
      }
    } else if (infeasible_ctrl_mode == 3) {
      // Mode 3: Standard feasible init + slacks for exploration
      // Keep traj_init as-is (from prepareForAlilqr — dynamically feasible)
      // Slacks start at zero, give optimizer freedom to escape wound local minima
      if (verbose) { cout << "Infeasible start mode 3: feasible start + slack exploration\n"; }
    }
  }

  ALILQR_OUTPUT_FORM alilqrOut = OldPlanner::alilqr(dt,traj_init, vecs_dt, costSettings_tmp,alilqrSettings,false);
  // Disable slacks for Pass 2 — Pass 2 must produce dynamically feasible trajectory.
  // If dt_tp == dt_tvlqr (same timestep), Pass 2 forward-sims Pass 1 controls → feasible.
  use_infeasible_start = false;
  auto t_end_pass1 = std::chrono::high_resolution_clock::now();
  auto pass1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_pass1 - t_start_pass1).count();
  cout << "TIMING: Pass 1 (dt=" << dt << "s, N=" << get<0>(traj_init).n_cols << "): " << pass1_ms << " ms\n";
  if(verbose){cout<<"out of alilqr\n";}

  //any disturbances?

  // OPT_FORM opt_tmp0 = get<0>(alilqrOut);
  // mat Xtmp0 = get<0>(opt_tmp0);
  // mat Utmp0 = get<1>(opt_tmp0);
  //
  // bool anyDist = bool(sat.plan_for_gg || sat.plan_for_srp || sat.plan_for_aero || sat.plan_for_prop || sat.plan_for_resdipole || sat.plan_for_gendist);
  // // assert(!anyDist);
  // if(anyDist){
  //   //if any disturbances, run the basic/larger-time-step alilqr again, with the disturbances on.
  //   OPT_FORM opt = get<0>(alilqrOut);
  //   TRAJECTORY_FORM traj_0 = make_tuple(get<0>(opt),get<1>(opt),get<5>(opt),get<2>(opt));
  //   alilqrOut = OldPlanner::alilqr(dt,traj_0, vecs_dt, costSettings_tmp,alilqrSettings,false);
  // }
  OPT_FORM opt_tmp = get<0>(alilqrOut);
  mat Xtmp = get<0>(opt_tmp);
  mat Utmp = get<1>(opt_tmp);
  mat Ktmp = get<3>(opt_tmp);
  if(verbose){
    cout<<"dist done\n";
    cout<<"Before trajOptAfter: Ktmp=("<<Ktmp.n_rows<<","<<Ktmp.n_cols<<")\n";
  }
  AFTER_OUTPUT_FORM results2 = OldPlanner::trajOptAfter(vecs, dt, time_start, time_end, alilqrOut);
  if(verbose){cout<<"trajOptAfter done\n";}
  return results2;
}

// Multi-start trajectory optimization: runs multiple Pass 1s with different initializations
// and picks the best one before running Pass 2
AFTER_OUTPUT_FORM OldPlanner::trajOptMultiStart(VECTOR_INFO_FORM &vecs, int N, TIME_FORM time_start, TIME_FORM time_end, vec x0, std::vector<int> bdotModes)
{
  if(bdotModes.empty()) {
    bdotModes = {0, 1, 4, 5};  // Default: random, bdot, PD, PD+noise
  }
  
  cout << "MULTI-START: Running " << bdotModes.size() << " Pass 1 attempts\n";
  
  // Run Pass 1 for each bdot mode and track results
  std::vector<ALILQR_OUTPUT_FORM> pass1Results;
  std::vector<double> pass1Costs;
  std::vector<double> pass1Violations;
  std::vector<VECTOR_INFO_FORM> pass1Vecs;
  std::vector<COST_SETTINGS_FORM> pass1CostSettings;
  
  for(size_t i = 0; i < bdotModes.size(); i++) {
    int bdotOn = bdotModes[i];
    auto t_start_p1 = std::chrono::high_resolution_clock::now();
    
    BEFORE_OUTPUT_FORM results = OldPlanner::trajOptBefore(vecs, dt, time_start, time_end, x0, bdotOn);
    TRAJECTORY_FORM traj_init = get<0>(results);
    VECTOR_INFO_FORM vecs_dt = get<1>(results);
    COST_SETTINGS_FORM costSettings_tmp = get<2>(results);
    
    ALILQR_OUTPUT_FORM alilqrOut = OldPlanner::alilqr(dt, traj_init, vecs_dt, costSettings_tmp, alilqrSettings, false);
    
    auto t_end_p1 = std::chrono::high_resolution_clock::now();
    auto p1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_p1 - t_start_p1).count();
    
    // Get cost and constraint violation for this result
    OPT_FORM opt = get<0>(alilqrOut);
    double cost = get<1>(alilqrOut);  // mu (penalty)
    double grad = get<2>(alilqrOut);
    
    // Compute max constraint violation
    mat lambdaSet = mat(sat.constraint_N(), get<0>(opt).n_cols).zeros();
    mat muSet0 = mat(sat.constraint_N(), get<0>(opt).n_cols).ones() * cost;
    AUGLAG_INFO_FORM auglag = make_tuple(lambdaSet, cost, muSet0);
    TRAJECTORY_FORM traj_tmp = make_tuple(get<0>(opt), get<1>(opt), get<5>(opt), get<2>(opt));
    auto maxViolResult = OldPlanner::maxViol(traj_tmp, vecs_dt, auglag);
    double cmax = get<1>(maxViolResult);
    
    // Compute actual cost
    double totalCost = cost2Func(traj_tmp, vecs_dt, auglag, &costSettings_tmp, true);
    
    pass1Results.push_back(alilqrOut);
    pass1Costs.push_back(totalCost);
    pass1Violations.push_back(cmax);
    pass1Vecs.push_back(vecs_dt);
    pass1CostSettings.push_back(costSettings_tmp);
    
    cout << "  Pass1[" << i << "] bdot=" << bdotOn << ": cost=" << totalCost 
         << ", cmax=" << cmax << ", time=" << p1_ms << "ms\n";
  }
  
  // Pick the best result (lowest cost with acceptable constraint violation)
  size_t bestIdx = 0;
  double bestScore = std::numeric_limits<double>::max();
  double cmaxThreshold = 1e-3;  // Acceptable constraint violation
  
  for(size_t i = 0; i < pass1Results.size(); i++) {
    // Score = cost + penalty for constraint violation
    double score = pass1Costs[i];
    if(pass1Violations[i] > cmaxThreshold) {
      score += 1e6 * pass1Violations[i];  // Heavy penalty for violations
    }
    if(score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  
  cout << "MULTI-START: Selected Pass1[" << bestIdx << "] with bdot=" << bdotModes[bestIdx] 
       << " (cost=" << pass1Costs[bestIdx] << ", cmax=" << pass1Violations[bestIdx] << ")\n";
  
  // Run Pass 2 on the best result
  cout << "TIMING: Pass 1 total (all " << bdotModes.size() << " attempts): combined\n";
  AFTER_OUTPUT_FORM results2 = OldPlanner::trajOptAfter(vecs, dt, time_start, time_end, pass1Results[bestIdx]);
  
  return results2;
}



/*This function generates bdot gains to initialize the trajectory optimizer
  Inputs:
    Bset - magnetic field over time
    x0 - initial x position
    bdotgain - bdot gain
    umax - max allowable dipole
    dt - delta t
*/
tuple<TRAJECTORY_FORM,double> OldPlanner::bdot(vec x0,double dt0, int N,VECTOR_INFO_FORM vecs,  COST_SETTINGS_FORM costSettings_tmp,double mu)
{
  mat lambdaSet = mat(sat.constraint_N(), N).zeros();
  mat muSet0 = mat(sat.constraint_N(), N).ones()*mu;
  AUGLAG_INFO_FORM auglag_vals = make_tuple(lambdaSet,mu,muSet0);
  //Initialize newX
  //int N = Bset.n_cols+1;
  mat Xset = mat(sat.state_N(), N);
  mat Uset = mat(sat.control_N(), N).zeros();
  mat TQset = mat(3, N).zeros();
  cout.precision(4);

  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);
  vec t = get<0>(vecs);
  //Xset.fill(datum::nan);

  //Copy initial state of x to xk
  vec xk = vec(x0);
  xk = sat.state_norm(xk);
  Xset.col(0) = xk;
  vec4 qk = xk.rows(3, 6);
  vec3 Bk = Bset.col(0);
  mat33 RmatT = rotMat(qk).t();
  vec uk = vec(sat.control_N()).zeros();
  vec umax = join_cols(vec(sat.MTQ_max),vec(sat.RW_max_torq),vec(sat.magic_max_torq));
  // vec umax = sat.MTQ_max;

  double ur = max(abs(uk/umax));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  uk.head(sat.number_MTQ) = -sat.mtq_ax_mat.t()*bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(1)-Bk)/dt0);
  uk.head(sat.number_MTQ) = -sat.mtq_ax_mat.t()*bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(1)-Bk)/dt0);

  ur = max(abs(uk/umax));
  
  ur = std::max(ur,1.0);
  uk = uk/ur;
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),0);
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;
  tuple<vec,vec> dynout;
  for(int k=1; k<N; k++)
  {

    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),Sset.col(k),0);
    RmatT = rotMat(qk).t();
    // uk = -bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(k)-Bk)/dt0);
    uk.zeros();
    uk.head(sat.number_MTQ) = -sat.mtq_ax_mat.t()*bdotgain*(-cross(xk.rows(0,2), RmatT*Bk) + RmatT*(Bset.col(k)-Bk)/dt0);///pow(norm(Bk),2);

    ur = max(abs(uk/umax));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    if(sat.number_RW>0){
      uk(span(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)) = 0*-diagmat(vec(sat.RW_J))*sat.rw_ax_mat.t()*sat.invJcom_noRW*-skewSymmetric(RmatT*Bk)*sat.mtq_ax_mat*uk.head(sat.number_MTQ);
    }
    ur = max(abs(uk/umax));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    Uset.col(k-1) = uk;
    //xprev = xk;
    Bk = Bset.col(k);
    dynout = rk4z(dt0,xk, uk, sat,dynamics_info_kn1, dynamics_info_k);
    vec dos = get<0>(dynout);
    xk = sat.state_norm(get<0>(dynout));
    // qk = normalise(xk.rows(3, 6));
    // xk.rows(3, 6) = qk;
    Xset.col(k) = xk;
    TQset.col(k-1) = get<1>(dynout);

  }
  TRAJECTORY_FORM traj = make_tuple(Xset,Uset,t,TQset);
  double bcost = OldPlanner::cost2Func(traj, vecs, auglag_vals, &costSettings_tmp);
  return make_tuple(traj,bcost);
}
/*This function generates a control trajectory to initialize the trajectory optimizer that accounts for goals
  Inputs:
    Bset - magnetic field over time
    ECIvec - desired pointing of satvec over time
    satvec - pointing vec over time
    x0 - initial x position
    dampgain - bdot gain (velocity damping), now scaled by norm(B)^2
    velgain - gain to control attempt to match desired angular velocity
    quatgain - gain to control attempt to match vector pointing
    umax - max allowable dipole
    dt0 - delta t
*/
tuple<TRAJECTORY_FORM,double> OldPlanner::smartbdot(vec x0,double dt0,int N,VECTOR_INFO_FORM vecs,COST_SETTINGS_FORM costSettings_tmp,double mu,bool invert)
{


  //double HLangleLimit = this->HLangleLimit;
  SMARTBDOT_SETTINGS_FORM sbSettings = highSettings;
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  vec ek = ECIvec.col(0);



  mat lambdaSet = mat(sat.constraint_N(), N).zeros();
  mat muSet0 = mat(sat.constraint_N(), N).ones()*mu;
  AUGLAG_INFO_FORM auglag_vals = make_tuple(lambdaSet,mu,muSet0);
  //int N = Bset.n_cols+1;
  mat Xset = mat(sat.state_N(), N);
  mat Uset = mat(sat.control_N(), N).zeros();
  mat TQset = mat(3, N).zeros();

  //Copy initial state of x to xk
  vec xk = vec(x0);
  xk = sat.state_norm(xk);
  Xset.col(0) = xk;
  vec4 qk = normalise(xk.rows(3, 6));

  //Initialize stuff for inside loop
  //vec xprev = vec(x0);
  //vec3 uk = vec(3).zeros();
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);
  vec t = get<0>(vecs);
  vec3 Bk = Bset.col(0);
  double nB2 = dot(Bk,Bk);
  vec3 satvk = normalise(satvec.col(0));
  double ang0;

  if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
    ek = ek.tail(3);
     ang0 = acos(norm_dot(satvk,rotMat(qk).t()*ek));
  }else{
     ang0  = acos(2.0*pow(norm_dot(qk,ek),2.0)-1.0);
  }
  if (ang0 >= HLangleLimit){
    sbSettings = lowSettings;
  }
  double umaxmult = get<5>(sbSettings);

  vec umax = join_cols(vec(sat.MTQ_max),0.01*vec(sat.RW_max_torq),0.01*vec(sat.magic_max_torq));
  vec umax2 = umaxmult*umax;//0.75*umax;



  mat33 RmatT = rotMat(qk).t();
  vec3 Bbody = RmatT*Bk;
  //uk = -bdotgain*(-cross(newX(1:3,1),rotT(newX(end-3:end,1))*Bset(:,1)) + rotT(newX(end-3:end,1))*(Bset(:,2)-Bset(:,1))/dt0);
  vec ECIvk = ECIvec.col(0);
  vec ECIvkp1 = ECIvec.col(1);
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),1);

  vec3 dist_torq = sat.dist_torque(xk,dynamics_info_kn1);
  vec uk = OldPlanner::smartbdot_rawmtq_finder(dt0,xk,nB2, ECIvk, ECIvkp1, satvk, Bbody,sbSettings,dist_torq);
  double ur = max(abs(uk/umax2));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  if(sat.number_RW>0){
    uk(span(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)) = -diagmat(vec(sat.RW_J))*sat.rw_ax_mat.t()*sat.invJcom_noRW*-skewSymmetric(RmatT*Bk)*sat.mtq_ax_mat*uk.head(sat.number_MTQ);
  }
  ur = max(abs(uk/umax2));
  ur = std::max(ur,1.0);
  uk = uk/ur;
  //uk = uk/clamp(max(abs(uk/umax)),1,datum::inf);
  Uset.col(0) = uk;

  //Loop from k = 1 to N-2 and fill in Xset, using rk4

  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;
  tuple<vec,vec> dynout;
  for(int k=1; k<N; k++)
  {
    Bk = Bset.col(k);
    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bk,Rset.col(k),pset(k),Vset.col(k),Sset.col(k),1);
    Uset.col(k) = uk;
    //uk = Uset.col(k-1);
    //xprev = xk;
    dynout = rk4z(dt0,xk, uk,sat,dynamics_info_kn1,dynamics_info_k);
    xk = sat.state_norm(get<0>(dynout));
    // qk = normalise(xk.rows(3, 6));
    // xk.rows(3, 6) = qk;
    Xset.col(k) = xk;
    TQset.col(k-1) = get<1>(dynout);

    nB2 = dot(Bk,Bk);
    //uk = -bdotgain*(-cross(newX(1:3,1),rotT(newX(end-3:end,1))*Bset(:,1)) + rotT(newX(end-3:end,1))*(Bset(:,2)-Bset(:,1))/dt0);
    ECIvk = ECIvec.col(k);
    if(k<N-1){ECIvkp1 = ECIvec.col(k+1);}
    satvk = satvec.col(k);
    RmatT = rotMat(qk).t();
    Bbody = RmatT*Bk;
    dist_torq = sat.dist_torque(xk,dynamics_info_k);
    uk = OldPlanner::smartbdot_rawmtq_finder(dt0,xk,nB2, ECIvk, ECIvkp1, satvk, Bbody,sbSettings,dist_torq);

    ur = max(abs(uk/umax2));
    ur = std::max(ur,1.0);
    uk = uk/ur;
    if(sat.number_RW>0){
      uk(span(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)) = -diagmat(vec(sat.RW_J))*sat.rw_ax_mat.t()*sat.invJcom_noRW*-skewSymmetric(RmatT*Bk)*sat.mtq_ax_mat*uk.head(sat.number_MTQ);
    }
    ur = max(abs(uk/umax2));
    ur = std::max(ur,1.0);
    uk = uk/ur;

  }
  TRAJECTORY_FORM traj = make_tuple(Xset,Uset,t,TQset);

  double bcost = OldPlanner::cost2Func(traj, vecs,auglag_vals,  &costSettings_tmp);
  return make_tuple(traj,bcost);
}

vec OldPlanner::smartbdot_rawmtq_finder(double dt0, vec xk, double nB2,vec ECIvk, vec ECIvkp1, vec3 satvk, vec3 Bbody,SMARTBDOT_SETTINGS_FORM sbSettings,vec3 dist_torq){

  double gyrogain = get<0>(sbSettings);
  double dampgain = get<1>(sbSettings);
  double velgain = get<2>(sbSettings);
  double quatgain = get<3>(sbSettings);
  double umaxmult = get<5>(sbSettings);
  bool ek_is_3 = ((ECIvk.n_elem==3)||((ECIvk.n_elem==4)&&(isnan(ECIvk(0)))));
  bool ekp1_is_3 = ((ECIvkp1.n_elem==3)||((ECIvkp1.n_elem==4)&&(isnan(ECIvkp1(0)))));

  xk = sat.state_norm(xk);
  vec4 qk = normalise(xk.rows(3, 6));
  mat33 RmatT = rotMat(qk).t();
  vec3 wk = xk.head(3);
  vec uk = vec(sat.control_N()).zeros();


  if(ek_is_3&&ekp1_is_3){
    ECIvk = ECIvk.tail(3);
    ECIvkp1 = ECIvkp1.tail(3);
    vec3 wkdes = cross(ECIvk,ECIvkp1);
    wkdes = asin(norm(wkdes))*normalise(wkdes)/dt0;
    wkdes = RmatT*wkdes;
    vec3 ECIvkBody = RmatT*ECIvk;
    vec3 qq = sat.invJcom*normalise(cross(normalise(ECIvkBody+satvk),normalise(cross(ECIvkBody,satvk))));
    qq = normalise(qq);
    vec3 x3 = normalise(Bbody);//normalise(sat.Jcom*cross(xx,x2));

      //uk = cross(Bbody,sat.Jcom*(dampgain*(wk-dot(wk,x3)*x3) + quatgain*acos(norm_dot(ECIvkBody,satvk))*normalise(cross(ECIvkBody,satvk))))/nB2;
    uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t()*cross(Bbody,(dampgain*(sat.Jcom*wk) + velgain*sat.Jcom*(wk-wkdes) + quatgain*acos(norm_dot(ECIvkBody,satvk))*normalise(sat.Jcom*cross(ECIvkBody,satvk))))/nB2;
    if(norm(cross((qq),x3))<0.02){
      vec3 x4 = normalise(cross(qq,x3));
      uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t()*cross(Bbody,(dampgain*(sat.Jcom*wk) + velgain*sat.Jcom*(wk-wkdes) + quatgain*acos(norm_dot(ECIvkBody,satvk))*x4*sign(norm_dot(x4,normalise(cross(ECIvkBody,satvk))))))/nB2;
    }
  }else{
    if(!ek_is_3 && ekp1_is_3){
      //current is a quaternion specification. make the other a quat specification nearby

      ECIvkp1 = ECIvkp1.tail(3);
      ECIvkp1 = normalise(ECIvkp1);
      ECIvkp1 = closestQuatForVecPoint(ECIvk,satvk,ECIvkp1);
    }
    if(!ekp1_is_3 && ek_is_3){
      ECIvk = ECIvk.tail(3);
      ECIvk = normalise(ECIvk);
      ECIvk = closestQuatForVecPoint(ECIvkp1,satvk,ECIvk);
    }
    vec4 dq = normquaterr(ECIvk,ECIvkp1);
    vec4 qerr = normquaterr(ECIvk,qk);
    if(as_scalar(qerr(0))!=0){
      qerr *= sign(qerr(0));
    }
    vec3 wkdes = normalise(dq.tail(3))*2.0*asin(norm(dq.tail(3)))/dt0;
    wkdes = RmatT*wkdes;

    vec3 qq = normalise(sat.Jcom*qerr.tail(3));
    qq = normalise(qq);
    vec3 bn = normalise(Bbody);//normalise(sat.Jcom*cross(xx,x2));

      //uk = cross(Bbody,sat.Jcom*(dampgain*(wk-dot(wk,x3)*x3) + quatgain*acos(norm_dot(ECIvkBody,satvk))*normalise(cross(ECIvkBody,satvk))))/nB2;
    uk.head(sat.number_MTQ) = (sat.mtq_ax_mat.t()*cross(Bbody,(-dist_torq + dampgain*(sat.Jcom*wk) + velgain*(sat.Jcom*(wk-wkdes)) + quatgain*2.0*acos(as_scalar(qerr(0)))*qq)))/nB2;
    // if(norm(cross((qq),x3))<0.02){
    //   vec3 x4 = normalise(cross(qq,bn));
    //   uk.head(sat.number_MTQ) = sat.mtq_ax_mat.t()*cross(Bbody,(dampgain*(sat.Jcom_noRW*wk) + velgain*sat.Jcom_noRW*(wk-wkdes) + quatgain*2*acos(qerr(0))*x4*sign(norm_dot(x4,qq))))/nB2;
    // }
  }
  return uk;
}

/* This function finds the TVLQR gains after alilqr is run
  Inputs:
    Xset, Uset, Rset, Vset, Bset - final states, control vectors, orbital position velocity and magfield
    QN - final time Q - 6 x 6 matrix
    R - control cost - 3 x 3 matrix
    dt0 - double
    satAlignVector, vNslew - 3 x 1 vectors for alignment
    costSettings - settings to find q
  Outputs:
   Kset - 3 x 6 x N-1 cube, TVLQR gains
   Sset - 6 x 6 x N cube, intermediate values used to find gains
*/
tuple<cube, cube> OldPlanner::findK(double dt_tvlqr0, TRAJECTORY_FORM& traj, VECTOR_INFO_FORM& vecs, COST_SETTINGS_FORM costSettings_tmp)
{
  //Initialize Sset and Kset
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat sunset = get<4>(vecs);
  mat Vset = get<2>(vecs);
  vec pset = get<7>(vecs);
  int N = Xset.n_cols;
  if(verbose) {
    cout<<"N is: "<<N<<"\n";
  }
  cube Kset_lqr = cube(sat.control_N(), sat.reduced_state_N(), N-1).zeros();
  cube Sset = cube(sat.reduced_state_N(), sat.reduced_state_N(), N).zeros();

  //Initialize various states & properties at time k
  int k = N-1;
  // int tk = dt_tvlqr0*(k-1)+1;
  //vec rk = Rset.col(tk);
  vec xk = Xset.col(k);
  vec xkp1 = xk;  // Terminal step: no next state
  vec3 bk = Bset.col(k);
  vec3 sk = satvec.col(k);
  vec ek = ECIvec.col(k);
  vec uk = vec(sat.control_N()).zeros();
  vec ukp = vec(sat.control_N()).zeros();
  vec4 qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
  //Find lkxx = LQR Q because it's the state cost matrix
  cost_jacs costJac = sat.costJacobians(k, N, xk, xkp1, uk,ukp, sk,ek,bk, &costSettings_tmp);
  mat lkxx = costJac.lxx;
  mat lkuu = costJac.luu;
  
  mat Sk = lkxx;//get<0>(weights);// mat66().zeros();
  mat Kk = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  Sset.slice(k) = Sk;//mat66().zeros();
  mat A = mat(sat.state_N(),sat.state_N()).zeros();
  mat B = mat(sat.state_N(),sat.control_N()).zeros();
  mat Aqk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  mat Bqk = mat(sat.reduced_state_N(),sat.control_N()).zeros();

  mat C = mat(sat.state_N(),3).zeros();
  //Find Gk and initialize Gkp1 and Skp1
  mat Gk = sat.findGMat(qk);
  mat Gkp1 = Gk;
  mat Skp1 = Sk;
  //vec3 prop_torq = this->prop_torq;

  //Loop backwards
  DYNAMICS_INFO_FORM dynamics_info_kp1 = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kp1;
  vec eigvals;
  mat eigvecs;

  for(int k=N-2; k>=0; k--)
  {
    dynamics_info_kp1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
    //Update states and stuff at time k
    Gkp1 = Gk;
    // tk = dt_tvlqr0*(k-1)+1;
    //rk = Rset.col(tk);
    xk = Xset.col(k);
    xkp1 = Xset.col(k+1);  // Next state for path length cost
    uk = Uset.col(k);
    ukp = ukp.zeros();
    if(k>0){ukp = Uset.col(k-1);}
    sk = satvec.col(k);
    ek = ECIvec.col(k);
    //vk = Vset.col(tk);
    qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
    bk = Bset.col(k);
    Skp1 = Sk;
    //Get lkxx = LQR Q because it's the state cost matrix
    costJac = sat.costJacobians(k, N, xk, xkp1, uk,ukp,sk,ek, bk,&costSettings_tmp);
    lkxx = costJac.lxx;
    lkuu = costJac.luu;
    //Get Gk
    Gk = sat.findGMat(qk);
    //Get A, B
    tuple<mat, mat,mat> AB = rk4zJacobians(dt_tvlqr0,xk, uk, sat,dynamics_info_k,dynamics_info_kp1);
    A = get<0>(AB);//px4MatToArma(&A_px4);
    B = get<1>(AB);//px4MatToArma(&B_px4);
    C = get<2>(AB);
    //Get Aqk and Bqk
    Aqk = Gkp1*A*trans(Gk);
    Bqk = Gkp1*B;
    mat Reff = lkuu + trans(Bqk)*Skp1*Bqk;
    Kk = solve(Reff, (trans(Bqk)*Skp1*Aqk),solve_opts::likely_sympd+solve_opts::fast);//inv(R + trans(Bqk)*Skp1*Bqk)*(trans(Bqk)*Skp1*Aqk);//
    
    Kset_lqr.slice(k) = Kk;
    // Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    // Sk = lkxx + trans(Kk)*lkuu*Kk + solve((Aqk-Bqk*Kk), Skp1*(Aqk-Bqk*Kk));
    // Sk = lkxx + trans(Kk)*lkuu*Kk + trans(Aqk-Bqk*Kk)*Skp1*(Aqk-Bqk*Kk);
    Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;

    Sk = 0.5*(Sk+trans(Sk));
    // Sk = 0.5*(Sk+lkxx);
    Sset.slice(k) = Sk;
  }
  if(verbose){cout<<size(Kset_lqr)<<"\n";}
  if(verbose){cout<<size(Sset)<<"\n";}
  return make_tuple(Kset_lqr, Sset);
}

// Variant of findK that uses a provided terminal S matrix instead of computing from terminal cost
// This enables proper cost propagation when computing K-gains in segments
tuple<cube, cube> OldPlanner::findKwithTerminalS(double dt_tvlqr0, TRAJECTORY_FORM& traj, VECTOR_INFO_FORM& vecs, COST_SETTINGS_FORM costSettings_tmp, mat terminal_S)
{
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat sunset = get<4>(vecs);
  mat Vset = get<2>(vecs);
  vec pset = get<7>(vecs);
  int N = Xset.n_cols;
  if(verbose) {
    cout<<"N is (with terminal S): "<<N<<"\n";
  }
  cube Kset_lqr = cube(sat.control_N(), sat.reduced_state_N(), N-1).zeros();
  cube Sset = cube(sat.reduced_state_N(), sat.reduced_state_N(), N).zeros();

  int k = N-1;
  vec xk = Xset.col(k);
  vec xkp1 = xk;
  vec3 bk = Bset.col(k);
  vec3 sk = satvec.col(k);
  vec ek = ECIvec.col(k);
  vec uk = vec(sat.control_N()).zeros();
  vec ukp = vec(sat.control_N()).zeros();
  vec4 qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
  
  // Use provided terminal S instead of computing from cost
  mat Sk = terminal_S;
  mat Kk = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  Sset.slice(k) = Sk;
  
  mat A = mat(sat.state_N(),sat.state_N()).zeros();
  mat B = mat(sat.state_N(),sat.control_N()).zeros();
  mat Aqk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  mat Bqk = mat(sat.reduced_state_N(),sat.control_N()).zeros();
  mat C = mat(sat.state_N(),3).zeros();
  mat Gk = sat.findGMat(qk);
  mat Gkp1 = Gk;
  mat Skp1 = Sk;

  DYNAMICS_INFO_FORM dynamics_info_kp1 = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kp1;

  for(int k=N-2; k>=0; k--)
  {
    dynamics_info_kp1 = dynamics_info_k;
    dynamics_info_k = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
    Gkp1 = Gk;
    xk = Xset.col(k);
    xkp1 = Xset.col(k+1);
    uk = Uset.col(k);
    ukp = ukp.zeros();
    if(k>0){ukp = Uset.col(k-1);}
    sk = satvec.col(k);
    ek = ECIvec.col(k);
    qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
    bk = Bset.col(k);
    Skp1 = Sk;
    
    cost_jacs costJac = sat.costJacobians(k, N, xk, xkp1, uk,ukp,sk,ek, bk,&costSettings_tmp);
    mat lkxx = costJac.lxx;
    mat lkuu = costJac.luu;
    
    Gk = sat.findGMat(qk);
    tuple<mat, mat,mat> AB = rk4zJacobians(dt_tvlqr0,xk, uk, sat,dynamics_info_k,dynamics_info_kp1);
    A = get<0>(AB);
    B = get<1>(AB);
    C = get<2>(AB);
    Aqk = Gkp1*A*trans(Gk);
    Bqk = Gkp1*B;
    Kk = solve((lkuu + trans(Bqk)*Skp1*Bqk), (trans(Bqk)*Skp1*Aqk),solve_opts::likely_sympd+solve_opts::fast);
    Kset_lqr.slice(k) = Kk;
    Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    Sk = 0.5*(Sk+trans(Sk));
    Sset.slice(k) = Sk;
  }
  if(verbose){cout<<size(Kset_lqr)<<"\n";}
  if(verbose){cout<<size(Sset)<<"\n";}
  return make_tuple(Kset_lqr, Sset);
}

tuple<cube, cube> OldPlanner::findKwDist(double dt_tvlqr0, TRAJECTORY_FORM& traj, VECTOR_INFO_FORM& vecs, COST_SETTINGS_FORM costSettings_tmp)
{
  //Initialize Sset and Kset
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  mat ECIvec = get<6>(vecs);
  mat satvec = get<5>(vecs);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  vec tset = get<0>(vecs);
  mat sunset = get<4>(vecs);
  mat Vset = get<2>(vecs);
  vec pset = get<7>(vecs);
  int N = Xset.n_cols;
  if(verbose) {
    cout<<"N is: "<<N<<"\n";
  }

  cube Kset_lqr = cube(sat.control_N(), sat.reduced_state_N()+3, N-1).zeros();
  cube Sset = cube(sat.reduced_state_N()+3, sat.reduced_state_N()+3, N).zeros();

  //Initialize various states & properties at time k
  int k = N-1;
  vec xk = Xset.col(k);
  vec xkp1 = xk;  // Terminal step: no next state
  vec3 ek = ECIvec.col(k);
  vec3 sk = satvec.col(k);
  vec3 bk = Bset.col(k);
  vec uk = vec(sat.control_N()).zeros();
  vec ukp = vec(sat.control_N()).zeros();
  //vec vk = Vset.col(tk);
  vec4 qk = xk.rows(sat.quat0index(),sat.quat0index()+3);

  //Find lkxx = LQR Q because it's the state cost matrix
  cost_jacs costJac = sat.costJacobians(k, N, xk, xkp1, uk,ukp, sk,ek,bk, &costSettings_tmp);
  mat lkxx = costJac.lxx;
  mat lkuu = costJac.luu;
  mat Sk = mat(sat.reduced_state_N()+3,sat.reduced_state_N()+3).zeros();
  Sk(span(0,sat.reduced_state_N()-1),span(0,sat.reduced_state_N()-1)) = lkxx;//get<0>(weights);// mat66().zeros();
  Sk(span(sat.reduced_state_N(),sat.reduced_state_N()+2),span(sat.reduced_state_N(),sat.reduced_state_N()+2)) += mat33().eye();

  mat Kk = mat(sat.control_N(),sat.reduced_state_N()+3).zeros();
  Sset.slice(k) = Sk;//mat66().zeros();
  mat A = mat(sat.state_N(),sat.state_N()).zeros();
  mat B = mat(sat.state_N(),sat.control_N()).zeros();
  mat Aqk = mat(sat.reduced_state_N()+3,sat.reduced_state_N()+3).zeros();
  mat Bqk = mat(sat.reduced_state_N()+3,sat.control_N()).zeros();

  mat C = mat(sat.state_N(),3).zeros();

  //Find Gk and initialize Gkp1 and Skp1
  mat Gk = sat.findGMat(qk);
  mat Gkp1 = Gk;
  mat Skp1 = Sk;
  //vec3 prop_torq = this->prop_torq;

  //Loop backwards
  DYNAMICS_INFO_FORM dynamics_info_kp1 = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kp1;
  vec eigvals;
  mat eigvecs;

  for(int k=N-2; k>=0; k--)
  {

    dynamics_info_kp1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),1);
    //Update states and stuff at time k
    Gkp1 = Gk;
    // tk = dt_tvlqr0*(k-1)+1;
    //rk = Rset.col(tk);
    xk = Xset.col(k);
    xkp1 = Xset.col(k+1);  // Next state for path length cost
    uk = Uset.col(k);
    ukp = ukp.zeros();
    if(k>0){ukp = Uset.col(k-1);}
    sk = satvec.col(k);
    ek = ECIvec.col(k);
    //vk = Vset.col(tk);
    qk = xk.rows(sat.quat0index(),sat.quat0index()+3);
    bk = Bset.col(k);
    Skp1 = Sk;
    //Get lkxx = LQR Q because it's the state cost matrix
    costJac = sat.costJacobians(k, N, xk, xkp1, uk,ukp,sk,ek, bk,&costSettings_tmp);

    lkxx = costJac.lxx;
    // lkxx = mat66().eye();//costJac.lxx;
    lkuu = costJac.luu;//#*1e5; //REMOVE BEFORE FLIGHT
    // //lkxx = join_rows(join_cols(mat33().eye()*swpoint, mat33().zeros()),join_cols(mat33().zeros(),mat33().eye()*sv1));
    //Get Gk
    Gk = sat.findGMat(qk);
    //Get A, B
    tuple<mat, mat, mat> AB = rk4zJacobians(dt_tvlqr0,xk, uk, sat,dynamics_info_k,dynamics_info_kp1);
    A = get<0>(AB);//px4MatToArma(&A_px4);
    B = get<1>(AB);//px4MatToArma(&B_px4);
    C = get<2>(AB);
    //Get Aqk and Bqk
    Aqk.zeros();
    Bqk.zeros();
    Bqk.rows(0,sat.reduced_state_N()-1) = Gkp1*B;
    Aqk(span(0,sat.reduced_state_N()-1),span(0,sat.reduced_state_N()-1)) = Gkp1*A*trans(Gk);
    Aqk(span(0,sat.reduced_state_N()-1),span(sat.reduced_state_N(),sat.reduced_state_N()+2)) = Gkp1*C;
    Aqk(span(sat.reduced_state_N(),sat.reduced_state_N()+2),span(sat.reduced_state_N(),sat.reduced_state_N()+2)) = mat33().eye();
    //Get Kk = (R+Bqk.'*Skp1*Bqk)\(Bqk.'*Skp1*Aqk)
    //mat tmpVal = lkuu + trans(Bqk)*Skp1*Bqk;
    //eig_gen(eigvals,eigvecs,tmpVal);
    //Kk = eigvecs*diagmat(1/clamp(abs(eigvals),1e-8,datum::inf))*eigvecs.t()*trans(Bqk)*Skp1*Aqk;
    Kk = solve((lkuu + trans(Bqk)*Skp1*Bqk), (trans(Bqk)*Skp1*Aqk),solve_opts::likely_sympd+solve_opts::fast);//inv(R + trans(Bqk)*Skp1*Bqk)*(trans(Bqk)*Skp1*Aqk);//

    Kset_lqr.slice(k) = Kk;
    // Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    // Sk = lkxx + trans(Kk)*lkuu*Kk + solve((Aqk-Bqk*Kk), Skp1*(Aqk-Bqk*Kk));
    // Sk = trans(Kk)*lkuu*Kk + trans(Aqk-Bqk*Kk)*Skp1*(Aqk-Bqk*Kk);
    Sk = lkxx + trans(Aqk)*Skp1*Aqk - trans(Aqk)*Skp1*Bqk*Kk;
    // Sk(span(0,sat.reduced_state_N()-1),span(0,sat.reduced_state_N()-1)) += lkxx;

    Sk = 0.5*(Sk+trans(Sk));
    // Sk = 0.5*(Sk+lkxx);
    Sset.slice(k) = Sk;
  }
  if(verbose){cout<<size(Kset_lqr)<<"\n";}
  if(verbose){cout<<size(Sset)<<"\n";}
  return make_tuple(Kset_lqr, Sset);
}


/*This function generates a (NOT initial) trajectory for the trajectory optimizer, using rk4, based on altering a previous trajectory
  Arguments:
    Xset - previous trajectory states - 7 x N matrix
    Uset - previous trajectory control inputs - 3 x N-1 matrix
    Kset - gain K at each timestep (from backwards pass) - 3 x 6 x N cube
    dset - from backwards pass - 3 x N matrix
    Rset - orbital position at each timestep - 3 x N matrix
    alph - double, (hyper)parameter
    lambdaSet - lambda vector - 6 x N matrix
  Returns:
    newX - new states of trajectory - 7 x N matrix
    newU - new control inputs of trajectory - 3 x N matrix
*/
 TRAJECTORY_FORM OldPlanner::generateTrajectory( double dt0,  double alpha, TRAJECTORY_FORM traj,  VECTOR_INFO_FORM& vecs,  const cube& Kset,  const mat& dset, bool useDist)
{
  //Initialize newU, newX
  mat newX;
  mat Xset = get<0>(traj);
  newX.copy_size(Xset);
  newX.zeros();

  // Initialize candidate slack storage for infeasible start
  if (use_infeasible_start && !slack_Sset.is_empty()) {
    slack_Sset_new = mat(slack_Sset.n_rows, slack_Sset.n_cols, fill::zeros);
  }
  //newX.fill(datum::nan);
  mat newU;
  mat Uset = get<1>(traj);
  newU.copy_size(Uset);
  newU.zeros();

  mat newTQ;
  mat TQset = get<3>(traj);
  newTQ.copy_size(TQset);
  newTQ.zeros();
  //newU.fill(datum::nan);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);


  //Copy initial state of x to newX
  vec newXk = Xset.col(0);
  newX.col(0) = newXk;

  vec newTQk = vec(3).zeros();
  // vec4 newQk = newXk.rows(sat.quat0index(),sat.quat0index()+3);
  vec4 Qkprev = vec(4,fill::zeros);
  vec dt_timevec = get<2>(traj);

  //Initialize stuff for inside loop
  //vec delX = vec(6).zeros();
  //vec newXprev = newX.col(0);
  vec oldXprev = mat(newXk);
  vec4 oldQprev = oldXprev.rows(sat.quat0index(),sat.quat0index()+3);
  //mat Kprev = Kset.slice(0);
  //vec3 delU = vec(3).zeros();
  vec newUprev = vec(sat.control_N()).zeros();
  //vec3 oldUprev = Uset.col(0);
  //vec3 dprev = dset.col(0);

  vec3 angErr;
  vec4 quatErr;
  vec otherErr = vec(sat.reduced_state_N()-6).zeros();
  vec3 avErr = vec3().zeros();
  int N = Xset.n_cols;
  //Loop from k=1 to k=N-1 and update newU and newX
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),int(useDist));
  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;

  for(int k=1; k<N; k++)
  {

    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),Sset.col(k),int(useDist));
    //Update all the "prev" variables
    //newXprev = newXk;
    oldXprev = Xset.col(k-1);
    oldQprev = oldXprev.rows(sat.quat0index(),sat.quat0index()+3);
    Qkprev =  newXk.rows(sat.quat0index(),sat.quat0index()+3);

    quatErr = normquaterr(oldQprev,Qkprev);//normalise(join_cols(vec(1).ones()*as_scalar(oldQprev.t()*Qkprev),oldQprev(0)*Qkprev.rows(1,3) - Qkprev(0)*oldQprev.rows(1,3)-cross(oldQprev.rows(1,3),Qkprev.rows(1,3))));
    // quatErr = normalise(join_cols(vec(1).ones()*as_scalar(oldQprev.t()*Qkprev),-oldQprev(0)*Qkprev.rows(1,3) + Qkprev(0)*oldQprev.rows(1,3)+cross(oldQprev.rows(1,3),Qkprev.rows(1,3))));

    // quatErr *= sign(quatErr(0));
    if(quaternionTo3VecMode >= 2 ){
      if (quaternionTo3VecMode ==2 )// Cayley
      {
        if(abs(quatErr(0))<EPSVAR){
          if(abs(quatErr(0))>0){
            quatErr(0) = EPSVAR*sign(quatErr(0));
          }
          else{
            quatErr(0) = EPSVAR;
          }
        }
        angErr = (quatErr.rows(1,3))/(quatErr(0));
      }else if(quaternionTo3VecMode ==3){//qev w/ qe0>0
        if(abs(quatErr(0))>0){
          angErr = quatErr.rows(1,3)*sign(quatErr(0));
        }
        else{angErr = quatErr.rows(1,3);}

      }else if(quaternionTo3VecMode ==4){// qev
        angErr = quatErr.rows(1,3);
      }else{
        // Modes 5 and 6 (2xMRP) have incorrect scaling (constant=2 instead of 1)
        // and are not consistent with the W^T linearization used in the backward pass.
        throw std::invalid_argument("quaternionTo3VecMode 5 and 6 are not supported. Use modes 0-4.");
      }
    }else{//mode is 0 or 1,(0 is MRP with qe0>0, 1 is MRP)
      if (quaternionTo3VecMode != 1)
      {
        if(abs(quatErr(0))>0){
          quatErr *= sign(quatErr(0));
        }
      } //qev w/ qe0>0
      angErr = 2.0*(quatErr.rows(1,3))/(1+quatErr(0));
    }

    otherErr = newXk.tail(sat.reduced_state_N()-6) - oldXprev.tail(sat.reduced_state_N()-6);
    avErr = newXk.head(3) - oldXprev.head(3);
    vec errVec = join_cols(avErr,angErr,otherErr);
    newUprev = Uset.col(k-1) + Kset.slice(k-1)*errVec + alpha*dset.col(k-1);
    // newUprev = Uset.col(k-1)+Kset.slice(k-1)*join_cols(newXk.rows(sat.avindex0(),sat.avindex0()+2)-oldXprev.rows(sat.avindex0(),sat.avindex0()+2),(oldQprev(0)*Qkprev.rows(1,3) - Qkprev(0)*oldQprev.rows(1,3)-cross(oldQprev.rows(1,3),Qkprev.rows(1,3)))/(as_scalar(oldQprev.t()*Qkprev))) + alpha*dset.col(k-1);

    // Compute slack update (infeasible start)
    vec s_new;
    if (use_infeasible_start && !slack_Kset.is_empty() && (k-1) < (int)slack_Sset.n_cols) {
      s_new = slack_Sset.col(k-1) + slack_Kset.slice(k-1)*errVec + alpha*slack_dset.col(k-1);
    }

    for(int i = 0; i < sat.control_N(); i++)
    {
      double ucheck = newUprev(i);
      if((isnan(ucheck)||isinf(ucheck)))//||abs(ucheck)>100000000))
      {
        if (use_infeasible_start && k <= 3) {
          cout << "  [infeas fwd] NaN ctrl at k=" << k << " i=" << i << "\n";
          cout << "    errVec=" << errVec.t();
          cout << "    K_max=" << Kset.slice(k-1).max() << " K_min=" << Kset.slice(k-1).min() << "\n";
          cout << "    d=" << dset.col(k-1).t();
          cout << "    u_prev=" << Uset.col(k-1).t();
        }
        return make_tuple(newX.fill(datum::nan), newU.fill(datum::nan),dt_timevec,newTQ.fill(datum::nan));
      }
    }
    //get newXk using integrator (RK4 or Euler)
    tuple<vec,vec> integOut = _useEuler
      ? eulerz(dt0, newXk, newUprev, sat, dynamics_info_kn1, dynamics_info_k)
      : rk4z(dt0, newXk, newUprev, sat, dynamics_info_kn1, dynamics_info_k);
    newXk = get<0>(integOut);
    newTQk = get<1>(integOut);

    // Apply slack correction to state (infeasible start)
    // x_{k+1} = f(x_k, u_k) + G'·s_k  (G maps full→reduced, G' maps reduced→full)
    if (use_infeasible_start && s_new.n_elem > 0) {
      mat Gkp1 = sat.findGMat(normalise(newXk.rows(3, 6)));
      newXk += Gkp1.t() * s_new;
      // Store new slack for cost evaluation
      if ((k-1) < (int)slack_Sset_new.n_cols) {
        slack_Sset_new.col(k-1) = s_new;
      }
    }

    newXk = sat.state_norm(newXk);

    // newQk = normalise(newXk.rows(sat.quat0index(), sat.quat0index()+3));
    // newXk.rows(sat.quat0index(), sat.quat0index()+3) = newQk;
    //Update newX and newU with the new state and control vector
    newX.col(k) = newXk;
    newTQ.col(k-1) = newTQk;
    newU.col(k-1) = newUprev;

  }
  return make_tuple(newX, newU,dt_timevec,newTQ);
}
// ========================================================================
// Infeasible start helper methods (ALTRO-style slack variables on dynamics)
// ========================================================================

/**
 * Compute augmented Lagrangian cost for slack variables: Σ_k [λ'·s + (μ/2)·||s||²]
 */
double OldPlanner::slackCost(const mat& Sset_in) const {
    if (!use_infeasible_start || Sset_in.is_empty()) return 0.0;
    double cost = 0.0;
    double mu_eff = slack_mu + slack_w;  // AL penalty + fixed cost
    for (int k = 0; k < (int)Sset_in.n_cols; k++) {
        vec sk = Sset_in.col(k);
        cost += dot(slack_lambdaSet.col(k), sk) + 0.5 * mu_eff * dot(sk, sk);
    }
    return cost;
}

/**
 * Compute dynamics defects of a given trajectory and store as initial slacks.
 * Defect at step k: s_k = G_{k+1} · (x_{k+1} - f(x_k, u_k))  [in reduced state space]
 */
void OldPlanner::initSlacksFromDefects(double dt0, TRAJECTORY_FORM& traj, VECTOR_INFO_FORM& vecs) {
    mat Xset = get<0>(traj);
    mat Uset = get<1>(traj);
    int N = Xset.n_cols;
    int n_red = sat.reduced_state_N();

    mat Bset = get<3>(vecs);
    mat Rset = get<1>(vecs);
    mat Vset = get<2>(vecs);
    mat Sset_env = get<4>(vecs);
    vec pset = get<7>(vecs);

    slack_Sset = mat(n_red, N, fill::zeros);

    DYNAMICS_INFO_FORM dyn_k, dyn_kp1;
    dyn_k = make_tuple(Bset.col(0), Rset.col(0), pset(0), Vset.col(0), Sset_env.col(0), 0);

    for (int k = 0; k < N - 1; k++) {
        dyn_kp1 = make_tuple(Bset.col(k+1), Rset.col(k+1), pset(k+1), Vset.col(k+1), Sset_env.col(k+1), 0);

        // Simulate one step from current state
        auto [x_dyn, tq] = _useEuler
            ? eulerz(dt0, Xset.col(k), Uset.col(k), sat, dyn_k, dyn_kp1)
            : rk4z(dt0, Xset.col(k), Uset.col(k), sat, dyn_k, dyn_kp1);
        x_dyn = sat.state_norm(x_dyn);

        // Compute defect in reduced state space using G matrix
        vec x_actual = Xset.col(k + 1);
        mat Gkp1 = sat.findGMat(x_actual.rows(3, 6));

        // Defect: difference between actual and predicted next state
        // For non-quaternion states: simple difference
        // For quaternion: use G to map to reduced space
        vec defect_full = x_actual - x_dyn;
        // Map to reduced space: ω difference + attitude error + other state difference
        vec defect_red(n_red, fill::zeros);
        defect_red.head(3) = defect_full.head(3);  // angular velocity defect
        // Attitude defect: use quaternion error mapped to 3-vector
        vec4 q_dyn = normalise(x_dyn.rows(3, 6));
        vec4 q_actual = normalise(x_actual.rows(3, 6));
        vec4 qe = normquaterr(q_dyn, q_actual);
        if (qe(0) < 0) qe = -qe;
        // Use same parameterization as backward pass (MRP by default, mode 0)
        defect_red.rows(3, 5) = 2.0 * qe.rows(1, 3) / (1.0 + qe(0));
        // Other states (RW momentum, etc.)
        if (n_red > 6) {
            defect_red.tail(n_red - 6) = defect_full.tail(n_red - 6);
        }

        slack_Sset.col(k) = defect_red;
        dyn_k = dyn_kp1;
    }

    if (verbose) {
        cout << "Infeasible start: initial slack norm = " << norm(slack_Sset, "fro")
             << ", max = " << abs(slack_Sset).max() << "\n";
    }
}

/**
 * Generate a SLERP trajectory from x0 toward goal quaternion.
 * States: SLERP quaternion with finite-difference angular velocity, constant RW momentum.
 * Controls: zero.
 */
TRAJECTORY_FORM OldPlanner::generateSlerpTrajectory(double dt0, vec x0, vec4 q_goal, int N, VECTOR_INFO_FORM& vecs, int ctrl_mode) {
    mat Xset(sat.state_N(), N, fill::zeros);
    mat Uset(sat.control_N(), N, fill::zeros);
    mat TQset(3, N, fill::zeros);
    vec t = get<0>(vecs);

    vec4 q0 = normalise(x0.rows(3, 6));
    vec4 qg = normalise(q_goal);

    // Ensure shortest path
    if (dot(q0, qg) < 0) qg = -qg;

    double dot_val = min(abs(dot(q0, qg)), 1.0);
    double theta = acos(dot_val);  // half-angle in quaternion space
    double sin_theta = sin(theta);

    // Generate SLERP quaternions with front-loaded profile:
    // Concentrate rotation in first ~30% of trajectory, hold at goal for remainder.
    // This prevents the optimizer from getting stuck at the constant-rate SLERP minimum.
    double slew_frac = 0.3;  // Do full rotation in first 30% of trajectory
    for (int k = 0; k < N; k++) {
        double t_norm = (double)k / (N - 1);
        double frac = min(1.0, t_norm / slew_frac);  // Reaches 1.0 at slew_frac of trajectory
        vec4 qk;
        if (theta < 1e-10) {
            qk = q0;
        } else {
            qk = normalise((sin((1.0 - frac) * theta) * q0 + sin(frac * theta) * qg) / sin_theta);
        }
        Xset(span(3, 6), k) = qk;
        // RW momentum: constant from initial state
        if (sat.state_N() > 7) {
            Xset.rows(7, sat.state_N() - 1).col(k) = x0.rows(7, sat.state_N() - 1);
        }
    }

    // Compute angular velocity from finite differences
    for (int k = 0; k < N - 1; k++) {
        vec4 qk = normalise(Xset(span(3, 6), k));
        vec4 qkp1 = normalise(Xset(span(3, 6), k + 1));
        vec4 dq = normquaterr(qk, qkp1);
        if (dq(0) < 0) dq = -dq;
        double half_angle = acos(min(abs(dq(0)), 1.0));
        vec3 omega_k;
        if (half_angle > 1e-12) {
            omega_k = (2.0 * half_angle / dt0) * normalise(dq.rows(1, 3));
        } else {
            omega_k = vec3(fill::zeros);
        }
        Xset(span(0, 2), k) = omega_k;
    }
    // Terminal: zero angular velocity (at goal, at rest)
    Xset(span(0, 2), N - 1) = vec3(fill::zeros);

    // ================================================================
    // Inverse dynamics: compute best-effort controls for SLERP path
    // τ_des = J·α + ω×(Jω + h)
    // Solve MTQ: m = (B × τ) / |B|², clamp to limits
    // Solve RW: residual torque along RW axis, clamp
    // ================================================================
    // ================================================================
    // Compute initial controls for SLERP path
    // ctrl_mode: 0=zero, 1=inverse dynamics, 2=random
    // ================================================================
    if (ctrl_mode == 1) {
      // Inverse dynamics: τ_des = J·α + ω×(J·ω+h), solve for MTQ/RW
      mat Bset = get<3>(vecs);
      mat33 J_body = sat.Jcom_noRW;
      double rw_scale = NONMTQ_TORQ_SCALE;

      for (int k = 0; k < N - 1; k++) {
          vec3 wk = Xset(span(0, 2), k);
          vec3 wkp1 = Xset(span(0, 2), k + 1);
          vec3 alpha_des = (wkp1 - wk) / dt0;

          vec3 Jw = J_body * wk;
          for (int j = 0; j < sat.number_RW; j++) {
              Jw += Xset(7 + j, k) * vec3(sat.rw_ax_mat.col(j));
          }
          vec3 tau_des = J_body * alpha_des + cross(wk, Jw);

          vec3 tau_remaining = tau_des;
          for (int j = 0; j < sat.number_RW; j++) {
              vec3 ax(sat.rw_ax_mat.col(j));
              double tau_proj = dot(tau_remaining, ax);
              double u_rw = -tau_proj / rw_scale;
              double u_lim = sat.RW_max_torq.at(j) / rw_scale;
              u_rw = std::max(-u_lim, std::min(u_lim, u_rw));
              Uset(sat.number_MTQ + j, k) = u_rw;
              tau_remaining -= (-u_rw * rw_scale) * ax;
          }

          vec3 Bk = Bset.col(k);
          double Bnorm2 = dot(Bk, Bk);
          if (Bnorm2 > 1e-20 && sat.number_MTQ > 0) {
              vec3 m_des = cross(Bk, tau_remaining) / Bnorm2;
              vec u_mtq = pinv(sat.mtq_ax_mat) * m_des;
              for (int j = 0; j < sat.number_MTQ; j++) {
                  double u_lim = sat.MTQ_max.at(j);
                  u_mtq(j) = std::max(-u_lim, std::min(u_lim, u_mtq(j)));
              }
              Uset(span(0, sat.number_MTQ - 1), k) = u_mtq;
          }
      }
    } else if (ctrl_mode == 2) {
      // Random controls within actuator limits
      double rw_scale = NONMTQ_TORQ_SCALE;
      arma::arma_rng::set_seed_random();
      for (int k = 0; k < N - 1; k++) {
          for (int j = 0; j < sat.number_MTQ; j++) {
              double u_lim = sat.MTQ_max.at(j);
              Uset(j, k) = u_lim * (2.0 * arma::randu() - 1.0);
          }
          for (int j = 0; j < sat.number_RW; j++) {
              double u_lim = sat.RW_max_torq.at(j) / rw_scale;
              Uset(sat.number_MTQ + j, k) = u_lim * (2.0 * arma::randu() - 1.0);
          }
      }
    }
    // ctrl_mode == 0: leave Uset as zeros

    if (verbose) {
        double ang_total = 2.0 * theta * 180.0 / datum::pi;
        cout << "SLERP trajectory: " << ang_total << "° rotation over " << N
             << " steps (dt=" << dt0 << "s), inv-dyn controls computed\n";
    }

    return make_tuple(Xset, Uset, t.head(N), TQset);
}


/*This function generates the initial trajectory for the trajectory optimizer, using rk4
  Arguments:
    x0 - initial state vector - 7 x 1 vector
    Uset - control inputs for the trajectory, size 3 x N-1
    dt - time between steps of trajectory - double
  Returns:
   Xset - set of states in trajectory, size 7 x N matrix
*/
TRAJECTORY_FORM OldPlanner::generateInitialTrajectory(double dt0, vec x0, mat Uset,VECTOR_INFO_FORM vecs) {
  //Initialize newX

  int N = Uset.n_cols;
  mat Xset = mat(sat.state_N(), N).fill(datum::nan);
  mat TQset = mat(3, N).fill(datum::nan);

  //Xset.fill(datum::nan);

  //Copy initial state of x to xk
  vec xk = vec(x0);
  xk = sat.state_norm(xk);
  vec4 qk = normalise(xk.rows(sat.quat0index(), sat.quat0index()+3));
  xk.rows(sat.quat0index(),sat.quat0index()+3) = qk;
  Xset.col(0) = xk;


  //Initialize stuff for inside loop
  //vec xprev = vec(x0);
  mat Bset = get<3>(vecs);
  mat Rset = get<1>(vecs);
  vec t = get<0>(vecs);

  mat Vset = get<2>(vecs);
  mat Sset = get<4>(vecs);
  vec pset = get<7>(vecs);
  vec uk = vec(sat.control_N()).zeros();
  vec3 Bk = Bset.col(0);


  //Loop from k = 1 to N-1 and fill in Xset, using rk4
  DYNAMICS_INFO_FORM dynamics_info_kn1 = make_tuple(Bset.col(0),Rset.col(0),pset(0),Vset.col(0),Sset.col(0),1);

  DYNAMICS_INFO_FORM dynamics_info_k = dynamics_info_kn1;
  tuple<vec,vec> dynout;
  for(int k=1; k<N; k++)
  {
    dynamics_info_kn1 = dynamics_info_k;
    dynamics_info_k =  make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),Sset.col(k),1);
    uk = Uset.col(k-1);
    //xprev = xk;
    // Bk = Bset.col(k-1);
    dynout = _useEuler
      ? eulerz(dt0, xk, uk, sat, dynamics_info_kn1, dynamics_info_k)
      : rk4z(dt0, xk, uk, sat, dynamics_info_kn1, dynamics_info_k);

    xk = sat.state_norm(get<0>(dynout));
    // xk.rows(3, 6)  = normalise(xk.rows(3, 6));
    Xset.col(k) = xk;
    TQset.col(k-1) = get<1>(dynout);
  }

  return make_tuple(Xset,Uset,t,TQset);
}



/*This method actually does alilqr
  Inputs:
    Xset - set of states so far - 7 x N matrix
    Uset - set of control inputs - 3 x N-1 matrix
    Rset - orbital position - 3 x N matrix
    Vset - orbital velocity - 3 x N matrix
    Bset - orbital magnetic field - 3 x N matrix
    R - control cost, 3x3 matrix
    QN - Q at t = N, 6x6 matrix
    costSettings - settings for Q
    forwardPassSettings - settings for forwardPass
    alilqrSettings - settings for alilqr - contains int lagMultInit, double penInit, int regInit, int maxOuterIter,
    double gradTol, double costTol, double cmax, int zCountLim, int maxIter, double penMax, double penScale, int lagMultMax,
    double ilqrCostTol
  Outputs:
    P - covariance - 6 x 6 x N cube
    K - gains - 3 x 6 x N cube
    dset - unclear - 3 x N matrix
    delV - 1 x 2 matrix
    rho,drho - same as above
*/
ALILQR_OUTPUT_FORM OldPlanner::alilqr(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, COST_SETTINGS_FORM costSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings_tmp,bool isFirstSearch)
{

  LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp = get<0>(alilqrSettings_tmp);
  AUGLAG_SETTINGS_FORM auglagSettings_tmp = get<1>(alilqrSettings_tmp);
  BREAK_SETTINGS_FORM breakSettings_tmp = get<2>(alilqrSettings_tmp);
  REG_SETTINGS_FORM regSettings_tmp = get<3>(alilqrSettings_tmp);

  double lagMultInit_tmp = get<0>(auglagSettings_tmp);
  double penInit_tmp = get<2>(auglagSettings_tmp);
  double penScale_tmp = get<4>(auglagSettings_tmp);
  double maxCost_tmp = get<8>(breakSettings_tmp);

  int maxOuterIter_tmp = get<0>(breakSettings_tmp);
  int maxIlqrIter_tmp = get<1>(breakSettings_tmp);
  int zCountLim_tmp = get<6>(breakSettings_tmp);
  double cmax_tmp = get<7>(breakSettings_tmp);

  double regInit_tmp = get<0>(regSettings);


  //double eps = this->eps;
  mat Xset = get<0>(traj);
  mat Uset = get<1>(traj);
  vec dt_timevec = get<2>(traj);

  int N = Xset.n_cols;
  //initialize grad, iter, lambdaSet, mu, LA0
  double grad = 1.0/EPSVAR;
  int iter = 0;
  mat lambdaSet = mat(sat.constraint_N(), N).ones()*lagMultInit_tmp;
  mat muSet = mat(sat.constraint_N(), N).ones()*penInit_tmp/penScale_tmp;
  double mu = penInit_tmp/penScale_tmp;
  AUGLAG_INFO_FORM auglag_vals = make_tuple(lambdaSet,mu,muSet);

  // mat clist = mat(sat.constraint_N(),N);

  // For infeasible start modes 0-2: keep SLERP trajectory as-is
  // For infeasible start mode 3: run forward sim (trajectory is already feasible)
  // For skip_initial_fwd_sim: keep trajectory as-is (used by same-dt Pass 2 to preserve topology)
  // For standard mode: re-simulate from controls to ensure feasibility
  bool skip_fwd_sim = skip_initial_fwd_sim || (use_infeasible_start && infeasible_ctrl_mode <= 2);
  if (!skip_fwd_sim) {
    traj = generateInitialTrajectory(dt0, Xset.col(0), Uset, vecs);
  }

  // Initialize slack variables for infeasible start
  if (use_infeasible_start) {
    int n_red = sat.reduced_state_N();
    double LMmax_tmp = get<1>(auglagSettings_tmp);
    // Initialize slack storage
    slack_lambdaSet = mat(n_red, N, fill::zeros);
    // slack_w provides baseline PSD guarantee: (P + (mu + w)I) is PD when w > |min eig(P)|
    // Pass 1 (mu=0→1): start low, slacks can be large to relax dynamics
    // Pass 2 (mu inherited): cap at moderate level so slacks can still help with warm-start defects
    if (slack_mu < 1.0) {
      slack_mu = 1.0;  // Pass 1 init
    } else {
      slack_mu = std::min(slack_mu, 100.0);  // Pass 2: cap inherited mu
    }
    slack_Kset = cube(n_red, n_red, N, fill::zeros);
    slack_dset = mat(n_red, N, fill::zeros);
    slack_Sset_new = mat(n_red, N, fill::zeros);
    // Compute initial slacks from dynamics defects
    initSlacksFromDefects(dt0, traj, vecs);
  }

  tuple<mat, double> viol = OldPlanner::maxViol(traj,vecs,auglag_vals);
  mat clist = get<0>(viol);
  auglag_vals = OldPlanner::incrementAugLag(auglag_vals,clist,auglagSettings_tmp);
  AUGLAG_INFO_FORM auglag_vals_clean = make_tuple(0*lambdaSet,0,0*muSet);
  double LA0 = OldPlanner::cost2Func(traj, vecs, auglag_vals, &costSettings_tmp);
  if (use_infeasible_start) { LA0 += slackCost(slack_Sset); }
  double LA = LA0;
  double LAnc = OldPlanner::cost2Func(traj, vecs, auglag_vals_clean, &costSettings_tmp);


  double cmaxtmp = 0.0;
  double dlaZcount = 0;
  double dLA = 0.0;
  double newLA = LA;
  REG_PAIR regs = make_tuple(regInit_tmp,0.0);
  //initialize Kset, Pset
  BACKWARD_PASS_RESULTS_FORM BPresults;
  tuple<double,double,mat,double,REG_PAIR,TRAJECTORY_FORM> ilqrRes;
  //Outer loop
  if(verbose) {
    cout<<"begin outer\n";
  }
  double stepsSinceRand = -1;

  for(int j = 0; j < maxOuterIter_tmp; j++)
  {
     if(verbose){cout<<"outer iter "<<j<<"\n";}
    //reset cmaxtmp, dlaZcount
    cmaxtmp = 0.0;
    dlaZcount = 0;
    clist.zeros();
    dLA = 0.0;
    stepsSinceRand = -1;
    //ILQR
    //set rho and drho to regInit
    regs = make_tuple(regInit_tmp,0.0);
    //Find initial cost and init dLA
    LA = OldPlanner::cost2Func(traj,vecs, auglag_vals, &costSettings_tmp);
    if (use_infeasible_start) { LA += slackCost(slack_Sset); }
    if(verbose){
      OldPlanner::costInfo(traj, vecs, auglag_vals,&costSettings_tmp);
      if (use_infeasible_start) {
        cout << "  slack cost: " << slackCost(slack_Sset) << " slack_mu: " << slack_mu
             << " slack_norm: " << norm(slack_Sset, "fro") << " slack_max: " << abs(slack_Sset).max() << "\n";
      }
    }
    //inner loop
    for(int ii = 0; ii < maxIlqrIter_tmp; ii++)
    {

      //update iter
      if(verbose){cout<<"ii: "<<ii<<endl;}
      iter++;
      // tuple<TRAJECTORY_FORM,double>  rnOut = OldPlanner::addRandNoise(dt0, traj,  dlaZcount,  stepsSinceRand, breakSettings_tmp, regSettings_tmp, &costSettings_tmp, auglag_vals, vecs);
      // traj = get<0>(rnOut);
      // stepsSinceRand = get<1>(rnOut);

      ilqrRes = OldPlanner::ilqrStep(dt0,traj,vecs,auglag_vals,regs,&costSettings_tmp,regSettings_tmp,lineSearchSettings_tmp,breakSettings_tmp,!isFirstSearch);

      newLA = get<0>(ilqrRes);
      cmaxtmp = get<1>(ilqrRes);
      clist = get<2>(ilqrRes);
      grad = get<3>(ilqrRes);
      regs = get<4>(ilqrRes);
      traj = get<5>(ilqrRes);

      dLA = abs(newLA-LA);
      // if(stepsSinceRand != 0){
        dlaZcount++;
        // Use relative tolerance: reset if cost changed by more than 0.1%
        // (matches Python: dLA / (abs(LA) + 1e-10) > 1e-3)
        double relative_change = dLA / (std::abs(LA) + 1e-10);
        if(relative_change > 1e-3)
        {
          dlaZcount = 0;
        }
      // }
      // if(stepsSinceRand>=0){stepsSinceRand++;}
      //update LA, Xset, Uset
      LA = newLA;
      LAnc = OldPlanner::cost2Func(traj,vecs, auglag_vals_clean, &costSettings_tmp);
      current_traj = traj;
      current_Xset = get<0>(traj);
      current_Uset = get<1>(traj);
      current_ilqr_iter = ii;
      current_outer_iter = j;
      if(verbose){cout<<ii<<" "<<j<<" cmaxtmp,LA,LA clean "<<cmaxtmp<<" "<<LA<<" "<<LAnc<<"\n";}

      //Check if we need to break out of the loop
      if(OldPlanner::ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp)){
        if(verbose){cout<<"innerbreak\n";}
        break;
      }
    }
    if(OldPlanner::outerBreak(auglag_vals,cmaxtmp,breakSettings_tmp,auglagSettings_tmp)&&j>2&&OldPlanner::ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp,true)) {
      if(verbose){cout<<"outerbreak\n";}
      break;
    }
    //update lambdaSet, etc.
    auglag_vals = OldPlanner::incrementAugLag(auglag_vals,clist,auglagSettings_tmp);

    // Update slack augmented Lagrangian (same schedule as regular constraints)
    if (use_infeasible_start) {
      double LMmax_tmp = get<1>(auglagSettings_tmp);
      double penMax_tmp = get<3>(auglagSettings_tmp);
      double penScale_aug = get<4>(auglagSettings_tmp);
      for (int k = 0; k < N - 1; k++) {
        slack_lambdaSet.col(k) += slack_mu * slack_Sset.col(k);
        slack_lambdaSet.col(k) = clamp(slack_lambdaSet.col(k), -LMmax_tmp, LMmax_tmp);
        // No non-negativity clamp: equality constraint s=0, not inequality
      }
      // Slack penalty uses its own max (decoupled from constraint penalty)
      slack_mu = min(slack_penalty_max, penScale_aug * slack_mu);
      if (verbose) {
        cout << "  Slack AL update: mu=" << slack_mu << " mu_eff=" << (slack_mu + slack_w)
             << " |lambda|=" << norm(slack_lambdaSet, "fro")
             << " |s|=" << norm(slack_Sset, "fro")
             << " max|s|=" << abs(slack_Sset).max() << "\n";
      }
    }
  }
  cout << "ALILQR: " << iter << " total iters, " << current_outer_iter+1 << " outer, final grad=" << grad << " dLA=" << dLA << " cost=" << LA << endl;
  if(verbose){cout<<"out of loops\n";}
  
  // Run final backward pass to get K-gains for warm-start
  // (The K-gains computed during ilqrStep iterations aren't returned)
  tuple<BACKWARD_PASS_RESULTS_FORM, REG_PAIR> finalBackwardPass = OldPlanner::backwardPass(
      dt0, traj, vecs, auglag_vals, regs, &costSettings_tmp, regSettings_tmp, !isFirstSearch);
  BPresults = get<0>(finalBackwardPass);
  
  cube Kcube = get<0>(BPresults);
  if(verbose){cout<<"Kcube: ("<<Kcube.n_rows<<","<<Kcube.n_cols<<","<<Kcube.n_slices<<")\n";}
  mat Kmat = packageK(Kcube);
  if(verbose){cout<<"Kmat done: ("<<Kmat.n_rows<<","<<Kmat.n_cols<<")\n";}
  OPT_FORM opt = make_tuple(get<0>(traj),get<1>(traj),get<3>(traj),Kmat,lambdaSet,dt_timevec);
  if(verbose){cout<<"opt packaged\n";}
  return make_tuple(opt, mu, grad);
}
tuple<double,double,mat,double,REG_PAIR,TRAJECTORY_FORM> OldPlanner::ilqrStep(double dt0,TRAJECTORY_FORM traj,VECTOR_INFO_FORM vecs,AUGLAG_INFO_FORM auglag_vals,REG_PAIR regs,COST_SETTINGS_FORM *costSettings_ptr,REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,BREAK_SETTINGS_FORM breakSettings_tmp,bool useDist){
  //Check if planner killed from python
  if (PyErr_CheckSignals() != 0) {
    throw py::error_already_set();
  }

  if (verbose){
    OldPlanner::costInfo(traj, vecs, auglag_vals,  costSettings_ptr);
  }

  double N = get<0>(traj).n_cols;


  tuple<BACKWARD_PASS_RESULTS_FORM, REG_PAIR> backwardPassResults = OldPlanner::backwardPass(dt0,traj,vecs,auglag_vals,regs,costSettings_ptr,regSettings_tmp,useDist);

  BACKWARD_PASS_RESULTS_FORM BPresults = get<0>(backwardPassResults);
  regs = get<1>(backwardPassResults);
  //call forward pass

  // mat newTQ = get<3>(traj);
  tuple<TRAJECTORY_FORM, double, REG_PAIR> forwardPassOut = OldPlanner::forwardPass(dt0,traj,vecs,auglag_vals,BPresults,regs,costSettings_ptr,regSettings_tmp,lineSearchSettings_tmp,useDist);
  double newLA = get<1>(forwardPassOut);
  regs = get<2>(forwardPassOut);
  traj = get<0>(forwardPassOut);
  double newLAnc = OldPlanner::cost2Func(traj, vecs, auglag_vals, costSettings_ptr,false);

  // double grad = sum( max(abs(get<1>(BPresults).cols(0,N-2))/(sqrt(sum(get<1>(traj).cols(0,N-2) % get<1>(traj).cols(0,N-2),0))+1),0))/(N-1);
  double grad = sum( max(abs(get<1>(BPresults).cols(0,N-2))/(abs(get<1>(traj).cols(0,N-2) )+1),0))/(N-1);
  // double grad = sum( vecnorm(get<1>(BPresults).cols(0,N-2),"inf",0)/(vecnorm(get<1>(traj).cols(0,N-2) )+1))/(N-1);
  // double grad = sum( abs(get<1>(BPresults).cols(0,N-2))/(abs(get<1>(traj).cols(0,N-2) )+1))/(N-1);



  //find dLA
  tuple<mat, double> viol = OldPlanner::maxViol(traj,vecs,auglag_vals);
  mat clist = get<0>(viol);
  double cmaxtmp = get<1>(viol);
  if(newLAnc>get<8>(breakSettings_tmp)){
    regs = increaseReg(regs,regSettings_tmp);
  }
  return make_tuple(newLA,cmaxtmp,clist,grad,regs,traj);
}
bool OldPlanner::ilqrBreak(double grad,double LA, double dLA, double dlaZcount, double cmaxtmp, double iter,BREAK_SETTINGS_FORM breakSettings_tmp,bool forOuter)
{
  if(verbose){cout<<"ilqrBreak\n";}
  int maxOuterIter_tmp = get<0>(breakSettings_tmp);
  int maxIter_tmp = get<2>(breakSettings_tmp);
  double gradTol_tmp = get<3>(breakSettings_tmp);
  double ilqrCostTol_tmp = get<4>(breakSettings_tmp);
  double costTol_tmp = get<5>(breakSettings_tmp);
  int zCountLim_tmp = get<6>(breakSettings_tmp);
  double cmax_tmp = get<7>(breakSettings_tmp);
  double  max_cost = get<8>(breakSettings_tmp);

  double useCostTol = ilqrCostTol_tmp;
  if((current_outer_iter>=maxOuterIter_tmp-1) || forOuter){
    useCostTol = costTol_tmp;
  }//
  if(verbose){cout<<"useCostTol "<<useCostTol<<"\n";}
  //if ((((cmaxtmp<cmax) || j < maxOuterIter) && grad<gradTol) ||dlaZcount > zCountLim ||(0 < dLA && dLA < ilqrCostTol && ((cmaxtmp<cmax) || j < maxOuterIter ) ))

  // (
  //   (
  //     (
  //       (cmaxtmp<cmax)
  //       ||
  //       (j<maxOuterIter)
  //     )
  //     &&
  //     (grad<gradTol)
  //   )
  //   ||
  //   (dlaZcount > zCountLim)
  //   ||
  //   (
  //     (0<=dLA)
  //     &&
  //     (dLA<ilqrCostTol)
  //     &&
  //     (
  //       (cmaxtmp<cmax)
  //       ||
  //       j < maxOuterIter
  //     )
  //   )
  // )
  // if(((grad<gradTol_tmp)||(0<=dLA && dLA<useCostTol)&&((!ls_failed)))//&&((!ls_failed)||(!forOuter)))
  //     ||((dlaZcount > zCountLim_tmp))//||((!forOuter)&&(dlaZcount > zCountLim_tmp))
  //     ||(LA>max_cost))
  // Break if (grad small AND cost change small) AND line search didn't fail
  // OR if cost hasn't changed for too many iterations
  if(((grad<gradTol_tmp)&&(0<dLA && dLA<useCostTol)&&(!ls_failed))||
    (dlaZcount > zCountLim_tmp) )
  {
    if(verbose) {
      cout<<"breaking inner loop alilqr with value j: "<<current_outer_iter<<" and value ii: "<<current_ilqr_iter<<"\n";
      cout<<"cmaxtmp "<<cmaxtmp<<"\n";
      cout<<"grad "<<grad<<"vs"<<gradTol_tmp<<"\n";
      cout<<"dLA "<<dLA<<"vs"<<useCostTol<<"\n";
      cout<<"line search failed?: "<<ls_failed<<"\n";
      cout<<"zcount "<<dlaZcount<<"vs"<<zCountLim_tmp<<"\n";
    }
    return true;
  }
  if(verbose){
    cout<<"checked break conditions\n";
    if(verbose) {
      cout<<" outer iter: "<<current_outer_iter<<" ilqr iter: "<<current_ilqr_iter<<" cmaxtmp: "<<cmaxtmp<<" grad: "<<grad<<" vs gradtol: "<<gradTol_tmp<<" dLA: "<<dLA<<" vs costTol: "<<useCostTol<<" zcount: "<<dlaZcount<<" vs zcountlim: "<<zCountLim_tmp<<"\n";
    }
  }
  if(iter == maxIter_tmp)
  {
    if(verbose){cout<<"breaking because iter == maxIter_tmp\n";}
    return true;
  }
  if(iter > maxIter_tmp)
  {
    if(verbose){cout<<"Total iteration limit exceeded in alilqr\n";}
    throw "Total iteration limit exceeded in alilqr";
  }
  if(verbose){cout<<"checked iteration limit\n";}
  return false;
}


tuple<TRAJECTORY_FORM,double>  OldPlanner::addRandNoise(double dt0, TRAJECTORY_FORM traj, double dlaZcount, double stepsSinceRand, BREAK_SETTINGS_FORM breakSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,COST_SETTINGS_FORM *costSettings_ptr, AUGLAG_INFO_FORM auglag_vals,VECTOR_INFO_FORM vecs){


  double randPercent = get<6>(regSettings_tmp);
  // if(verbose) {
  // }
  if(dlaZcount>std::max(2.0,get<6>(breakSettings_tmp)*0.5) && stepsSinceRand<0 && randPercent > 0) {
    mat Xset = get<0>(traj);
    int N = Xset.n_cols;
    mat Uset = get<1>(traj);
    // TRAJECTORY_FORM newTraj = generateInitialTrajectory(dt0,Xset.col(0), Uset.cols(0,N-2) + randPercent*diagmat(max(abs(Uset.cols(0,N-2)),1))*2*(randu(size(Uset.cols(0,N-2)))-0.5),vecs);
    mat Unoise = randPercent*abs(Uset) % (2*randu(size(Uset))-1.0);
    // if(verbose) {
    // }
    TRAJECTORY_FORM newTraj = generateInitialTrajectory(dt0,Xset.col(0), Uset + Unoise,vecs);


    double testLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_ptr);
    if(!(isnan(testLA)||isinf(testLA)) && randPercent > 0) {
      traj = newTraj;//make_tuple(get<0>(newTraj),get<1>(newTraj),get<2>(traj),get<3>(newTraj));
      if(verbose) {
        cout<<"a bit of random added\n";
      }
      stepsSinceRand = 0;
    }

  }
  return make_tuple(traj,stepsSinceRand);
}


bool OldPlanner::outerBreak(AUGLAG_INFO_FORM auglag_vals, double cmaxtmp,BREAK_SETTINGS_FORM breakSettings_tmp,AUGLAG_SETTINGS_FORM auglagSettings_tmp)
{
  if(verbose){cout<<"outerBreak\n";}
  double mu = get<1>(auglag_vals);
  mat muSet = get<2>(auglag_vals);
  double cmax_tmp = get<7>(breakSettings_tmp);
  double penMax_tmp = get<3>(auglagSettings_tmp);

  // When using infeasible start, don't break if slacks are still large
  if (use_infeasible_start && slack_Sset.n_elem > 0) {
    double slack_max = abs(slack_Sset).max();
    double slack_tol = 1e-3;  // Must drive slacks below this to exit
    if (slack_max > slack_tol) {
      if (verbose) {
        cout << "outerBreak blocked: slack_max=" << slack_max << " > " << slack_tol << "\n";
      }
      return false;
    }
  }

  if (((cmaxtmp<cmax_tmp)|| (muSet.max() >= penMax_tmp)))
  {
    if(verbose) {
      cout<<"breaking outer loop alilqr with value j: "<<current_outer_iter<<"\n";
      cout<<"cmaxtmp: "<<cmaxtmp<<"vs"<<cmax_tmp<<"\n";
      cout<<"penMax: "<<mu<<"vs"<<penMax_tmp<<"\n";
      cout<<"penMax2: "<<muSet.max()<<"vs"<<penMax_tmp<<"\n";
    }
    return true;
  }
  return false;
}

void OldPlanner::costInfo(TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, AUGLAG_INFO_FORM auglag_vals,  COST_SETTINGS_FORM *costSettings_ptr){
    mat Xset = get<0>(traj);
    mat Uset = get<1>(traj);
    mat TQset = get<3>(traj);
    double N = Xset.n_cols;
    vec dt_timevec = get<2>(traj);
    mat U0 = mat(arma::size(Uset)).zeros();
    mat Unomtq = Uset;
    Unomtq.head_rows(sat.number_MTQ) *= 0;
    mat lambdaSet = get<0>(auglag_vals);
    double mu = get<1>(auglag_vals);
    mat muSet = get<2>(auglag_vals);
    mat lambda0 = mat(arma::size(lambdaSet)).zeros();
    double pen0 = 0.0;
    mat penSet0 = mat(arma::size(muSet)).zeros();

    mat satvec = get<5>(vecs);
    mat ECIvec = get<6>(vecs);
    vec angs = vec(N);
    vec ek;
    for(int k =0;k<N;k++){
      ek = ECIvec.col(k);
      if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
        ek = ek.tail(3);
        angs(k) = (180.0/datum::pi)*acos(norm_dot(satvec.col(k),rotMat(Xset(sat.quat0index(),k,size(4,1))).t()*ek));
      }else{
        angs(k) = (180.0/datum::pi)*acos(2.0*pow(norm_dot(Xset(sat.quat0index(),k,size(4,1)),ek),2.0)-1.0);
      }

    }
    vec angdiffs = diff(angs);
    vec avs = sum(square(Xset.head_rows(3))).t();
    vec avdiffs = diff(avs);
    vec hs = vec(N).zeros();
    vec hdiffs = vec(N-1).zeros();
    vec urws = vec(N).zeros();
    vec urwdiffs = vec(N-1).zeros();
    if(sat.number_RW>0){
      hs = sqrt(sum(square(Xset.tail_rows(sat.number_RW)))).t();
      hdiffs = diff(hs);
      urws = sqrt(sum(square(Uset.rows(sat.number_MTQ,sat.number_MTQ+sat.number_RW-1)))).t();
      urwdiffs = diff(urws);
    }
    vec umags = vec(N).zeros();
    vec umagdiffs = vec(N-1).zeros();
    if(sat.number_magic>0){
      umags = sqrt(sum(square(Uset.tail_rows(sat.number_magic)))).t();
      umagdiffs = diff(umags);
    }
    vec umtqs = vec(N).zeros();
    vec umtqdiffs = vec(N-1).zeros();
    if(sat.number_MTQ>0){
      umtqs = sqrt(sum(square(Uset.head_rows(sat.number_MTQ)))).t();
      umtqdiffs = diff(umtqs);
    }


      //Extract costSettings
    COST_SETTINGS_FORM costSettings_tmp = *costSettings_ptr;
    double angle_weight_tmp = get<0>(costSettings_tmp);
    double angvel_weight_tmp = get<1>(costSettings_tmp);
    double u_weight_tmp = get<2>(costSettings_tmp);
    double av_with_mag_weight_tmp = get<3>(costSettings_tmp);
    double ang_av_weight_tmp = get<4>(costSettings_tmp);
    double angle_weight_N_tmp = get<5>(costSettings_tmp);
    double angvel_weight_N_tmp = get<6>(costSettings_tmp);
    double av_with_mag_weight_N_tmp = get<7>(costSettings_tmp);
    double ang_av_weight_N_tmp = get<8>(costSettings_tmp);
    int whichAngCostFunc_tmp = get<9>(costSettings_tmp);
    int useRawControlCost_tmp = get<10>(costSettings_tmp);
    int useFullCostHess_tmp = get<11>(costSettings_tmp);
    COST_SETTINGS_FORM nou_Settings = make_tuple(angle_weight_tmp,angvel_weight_tmp,0.0,av_with_mag_weight_tmp,ang_av_weight_tmp,angle_weight_N_tmp,angvel_weight_N_tmp,av_with_mag_weight_N_tmp,ang_av_weight_N_tmp,whichAngCostFunc_tmp,useRawControlCost_tmp,useFullCostHess_tmp);
    COST_SETTINGS_FORM only_av_Settings = make_tuple(0.0,angvel_weight_tmp,0.0,0.0,0.0,0.0,angvel_weight_N_tmp,0.0,0.0,whichAngCostFunc_tmp,useRawControlCost_tmp,useFullCostHess_tmp);
    COST_SETTINGS_FORM only_ang_Settings = make_tuple(angle_weight_tmp,0.0,0.0,0.0,0.0,angle_weight_N_tmp,0.0,0.0,0.0,whichAngCostFunc_tmp,useRawControlCost_tmp,useFullCostHess_tmp);

    mat clearvel = mat(sat.state_N(),sat.state_N()).eye();
    clearvel(span(0,2),span(0,2)).zeros();

    TRAJECTORY_FORM cleartraj = make_tuple(clearvel*Xset,U0,dt_timevec,TQset);
    TRAJECTORY_FORM noutraj = make_tuple(Xset,U0,dt_timevec,TQset);
    TRAJECTORY_FORM nomtqtraj = make_tuple(Xset,Unomtq,dt_timevec,TQset);
    AUGLAG_INFO_FORM auglag_vals_zero = make_tuple(lambda0,pen0,penSet0);

    double LA = OldPlanner::cost2Func(traj, vecs, auglag_vals, costSettings_ptr);
    double LAnc = OldPlanner::cost2Func(traj, vecs, auglag_vals_zero, costSettings_ptr);
    double LAnou = OldPlanner::cost2Func(noutraj, vecs, auglag_vals_zero, &nou_Settings);
    double LAnomtq = OldPlanner::cost2Func(nomtqtraj, vecs, auglag_vals_zero, &nou_Settings);
    double LAang = OldPlanner::cost2Func(cleartraj,vecs,auglag_vals_zero, &only_ang_Settings);
    double LAnouav = OldPlanner::cost2Func(cleartraj,vecs,auglag_vals_zero, &nou_Settings);
    double LAav = LAnou-LAnouav;//OldPlanner::cost2Func(noutraj,vecs,auglag_vals_zero, &only_av_Settings);
    double LAu = LAnc - LAnou;
    double LAmtq = LAnc-LAnomtq;
    // double LAav = LAnc-LAu-LAang;
    double avg_ang = LAang/((N-1)*angle_weight_tmp+angle_weight_N_tmp);

    cout<<"LA: "<<LA<<" and LA w/o constraints: "<< LAnc <<"\n";
    cout<<"control strength "<<max(abs(Uset))<<"\n";
    // if(sat.number_RW>0){

    // }
    // if(sat.number_MTQ>0){
    // }
    // if(sat.number_magic>0){
    // }
    cout<<"ang/h cost: "<<LAang<<" omega: "<<LAav<<" mtq: "<<LAmtq<<" and u: "<<LAu<<"\n";

    return;
}

AUGLAG_INFO_FORM OldPlanner::incrementAugLag(AUGLAG_INFO_FORM auglag_vals, mat clist, AUGLAG_SETTINGS_FORM auglagSettings_tmp){
    mat lambdaSet = get<0>(auglag_vals);
    double mu = get<1>(auglag_vals);
    mat muSet = get<2>(auglag_vals);

    double LMmax = get<1>(auglagSettings_tmp);
    double muMax = get<3>(auglagSettings_tmp);
    double muScale = get<4>(auglagSettings_tmp);

    double N = lambdaSet.n_cols;
    for(int k = 0; k < N; k++)
    {
      for(int i = 0; i < sat.constraint_N(); i++)
      {
        // if(clist(i,k)>0){
          lambdaSet(i,k) = lambdaSet(i,k) + muSet(i,k)*clist(i,k);
        // }
          lambdaSet(i,k) = min(LMmax*1.0, lambdaSet(i,k));
          //double minTmp = min(lagMultMax*1.0, (lambdaSet(i,k) + muSet(i,k)*max(0.0,clist(i,k))));
          lambdaSet(i, k) = max(-LMmax*1.0, lambdaSet(i,k));
          if(i < sat.ineq_constraint_N()) //because all of our constraints are limits, not equality constraints. If it was an equality constraint, we would allow it to be negative.
          {
            lambdaSet(i, k) = max(0.0, lambdaSet(i,k));
          }
          // if(clist(i,k)>=-cmax){
          // if(clist(i,k)<=cmax){
            muSet(i,k) = max(0.0,min(muMax*1.0, muScale*muSet(i,k)));
          // }
      }
    }
  //update mu
  mu = max(0.0,min(muMax*1.0, muScale*mu));
  return make_tuple(lambdaSet,mu,muSet);

}

/* This method gets the max violations and constraint list */
tuple<mat, double> OldPlanner::maxViol(TRAJECTORY_FORM &traj, VECTOR_INFO_FORM &vecs,AUGLAG_INFO_FORM &auglag)
{
  mat lambdaSet = get<0>(auglag);
  double mu = get<1>(auglag);
  mat muSet = get<2>(auglag);

  mat Uset = get<1>(traj);
  mat Xset = get<0>(traj);
  Uset = join_rows(Uset,vec(sat.control_N()).zeros());
  int N = Xset.n_cols;
  mat sunset = get<4>(vecs);
  mat clist = mat(sat.constraint_N(), N);
  vec uk;
  vec xk;
  vec3 sunk;
  //loop over trajectory and fill in clist
  for(int k = 0; k < N; k++)
  {
    uk = Uset.col(k);
    xk = Xset.col(k);
    sunk = normalise(sunset.col(k));
    vec ck = sat.getConstraints(k, N, uk,xk,sunk);
    clist.col(k) = ck;
  }
  //clist = clamp(clist,0.0,datum::inf);
  mat corrected_clist(arma::size(clist));
  if(sat.eq_constraint_N()>0){
    corrected_clist = join_cols(clamp(clist.rows(0,sat.ineq_constraint_N()-1),0.0,datum::inf),clist.rows(sat.ineq_constraint_N(),sat.constraint_N()-1));
  }else{
    corrected_clist = clamp(clist.rows(0,sat.ineq_constraint_N()-1),0.0,datum::inf);
  }
  double cmaxtmp = abs(corrected_clist).max();

  //DEBUG
  //mat w2 = sum((Xset.rows(0,2) % Xset.rows(0,2)),0);
  uvec::fixed<2> ss=ind2sub(arma::size(corrected_clist),abs(corrected_clist).index_max());
  if(verbose){cout<<": "<<cmaxtmp<<" at subscript "<<ss(0)<<", "<<ss(1)<<"\n";}
  if(verbose){cout<<"state at max viol: "<<Xset.col(ss(1)).t()<<"\n";}
  if(verbose){cout<<"ctrl at max viol: "<<Uset.col(ss(1)).t()<<"\n";}
  if(verbose){cout<<corrected_clist.col(ss(1)).t()<<"\n";}
  if(cmaxtmp>1e10 && verbose){
    cout<<Xset.cols(0,ss(1))<<"\n";
    cout<<Uset.cols(0,ss(1))<<"\n";
  }

  return make_tuple(clist, cmaxtmp);
}

/*This method does backwardpass for AL-iLQR
  Inputs:
    Xset - set of states so far - 7 x N matrix
    Uset - set of control inputs - 3 x N-1 matrix
    Rset - orbital position - 3 x N matrix
    Vset - orbital velocity - 3 x N matrix
    Bset - orbital magnetic field - 3 x N matrix
    lambdaSet - lambda - 6 x N matrix
    rho, drho, mu, dt, regMin, regScale - various parameters, all doubles
    R - control cost, 3x3 matrix
    QN - Q at t = N, 6x6 matrix
    umax - max allowable u for constraints
    int Nslew - time at which slew turns to point - int
    vec satAlignVector - vector of satellite to align with goal - 3 x 1 vector
    qSettings - settings for Q
  Outputs:
    P - covariance - 6 x 6 x N cube
    K - gains - 3 x 6 x N cube
    dset - unclear - 3 x N matrix
    delV - 1 x 2 matrix
    rho,drho - same as above
*/
// Does qSettings need to be a pointer? Can't we use a reference?
tuple<BACKWARD_PASS_RESULTS_FORM, REG_PAIR> OldPlanner::backwardPass(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, AUGLAG_INFO_FORM auglag_vals,REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,bool useDist)
{
  mat Uset = get<1>(traj);
  mat Xset = get<0>(traj);
  Uset = join_rows(Uset,vec(sat.control_N()).zeros());
  int N = Xset.n_cols;

  //Initialize return items
  cube Kset = cube(sat.control_N(), sat.reduced_state_N(), N-1).zeros();
  //cube Pset = cube(6, 6, N).zeros();
  mat dset = mat(sat.control_N(), N-1).zeros();
  vec2 delV = vec2().zeros();

  double regMin_tmp = get<1>(regSettings_tmp);
  bool useDynamicsHess_tmp = bool(get<7>(regSettings_tmp));
  bool useConstraintsHess_tmp = bool(get<8>(regSettings_tmp));
  int regMode_tmp = int(get<6>(regSettings_tmp));  // 0=control-space, 1=state-space, 2=both (index 6, was rand_add_ratio)

  //Initialize xk, uk, rk, etc
  mat lambdaSet = get<0>(auglag_vals);
  double mu = get<1>(auglag_vals);
  mat muSet = get<2>(auglag_vals);

  mat Rset = get<1>(vecs);
  mat Vset = get<2>(vecs);
  mat Bset = get<3>(vecs);
  mat sunset = get<4>(vecs);
  mat satvec = get<5>(vecs);
  mat ECIvec = get<6>(vecs);
  mat pset = get<7>(vecs);
  bool reset = false;
  double rho = 0.0;

  vec xk = vec(sat.state_N()).zeros();
  vec4 qk = vec(4).zeros();
  vec3 sunk = vec(3).zeros();
  vec dk = vec(sat.control_N()).zeros();
  mat Kk = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  //get ck (constraints)
  vec ck = vec(sat.constraint_N()).zeros();
  //get ImuK given ck
  mat Imuk = mat(sat.constraint_N(),sat.constraint_N()).zeros();
  mat Ilamk = mat(sat.constraint_N(),sat.constraint_N()).zeros();

  vec ukp = vec(sat.control_N()).zeros();
  vec viol = vec(sat.constraint_N()).zeros();
  cube violcxx = cube(sat.reduced_state_N(),sat.reduced_state_N(),sat.constraint_N()).zeros();
  cube violcux = cube(sat.control_N(),sat.reduced_state_N(),sat.constraint_N()).zeros();
  cube violcuu = cube(sat.control_N(),sat.control_N(),sat.constraint_N()).zeros();
  vec pk = vec(sat.reduced_state_N()).zeros();
  mat Pk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();

  mat Gk = mat(sat.reduced_state_N(), sat.state_N()).zeros();
  mat Gkp1 = mat(sat.reduced_state_N(), sat.state_N()).zeros();

  tuple<mat, mat,mat> AB;
  tuple<cube, cube, cube> hesses;
  mat Aqk = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  mat Bqk = mat(sat.reduced_state_N(),sat.control_N()).zeros();
  cube ddxd__dxdx = cube(sat.state_N(),sat.state_N(),sat.state_N()).zeros();
  cube ddxd__dudx = cube(sat.control_N(),sat.state_N(),sat.state_N()).zeros();
  cube ddxd__dudu = cube(sat.control_N(),sat.control_N(),sat.state_N()).zeros();
  cube ddxd__dxdxQ = cube(sat.reduced_state_N(),sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  cube ddxd__dudxQ = cube(sat.control_N(),sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  cube ddxd__duduQ = cube(sat.control_N(),sat.control_N(),sat.reduced_state_N()).zeros();

  cost_jacs costJac;
  tuple<mat, mat> cnstrJac;
  tuple<cube,cube,cube> cnstrHess;

  mat cku = mat(sat.constraint_N(),sat.control_N()).zeros();
  mat ckx = mat(sat.constraint_N(),sat.reduced_state_N()).zeros();
  mat Qkuu = mat(sat.control_N(),sat.control_N()).zeros();
  mat Qkuureg = mat(sat.control_N(),sat.control_N()).zeros();
  mat Qkuureg_chol = mat(sat.control_N(),sat.control_N()).zeros();
  umat Qkuureg_chol_piv;// = mat(sat.control_N(),sat.control_N()).zeros();
  mat Qkux = mat(sat.control_N(),sat.reduced_state_N()).zeros();
  mat Qkxx = mat(sat.reduced_state_N(),sat.reduced_state_N()).zeros();
  vec Qku = vec(sat.control_N()).zeros();
  vec Qkx = vec(sat.reduced_state_N()).zeros();

  vec eigs = vec(sat.control_N()).zeros();
  vec modeig = vec(sat.control_N()).zeros();

  cx_vec cxeigs = cx_vec(vec(eigs),vec(eigs));
  vec eigsreg = vec(sat.control_N()).zeros();
  mat eigvecs = mat(sat.control_N(),sat.control_N()).zeros();

  DYNAMICS_INFO_FORM dynamics_info_k;
  DYNAMICS_INFO_FORM dynamics_info_kp1;
  // dynamics_info_k = make_tuple(Bset.col(N-1),Rset.col(N-1),prop_torq*plan_for_prop);
  //looping back from k = N-1 to 0:
  double regComp;
  double EVreg;
  double regAddComp;
  vec ek;
  int k = N-1;

  while(k >= 0)
  {
    //store Gk, Pk, pk
    Gkp1 = Gk;
    dynamics_info_kp1 = dynamics_info_k;
    xk = Xset.col(k);
    qk = xk.rows(3, 6);
    sunk = normalise(sunset.col(k));
    //use rk4 to find Ak, Bk
    Gk = sat.findGMat(qk);
    dynamics_info_k = make_tuple(Bset.col(k),Rset.col(k),pset(k),Vset.col(k),sunset.col(k),int(useDist));
    //rk4Jacobians
    //find Aqk, Bqk
    Aqk = Aqk.zeros();
    Bqk = Bqk.zeros();
    ukp = ukp.zeros();
    if(k<N-1)
    {
      AB = _useEuler
        ? eulerzJacobians(dt0, xk, Uset.col(k), sat, dynamics_info_k, dynamics_info_kp1)
        : rk4zJacobians(dt0, xk, Uset.col(k), sat, dynamics_info_k, dynamics_info_kp1);
      Aqk = Gkp1*get<0>(AB)*trans(Gk);
      Bqk = Gkp1*get<1>(AB);
      if(useDynamicsHess_tmp && !_useEuler){
        hesses = rk4zHessians(dt0,xk, Uset.col(k),sat,dynamics_info_k, dynamics_info_kp1);
        ddxd__dxdx = get<0>(hesses);
        ddxd__dudx = get<1>(hesses);
        ddxd__dudu = get<2>(hesses);
        ddxd__dxdxQ = matOverCube(Gkp1,matTimesCube(Gk,cubeTimesMat(ddxd__dxdx,trans(Gk))));
        ddxd__dudxQ = matOverCube(Gkp1,cubeTimesMat(ddxd__dudx,trans(Gk)));
        ddxd__duduQ =  matOverCube(Gkp1,ddxd__dudu);
      }
    }
    //
    if (k>0){ukp = Uset.col(k-1);}else{ukp = Uset.col(0);}
    vec xkp1 = (k < N-1) ? vec(Xset.col(k+1)) : xk;  // Next state for path length cost
    vec xkm1 = (k > 0) ? vec(Xset.col(k-1)) : vec();  // Previous state for backward path length gradient
    ek = ECIvec.col(k);
    if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
      ek = ek.tail(3);
      costJac = sat.veccostJacobians(k, N, xk, xkp1, Uset.col(k), ukp,satvec.col(k),ek,Bset.col(k), costSettings_tmp, xkm1);
    }else{
      costJac = sat.quatcostJacobians(k, N, xk, xkp1, Uset.col(k), ukp,satvec.col(k),ek,Bset.col(k), costSettings_tmp, xkm1);
    }

    cnstrJac = sat.constraintJacobians(k, N,Uset.col(k), xk,sunk);


    cku = get<0>(cnstrJac);
    ckx = get<1>(cnstrJac);
    ck = sat.getConstraints(k, N, Uset.col(k), xk,sunk);
    //update Imuk
    Imuk = sat.getImu(mu, muSet.col(k), ck, lambdaSet.col(k));
    Ilamk = sat.getIlam(mu, muSet.col(k), ck, lambdaSet.col(k));


    viol = (Ilamk*lambdaSet.col(k)+Imuk*ck);
    if(useConstraintsHess_tmp){
      cnstrHess = sat.constraintHessians(k, N, Uset.col(k),xk,sunk);//join_cols(mat33().eye(), -1*mat33().eye());
      // for(int i = 0; i < sat.constraint_N(); ++i)
      // {
      //   violcxx.slice(i) = mat(sat.reduced_state_N(),sat.reduced_state_N(),fill::ones)*viol(i);
      //   violcux.slice(i) = mat(sat.control_N(),sat.reduced_state_N(),fill::ones)*viol(i);
      //   violcuu.slice(i) = mat(sat.control_N(),sat.control_N(),fill::ones)*viol(i);
      // }
    }

    // State-space regularization: add rho*I to Pk before computing Q-functions
    // This helps when B matrix is rank-deficient (e.g., MTQ-only satellites)
    mat Pk_reg = Pk;
    bool useStateReg = (regMode_tmp == 1 || regMode_tmp == 2);
    if (useStateReg && rho > 0) {
      Pk_reg = Pk + rho * mat(sat.reduced_state_N(), sat.reduced_state_N()).eye();
    }

    // Infeasible start: use augmented controls approach (not Schur complement).
    // Augment ū = [u; s], B̄ = [B; I]. Solve the larger (m+n)×(m+n) Q̄uu directly.
    // Better conditioned than Schur because μI only appears in slack portion.
    // No state-space regularization needed: B̄=[B;I] is always full rank.
    mat Pk_use = Pk_reg;  // for standard iLQR (no slacks)
    vec pk_use = pk;

    Qkxx = costJac.lxx + trans(Aqk)*Pk*Aqk + trans(ckx)*Imuk*ckx ;
    Qkx = costJac.lx + trans(Aqk)*pk + trans(ckx)*viol;
    if(useDynamicsHess_tmp){
      Qkxx += vecOverCube(pk_use,ddxd__dxdxQ);
    }
    if(useConstraintsHess_tmp){
      Qkxx += vecOverCube(viol,get<2>(cnstrHess));//mat(sum(get<2>(cnstrHess) % violcxx,2));
    }
    // Qkxx_full = costJac.lxx + trans(Aqk)*Pk*Aqk + trans(dAqkdx)*pk + mat(sum(get<2>(cnstrHess) % violcxx,2)) + trans(ckx)*Imuk*ckx;
    if(k == N-1){
      pk = Qkx;
      Pk = Qkxx;
      Pk = 0.5*(Pk + trans(Pk));

      if(verbose){
        vec Pk_eigs;
        eig_sym(Pk_eigs, Pk);
        cout << "Terminal Pk eigenvalues: " << Pk_eigs.t() << endl;
        if(min(Pk_eigs) < 0) {
          cout << "WARNING: Terminal cost Hessian is not PSD!" << endl;
        }
        cout << "costJac.lxx at terminal:\n" << costJac.lxx << endl;
        
        // Also helpful - eigendecomposition to see which directions are negative
        vec lxx_eigs;
        mat lxx_eigvecs;
        eig_sym(lxx_eigs, lxx_eigvecs, costJac.lxx);
        cout << "lxx eigenvalues: " << lxx_eigs.t() << endl;
        cout << "lxx eigenvectors (columns):\n" << lxx_eigvecs << endl;
        
        cout << "CostJac lkx: " << costJac.lx << endl;
        cout<<"Costjac "<<costJac.lxx<<"\n";

        if(sat.number_RW > 0) {
          cout << "RW speeds in state: " << xk.rows(7, 7+sat.number_RW-1).t() << endl;
          for(int j = 0; j < sat.number_RW; j++) {
            double z = xk(7+j);
            cout << "RW " << j << ": z=" << z
                  << " threshold=" << sat.RW_AM_cost_threshold.at(j)
                  << " softplus''=" << shifted_softplus_deriv2(abs(z), sat.RW_AM_cost_threshold.at(j))
              << endl;
          }
        }
      }

      k--;
      continue;
    }
    Qkux = costJac.lux + trans(Bqk)*Pk*Aqk + trans(cku)*Imuk*ckx;
    Qku = costJac.lu + trans(Bqk)*pk + trans(cku)*viol;

    //find Qkuu and Qkuureg
    Qkuu = costJac.luu + trans(Bqk)*Pk*Bqk + trans(cku)*Imuk*cku;


    if(useDynamicsHess_tmp){
      Qkux += vecOverCube(pk,ddxd__dudxQ);
      Qkuu += vecOverCube(pk,ddxd__duduQ);
    }
    if(useConstraintsHess_tmp){
      Qkux += vecOverCube(viol,get<1>(cnstrHess));//mat(sum(get<1>(cnstrHess) % violcux,2));
      Qkuu += vecOverCube(viol,get<0>(cnstrHess));//mat(sum(get<0>(cnstrHess) % violcuu,2));
    }

    // Augmented controls for infeasible start: ū = [u; s], B̄ = [B; I]
    // Build augmented Q̄uu (m+n × m+n), Q̄ux (m+n × n), Q̄u (m+n × 1)
    // Slack portion: Q_ss = P + μ_eff·I, Q_sx = P·A, Q_su = P·B, Q_s = p + λ + μ_eff·s
    bool augmented = use_infeasible_start && k < N-1;
    mat Qkuu_aug, Qkux_aug, Qkuureg_aug;
    vec Qku_aug;
    if (augmented) {
      int m = sat.control_N();
      int n_red = sat.reduced_state_N();
      double mu_eff = slack_mu + slack_w;

      // Q̄uu = [Quu,     B^T P  ]   (m+n × m+n)
      //        [P B,   P + μ_eff I]
      Qkuu_aug = mat(m + n_red, m + n_red, fill::zeros);
      Qkuu_aug(span(0, m-1), span(0, m-1)) = Qkuu;
      Qkuu_aug(span(0, m-1), span(m, m+n_red-1)) = trans(Bqk) * Pk;
      Qkuu_aug(span(m, m+n_red-1), span(0, m-1)) = Pk * Bqk;
      Qkuu_aug(span(m, m+n_red-1), span(m, m+n_red-1)) = Pk + mu_eff * mat(n_red, n_red, fill::eye);
      Qkuu_aug = 0.5 * (Qkuu_aug + Qkuu_aug.t());

      // Q̄ux = [Qux ]   (m+n × n)
      //        [P A ]
      Qkux_aug = mat(m + n_red, n_red, fill::zeros);
      Qkux_aug.rows(0, m-1) = Qkux;
      Qkux_aug.rows(m, m+n_red-1) = Pk * Aqk;

      // Q̄u = [Qu              ]   (m+n × 1)
      //       [p + λ + μ_eff·s ]
      vec lam_s = (!slack_lambdaSet.is_empty() && k < (int)slack_lambdaSet.n_cols)
                  ? vec(slack_lambdaSet.col(k)) : vec(n_red, fill::zeros);
      vec s_k = (!slack_Sset.is_empty() && k < (int)slack_Sset.n_cols)
                ? vec(slack_Sset.col(k)) : vec(n_red, fill::zeros);
      Qku_aug = vec(m + n_red, fill::zeros);
      Qku_aug.head(m) = Qku;
      Qku_aug.tail(n_red) = pk + lam_s + mu_eff * s_k;
    }

    rho = get<0>(regs);
    // Choose which Q matrices to use for the solve
    mat& Qsolve_uu = augmented ? Qkuu_aug : Qkuu;
    mat& Qsolve_ux = augmented ? Qkux_aug : Qkux;
    vec& Qsolve_u  = augmented ? Qku_aug  : Qku;

    reset |= (Qsolve_uu.has_nan()||Qsolve_uu.has_inf());
    if(verbose&&reset){
      cout<<"Qkuu has nan or inf: "<<Qsolve_uu.has_nan()<<" "<<Qsolve_uu.has_inf()<<"\n";
    }
    if(!reset){

        Qsolve_uu = 0.5*(Qsolve_uu+Qsolve_uu.t());

        if(verbose){
          double cond_num = arma::cond(Qsolve_uu);
          if(cond_num > 1e10){
            cout << "WARNING: Qkuu condition number = " << cond_num << " at k=" << k << endl;
          }
        }
        // if(!reset){
        //   // reset |= !eig_sym(eigs,eigvecs,Qkuu);//eigs = eig_sym(Qkuu);
        //   reset |= !eig_sym(eigs,eigvecs,Qkuu);
        //   reset |= (min(eigs) <= -rho);
        // }
      // if(useDynamicsHess_tmp || useConstraintsHess_tmp){
      //   reset |= Qkuu.is_sympd()
      //   reset |= !eig_gen(cxeigs,Qkuu);
      //   reset |= (min(real(cxeigs)) < -rho);
      // }else{
      //   Qkuu = 0.5*(Qkuu+Qkuu.t());//trimatu(Qkuu)+trans(trimatu(Qkuu,1));
      //   reset |= !eig_sym(eigs,Qkuu);//eigs = eig_sym(Qkuu);
      //   reset |= (min(eigs) < -rho);
      // }
//
      // eig_sym(eigs,eigvecs,Qkuureg);
      // eig_sym(eigs,eigvecs,Qkuu);
      // Qkuureg = eigvecs*diagmat(clamp(eigs,rho,datum::inf))*eigvecs.t();
      // Qkuureg = eigvecs*diagmat(clamp(abs(eigs),rho,datum::inf))*eigvecs.t();

      // Control-space regularization: conditional based on regMode_tmp
      // 0=control-space only, 1=state-space only, 2=both
      bool useControlReg = (regMode_tmp == 0 || regMode_tmp == 2);
      if (augmented) {
        // For augmented system: regularize entire matrix with rho·I
        // The slack portion has μ_eff·I but coupling B^T·P / P·B can make
        // the full matrix indefinite when P has negative eigenvalues (cross-term).
        int m = sat.control_N();
        int n_red = sat.reduced_state_N();
        Qkuureg_aug = Qkuu_aug + rho * mat(m + n_red, m + n_red, fill::eye);
      } else if (useControlReg) {
        Qkuureg = Qkuu + rho*mat(sat.control_N(),sat.control_N()).eye();
      } else {
        Qkuureg = Qkuu;
      }
      mat& Qsolve_uureg = augmented ? Qkuureg_aug : Qkuureg;





      //regularization to all
      // modeig = eigs+rho; //pure inverse of Qkuu+rho*eye
      // modeig = abs(eigs)+rho; //pure inverse but abs(eigs)
      // modeig = rho+clamp(eigs,0,datum::inf);  //pure inverse but first negative eigs are eliminated.

      //clamping
      // modeig = clamp(eigs,rho,datum::inf);  //all eigs greater than rho
      // modeig = clamp(abs(eigs),rho,datum::inf);


      //unlcear
      // modeig = clamp(eigs+rho,rho,datum::inf);
      // modeig = clamp(eigs+rho,0,datum::inf);


      // reset |= (min(modeig) <= 0);
      // Qkuureg_chol = eigvecs*diagmat(1.0/modeig)*eigvecs.t();

      // Qkuureg_chol = eigvecs*diagmat(clamp(1.0/(eigs+rho),0,1.0/rho))*eigvecs.t();
      // Qkuureg_chol = eigvecs*diagmat(clamp(1.0/abs(eigs+rho),0,datum::inf))*eigvecs.t();
      // Qkuureg_chol = eigvecs*diagmat(1.0/clamp(eigs+rho,rho,datum::inf))*eigvecs.t();
      // eigsreg = eigs + rho;
      // reset |= !chol(Qkuureg_chol,Qkuureg_chol_piv,Qkuureg,"lower","matrix"); //cheap check for positive-definiteness
      mat Qkuureg_chol_solve;
      reset |= !chol(Qkuureg_chol_solve, Qsolve_uureg);

      reset |= (Qkuureg_chol_solve.has_nan()||Qkuureg_chol_solve.has_inf());
      if(verbose&&reset){
        cout<<"Qkuu_reg not PD!\n";
        cout<<"k "<<k<<"\n";
        cout<<Qsolve_uu<<"\n";
        cout<<Qsolve_u.t()<<"\n";
      }
      if(!reset){
        if(Qsolve_uureg.has_nan()||Qsolve_uureg.has_inf()){
          if(verbose){
            cout<<"somehow regularized Qkuu has nan/inf\n";
          }
          throw("somehow regularized Qkuu has nan/inf but nonregularized does not");
        }
        // Solve augmented or standard system
        mat Kk_full;
        vec dk_full;
        reset |= !solve(Kk_full, Qsolve_uureg, Qsolve_ux, solve_opts::likely_sympd+solve_opts::fast);
        if(verbose&&reset){ cout<<"Solving Kk failed \n"; }
        reset |= !solve(dk_full, Qsolve_uureg, Qsolve_u, solve_opts::likely_sympd+solve_opts::fast);
        if(verbose&&reset){ cout<<"Solving dk failed \n"; }
        reset |= (dk_full.has_nan()||dk_full.has_inf());
        reset |= (Kk_full.has_nan()||Kk_full.has_inf());

        if (!reset) {
          if (augmented) {
            // Extract control gains (top m rows) and slack gains (bottom n rows)
            int m = sat.control_N();
            int n_red = sat.reduced_state_N();
            Kk = -Kk_full.rows(0, m-1);
            dk = -dk_full.head(m);
            slack_Kset.slice(k) = -Kk_full.rows(m, m+n_red-1);
            slack_dset.col(k) = -dk_full.tail(n_red);
          } else {
            Kk = -Kk_full;
            dk = -dk_full;
          }
        }
      }
    }
    if(!reset){
      Kset.slice(k) = Kk;
      dset.col(k) = dk;

      if (augmented) {
        // Ricatti update using full augmented K̄ = [K; Ks], d̄ = [d; ds]
        int m = sat.control_N();
        int n_red = sat.reduced_state_N();
        mat Kbar = join_vert(Kk, mat(slack_Kset.slice(k)));  // (m+n) × n
        vec dbar = join_vert(dk, vec(slack_dset.col(k)));     // (m+n) × 1
        pk = Qkx + trans(Kbar)*Qkuu_aug*dbar + trans(Kbar)*Qku_aug + trans(Qkux_aug)*dbar;
        Pk = Qkxx + trans(Kbar)*Qkuu_aug*Kbar + trans(Kbar)*Qkux_aug + trans(Qkux_aug)*Kbar;
      } else {
        pk = Qkx + trans(Kk)*Qkuu*dk + trans(Kk)*Qku + trans(Qkux)*dk;
        Pk = Qkxx + trans(Kk)*Qkuu*Kk + trans(Kk)*Qkux + trans(Qkux)*Kk;
      }
      if(Pk.has_nan()||Pk.has_inf()){
        if(verbose){
          cout<<"Costjac "<<costJac.lxx<<"\n";
          for(int j = 0;j<sat.number_RW;j++){
            double z = xk(7+j);
            double sz = sign(z);
            cout<<"RW "<<j<<": z="<<z<<", |z|="<<z*sz<<", threshold="<<sat.RW_AM_cost_threshold.at(j)<<"\n";
            cout<<"  shifted_softplus="<<shifted_softplus(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";
            cout<<"  shifted_softplus_deriv="<<shifted_softplus_deriv(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";
            cout<<"  shifted_softplus_deriv2="<<shifted_softplus_deriv2(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";

            double stic_arg = (sat.RW_stiction_threshold.at(j)-z*sz)/sat.RW_stiction_threshold.at(j);
            cout<<"  stiction_arg="<<stic_arg<<", threshold="<<sat.RW_stiction_threshold.at(j)<<"\n";
            cout<<"  smoothstep="<<smoothstep(stic_arg)<<"\n";
            cout<<"  smoothstep_deriv="<<smoothstep_deriv(stic_arg)<<"\n";
            cout<<"  smoothstep_deriv2="<<smoothstep_deriv2(stic_arg)<<"\n";
          }
          cout<<"Pk has nan/inf\n";
        }
        reset = true;
      }
    }
    if (reset)
    {
      k = N-1;
      delV.zeros();
      Gk.zeros();
      Pk.zeros();
      pk.zeros();
      Kset.zeros();
      dset.zeros();
      dk.zeros();
      Kk.zeros();
      reset = false;
      regs = increaseReg(regs,regSettings_tmp);
      continue;
    }
    Pk = 0.5*(Pk + trans(Pk));
    delV += join_cols(trans(dk)*Qku, 0.5*trans(dk)*Qkuu*dk);//*(get<0>(regs)+mean(eigs))/mean(eigs);//delV_int;
    dk.zeros();
    Kk.zeros();
    k--;
  }
  //find rho and drho
  regs = decreaseReg(regs,regSettings_tmp);
  return make_tuple(make_tuple(Kset,dset,delV), regs);
}
/*This function is forward pass!!
  Arguments:
    Xset - states of previous trajectory, 7 x N mat
    Uset - control vectors of previous trajectory, 3 x N-1 mat
    Kset - K gain matrices from backwards pass, 6 x 3 x N mat
    dset - from previous backwards pass, 3 x N mat
    delV - from previous backwards pass, 1 x 2 mat
    LA - cost of previous trajectory - double
    lambdaSet - lambda - 6 x N mat
    rho,drho,mu - various params
    Rset - orbital position - 3 x N mat
    Vset - orbital velocity - 3 x N mat
    QN - matrices for Q function, 3x3 mat and 6x6 mat respectively
    costSettings - settings for finding Q matrix
    forwardPassSettings - contains maxLsIter beta1 beta2 regScale regMin regBump umax xmax epsilon vNslew satAlignVector (all parameters or max constr or parameters for
      finding cost wrt alignment vectors)
  Returns:
    newX - new trajectory - 7 x N
    newU - new control states - 3 x N
    newLA - new cost - double
    rho,drho - updated params (doubles)
*/
tuple<TRAJECTORY_FORM,double, REG_PAIR> OldPlanner::forwardPass(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, AUGLAG_INFO_FORM auglag_vals, BACKWARD_PASS_RESULTS_FORM BPresults, REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp_ptr, REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,bool useDist)
{
  int maxLsIter_tmp = get<0>(lineSearchSettings_tmp);
  double beta1_tmp = get<1>(lineSearchSettings_tmp);
  double beta2_tmp = get<2>(lineSearchSettings_tmp);

  cube Kset = get<0>(BPresults);
  mat dset = get<1>(BPresults);
  vec2 delV = get<2>(BPresults);

  //Get N
  mat Xset = get<0>(traj);
  int N = Xset.n_cols;
  //Initialize newU, newX
  mat newX = mat(sat.state_N(), N).zeros();
  mat newU = mat(sat.control_N(), N).zeros();
  //Initialize cost, alpha, z, exp
  double alph = 1.0;
  double newLA = 1.79769e+308;//1/eps;
  double z = -1.0;
  int lsiter = 0;
  double exp = 0.0;

  TRAJECTORY_FORM newTraj = traj;
  bool everythingOK = true;

  double LA = cost2Func(traj,vecs,auglag_vals, costSettings_tmp_ptr);
  if (use_infeasible_start) { LA += slackCost(slack_Sset); }
  // newTraj = generateTrajectory(dt0,0.0,traj,vecs, Kset, dset,useDist);
  // newX = get<0>(newTraj);
  // newU = get<1>(newTraj);
  // newLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_tmp_ptr);;

  //if(verbose){cout<<LA<<" "<<newLA<<"\n";}//here overall
  ls_failed = false;
  //Loop while z is NOT between beta2 and beta1, and the new cost is higher than the original cost
  if(verbose){cout<<delV.t()<<"\n";}

  newTraj = generateTrajectory(dt0,0,traj,vecs, Kset, dset,useDist);
  newLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_tmp_ptr);
  // if(verbose){cout<<"ls iter, LA-nLA, nLA,TEST, z,alph,reg "<<-1<<" "<<LA-newLA<<" "<<newLA<<" "<<0<<" "<<nan("1")<<" "<<0<<" "<<get<0>(regs)<<"\n";}
  newLA = 1.79769e+308;//1/eps;

  while((z<=beta1_tmp||z>beta2_tmp)||(newLA>=LA))
  {

    //If iter > maxLsIter, need to give up and just return the original trajectory if we haven't found a better new one
    if(lsiter > maxLsIter_tmp)
    {
      //double drho0 = get<1>(regs);
      regs = increaseReg(regs,regSettings_tmp);
      //increase regularization, so a second try does better
      //regs = increaseReg(regs,regSettings_tmp);
      //bump regularization even more
      regs = make_tuple(get<0>(regs) + get<4>(regSettings_tmp),get<1>(regs));//1.0/get<3>(regSettings_tmp));

      regs = increaseReg(regs,regSettings_tmp); //do 2 increases otherwise the backwardpass just undoes it
      if(verbose){cout<<"*************** z denied\n";}
      ls_failed = true;
      return make_tuple(traj,LA, regs);

    }
    //Call generateTrajectory to get a new trajectory
    newTraj = generateTrajectory(dt0,alph,traj,vecs, Kset, dset,useDist);
    newX = get<0>(newTraj);
    newU = get<1>(newTraj);

    //If we have violated the constraints, or there was an error causing NaN in some matrix, we need to skip to the next iteration with updated iter and alph
    //Check newX for issues
    //Not sure if correctly checking for constraint violations
    // everythingOK = ;
    z = -1.0;
    if(!(newX.has_nan()||newX.has_inf()||newU.has_nan()||newU.has_inf())){//||(abs(newX).max()>100000000.0))) {
      newLA = cost2Func(newTraj,vecs,auglag_vals, costSettings_tmp_ptr);
      if (use_infeasible_start) { newLA += slackCost(slack_Sset_new); }
      double newLAtest = cost2Func(traj,vecs,auglag_vals, costSettings_tmp_ptr);
        // if(verbose){cout<<"calced new LA\n";}
      // everythingOK |= !isnan(newLA);
      if(isnan(newLA)||isinf(newLA)){
        newTraj = traj;
        newLA = LA;
        if(verbose){cout<<"newLA is nan\n";}
        lsiter++;
        alph /= 2.0;
        continue;
      }
    }
    // else{cout<<" issue with trajectory!\n";
    //   }
      //Update exp
    exp = -alph*(delV(0) + alph*delV(1));
    // if((exp > 0.0)&&(exp<1.0e-14))
    // {
    //   //double drho0 = get<1>(regs);
    //   regs = increaseReg(regs,regSettings_tmp);
    //   //increase regularization, so a second try does better
    //   //regs = increaseReg(regs,regSettings_tmp);
    //   //bump regularization even more
    //   return make_tuple(traj,LA, regs);
    //
    // } else
    if (exp > 0.0)
    {
      z = (LA-newLA)/exp;
    }

    // if(lsiter < 5 || lsiter % 10 == 0){cout<<"  [LS] iter:"<<lsiter<<" dLA:"<<LA-newLA<<" exp:"<<exp<<" z:"<<z<<" alph:"<<alph<<" rho:"<<get<0>(regs)<<"\n";}
    lsiter++;
    alph /= 2.0;
  }
  //If we somehow increased the cost, we need to throw an exception. This is not supposed to happen! (because we will just keep the old traj if can't find a better one)
  if(newLA > LA)
  {
    if(verbose){cout<<"Increased cost in forwardpass\n";}
    throw("Increased cost in forwardpass");
  }
  if(verbose){cout<<"*************** z "<<z<<"\n";}
  // Commit new slacks on acceptance
  if (use_infeasible_start && !slack_Sset_new.is_empty()) {
    slack_Sset = slack_Sset_new;
  }
  return make_tuple(newTraj, newLA, regs);
}

/*This is the cost function, which finds the cost over a preset trajectory
  Arguments:
   Xset - preset states of trajectory, 7 x N
   Uset - preset control inputs of trajectory, 3 x N-1
   Vset - preset orbit velocity of trajectory, 3 x N
   Rset - preset orbital position of trajectory, 3 x N
   lambdaSet - preset lambda vector, 6 x N
   mu - parameter - double
   dt0 - time between steps in trajectory - double
   QN - Q function at final timestep - 6x6 matrix
   R - control cost - 3 x 3 matrix
   umax_ptr - maximum allowable u, for constraints - 3 x 1 vector
   costSettings - settings for finding Q function -- tuple<int, double, double, mat, double, double> costSettings contains: an  which describes the time to go from slew mode to pointing mode,
   double sv1, a scaling factor for the quaternion cost in pointing mode, double swpoint, a scaling factor for the
   rotation rate cost in pointing mode, rNslew, the r_ECI at time Nslew, swslew,
   a scaling factor for the rotation rate cost in slew mode, and sratioslew, a scaling factor for the overall Q
   matrix in slew mode
  Returns:
  LA - vector of cost at each timestep - double
*/
 double OldPlanner::cost2Func( TRAJECTORY_FORM &traj,  VECTOR_INFO_FORM &vecs,  AUGLAG_INFO_FORM &auglag_vals,  COST_SETTINGS_FORM *costSettings_ptr,bool useConstraints)
{
  mat Xset = get<0>(traj);
  mat Uset0 = get<1>(traj);
  mat Uset = mat(Uset0);
  if(Uset.n_cols<Xset.n_cols){
    Uset = join_rows(Uset,vec(sat.control_N()).zeros());
  }

  COST_SETTINGS_FORM costSettings_tmp = *costSettings_ptr;
  int N = Xset.n_cols;
  mat sunset = get<4>(vecs);
  mat Bset = get<3>(vecs);
  mat satvec = get<5>(vecs);
  mat ECIvec = get<6>(vecs);
  mat lambdaSet = get<0>(auglag_vals);
  double mu = get<1>(auglag_vals);
  mat muSet = get<2>(auglag_vals);
  vec ck;
  vec xk;
  vec uk;
  vec ukp;
  vec lamk;
  vec muk;
  vec3 bk;
  vec ek;
  vec3 sunk;
  vec3 sk;

  mat Imuk;
  mat Ilamk;
  double dLA = 0.0;

  double LA = 0.0;
  ukp = Uset.col(0);
  for(int k = 0; k < N; k++)
  {
    xk = Xset.col(k);
    vec xkp1 = (k < N-1) ? vec(Xset.col(k+1)) : xk;  // Next state for path length cost
    uk = Uset.col(k);
    bk = Bset.col(k);
    sunk = normalise(sunset.col(k));
    sk = satvec.col(k);
    ek = ECIvec.col(k);
    //Update ck and Imuk
    if((ek.n_elem==3)||((ek.n_elem==4)&&(isnan(ek(0))))){
      ek = ek.tail(3);
      dLA = sat.stepcost_vec(k, N, xk, xkp1, uk, ukp,sk, ek,bk, costSettings_ptr);
    }else{
      dLA = sat.stepcost_quat(k, N, xk, xkp1, uk, ukp,sk, ek,bk, costSettings_ptr);
    }
    LA += dLA;
    if(isinf(dLA) || isnan(dLA)){
      if(verbose){
        cout<<dLA<<"\n";
        cout<<xk.t()<<"\n";
        cout<<uk.t()<<"\n";
        cout<<bk.t()<<"\n";
        cout<<sk.t()<<"\n";
        cout<<ek.t()<<"\n";
        cout<<k<<"\n";
        if(k>0){
          cout<<"prev\n";
          cout<<Xset.col(k-1).t()<<"\n";
          cout<<Uset.col(k-1).t()<<"\n";
          cout<<Bset.col(k-1).t()<<"\n";
        }
        for(int j = 0;j<sat.number_RW;j++)
        {
          double z = xk(7+j);
          double sz = sign(z);
          cout<<"RW "<<j<<": |z|="<<z*sz<<", threshold="<<sat.RW_AM_cost_threshold.at(j)<<"\n";
          cout<<"  shifted_softplus="<<shifted_softplus(z*sz,sat.RW_AM_cost_threshold.at(j))<<"\n";
          cout<<"  AM_cost="<<0.5*sat.RW_AM_cost.at(j)*pow(shifted_softplus(z*sz,sat.RW_AM_cost_threshold.at(j)),2)<<"\n";
          double stic_arg = (sat.RW_stiction_threshold.at(j)-z*sz)/sat.RW_stiction_threshold.at(j);
          cout<<"  stiction_arg="<<stic_arg<<", stiction_cost="<< 0.5*sat.RW_stiction_cost.at(j)*pow(smoothstep(stic_arg)*sat.RW_stiction_threshold.at(j),2)<<"\n";
        }
        cout<<"broken, infinite/nan cost on step.\n";
      }
      break;
    }


    //assert (!isnan(stepcost_vec(k, N, xk,uk,  sk, ek,bk, costSettings_ptr)) || !(std::cerr<<k<<" "<<N<<" "<<xk.t()<<" "<<uk.t()<<" "<<sk.t()<<" "<<ek.t()<<" "<<bk.t()<<" "<<"\n"));
    if(useConstraints){
      lamk = lambdaSet.col(k);
      muk = muSet.col(k);
      ck = sat.getConstraints(k, N, uk, xk,sunk);
      Ilamk = sat.getIlam(mu, muk, ck, lamk);
      Imuk = sat.getImu(mu, muk, ck, lamk);
      LA += as_scalar(trans(lamk)*Ilamk*ck+trans(0.5*Imuk*ck)*ck);
    }

    ukp = uk;
  }
  return LA;
}
