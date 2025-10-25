#ifndef TPR_OldPlanner_HPP
#define TPR_OldPlanner_HPP

#include <armadillo>
#include <stdexcept>
#include <typeinfo>
#include "Satellite.hpp"
// #include "../ArmaNumpy.hpp"
#include <string>
#include <tuple>
#include "PlannerUtil.hpp"
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
//#include "PlannerUtilPy.hpp"
// #include "PlannerPython.hpp"


//
class OldPlanner {
public:
    OldPlanner();
    OldPlanner(Satellite sat_in, ALL_SETTINGS_FORM allSettings);
     ALL_SETTINGS_FORM readParameters() ;

    void setVerbosity(bool verbosity);
    void updateParameters_notsat(SYSTEM_SETTINGS_FORM systemSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings2_tmp,  INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp);

    BEFORE_OUTPUT_FORM trajOptBefore(VECTOR_INFO_FORM vecs_w_time,double dt_use, TIME_FORM time_start, TIME_FORM time_end, arma::vec x0, int bdotOn);
    AFTER_OUTPUT_FORM trajOptAfter(VECTOR_INFO_FORM vecs_w_time,double dt_prev, TIME_FORM time_start, TIME_FORM time_end, ALILQR_OUTPUT_FORM alilqrOut);
    AFTER_OUTPUT_FORM trajOpt(VECTOR_INFO_FORM &vecs,int N, TIME_FORM time_start, TIME_FORM time_end, arma::vec x0, int bdotOn);
    std::tuple<TRAJECTORY_FORM,double> bdot(arma::vec x0,double dt, int N,VECTOR_INFO_FORM vecs,  COST_SETTINGS_FORM costSettings_tmp,double mu);
    std::tuple<TRAJECTORY_FORM,double> smartbdot(arma::vec x0,double dt,int N,VECTOR_INFO_FORM vecs,COST_SETTINGS_FORM costSettings_tmp,double mu,bool invert);
    arma::vec smartbdot_rawmtq_finder(double dt0,arma::vec wk, double nB2, arma::vec ECIvk, arma::vec ECIvkp1, arma::vec3 satvk, arma::vec3 Bbody,SMARTBDOT_SETTINGS_FORM sbSettings,arma::vec3 dist_torq);

    std::tuple<arma::cube, arma::cube> findK(double dt_tvlqr, TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, COST_SETTINGS_FORM costSettings_tmp);
    std::tuple<arma::cube, arma::cube> findKwDist(double dt_tvlqr, TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, COST_SETTINGS_FORM costSettings_tmp);

    TRAJECTORY_FORM generateTrajectory( double dt,  double alpha,  TRAJECTORY_FORM traj,  VECTOR_INFO_FORM vecs,  arma::cube Kset,  arma::mat dset,  bool useDist);
    TRAJECTORY_FORM generateInitialTrajectory(double dt, arma::vec x0, arma::mat Uset,VECTOR_INFO_FORM vecs);
  //arma::vec dynamics(arma::vec x, arma::vec::fixed<CONTROL_NUM> u, arma::vec3 Bk, arma::vec3 Rk);
    ALILQR_OUTPUT_FORM alilqr(double dt0, TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, COST_SETTINGS_FORM costSettings_tmp, ALILQR_SETTINGS_FORM alilqrSettings_tmp,bool isFirstSearch);
    std::tuple<arma::mat, double> maxViol(TRAJECTORY_FORM &traj, VECTOR_INFO_FORM &vecs,AUGLAG_INFO_FORM &auglag);
    std::tuple<BACKWARD_PASS_RESULTS_FORM, REG_PAIR> backwardPass(double dt,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs,AUGLAG_INFO_FORM auglag_vals,REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,bool useDist);
    std::tuple<TRAJECTORY_FORM,double, REG_PAIR> forwardPass(double dt0,TRAJECTORY_FORM traj, VECTOR_INFO_FORM &vecs, AUGLAG_INFO_FORM auglag_vals, BACKWARD_PASS_RESULTS_FORM BPresults, REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp_ptr, REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,bool useDist);
    //std::tuple<TRAJECTORY_FORM, double, REG_PAIR> forwardPass(double dt,TRAJECTORY_FORM traj, VECTOR_INFO_FORM vecs, AUGLAG_INFO_FORM auglag_vals, BACKWARD_PASS_RESULTS_FORM BPresults, REG_PAIR regs, COST_SETTINGS_FORM *costSettings_tmp, REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp);
    std::tuple<double,double,arma::mat,double,REG_PAIR,TRAJECTORY_FORM> ilqrStep(double dt0,TRAJECTORY_FORM traj,VECTOR_INFO_FORM vecs,AUGLAG_INFO_FORM auglag_vals,REG_PAIR regs,COST_SETTINGS_FORM *costSettings_ptr,REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,BREAK_SETTINGS_FORM breakSettings_tmp,bool useDist);

    double cost2Func( TRAJECTORY_FORM &traj, VECTOR_INFO_FORM &vecs,  AUGLAG_INFO_FORM &auglag_vals,  COST_SETTINGS_FORM *costSettings_ptr,bool useConstraints = true);
    AUGLAG_INFO_FORM incrementAugLag(AUGLAG_INFO_FORM auglag_vals, arma::mat clist, AUGLAG_SETTINGS_FORM auglagSettings_tmp);
    void costInfo(TRAJECTORY_FORM traj,  VECTOR_INFO_FORM vecs, AUGLAG_INFO_FORM auglag_vals,COST_SETTINGS_FORM *costSettings_ptr);
    bool outerBreak(AUGLAG_INFO_FORM auglag_vals, double cmaxtmp,BREAK_SETTINGS_FORM breakSettings_tmp,AUGLAG_SETTINGS_FORM auglagSettings_tmp);
    bool ilqrBreak(double grad, double LA,double dLA, double dlaZcount, double cmaxtmp, double iter,BREAK_SETTINGS_FORM breakSettings_tmp,bool forOuter = false);
    std::tuple<TRAJECTORY_FORM,double> addRandNoise(double dt0, TRAJECTORY_FORM traj, double dlaZcount, double stepsSinceRand, BREAK_SETTINGS_FORM breakSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,COST_SETTINGS_FORM *costSettings_ptr, AUGLAG_INFO_FORM auglag_vals,VECTOR_INFO_FORM vecs);


    double dt;

    arma::mat current_Xset;
    arma::mat current_Uset;
    TRAJECTORY_FORM current_traj;
    int current_ilqr_iter;
    int current_outer_iter;

    // arma::mat33 J;
    // arma::mat33 invJ;
    Satellite sat;



    //remove before flight
    bool verbose;

    int quaternionTo3VecMode = 0;

private:


  double bdotgain;
  double HLangleLimit;

  double gyrogainH;
  double dampgainH;
  double velgainH;
  double quatgainH;
  double randvalH;
  double umaxmultH;

  double gyrogainL;
  double dampgainL;
  double velgainL;
  double quatgainL;
  double randvalL;
  double umaxmultL;


  double angle_weight;
  double angvel_weight;
  double u_weight;
  double u_with_mag_weight;
  double av_with_mag_weight;
  double ang_av_weight;
  double angle_weight_N;
  double angvel_weight_N;
  double av_with_mag_weight_N;
  double ang_av_weight_N;
  int whichAngCostFunc;
  int useRawControlCost;

  double angle_weight2;
  double angvel_weight2;
  double u_weight2;
  double u_with_mag_weight2;
  double av_with_mag_weight2;
  double ang_av_weight2;
  double angle_weight_N2;
  double angvel_weight_N2;
  double av_with_mag_weight_N2;
  double ang_av_weight_N2;
  int whichAngCostFunc2;
  int useRawControlCost2;

  double angle_weight_tvlqr;
  double angvel_weight_tvlqr;
  double u_weight_tvlqr;
  double u_with_mag_weight_tvlqr;
  double av_with_mag_weight_tvlqr;
  double ang_av_weight_tvlqr;
  double angle_weight_N_tvlqr;
  double angvel_weight_N_tvlqr;
  double av_with_mag_weight_N_tvlqr;
  double ang_av_weight_N_tvlqr;
  int whichAngCostFunc_tvlqr;
  int tracking_LQR_formulation;
  int useRawControlCost_tvlqr;


  double dt_tvlqr;
  double tvlqr_len;
  double tvlqr_overlap;
  double eps;

  double maxLsIter;
  double beta1;
  double beta2;

  double lagMultInit;
  double lagMultMax;
  double penInit;
  double penMax;
  double penScale;

  int maxOuterIter;
  int maxIlqrIter;
  int maxIter;
  double gradTol;
  double ilqrCostTol;
  double costTol;
  int zCountLim;
  double cmax;
  double maxCost;
  arma::vec xmax;

  double regInit;
  double regMin;
  double regMax;
  double regScale;
  double regBump;
  int regMinConds;
  double regBumpRandAddRatio;

  int useEVmagic;
  int SPDEVreg;
  int SPDEVregAll;
  int rhoEVregTest;
  int EVregTestpreabs;
  int EVaddreg;
  int EVregIsRho;
  int EVrhoAdd;
  double useDynamicsHess = 1;
  int useConstraintHess = 1;

  int maxLsIter2;
  double beta12;
  double beta22;

  double lagMultInit2;
  double lagMultMax2;
  double penInit2;
  double penMax2;
  double penScale2;

  int maxOuterIter2;
  int maxIlqrIter2;
  int maxIter2;
  double gradTol2;
  double ilqrCostTol2;
  double costTol2;
  int zCountLim2;
  double cmax2;
  double maxCost2;
  arma::vec xmax2;

  double regInit2;
  double regMin2;
  double regMax2;
  double regScale2;
  double regBump2;
  int regMinConds2;
  double regBumpRandAddRatio2;

  int useEVmagic2;
  int SPDEVreg2;
  int SPDEVregAll2;
  int rhoEVregTest2;
  int EVregTestpreabs2;
  int EVaddreg2;
  int EVregIsRho2;
  int EVrhoAdd2;
  double useDynamicsHess2;
  int useConstraintHess2;

  bool ls_failed;

  SYSTEM_SETTINGS_FORM systemSettings;

  SMARTBDOT_SETTINGS_FORM lowSettings;
  SMARTBDOT_SETTINGS_FORM highSettings;
  INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings;

  COST_SETTINGS_FORM costSettings;
  COST_SETTINGS_FORM costSettings2;
  COST_SETTINGS_FORM costSettings_tvlqr;
  LQR_COST_SETTINGS_FORM costSettings_tvlqr_full;


  REG_SETTINGS_FORM regSettings;
  BREAK_SETTINGS_FORM breakSettings;
  AUGLAG_SETTINGS_FORM auglagSettings;
  LINE_SEARCH_SETTINGS_FORM lineSearchSettings;
  ALILQR_SETTINGS_FORM alilqrSettings;

  REG_SETTINGS_FORM regSettings2;
  BREAK_SETTINGS_FORM breakSettings2;
  AUGLAG_SETTINGS_FORM auglagSettings2;
  LINE_SEARCH_SETTINGS_FORM lineSearchSettings2;
  ALILQR_SETTINGS_FORM alilqrSettings2;




  //DEBUG (TODO: remove before flight)
};

#endif
