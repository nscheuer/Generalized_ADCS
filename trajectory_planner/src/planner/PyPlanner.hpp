#ifndef TPR_PYPLANNER_HPP
#define TPR_PYPLANNER_HPP

//#include <armadillo>
//#include "../ArmaCSV.hpp"
//#include "../ArmaJSON.hpp"
//#include "../Util.hpp"
//#include "../ArmaNumpy.hpp"
#include <string>
#include <tuple>
#include <stdexcept>
#include <typeinfo>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
//#include "PlannerUtil.hpp"
// #include "PlannerPython.hpp"
#include "OldPlanner.hpp"
#include "PlannerUtilPy.hpp"


// //
class PyPlanner {
public:
  PyPlanner(Satellite sat, SYSTEM_SETTINGS_PY_FORM systemSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings2_tmp, INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp);
  void updateParameters(SYSTEM_SETTINGS_PY_FORM systemSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings2_tmp,  INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp);

     py::tuple readParameters();
     py::tuple readForDebugPlotting();
//     BEFORE_OUTPUT_FORM trajOptBefore(VECTOR_INFO_FORM vecs_w_time,int N, TIME_FORM time_start, TIME_FORM time_end, arma::mat x0, int bdotOn);
//     py::tuple trajOptAfter(VECTOR_INFO_FORM vecs_w_time,int N, TIME_FORM time_start, TIME_FORM time_end, ALILQR_OUTPUT_FORM alilqrOut);
    py::tuple trajOpt(VECTOR_INFO_PY_FORM vecsPy,int N, TIME_FORM time_start, TIME_FORM time_end, py::array_t<double> x0Numpy, int bdotOn);
    py::tuple trajOptBeforePython(VECTOR_INFO_PY_FORM vecs_w_timePy,double dt_use, TIME_FORM time_start,TIME_FORM time_end, py::array_t<double> x0Numpy, int bdotOn);
    py::tuple trajOptAfterPython(VECTOR_INFO_PY_FORM vecs_w_timePy,double dt_prev, TIME_FORM time_start, TIME_FORM time_end, ALILQR_OUTPUT_PY_FORM alilqrOut);
    py::array_t<double> rk4zPython(double dt, py::array_t<double> x, py::array_t<double> u,  DYNAMICS_INFO_PY_FORM dynamics_info_k_py, DYNAMICS_INFO_PY_FORM dynamics_info_kp1_py);
    py::array_t<double> dynamicsPython(py::array_t<double> x, py::array_t<double> u,DYNAMICS_INFO_PY_FORM dynamics_info_py);
    double cost2FuncPython(TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy, COST_SETTINGS_FORM costSettings_tmp);
    py::tuple backwardPassPython(double dt, TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy,   AUGLAG_INFO_PY_FORM auglag_valsPy,REG_PAIR regs, COST_SETTINGS_FORM costSettings_tmp, REG_SETTINGS_FORM regSettings_tmp,bool useDist);
    py::tuple forwardPassPython(double dt, TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy, BACKWARD_PASS_RESULTS_PY_FORM BPresultsPy, REG_PAIR regs, COST_SETTINGS_FORM costSettings_tmp, REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,bool useDist);
    AUGLAG_INFO_PY_FORM incrementAugLagPython(AUGLAG_INFO_PY_FORM auglag_valsPy, py::array_t<double> clistPy, AUGLAG_SETTINGS_FORM auglagSettings_tmp);
    ALILQR_OUTPUT_PY_FORM alilqrPython(double dt0, TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, COST_SETTINGS_FORM costSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings_tmpPy,bool isFirstSearch);
    bool ilqrBreakPython(double grad, double LA,double dLA, double dlaZcount, double cmaxtmp, double iter,BREAK_SETTINGS_PY_FORM breakSettingsPy, int oi, int ini,bool forOuter);
    bool outerBreakPython(AUGLAG_INFO_PY_FORM auglag_valsPy, double cmaxtmp,BREAK_SETTINGS_PY_FORM breakSettingsPy,AUGLAG_SETTINGS_FORM auglagSettings_tmp,int oi);
    //void costInfoPython(TRAJECTORY_PY_FORM trajPy,VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy,  COST_SETTINGS_FORM costSettings_tmp);
    py::tuple maxViolPython(TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy,AUGLAG_INFO_PY_FORM alpy);
    py::tuple ilqrStepPython(double dt0,TRAJECTORY_PY_FORM trajPy,VECTOR_INFO_PY_FORM vecsPy,AUGLAG_INFO_PY_FORM auglag_valsPy,REG_PAIR regs,COST_SETTINGS_FORM costSettings_tmp,REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,BREAK_SETTINGS_PY_FORM breakSeettingsPy,bool useDist);
    TRAJECTORY_PY_FORM generateInitialTrajectoryPython(double dt0, py::array_t<double> x0Py, py::array_t<double> UsetPy,VECTOR_INFO_PY_FORM vecsPy);
    double getdt();
    void setquaternionTo3VecMode(int val);
    py::tuple addRandNoisePython(double dt0, TRAJECTORY_PY_FORM trajPy, double dlaZcount, double stepsSinceRand, BREAK_SETTINGS_PY_FORM breakSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,COST_SETTINGS_FORM costSettings_tmp, AUGLAG_INFO_PY_FORM auglag_vals,VECTOR_INFO_PY_FORM vecs);
    void setPlannerVerbosity(bool verbosity);

  mat current_Xset;
  mat current_Uset;
  TRAJECTORY_FORM current_traj;
  int current_ilqr_iter;
  int current_outer_iter;
  OldPlanner op;
 };
#endif
