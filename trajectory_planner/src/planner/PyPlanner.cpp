// #include "OldPlanner.hpp"
// #include "PlannerUtil.hpp"
// #include "PlannerPython.hpp"
// #include "PlannerUtilPy.hpp"
// #include "../ArmaNumpy.hpp"
#include <stdexcept>
#include <typeinfo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "PyPlanner.hpp"



namespace py = pybind11;
using namespace arma;
using namespace std;

PyPlanner::PyPlanner(Satellite sat, SYSTEM_SETTINGS_PY_FORM systemSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings2_tmp,  INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp)
: op(sat,ParametersPy2Cpp(std::make_tuple(systemSettings_tmp,alilqrSettings_tmp,alilqrSettings2_tmp,initialTrajSettings_tmp,costSettings_tmp,costSettings2_tmp,costSettings_tvlqr_tmp)))
{
  // cout<<"in pyplanner\n";
}

void PyPlanner::updateParameters(SYSTEM_SETTINGS_PY_FORM systemSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings2_tmp, INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp, COST_SETTINGS_FORM costSettings_tmp,COST_SETTINGS_FORM costSettings2_tmp,LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp) {
    ALL_SETTINGS_PY_FORM allSettingsPy = std::make_tuple(systemSettings_tmp,alilqrSettings_tmp,alilqrSettings2_tmp,initialTrajSettings_tmp,costSettings_tmp,costSettings2_tmp,costSettings_tvlqr_tmp);
    ALL_SETTINGS_FORM allSettings = ParametersPy2Cpp(allSettingsPy);
    op.updateParameters_notsat(get<0>(allSettings),get<1>(allSettings),get<2>(allSettings),get<3>(allSettings),get<4>(allSettings),get<5>(allSettings),get<6>(allSettings));

}


py::tuple PyPlanner::readParameters() {

  ALL_SETTINGS_FORM  allSettings = op.readParameters();
  ALL_SETTINGS_PY_FORM allSettingsPy = ParametersCpp2Py(allSettings);
  return py::make_tuple( get<0>(allSettingsPy),  get<1>(allSettingsPy),  get<2>(allSettingsPy),  get<3>(allSettingsPy),  get<4>(allSettingsPy),  get<5>(allSettingsPy), get<6>(allSettingsPy));
}

py::tuple PyPlanner::readForDebugPlotting() {
  try{
    py::array_t<double> Xset_out = armaMatrixToNumpy(op.current_Xset);
    py::array_t<double> Uset_out = armaMatrixToNumpy(op.current_Uset);
    int ilqr_out = op.current_ilqr_iter;
    int outer_out = op.current_outer_iter;
    return py::make_tuple(Xset_out, Uset_out, ilqr_out, outer_out);
  } catch(...) {
    py::array_t<double> Xset_out = armaMatrixToNumpy(mat22().zeros());
    py::array_t<double> Uset_out = armaMatrixToNumpy(mat22().zeros());
    int ilqr_out = -1;
    int outer_out = -1;
    return py::make_tuple(Xset_out, Uset_out, ilqr_out, outer_out);
  }
}

py::tuple PyPlanner::trajOptBeforePython(VECTOR_INFO_PY_FORM vecs_w_timePy,double dt_use, TIME_FORM time_start, TIME_FORM time_end, py::array_t<double> x0Numpy, int bdotOn){
  vec x0 = numpyToArmaVector(x0Numpy);
  // cout<<"x0"<<x0.t()<<endl;
  VECTOR_INFO_FORM vecs_w_time = vecsPy2Cpp(vecs_w_timePy);

  BEFORE_OUTPUT_FORM results = op.trajOptBefore(vecs_w_time, dt_use, time_start, time_end, x0, bdotOn);
  VECTOR_INFO_FORM vecs_dt = get<1>(results);
  TRAJECTORY_PY_FORM trajPy = trajCpp2Py(get<0>(results));
  TRAJECTORY_FORM traj = get<0>(results);
  mat Xset = get<0>(traj);
  // cout<<"Xset0"<<Xset.col(0).t()<<endl;
  VECTOR_INFO_PY_FORM vecs_dtPy = vecsCpp2Py(vecs_dt);
  COST_SETTINGS_FORM costSettings_tmp = get<2>(results);
  // py::array_t<double> dt_timevecPy = armaMatrixToNumpy(get<0>(vecs_dt));

  return py::make_tuple(trajPy,vecs_dtPy,costSettings_tmp);
}

py::tuple PyPlanner::trajOpt(VECTOR_INFO_PY_FORM vecs_w_timePy,int N, TIME_FORM time_start, TIME_FORM time_end, py::array_t<double> x0Numpy, int bdotOn){
  // cout << std::format("{}", time_start)<<"\n";
  // cout << std::format("{}", time_end)<<"\n";

  // cout<<"numpy: "<<x0Numpy<<"\n";
  vec x0 = numpyToArmaVector(x0Numpy);
  // cout<<x0<<"\n";
  //   cout<<"numpy: "<<x0Numpy<<"\n";
  VECTOR_INFO_FORM vecs_w_time = vecsPy2Cpp(vecs_w_timePy);
  AFTER_OUTPUT_FORM results = op.trajOpt(vecs_w_time, N, time_start, time_end, x0, bdotOn);

  py::tuple resultsPy = afterOutputCpp2Py(results);
  //return success
  return resultsPy;
}



py::tuple PyPlanner::trajOptAfterPython(VECTOR_INFO_PY_FORM vecs_w_timePy,double dt_prev, TIME_FORM time_start, TIME_FORM time_end, ALILQR_OUTPUT_PY_FORM alilqrOutPy)
{
  VECTOR_INFO_FORM vecs_w_times = vecsPy2Cpp(vecs_w_timePy);
  ALILQR_OUTPUT_FORM alilqrOut = alilqrOutPy2Cpp(alilqrOutPy);
  AFTER_OUTPUT_FORM results = op.trajOptAfter(vecs_w_times, dt_prev, time_start, time_end, alilqrOut);
  py::tuple resultsPy = afterOutputCpp2Py(results);
  //return success
  return resultsPy;
}

TRAJECTORY_PY_FORM PyPlanner::generateInitialTrajectoryPython(double dt0, py::array_t<double> x0Py, py::array_t<double> UsetPy,VECTOR_INFO_PY_FORM vecsPy){
  vec x0 = numpyToArmaVector(x0Py);
  // cout<<"x0 in genIT call py "<<x0.rows(3,6).t()<<endl;
  mat Uset = numpyToArmaMatrix(UsetPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  TRAJECTORY_FORM traj = op.generateInitialTrajectory(dt0,x0,Uset,vecs);
  return trajCpp2Py(traj);
}



py::array_t<double> PyPlanner::rk4zPython(double dt0, py::array_t<double> x, py::array_t<double> u, DYNAMICS_INFO_PY_FORM dynamics_info_k_py, DYNAMICS_INFO_PY_FORM dynamics_info_kp1_py) {
  vec xvec = numpyToArmaVector(x);
  vec uvec = numpyToArmaVector(u);
  ////cout<<"test\n";
  vec3 Bvec = numpyToArmaVector(get<0>(dynamics_info_k_py));
  vec3 Rvec = numpyToArmaVector(get<1>(dynamics_info_k_py));
  vec3 Bp1vec = numpyToArmaVector(get<0>(dynamics_info_kp1_py));
  vec3 Rp1vec = numpyToArmaVector(get<1>(dynamics_info_kp1_py));
  int pTF = get<2>(dynamics_info_k_py);
  int pTFp1 = get<2>(dynamics_info_kp1_py);
  vec3 Vvec = numpyToArmaVector(get<3>(dynamics_info_k_py));
  vec3 Svec = numpyToArmaVector(get<4>(dynamics_info_k_py));
  vec3 Vp1vec = numpyToArmaVector(get<3>(dynamics_info_kp1_py));
  vec3 Sp1vec = numpyToArmaVector(get<4>(dynamics_info_kp1_py));
  int distTF = get<5>(dynamics_info_k_py);
  int distTFp1 = get<5>(dynamics_info_kp1_py);

  DYNAMICS_INFO_FORM dynamics_info_k = make_tuple(Bvec,Rvec,pTF,Vvec,Svec,distTF);
  DYNAMICS_INFO_FORM dynamics_info_kp1 = make_tuple(Bp1vec,Rp1vec,pTFp1,Vp1vec,Sp1vec,distTFp1);

  vec xp1 = rk4z_pure(dt0,xvec, uvec, op.sat,dynamics_info_k, dynamics_info_kp1);
  return armaVectorToNumpy(xp1);
}

py::array_t<double> PyPlanner::dynamicsPython(py::array_t<double> x, py::array_t<double> u, DYNAMICS_INFO_PY_FORM dynamics_info_py) {
  vec xvec = numpyToArmaVector(x);
  vec uvec = numpyToArmaVector(u);
  vec3 Bvec = numpyToArmaVector(get<0>(dynamics_info_py));
  vec3 Rvec = numpyToArmaVector(get<1>(dynamics_info_py));
  int pTF = get<2>(dynamics_info_py);
  vec3 Vvec = numpyToArmaVector(get<3>(dynamics_info_py));
  vec3 Svec = numpyToArmaVector(get<4>(dynamics_info_py));
  int distTF = get<5>(dynamics_info_py);


  DYNAMICS_INFO_FORM dynamics_info = make_tuple(Bvec,Rvec,pTF,Vvec,Svec,distTF);

  vec xdot = op.sat.dynamics_pure(xvec, uvec, dynamics_info);
  return armaVectorToNumpy(xdot);
}



py::tuple PyPlanner::addRandNoisePython(double dt0, TRAJECTORY_PY_FORM trajPy, double dlaZcount, double stepsSinceRand, BREAK_SETTINGS_PY_FORM breakSettingsPy,REG_SETTINGS_FORM regSettings_tmp,COST_SETTINGS_FORM costSettings_tmp, AUGLAG_INFO_PY_FORM auglag_valsPy,VECTOR_INFO_PY_FORM vecsPy){
  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
  BREAK_SETTINGS_FORM breakSettings_tmp = breakSettingsPy2Cpp(breakSettingsPy);

  tuple<TRAJECTORY_FORM,double> rnOut = op.addRandNoise(dt0, traj,  dlaZcount,  stepsSinceRand, breakSettings_tmp, regSettings_tmp, &costSettings_tmp, auglag_vals, vecs);
  traj = get<0>(rnOut);
  TRAJECTORY_PY_FORM trajOut = trajCpp2Py(traj);
  stepsSinceRand = get<1>(rnOut);
  return py::make_tuple(trajOut,stepsSinceRand);
}

py::tuple PyPlanner::ilqrStepPython(double dt0,TRAJECTORY_PY_FORM trajPy,VECTOR_INFO_PY_FORM vecsPy,AUGLAG_INFO_PY_FORM auglag_valsPy,REG_PAIR regs,COST_SETTINGS_FORM costSettings_tmp,REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,BREAK_SETTINGS_PY_FORM breakSettingsPy,bool useDist){
  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
  BREAK_SETTINGS_FORM breakSettings_tmp = breakSettingsPy2Cpp(breakSettingsPy);
  tuple<double,double,mat,double,REG_PAIR,TRAJECTORY_FORM> ilqrRes =  op.ilqrStep(dt0,traj,vecs,auglag_vals,regs,&costSettings_tmp,regSettings_tmp,lineSearchSettings_tmp,breakSettings_tmp,useDist);
  double newLA = get<0>(ilqrRes);
  double cmaxtmp = get<1>(ilqrRes);
  mat clist = get<2>(ilqrRes);
  double grad = get<3>(ilqrRes);
  regs = get<4>(ilqrRes);
  traj = get<5>(ilqrRes);
  return py::make_tuple(newLA,cmaxtmp, armaMatrixToNumpy(clist),grad,regs,trajCpp2Py(traj));

}


bool PyPlanner::ilqrBreakPython(double grad, double LA, double dLA, double dlaZcount, double cmaxtmp, double iter,BREAK_SETTINGS_PY_FORM breakSettingsPy,int outer_iter, int ilqr_iter,bool forOuter)
{
  op.current_ilqr_iter = ilqr_iter;
  op.current_outer_iter = outer_iter;
  BREAK_SETTINGS_FORM breakSettings_tmp = breakSettingsPy2Cpp(breakSettingsPy);
  return op.ilqrBreak(grad,LA,dLA,dlaZcount,cmaxtmp,iter,breakSettings_tmp,forOuter);
}


bool PyPlanner::outerBreakPython(AUGLAG_INFO_PY_FORM auglag_valsPy, double cmaxtmp,BREAK_SETTINGS_PY_FORM breakSettingsPy,AUGLAG_SETTINGS_FORM auglagSettings_tmp,int outer_iter)
{
  AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
  op.current_outer_iter = outer_iter;
  BREAK_SETTINGS_FORM breakSettings_tmp = breakSettingsPy2Cpp(breakSettingsPy);
  return op.outerBreak(auglag_vals,cmaxtmp,breakSettings_tmp,auglagSettings_tmp);
}

/*void PyPlanner::costInfoPython(TRAJECTORY_PY_FORM trajPy,VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy,  COST_SETTINGS_FORM costSettings_tmp){
    TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
    VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
    AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
    op.costInfo(traj,vecs,auglag_vals,&costSettings_tmp);
}*/



AUGLAG_INFO_PY_FORM PyPlanner::incrementAugLagPython(AUGLAG_INFO_PY_FORM auglag_valsPy, py::array_t<double> clistPy, AUGLAG_SETTINGS_FORM auglagSettings_tmp){
    AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
    mat clist = numpyToArmaMatrix(clistPy);
    auglag_vals = op.incrementAugLag(auglag_vals,clist,auglagSettings_tmp);
    return auglagCpp2Py(auglag_vals);
}



ALILQR_OUTPUT_PY_FORM PyPlanner::alilqrPython(double dt0, TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, COST_SETTINGS_FORM costSettings_tmp, ALILQR_SETTINGS_PY_FORM alilqrSettings_tmpPy,bool isFirstSearch)
{
  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  ALILQR_SETTINGS_FORM alilqrSettings_tmp = alilqrSettingsPy2Cpp(alilqrSettings_tmpPy);
  ALILQR_OUTPUT_FORM output = op.alilqr(dt0,traj,vecs,costSettings_tmp,alilqrSettings_tmp,isFirstSearch);
  ALILQR_OUTPUT_PY_FORM outputPy = alilqrOutCpp2Py(output);
  return outputPy;
}


double PyPlanner::cost2FuncPython(TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy, COST_SETTINGS_FORM costSettings_tmp) {
  AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  double cost = op.cost2Func(traj, vecs,auglag_vals,  &costSettings_tmp);
  return cost;
}



py::tuple PyPlanner::backwardPassPython(double dt0, TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy,  REG_PAIR regs,COST_SETTINGS_FORM costSettings_tmp,REG_SETTINGS_FORM regSettings_tmp,bool useDist) {

  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);

  tuple<BACKWARD_PASS_RESULTS_FORM,REG_PAIR> backwardPassResults = op.backwardPass(dt0,traj,vecs, auglag_vals, regs,&costSettings_tmp,regSettings_tmp,useDist);
  BACKWARD_PASS_RESULTS_FORM BPresults = get<0>(backwardPassResults);
  regs = get<1>(backwardPassResults);

  BACKWARD_PASS_RESULTS_PY_FORM BPresultsPy = bpOutCpp2Py(BPresults);
  return py::make_tuple(BPresultsPy, regs);
}

py::tuple PyPlanner::forwardPassPython(double dt0, TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM auglag_valsPy, BACKWARD_PASS_RESULTS_PY_FORM BPresultsPy, REG_PAIR regs, COST_SETTINGS_FORM costSettings_tmp, REG_SETTINGS_FORM regSettings_tmp, LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp,bool useDist) {
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  AUGLAG_INFO_FORM auglag_vals = auglagPy2Cpp(auglag_valsPy);
  BACKWARD_PASS_RESULTS_FORM BPresults = bpOutPy2Cpp(BPresultsPy);

  tuple<TRAJECTORY_FORM, double, REG_PAIR> forwardPassOut = op.forwardPass(dt0,traj,  vecs,auglag_vals,BPresults,regs, &costSettings_tmp, regSettings_tmp,lineSearchSettings_tmp,useDist);
  TRAJECTORY_FORM trajOut = get<0>(forwardPassOut);
  double newLA = get<1>(forwardPassOut);
  regs = get<2>(forwardPassOut);

  TRAJECTORY_PY_FORM traj_out = trajCpp2Py(trajOut);

  return py::make_tuple(traj_out, newLA, regs);
}

py::tuple PyPlanner::maxViolPython(TRAJECTORY_PY_FORM trajPy, VECTOR_INFO_PY_FORM vecsPy, AUGLAG_INFO_PY_FORM alpy) {
  TRAJECTORY_FORM traj = trajPy2Cpp(trajPy);
  VECTOR_INFO_FORM vecs = vecsPy2Cpp(vecsPy);
  AUGLAG_INFO_FORM al = auglagPy2Cpp(alpy);

  tuple<mat, double> viol = op.maxViol(traj,vecs,al);
  double cmaxtmp = get<1>(viol);
  mat clist = get<0>(viol);
  py::array_t<double> clist_out = armaMatrixToNumpy(clist);
  return py::make_tuple(clist_out, cmaxtmp);
}

double PyPlanner::getdt(){
  return op.dt;
}

void PyPlanner::setquaternionTo3VecMode(int val){
  op.quaternionTo3VecMode = val;
}

void PyPlanner::setPlannerVerbosity(bool verbosity){
  op.setVerbosity(verbosity);
}


PYBIND11_MODULE(tplaunch, m) {
    py::class_<PyPlanner>(m, "Planner")
        .def(py::init<Satellite,SYSTEM_SETTINGS_PY_FORM, ALILQR_SETTINGS_PY_FORM, ALILQR_SETTINGS_PY_FORM, INITIAL_TRAJ_SETTINGS_FORM, COST_SETTINGS_FORM,COST_SETTINGS_FORM,LQR_COST_SETTINGS_FORM>())
        //.def(py::init<ALL_SETTINGS_PY_FORM>())
        .def("readParameters", &PyPlanner::readParameters)
        .def("readDebug", &PyPlanner::readForDebugPlotting)
        .def("updateParameters", &PyPlanner::updateParameters)
        .def("trajOpt", &PyPlanner::trajOpt)
        .def("prepareForAlilqr", &PyPlanner::trajOptBeforePython)
        .def("cleanUpAfterAlilqr", &PyPlanner::trajOptAfterPython)
        .def("alilqr", &PyPlanner::alilqrPython)
        .def("cost2Func", &PyPlanner::cost2FuncPython)
        .def("backwardPass", &PyPlanner::backwardPassPython)
        .def("dynamics", &PyPlanner::dynamicsPython)
        .def("rk4z", &PyPlanner::rk4zPython)
        .def("forwardPass", &PyPlanner::forwardPassPython)
        .def("maxViol", &PyPlanner::maxViolPython)
        //.def("costInfo", &PyPlanner::costInfoPython)
        .def("outerBreak", &PyPlanner::outerBreakPython)
        .def("ilqrBreak", &PyPlanner::ilqrBreakPython)
        .def("incrementAugLag", &PyPlanner::incrementAugLagPython)
        .def("ilqrStep", &PyPlanner::ilqrStepPython)
        .def("generateInitialTrajectory", &PyPlanner::generateInitialTrajectoryPython)
        .def("addRandNoise", &PyPlanner::addRandNoisePython)
        .def("getdt", &PyPlanner::getdt)
        .def("setVerbosity", &PyPlanner::setPlannerVerbosity)
        .def("setquaternionTo3VecMode", &PyPlanner::setquaternionTo3VecMode)
        .def(py::pickle(
          []( PyPlanner &p) { // __getstate__
              /* Return a tuple that fully encodes the state of the object */
               OldPlanner op = p.op;
              Satellite sat = op.sat;
              ALL_SETTINGS_FORM outCpp = op.readParameters();
               ALL_SETTINGS_PY_FORM out = ParametersCpp2Py(outCpp);
              SYSTEM_SETTINGS_PY_FORM systemSettings_tmp = (SYSTEM_SETTINGS_PY_FORM)get<0>(out);
              ALILQR_SETTINGS_PY_FORM alilqrSettings_tmp = (ALILQR_SETTINGS_PY_FORM )get<1>(out);
              ALILQR_SETTINGS_PY_FORM alilqrSettings2_tmp = (ALILQR_SETTINGS_PY_FORM )get<2>(out);
              INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp = (INITIAL_TRAJ_SETTINGS_FORM)get<3>(out);
              COST_SETTINGS_FORM costSettings_tmp = (COST_SETTINGS_FORM)get<4>(out);
              COST_SETTINGS_FORM costSettings2_tmp = (COST_SETTINGS_FORM)get<5>(out);
              LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp = (LQR_COST_SETTINGS_FORM)get<6>(out);
              return py::make_tuple(sat,systemSettings_tmp, alilqrSettings_tmp, alilqrSettings2_tmp,
                initialTrajSettings_tmp, costSettings_tmp, costSettings2_tmp, costSettings_tvlqr_tmp);
          },
          [](py::tuple t) { // __setstate__
              Satellite sat = t[0].cast<Satellite>();
              SYSTEM_SETTINGS_PY_FORM systemSettings_tmp = t[1].cast<SYSTEM_SETTINGS_PY_FORM>();
              ALILQR_SETTINGS_PY_FORM alilqrSettings_tmp = t[2].cast<ALILQR_SETTINGS_PY_FORM>();
              ALILQR_SETTINGS_PY_FORM alilqrSettings2_tmp = t[3].cast<ALILQR_SETTINGS_PY_FORM>();
              INITIAL_TRAJ_SETTINGS_FORM initialTrajSettings_tmp = t[4].cast<INITIAL_TRAJ_SETTINGS_FORM>();
              COST_SETTINGS_FORM costSettings_tmp = t[5].cast<COST_SETTINGS_FORM>();
              COST_SETTINGS_FORM costSettings2_tmp = t[6].cast<COST_SETTINGS_FORM>();
              LQR_COST_SETTINGS_FORM costSettings_tvlqr_tmp = t[7].cast<LQR_COST_SETTINGS_FORM>();
              //if (t.size() != 9)
                  //throw std::runtime_error("Invalid state!");

              /* Create a new C++ instance */
              PyPlanner p(sat,systemSettings_tmp, alilqrSettings_tmp, alilqrSettings2_tmp,
                initialTrajSettings_tmp, costSettings_tmp,
                costSettings2_tmp, costSettings_tvlqr_tmp);

              return p;
        }));
        // .def_readwrite("dt", &PyPlanner::dt);
}
