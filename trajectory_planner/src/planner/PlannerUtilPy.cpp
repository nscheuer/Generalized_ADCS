// #define earth_mu 3.986e14
// #include "PlannerUtil.hpp"
#include "PlannerUtilPy.hpp"




using namespace arma;
using namespace std;

// ---- HELPERS ---


TRAJECTORY_FORM trajPy2Cpp(TRAJECTORY_PY_FORM py_traj){

  // cout<<"x0 in p2c call "<<x.col(0).rows(3,6).t()<<endl;
  mat x = numpyToArmaMatrix(get<0>(py_traj));
  mat u = numpyToArmaMatrix(get<1>(py_traj));
  vec t = numpyToArmaVector(get<2>(py_traj));
  mat tq = numpyToArmaMatrix(get<3>(py_traj));
  // cout<<"size"<<tq.n_cols<<tq.n_rows<<endl;
  return std::make_tuple(x,u,t,tq);
}

TRAJECTORY_PY_FORM trajCpp2Py(TRAJECTORY_FORM cpp_traj){

    // mat x = get<0>(cpp_traj);
    // cout<<"x0 in c2p call "<<x.col(0).rows(3,6).t()<<endl;
  py::array_t<double> x = armaMatrixToNumpy(get<0>(cpp_traj));
  py::array_t<double> u = armaMatrixToNumpy(get<1>(cpp_traj));
  py::array_t<double> t = armaVectorToNumpy(get<2>(cpp_traj));
  py::array_t<double> tq = armaMatrixToNumpy(get<3>(cpp_traj));
  return std::make_tuple(x,u,t,tq);
}

py::tuple afterOutputCpp2Py(AFTER_OUTPUT_FORM afterCpp){
  int success = get<0>(afterCpp);
  double gradOut = get<1>(afterCpp);
  // OPT_TIMES_FORM main_opt_times = get<2>(afterCpp);
  // OPT_TIMES_FORM lqr_opt_times = get<3>(afterCpp);
  OPT_FORM main_opt = get<2>(afterCpp);
  OPT_FORM lqr_opt = get<3>(afterCpp);
  TRAJECTORY_FORM traj = get<4>(afterCpp);
  // OPT_TIMES_PY_FORM main_opt_timesPy = optTimesCpp2Py(main_opt_times);
  // OPT_TIMES_PY_FORM lqr_opt_timesPy = optTimesCpp2Py(lqr_opt_times);
  OPT_PY_FORM main_optPy = optCpp2Py(main_opt);
  OPT_PY_FORM lqr_optPy = optCpp2Py(lqr_opt);
  TRAJECTORY_PY_FORM  trajPy = trajCpp2Py(traj);
  // return py::make_tuple(success, gradOut, main_opt_timesPy, lqr_opt_timesPy, trajPy);
  return py::make_tuple(success, gradOut, main_optPy, lqr_optPy, trajPy);
}

ALILQR_SETTINGS_FORM alilqrSettingsPy2Cpp(ALILQR_SETTINGS_PY_FORM alilqrPy){
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp = get<0>(alilqrPy);
    AUGLAG_SETTINGS_FORM auglagSettings_tmp = get<1>(alilqrPy);
    BREAK_SETTINGS_PY_FORM breakSettings_tmpPy = get<2>(alilqrPy);
    REG_SETTINGS_FORM regSettings_tmp = get<3>(alilqrPy);

    BREAK_SETTINGS_FORM breakSettings_tmp = breakSettingsPy2Cpp(breakSettings_tmpPy);
    ALILQR_SETTINGS_FORM alilqrCpp = std::make_tuple(lineSearchSettings_tmp,auglagSettings_tmp,breakSettings_tmp,regSettings_tmp);
    return alilqrCpp;
}


ALILQR_SETTINGS_PY_FORM alilqrSettingsCpp2Py(ALILQR_SETTINGS_FORM alilqrCpp){
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings_tmp = get<0>(alilqrCpp);
    AUGLAG_SETTINGS_FORM auglagSettings_tmp = get<1>(alilqrCpp);
    BREAK_SETTINGS_FORM breakSettings_tmp = get<2>(alilqrCpp);
    REG_SETTINGS_FORM regSettings_tmp = get<3>(alilqrCpp);

    BREAK_SETTINGS_PY_FORM breakSettings_tmpPy = breakSettingsCpp2Py(breakSettings_tmp);
    ALILQR_SETTINGS_PY_FORM alilqrPy = std::make_tuple(lineSearchSettings_tmp,auglagSettings_tmp,breakSettings_tmpPy,regSettings_tmp);
    return alilqrPy;
}

BREAK_SETTINGS_FORM breakSettingsPy2Cpp(BREAK_SETTINGS_PY_FORM breakPy){
  int maxOuterIter_tmp = get<0>(breakPy);
  int maxIlqrIter_tmp = get<1>(breakPy);
  int maxIter_tmp = get<2>(breakPy);
  double gradTol_tmp = get<3>(breakPy);
  double ilqrCostTol_tmp = get<4>(breakPy);
  double costTol_tmp = get<5>(breakPy);
  int zCountLim_tmp = get<6>(breakPy);
  double cmax_tmp = get<7>(breakPy);
  double maxCost_tmp = get<8>(breakPy);
  py::array_t<double> xmax_tmpPy = get<9>(breakPy);

  vec xmax_tmp = numpyToArmaVector(xmax_tmpPy);

  BREAK_SETTINGS_FORM breakCpp = std::make_tuple(maxOuterIter_tmp,maxIlqrIter_tmp,maxIter_tmp,gradTol_tmp,ilqrCostTol_tmp,costTol_tmp,zCountLim_tmp,cmax_tmp,maxCost_tmp,xmax_tmp);
  return breakCpp;

}


BREAK_SETTINGS_PY_FORM breakSettingsCpp2Py(BREAK_SETTINGS_FORM breakCpp){
  int maxOuterIter_tmp = get<0>(breakCpp);
  int maxIlqrIter_tmp = get<1>(breakCpp);
  int maxIter_tmp = get<2>(breakCpp);
  double gradTol_tmp = get<3>(breakCpp);
  double ilqrCostTol_tmp = get<4>(breakCpp);
  double costTol_tmp = get<5>(breakCpp);
  int zCountLim_tmp = get<6>(breakCpp);
  double cmax_tmp = get<7>(breakCpp);
  double maxCost_tmp = get<8>(breakCpp);
  vec xmax_tmp = get<9>(breakCpp);

  py::array_t<double> xmax_tmpPy = armaVectorToNumpy(xmax_tmp);

  BREAK_SETTINGS_PY_FORM breakPy = std::make_tuple(maxOuterIter_tmp,maxIlqrIter_tmp,maxIter_tmp,gradTol_tmp,ilqrCostTol_tmp,costTol_tmp,zCountLim_tmp,cmax_tmp,maxCost_tmp,xmax_tmpPy);
  return breakPy;

}


SYSTEM_SETTINGS_FORM systemSettingsPy2Cpp(SYSTEM_SETTINGS_PY_FORM sysPy){
    arma::mat33 J = numpyToArmaMatrix(get<0>(sysPy));
    // invJ = inv(J);
    double dt = get<1>(sysPy);
    double dt_tvlqr = get<2>(sysPy);
    double eps = get<3>(sysPy);
    double tvlqr_len = get<4>(sysPy);
    double tvlqr_overlap = get<5>(sysPy);
    SYSTEM_SETTINGS_FORM sysCpp = std::make_tuple(J,dt,dt_tvlqr,eps,tvlqr_len,tvlqr_overlap);
    return sysCpp;
}

SYSTEM_SETTINGS_PY_FORM systemSettingsCpp2Py(SYSTEM_SETTINGS_FORM sysCpp){
    py::array_t<double> J = armaMatrixToNumpy(get<0>(sysCpp));
    // invJ = inv(J);
    double dt = get<1>(sysCpp);
    double dt_tvlqr = get<2>(sysCpp);
    double eps = get<3>(sysCpp);
    double tvlqr_len = get<4>(sysCpp);
    double tvlqr_overlap = get<5>(sysCpp);
    SYSTEM_SETTINGS_PY_FORM sysPy = std::make_tuple(J,dt,dt_tvlqr,eps,tvlqr_len,tvlqr_overlap);
    return sysPy;
}


ALL_SETTINGS_FORM ParametersPy2Cpp(ALL_SETTINGS_PY_FORM allPy){
    SYSTEM_SETTINGS_FORM systemSettings = systemSettingsPy2Cpp(get<0>(allPy));
    ALILQR_SETTINGS_FORM alilqrSettings = alilqrSettingsPy2Cpp(get<1>(allPy));
    ALILQR_SETTINGS_FORM alilqrSettings2 = alilqrSettingsPy2Cpp(get<2>(allPy));

    return std::make_tuple(systemSettings,alilqrSettings,alilqrSettings2,get<3>(allPy),get<4>(allPy),get<5>(allPy),get<6>(allPy));
}


ALL_SETTINGS_PY_FORM ParametersCpp2Py(ALL_SETTINGS_FORM allCpp)
{
    SYSTEM_SETTINGS_PY_FORM systemSettings = systemSettingsCpp2Py(get<0>(allCpp));

    ALILQR_SETTINGS_PY_FORM alilqrSettings = alilqrSettingsCpp2Py(get<1>(allCpp));
    ALILQR_SETTINGS_PY_FORM alilqrSettings2 = alilqrSettingsCpp2Py(get<2>(allCpp));

    return std::make_tuple(systemSettings,alilqrSettings,alilqrSettings2,get<3>(allCpp),get<4>(allCpp),get<5>(allCpp),get<6>(allCpp));
}


VECTOR_INFO_FORM vecsPy2Cpp(VECTOR_INFO_PY_FORM py_vecs){
  vec t = numpyToArmaVector(get<0>(py_vecs));
  mat r = numpyToArmaMatrix(get<1>(py_vecs));
  mat v = numpyToArmaMatrix(get<2>(py_vecs));
  mat b = numpyToArmaMatrix(get<3>(py_vecs));
  mat s = numpyToArmaMatrix(get<4>(py_vecs));
  mat a = numpyToArmaMatrix(get<6>(py_vecs));
  mat e = numpyToArmaMatrix(get<7>(py_vecs));
  vec p = numpyToArmaVector(get<8>(py_vecs));
  vec rho = numpyToArmaVector(get<5>(py_vecs));
  return std::make_tuple(t,r,v,b,s,a,e,p,rho);
}

VECTOR_INFO_PY_FORM vecsCpp2Py(VECTOR_INFO_FORM cpp_vecs){
  py::array_t<double> t = armaVectorToNumpy(get<0>(cpp_vecs));
  py::array_t<double> r = armaMatrixToNumpy(get<1>(cpp_vecs));
  py::array_t<double> v = armaMatrixToNumpy(get<2>(cpp_vecs));
  py::array_t<double> b = armaMatrixToNumpy(get<3>(cpp_vecs));
  py::array_t<double> s = armaMatrixToNumpy(get<4>(cpp_vecs));
  py::array_t<double> a = armaMatrixToNumpy(get<5>(cpp_vecs));
  py::array_t<double> e = armaMatrixToNumpy(get<6>(cpp_vecs));
  py::array_t<double> p = armaVectorToNumpy(get<7>(cpp_vecs));
  py::array_t<double> rho = armaVectorToNumpy(get<8>(cpp_vecs));
  return std::make_tuple(t,r,v,b,s,rho,a,e,p);
}

AUGLAG_INFO_FORM auglagPy2Cpp(AUGLAG_INFO_PY_FORM py_auglag){
  mat l = numpyToArmaMatrix(get<0>(py_auglag));
  double mu = get<1>(py_auglag);
  mat m = numpyToArmaMatrix(get<2>(py_auglag));
  return std::make_tuple(l,mu,m);
}

AUGLAG_INFO_PY_FORM auglagCpp2Py(AUGLAG_INFO_FORM cpp_auglag){
  py::array_t<double> l = armaMatrixToNumpy(get<0>(cpp_auglag));
  double mu = get<1>(cpp_auglag);
  py::array_t<double> m = armaMatrixToNumpy(get<2>(cpp_auglag));
  return std::make_tuple(l,mu,m);
}

ALILQR_OUTPUT_FORM alilqrOutPy2Cpp(ALILQR_OUTPUT_PY_FORM aloutPy){
  OPT_FORM opt = optPy2Cpp(get<0>(aloutPy));
  double mu = get<1>(aloutPy);
  double lastGrad = get<2>(aloutPy);
  return std::make_tuple(opt,mu,lastGrad);
}

ALILQR_OUTPUT_PY_FORM alilqrOutCpp2Py(ALILQR_OUTPUT_FORM alout){
  OPT_PY_FORM optPy = optCpp2Py(get<0>(alout));
  double mu = get<1>(alout);
  double lastGrad = get<2>(alout);
  return std::make_tuple(optPy,mu,lastGrad);
}

BACKWARD_PASS_RESULTS_PY_FORM bpOutCpp2Py(BACKWARD_PASS_RESULTS_FORM bpout){
  py::array_t<double> K = armaCubeToNumpy(get<0>(bpout));
  py::array_t<double> d = armaMatrixToNumpy(get<1>(bpout));
  py::array_t<double> dV = armaMatrixToNumpy(get<2>(bpout));
  return std::make_tuple(K,d,dV);
}

BACKWARD_PASS_RESULTS_FORM bpOutPy2Cpp(BACKWARD_PASS_RESULTS_PY_FORM bpoutPy){
  cube K = numpyToArmaCube(get<0>(bpoutPy));
  mat d = numpyToArmaMatrix(get<1>(bpoutPy));
  mat dV = numpyToArmaMatrix(get<2>(bpoutPy));
  return std::make_tuple(K,d,dV);
}


OPT_FORM optPy2Cpp(OPT_PY_FORM py_opt){
  mat x = numpyToArmaMatrix(get<0>(py_opt));
  mat u = numpyToArmaMatrix(get<1>(py_opt));
  mat tq = numpyToArmaMatrix(get<2>(py_opt));
  mat k = numpyToArmaMatrix(get<3>(py_opt));
  mat l = numpyToArmaMatrix(get<4>(py_opt));
  vec t = numpyToArmaVector(get<5>(py_opt));
  return std::make_tuple(x,u,tq,k,l,t);
}

OPT_PY_FORM optCpp2Py(OPT_FORM cpp_opt){
  py::array_t<double> x = armaMatrixToNumpy(get<0>(cpp_opt));
  py::array_t<double> u = armaMatrixToNumpy(get<1>(cpp_opt));
  py::array_t<double> tq = armaMatrixToNumpy(get<2>(cpp_opt));
  py::array_t<double> k = armaMatrixToNumpy(get<3>(cpp_opt));
  py::array_t<double> l = armaMatrixToNumpy(get<4>(cpp_opt));
  py::array_t<double> t = armaVectorToNumpy(get<5>(cpp_opt));
  return std::make_tuple(x,u,tq,k,l,t);
}

// OPT_TIMES_FORM optTimesPy2Cpp(OPT_TIMES_PY_FORM py_opt){
//   mat x = numpyToArmaMatrix(get<0>(py_opt));
//   mat u = numpyToArmaMatrix(get<1>(py_opt));
//   mat k = numpyToArmaMatrix(get<2>(py_opt));
//   mat l = numpyToArmaMatrix(get<3>(py_opt));
//   return std::make_tuple(x,u,k,l);
// }

// OPT_TIMES_PY_FORM optTimesCpp2Py(OPT_TIMES_FORM cpp_opt){
//   py::array_t<double> x = armaMatrixToNumpy(get<0>(cpp_opt));
//   py::array_t<double> u = armaMatrixToNumpy(get<1>(cpp_opt));
//   py::array_t<double> k = armaMatrixToNumpy(get<2>(cpp_opt));
//   py::array_t<double> l = armaMatrixToNumpy(get<3>(cpp_opt));
//   return std::make_tuple(x,u,k,l);
// }

// --- END HELPERS ---
