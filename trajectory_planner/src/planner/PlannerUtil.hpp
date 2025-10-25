#ifndef TPR_PLANNERUTIL_HPP
#define TPR_PLANNERUTIL_HPP

#include <armadillo>
#include <tuple>
#include <assert.h>
#include "Satellite.hpp"
#include "../ArmaCSV.hpp"
#include "../ArmaNumpy.hpp"
#include <string>
//namespace py = pybind11;

// #define earth_mu 3.986e14
// #define EPSVAR 2.22e-16
#define REG_PAIR std::tuple<double,double>

#define LQR_COST_SETTINGS_FORM std::tuple<double, double, double, double, double,double,double,double,double,double,int,int,int>
#define CONSTRAINT_SETTINGS_FORM std::tuple<double, arma::vec, double, arma::vec3,bool,bool>
#define SMARTBDOT_SETTINGS_FORM std::tuple<double, double, double, double,double,double>
#define INITIAL_TRAJ_SETTINGS_FORM std::tuple<double,double,SMARTBDOT_SETTINGS_FORM,SMARTBDOT_SETTINGS_FORM>

#define VECTOR_INFO_FORM std::tuple<arma::vec,arma::mat,arma::mat,arma::mat,arma::mat,arma::mat,arma::mat,arma::vec,arma::vec>
#define AUGLAG_INFO_FORM std::tuple<arma::mat,double,arma::mat>
#define OPT_FORM std::tuple<arma::mat,arma::mat,arma::mat,arma::mat,arma::mat,arma::vec>
// #define OPT_TIMES_FORM std::tuple<arma::mat,arma::mat,arma::mat,arma::mat,arma::vec>
#define FORWARD_PASS_SETTINGS_FORM std::tuple<int,double,double,double,double,double,double>

#define BACKWARD_PASS_RESULTS_FORM std::tuple<arma::cube,arma::mat,arma::mat>

#define TRAJECTORY_FORM std::tuple<arma::mat,arma::mat,arma::vec,arma::mat>
#define SYSTEM_SETTINGS_FORM std::tuple<arma::mat33,double,double,double,double,double>
#define AUGLAG_SETTINGS_FORM std::tuple<double,double,double,double,double>
#define LINE_SEARCH_SETTINGS_FORM std::tuple<int,double,double>
#define REG_SETTINGS_FORM std::tuple<double,double,double,double,double,int,double,int,int,int,int,int,int,int,int,double,int>
#define ALILQR_OUTPUT_FORM std::tuple<OPT_FORM, double, double>
#define BREAK_SETTINGS_FORM std::tuple<int,int,int,double,double,double,int,double,double,arma::vec>
#define ALILQR_SETTINGS_FORM std::tuple<LINE_SEARCH_SETTINGS_FORM,AUGLAG_SETTINGS_FORM,BREAK_SETTINGS_FORM,REG_SETTINGS_FORM>
#define BEFORE_OUTPUT_FORM std::tuple<TRAJECTORY_FORM,VECTOR_INFO_FORM,COST_SETTINGS_FORM>
#define TIME_FORM double //std::tuple<double, int, int, int>

#define ALL_SETTINGS_FORM std::tuple<SYSTEM_SETTINGS_FORM, ALILQR_SETTINGS_FORM, ALILQR_SETTINGS_FORM , INITIAL_TRAJ_SETTINGS_FORM , COST_SETTINGS_FORM ,COST_SETTINGS_FORM ,LQR_COST_SETTINGS_FORM>
#define AFTER_OUTPUT_FORM std::tuple<int,double, OPT_FORM,OPT_FORM, TRAJECTORY_FORM>

//#define COST_SETTINGS_FORM_CPP tuple<double, double, double, double, double,double,double,double>


arma::mat packageK(arma::cube Kcube);
arma::mat packageS(arma::cube Scube);
VECTOR_INFO_FORM findVecTimes(VECTOR_INFO_FORM vecs_w_time,double dt, TIME_FORM time_start, TIME_FORM time_end);
// TRAJECTORY_FORM addTrajTimes(TRAJECTORY_FORM clean_traj );

REG_PAIR increaseReg(REG_PAIR reg0, REG_SETTINGS_FORM reg_settings_tmp);
REG_PAIR decreaseReg(REG_PAIR reg0, REG_SETTINGS_FORM reg_settings_tmp);

arma::mat interp_vector(arma::mat m,arma::vec t_of_m,arma::vec t_des);

std::tuple<arma::vec,arma::vec> rk4z(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4z_pure(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);

std::tuple<arma::mat, arma::mat,arma::mat> rk4zJacobians(double dt,arma::vec xk, arma::vec uk, Satellite sat,DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::mat, arma::mat,arma::mat> rk4zx2Jacobians(double dt,arma::vec xk, arma::vec uk, Satellite sat,DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zxkp1r(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);

std::tuple<arma::cube, arma::cube,arma::cube> rk4zHessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube,arma::mat,arma::mat,arma::cube,arma::cube, arma::cube,arma::cube,arma::cube> rk4zxkp1rHessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);

std::tuple<arma::cube, arma::cube,arma::cube> rk4zx3Hessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zx3(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zx2(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube,arma::mat,arma::mat> rk4zx2Hessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube,arma::mat,arma::mat> rk4zx1Hessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zx1(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zxd0(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zxd3(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);

std::tuple<arma::cube, arma::cube,arma::cube> rk4zxd0Hessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zxd1(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube> rk4zxd1Hessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zx2r(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube,arma::mat,arma::mat> rk4zx2rHessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zx3r(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube,arma::mat,arma::mat> rk4zx3rHessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);
arma::vec rk4zxd2(double dt, arma::vec xk, arma::vec uk, Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k,DYNAMICS_INFO_FORM dynamics_info_kp1);
std::tuple<arma::cube, arma::cube,arma::cube> rk4zxd2Hessians(double dt0,arma::vec xk, arma::vec uk,Satellite sat, DYNAMICS_INFO_FORM dynamics_info_k, DYNAMICS_INFO_FORM dynamics_info_kp1);


arma::vec4 qdes(arma::vec3 satvk, arma::vec3 ECIvk, arma::vec4 q, arma::vec3 w, arma::vec3 Bbody, arma::mat33 wt);

#endif
