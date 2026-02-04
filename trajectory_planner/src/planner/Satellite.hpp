#ifndef TPR_SATELLITE_HPP
#define TPR_SATELLITE_HPP

#include <armadillo>
#include <tuple>
#include "../ArmaNumpy.hpp"
#include <string>
#include "GeneralUtil.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <list>
#include <vector>

// Cost settings tuple: (angle, ang_vel, control_mult, ang_vel_mag, ang_vel_err_dir,
//                       angle_N, ang_vel_N, ang_vel_mag_N, ang_vel_err_dir_N,
//                       ang_cost_func_type, use_raw_control_cost, use_full_cost_hess)
#define COST_SETTINGS_FORM std::tuple<double, double, double, double, double, double, double, double, double, int, int, int>
#define DYNAMICS_INFO_FORM std::tuple<arma::vec3,arma::vec3,int,arma::vec3,arma::vec3,int>
// NONMTQ_TORQ_SCALE: Scales RW/magic torque commands for better optimization conditioning
// 
// Physical interpretation: RW commands u_rw are in Nm (direct torque), while MTQ
// commands u_mtq are in Am² and produce torque τ = m × B ≈ u_mtq * |B|.
// With |B| ≈ 30μT, MTQ torque per unit ≈ 3e-5 Nm/Am².
// RW torque per unit = 1 Nm/Nm, so RW is ~33,000x more effective per unit control.
//
// Setting NONMTQ_TORQ_SCALE ≈ |B| makes RW and MTQ have similar effectiveness
// in the optimizer's B matrix, improving conditioning.
//
// The optimizer works with scaled controls: u_scaled = u_physical / NONMTQ_TORQ_SCALE
// After optimization, scale back: u_physical = u_scaled * NONMTQ_TORQ_SCALE
//
// Value = 3e-5: Matches typical B-field, RW/MTQ have similar B matrix entries
// Value = 1.0:  No scaling (original behavior)
//
// NOTE: For TVLQR K-gain computation, we use original (unscaled) costs via
// save_original_costs() and use_original_costs() to ensure proper K-gain scaling.
static const double NONMTQ_TORQ_SCALE = 3e-5;  // ≈ typical |B| in Tesla



static const double constraint_hess_mult=1.0;

namespace py = pybind11;
// using namespace arma;

struct cost_jacs{
  arma::vec lx;
  arma::mat lxx;
  arma::mat lux;
  arma::vec lu;
  arma::mat luu;
};

// Struct for extracted cost settings from COST_SETTINGS_FORM tuple
struct ExtractedCostSettings {
    double w_ang, w_av, w_u_mult, w_avmag, w_avang;
    double w_ang_N, w_av_N, w_avmag_N, w_avang_N;
    int whichAngCostFunc;
    int useRawControlCost;
    int useFullCostHess;

    static ExtractedCostSettings fromTuple(const COST_SETTINGS_FORM& settings);
    void applyTerminalWeights();
};

// Struct for RW cost computation results
struct RWCostResults {
    double ang_mom_cost;
    double stiction_cost;
};

class Satellite {
public:
    Satellite();
    Satellite(arma::mat33 Jcom_in);
    ~Satellite();

    void change_Jcom(arma::mat33 Jcom_in);
    void change_Jcom_py(py::array_t<double> Jcom_in_py);
    void readJcom();

    int constraint_N() const;
    int ineq_constraint_N() const;
    int control_N() const;
    int state_N() const;
    int reduced_state_N() const;
    int quat0index() const;
    int redang0index() const;
    int avindex0() const;
    int eq_constraint_N() const;

    std::tuple<double> constraintSettingsTuple();

    void add_gg_torq();
    void remove_gg_torq();
    void add_prop_torq(arma::vec3 prop_torq_in);
    void add_prop_torq_py(py::array_t<double> prop_torq_in_py);
    void remove_prop_torq();
    void add_srp_torq(arma::mat srp_coeff_in,int coeff_len);
    void add_srp_torq_py(py::array_t<double> srp_coeff_in_py,int coeff_len);
    void remove_srp_torq();
    void add_aero_torq(arma::mat drag_coeff_in,int coeff_len);
    void add_aero_torq_py(py::array_t<double> drag_coeff_in_py,int coeff_len);
    void remove_aero_torq();
    void add_resdipole_torq(arma::vec3 rd_in);
    void add_resdipole_torq_py(py::array_t<double> rd_in_py);
    void remove_resdipole_torq();
    void add_gendist_torq(arma::vec3 gd_torq_in);
    void add_gendist_torq_py(py::array_t<double> gd_torq_in_py);
    void remove_gendist_torq();

    void set_AV_constraint(double wmax);
    void clear_AV_constraint();

    void add_sunpoint_constraint(arma::vec3 body_ax, double angle,bool useACOS);
    void add_sunpoint_constraint_py(py::array_t<double> body_ax_py, double angle,bool useACOS);
    void clear_sunpoint_constraints();

    void add_MTQ(arma::vec3 body_ax, double max_moment, double cost);
    void add_MTQ_py(py::array_t<double> body_ax_py, double max_moment, double cost);
    void clear_MTQs();

    void update_invJ_noRW();
    void add_RW(arma::vec3 body_ax,double J, double max_torq, double max_ang_mom, double cost,double AM_cost, double AM_cost_threshold,double stiction_cost, double stiction_threshold);
    void add_RW_py(py::array_t<double> body_ax_py, double J,double max_torq, double max_ang_mom, double cost,double AM_cost,double AM_cost_threshold,double stiction_cost, double stiction_threshold);
    void clear_RWs();

    void add_magic(arma::vec3 body_ax, double max_torq, double cost);
    void add_magic_py(py::array_t<double> body_ax_py, double max_torq, double cost);
    void clear_magics();

    void auto_scale_control_costs(double base_weight = 1.0);
    void save_original_costs();  // Call before auto_scale to save unscaled costs
    void use_original_costs(bool use_orig);  // Switch between scaled/original costs for TVLQR

    double get_nonmtq_torq_scale();

    arma::vec state_norm(arma::vec x) const;
    arma::mat state_norm_jacobian(arma::vec x) const;
    arma::cube state_norm_hessian(arma::vec x) const;

    arma::mat findGMat(arma::vec4 qk) const;

    std::tuple<arma::cube,arma::cube,arma::cube> constraintHessians(int k, int N,arma::vec uk, arma::vec xk,arma::vec3 sunk) const;
    std::tuple<arma::mat,arma::mat> constraintJacobians(int k, int N, arma::vec uk,arma::vec xk,arma::vec3 sunk) const;
    arma::vec getConstraints(int t, int tmax, arma::vec u, arma::vec x, arma::vec3 sunECIvec) const;

    double stepcost_vec(int k, int N, arma::vec xk, arma::vec xkp1, arma::vec uk, arma::vec ukprev, arma::vec3 satvec_k, arma::vec3 ECIvec_k, arma::vec3 BECI_k,COST_SETTINGS_FORM *costSettings_ptr) const;
    double stepcost_quat(int k, int N, arma::vec xk, arma::vec xkp1, arma::vec uk, arma::vec ukprev, arma::vec3 satvec_k, arma::vec4 ECIvec_k, arma::vec3 BECI_k,COST_SETTINGS_FORM *costSettings_ptr) const;

    //std::tuple<arma::vec6, arma::mat66> veccostJacobians(int k, int N, arma::vec xk, arma::vec uk, arma::vec vk, arma::mat QN, arma::mat satvec, arma::mat ECIvec, arma::vec vNslew, std::tuple<int, double, double, double, double> *costSettings_ptr);
    cost_jacs veccostJacobians(int k, int N, arma::vec xk, arma::vec xkp1, arma::vec uk,arma::vec ukprev, arma::vec3 satvec_k, arma::vec3 ECIvec_k, arma::vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const;
    cost_jacs quatcostJacobians(int k, int N, arma::vec xk, arma::vec xkp1, arma::vec uk,arma::vec ukprev, arma::vec3 satvec_k, arma::vec4 ECIvec_k, arma::vec3 BECI_k, COST_SETTINGS_FORM *costSettings_ptr) const;
    cost_jacs costJacobians(int k, int N, arma::vec xk, arma::vec xkp1, arma::vec uk,arma::vec ukprev,arma::vec3 satvec, arma::vec ECIvec, arma::vec3 BECI_k,COST_SETTINGS_FORM *costSettings_ptr) const;

    // Path length cost helper - computes geodesic rotation cost between consecutive quaternions
    // Returns (cost, grad_qk, grad_qkp1)
    // Returns (cost, grad_qk, grad_qkp1, dtheta_dqd) for path length cost
    // dtheta_dqd is needed for proper Gauss-Newton Hessian: H = 2*weight * dtheta * dtheta^T
    std::tuple<double, arma::vec4, arma::vec4, arma::vec4> pathLengthCost(arma::vec4 qk, arma::vec4 qkp1, double weight) const;

    arma::mat getImu(double mu, arma::vec muk, arma::vec ck, arma::vec lamk);
    arma::mat getIlam(double mu, arma::vec muk, arma::vec ck, arma::vec lamk);

    std::tuple<arma::vec,arma::vec3> dynamics(arma::vec x, arma::vec u, const DYNAMICS_INFO_FORM& dynamics_info) const;
    arma::vec3 dist_torque(arma::vec x, const DYNAMICS_INFO_FORM& dynamics_info) const;
    arma::vec dynamics_pure(arma::vec x, arma::vec u, const DYNAMICS_INFO_FORM& dynamics_info) const;
    std::tuple<arma::mat, arma::mat,arma::mat> dynamicsJacobians( arma::vec x,  arma::vec u,  const DYNAMICS_INFO_FORM& dynamics_info) const;
    std::tuple<arma::cube, arma::cube,arma::cube> dynamicsHessians( arma::vec x,  arma::vec u,  const DYNAMICS_INFO_FORM& dynamics_info) const;

    py::tuple py_tuple_out() const;

    //NEXT TO DOS:
    // test with and magic!!
    // move rk4z, etc into here
    // test
    // add spherical harmonics for real
    // test
    // add MTQ axes...-->bdot in OldPlanner
    // add RW,magic to oldPlanner bdot so that they're nonzero at first.
    // add cost object and do it in here.
    // modify how the cost for different traj lengths are calculated (satellite object has fixerd costs for control, ang_mom, stiction. should there be a way to inflate values for the final time step? Or deflate values when using a denser timestep)

    arma::mat33 Jcom;
    arma::mat33 invJcom;
    arma::mat33 invJcom_noRW;
    arma::mat33 Jcom_noRW;
    int plan_for_aero = 0;
    int plan_for_prop = 0;
    int plan_for_srp = 0;
    int plan_for_gg = 0;
    int plan_for_resdipole = 0;
    int plan_for_gendist = 0;

    arma::mat srp_coeff;
    arma::mat drag_coeff;
    int coeff_N;

    arma::vec3 prop_torq = arma::vec3().zeros();
    arma::vec3 gen_dist_torq = arma::vec3().zeros();
    arma::vec3 res_dipole = arma::vec3().zeros();

    bool useAVconstraint = false;
    double AVmax = nan("1");

    int number_sunpoints = 0;
    std::vector<arma::vec3> sunpoint_axes = {};
    std::vector<double> sunpoint_angs = {};
    std::vector<bool> sunpoint_useACOSs = {};

    int number_MTQ = 0;
    int number_RW = 0;
    int number_magic = 0;

    std::vector<arma::vec3> MTQ_axes = {};
    std::vector<arma::vec3> RW_axes = {};
    std::vector<arma::vec3> magic_axes = {};

    arma::mat mtq_ax_mat = arma::mat(3,0).zeros();
    arma::mat rw_ax_mat = arma::mat(3,0).zeros();
    arma::mat magic_ax_mat = arma::mat(3,0).zeros();

    std::vector<double> MTQ_cost = {};
    std::vector<double> RW_cost = {};
    std::vector<double> RW_AM_cost = {};
    std::vector<double> RW_AM_cost_threshold = {};
    std::vector<double> RW_stiction_cost = {};
    std::vector<double> RW_stiction_threshold = {};
    std::vector<double> magic_cost = {};
    
    // Original (unscaled) costs for TVLQR - stored before auto_scale_control_costs
    std::vector<double> MTQ_cost_orig = {};
    std::vector<double> RW_cost_orig = {};
    std::vector<double> magic_cost_orig = {};
    bool use_original_costs_for_tvlqr = false;  // Flag to use unscaled costs

    std::vector<double> RW_J = {};

    std::vector<double> MTQ_max = {};
    std::vector<double> magic_max_torq = {};
    std::vector<double> RW_max_torq = {};
    std::vector<double> RW_max_ang_mom = {};

private:
    // Helper functions for cost Jacobian computation
    arma::mat setupActuationCostMatrix(int k, int N, ExtractedCostSettings& settings) const;
    RWCostResults computeRWCostsAndJacobians(
        const arma::vec& xk,
        arma::vec& lkx,
        arma::mat& lkxx
    ) const;
};

#endif // TPR_SATELLITE_HPP
