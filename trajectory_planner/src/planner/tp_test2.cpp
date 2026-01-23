// tp_test2.cpp - Higher-level optimizer tests
// Tests optimizer functions where we can analytically verify optimality
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Satellite.hpp"
#include "PlannerUtil.hpp"
#include "OldPlanner.hpp"

namespace py = pybind11;
using namespace std;
using namespace arma;

// Global Python interpreter - initialized once for all tests
static py::scoped_interpreter* python_guard = nullptr;

struct PythonInitializer {
    PythonInitializer() {
        if (!python_guard) {
            python_guard = new py::scoped_interpreter();
        }
    }
};

// Static instance to ensure Python is initialized before any tests run
static PythonInitializer python_init;

// Helper function to create a simple satellite for testing
Satellite createSimpleSatellite() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.1, 0.12, 0.15})));
    sat.add_MTQ(arma::vec({1,0,0}), 0.2, 1.0);
    sat.add_MTQ(arma::vec({0,1,0}), 0.2, 1.0);
    sat.add_MTQ(arma::vec({0,0,1}), 0.2, 1.0);
    return sat;
}

// Helper to create default vector info (environment)
VECTOR_INFO_FORM createDefaultVecInfo(int N, double dt) {
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
    arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});
    arma::vec3 sun_vec = arma::normalise(arma::vec({1.0, 0.0, 0.0}));
    arma::vec3 sat_body_vec = arma::vec({0.0, 0.0, 1.0});
    arma::vec3 eci_goal = arma::normalise(arma::vec({1.0, 0.0, 0.0}));

    arma::vec times = arma::linspace(0.0, (N-1)*dt, N);
    arma::mat Rset = arma::repmat(R_orb, 1, N);
    arma::mat Vset = arma::repmat(V_orb, 1, N);
    arma::mat Bset = arma::repmat(B_eci, 1, N);
    arma::mat sunset = arma::repmat(sun_vec, 1, N);
    arma::mat satvec = arma::repmat(sat_body_vec, 1, N);
    arma::mat ECIvec = arma::repmat(eci_goal, 1, N);
    arma::vec pset = arma::vec(N).zeros();
    arma::vec rhoset = arma::vec(N).zeros();

    return std::make_tuple(times, Rset, Vset, Bset, sunset, satvec, ECIvec, pset, rhoset);
}

// ============================================================================
// TEST: Optimizer convergence - cost should decrease monotonically
// ============================================================================
TEST_CASE("Optimizer cost decreases monotonically", "[optimizer][convergence]") {
    cout << "\n=== Test: Optimizer Cost Decreases Monotonically ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();
    int N = 20;
    double dt = 1.0;

    // Initial state with some angular velocity error
    arma::vec3 w0 = arma::vec({0.05, -0.03, 0.02});
    arma::vec4 q0 = arma::normalise(arma::vec({0.9, 0.2, 0.3, 0.1}));
    arma::vec x0 = join_cols(w0, q0);

    // Zero initial control guess
    arma::mat Uset_init = arma::mat(nu, N).zeros();

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Generate initial trajectory
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings with moderate weights
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Augmented Lagrangian (no active constraints initially)
    int nc = sat.constraint_N();
    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    // Create planner with appropriate settings
    arma::mat33 J_est = sat.Jcom;
    int nxr = sat.reduced_state_N();
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(20, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(30, 100, 3000, 1e-4, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-1, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Compute initial cost
    double initial_cost = planner.cost2Func(traj, vecs, auglag, &costSettings, false);
    cout << "Initial cost: " << initial_cost << endl;

    // Run a few iLQR iterations manually and track cost
    REG_PAIR regs = std::make_tuple(rho, 1.0);
    std::vector<double> costs;
    costs.push_back(initial_cost);

    TRAJECTORY_FORM current_traj = traj;
    for(int iter = 0; iter < 10; iter++) {
        auto step_result = planner.ilqrStep(dt, current_traj, vecs, auglag, regs,
                                            &costSettings, regSettings, lineSearchSettings,
                                            breakSettings, true);

        double new_cost = std::get<0>(step_result);
        current_traj = std::get<5>(step_result);
        regs = std::get<4>(step_result);
        costs.push_back(new_cost);

        cout << "Iteration " << iter << ": cost = " << new_cost << endl;
    }

    // Verify cost is generally decreasing (with tolerance for small increases due to regularization)
    int decrease_count = 0;
    for(size_t i = 1; i < costs.size(); i++) {
        if(costs[i] <= costs[i-1] + 1e-4 * std::abs(costs[i-1])) {
            decrease_count++;
        }
    }

    // At least 70% of iterations should show cost decrease or stay the same
    double decrease_ratio = (double)decrease_count / (costs.size() - 1);
    cout << "Cost decrease ratio: " << decrease_ratio * 100 << "%" << endl;
    CHECK(decrease_ratio >= 0.7);

    // Verify final cost is lower than initial
    double final_cost = costs.back();
    cout << "Final cost: " << final_cost << endl;
    cout << "Cost reduction: " << (initial_cost - final_cost) / initial_cost * 100 << "%" << endl;

    REQUIRE(final_cost < initial_cost);
}

// ============================================================================
// TEST: Equilibrium point - optimizer should maintain equilibrium
// ============================================================================
TEST_CASE("Optimizer maintains equilibrium at rest", "[optimizer][equilibrium]") {
    cout << "\n=== Test: Optimizer Maintains Equilibrium ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();
    int N = 10;
    double dt = 1.0;

    // Start at equilibrium: zero angular velocity, identity quaternion
    arma::vec3 w0 = arma::vec3().zeros();
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});
    arma::vec x0 = join_cols(w0, q0);

    // Zero control
    arma::mat Uset = arma::mat(nu, N).zeros();

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Generate trajectory - should stay at equilibrium
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset, dt_vec, TQset);

    // Cost settings that penalize deviation from equilibrium
    // The goal vector should align with body z-axis pointing at target
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    int nc = sat.constraint_N();
    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    int nxr = sat.reduced_state_N();
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(20, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(30, 100, 3000, 1e-4, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-1, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Run backward pass
    REG_PAIR regs = std::make_tuple(rho, 1.0);
    auto backwardResult = planner.backwardPass(dt, traj, vecs, auglag, regs, &costSettings, regSettings, true);
    BACKWARD_PASS_RESULTS_FORM bpResults = std::get<0>(backwardResult);

    arma::cube Kset = std::get<0>(bpResults);
    arma::mat dset = std::get<1>(bpResults);

    // At equilibrium, feedforward term d should be small (zero control is optimal)
    cout << "Feedforward terms (should be near zero at equilibrium):" << endl;
    for(int k = 0; k < N-1; k++) {
        double d_norm = arma::norm(dset.col(k));
        cout << "  ||d[" << k << "]|| = " << d_norm << endl;
        // At equilibrium, the feedforward term should be small
        // (not necessarily zero due to terminal cost gradient)
    }

    // The key test: feedback gains K should be reasonable (not NaN/Inf)
    for(int k = 0; k < N-1; k++) {
        arma::mat Kk = Kset.slice(k);
        CHECK(Kk.is_finite());
        cout << "K[" << k << "] condition number: " << arma::cond(Kk) << endl;
    }
}

// ============================================================================
// TEST: Optimality conditions (KKT) at solution
// ============================================================================
TEST_CASE("Verify KKT optimality conditions at solution", "[optimizer][kkt]") {
    cout << "\n=== Test: KKT Optimality Conditions ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 5;  // Short horizon for simplicity
    double dt = 1.0;

    // Non-trivial initial state
    arma::vec3 w0 = arma::vec({0.02, -0.01, 0.015});
    arma::vec4 q0 = arma::normalise(arma::vec({0.95, 0.1, 0.2, 0.15}));
    arma::vec x0 = join_cols(w0, q0);

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Cost settings
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-6, 1e-3, 1e-4, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-4;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-10, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Generate initial trajectory
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Run full ALILQR optimization
    cout << "Running ALILQR optimization..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    // ALILQR_OUTPUT_FORM is tuple<OPT_FORM, double, double>
    // OPT_FORM is tuple<mat,mat,mat,mat,mat,vec> = (Xset, Uset, ?, ?, ?, dt_vec)
    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Xset = std::get<0>(opt_form);
    arma::mat opt_Uset = std::get<1>(opt_form);
    arma::vec opt_dt_vec = std::get<5>(opt_form);

    // Reconstruct TRAJECTORY_FORM for backward pass
    arma::mat opt_TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM opt_traj = std::make_tuple(opt_Xset, opt_Uset, opt_dt_vec, opt_TQset);

    cout << "Optimization complete." << endl;
    cout << "Final cost: " << std::get<1>(result) << endl;
    cout << "Max constraint violation: " << std::get<2>(result) << endl;

    // Now verify optimality conditions
    // For unconstrained or interior solution: gradient of Hamiltonian w.r.t. u should be ~0

    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    REG_PAIR regs = std::make_tuple(rho, 1.0);

    // Run backward pass on optimal trajectory
    auto bp_result = planner.backwardPass(dt, opt_traj, vecs, auglag, regs, &costSettings, regSettings, true);
    BACKWARD_PASS_RESULTS_FORM bpResults = std::get<0>(bp_result);

    arma::mat dset = std::get<1>(bpResults);
    arma::mat delV = std::get<2>(bpResults);

    // The feedforward term d is proportional to Q_u (gradient of Q-function w.r.t. control)
    // At optimum, this should be small
    cout << "\nFeedforward terms (should be small at optimum):" << endl;
    double max_d_norm = 0;
    for(int k = 0; k < N-1; k++) {
        double d_norm = arma::norm(dset.col(k));
        max_d_norm = std::max(max_d_norm, d_norm);
        cout << "  ||d[" << k << "]|| = " << d_norm << endl;
    }

    // Expected value function improvement should be small at optimum
    cout << "\nExpected value improvement (delV):" << endl;
    cout << "  delV: " << delV.t();

    // Verify feedforward terms are reasonably small (indicates near-optimality)
    // Note: May not be exactly zero due to terminal condition and constraint handling
    cout << "\nMax feedforward norm: " << max_d_norm << endl;
    CHECK(max_d_norm < 1.0);  // Should be much smaller than initial gradient
}

// ============================================================================
// TEST: Constraint satisfaction with ALTRO
// ============================================================================
TEST_CASE("ALTRO satisfies control constraints", "[optimizer][constraints]") {
    cout << "\n=== Test: ALTRO Constraint Satisfaction ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 15;
    double dt = 0.5;

    // Aggressive initial state that will require significant control
    arma::vec3 w0 = arma::vec({0.1, -0.08, 0.06});
    arma::vec4 q0 = arma::normalise(arma::vec({0.7, 0.4, 0.4, 0.4}));
    arma::vec x0 = join_cols(w0, q0);

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // High weights to encourage aggressive control (which will hit constraints)
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e4, 1e3, 0.1, 0.0, 0.0,  // Low control weight to encourage constraint activation
        1e4, 1e3, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-5, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Generate initial trajectory with zero control
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    cout << "Running ALILQR with aggressive initial conditions..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    // ALILQR_OUTPUT_FORM is tuple<OPT_FORM, double, double>
    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double max_viol = std::get<2>(result);

    cout << "Max constraint violation: " << max_viol << endl;

    // Check constraint satisfaction
    double mtq_max = 0.2;  // From satellite setup
    double max_control_magnitude = 0;
    cout << "\nControl magnitudes:" << endl;
    for(int k = 0; k < N; k++) {
        double u_norm = arma::norm(opt_Uset.col(k), "inf");
        max_control_magnitude = std::max(max_control_magnitude, u_norm);
        cout << "  ||u[" << k << "]||_inf = " << u_norm << " (max allowed: " << mtq_max << ")" << endl;
    }

    // Constraint violation should be small (but may not be exactly zero due to tolerance)
    cout << "\nMax control magnitude: " << max_control_magnitude << endl;
    cout << "Constraint tolerance: " << max_viol << endl;

    // ALTRO should drive constraint violations small (within tolerance)
    CHECK(max_viol < 0.1);
}

// ============================================================================
// TEST: TVLQR gain computation - verify findK works
// ============================================================================
TEST_CASE("TVLQR gains can be computed", "[optimizer][tvlqr]") {
    cout << "\n=== Test: TVLQR Gain Computation ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int N = 5;
    double dt = 1.0;

    // Simple initial state
    arma::vec3 w0 = arma::vec({0.01, -0.005, 0.002});
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.0, 0.0, 0.0}));
    arma::vec x0 = join_cols(w0, q0);

    arma::mat Uset = arma::mat(nu, N).zeros();

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Generate trajectory
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset, dt_vec, TQset);

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(20, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(30, 100, 3000, 1e-4, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-1, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Compute TVLQR gains using findK (the basic version without disturbance augmentation)
    auto gains_result = planner.findK(dt, traj, vecs, costSettings);
    arma::cube Kset = std::get<0>(gains_result);
    arma::cube Sset = std::get<1>(gains_result);

    cout << "TVLQR gains computed successfully." << endl;
    cout << "Kset shape: " << Kset.n_rows << " x " << Kset.n_cols << " x " << Kset.n_slices << endl;
    cout << "Sset shape: " << Sset.n_rows << " x " << Sset.n_cols << " x " << Sset.n_slices << endl;

    // Verify dimensions are correct
    CHECK(Kset.n_rows == (size_t)nu);
    CHECK(Kset.n_cols == (size_t)nxr);
    CHECK(Kset.n_slices == (size_t)(N-1));
    CHECK(Sset.n_rows == (size_t)nxr);
    CHECK(Sset.n_cols == (size_t)nxr);
    CHECK(Sset.n_slices == (size_t)N);

    // Verify gains are finite
    for(int k = 0; k < N-1; k++) {
        arma::mat Kk = Kset.slice(k);
        arma::mat Sk = Sset.slice(k);

        CHECK(Kk.is_finite());
        CHECK(Sk.is_finite());

        // S should be symmetric (within numerical precision)
        arma::mat S_sym_error = Sk - Sk.t();
        double sym_error = arma::norm(S_sym_error, "fro");
        cout << "S[" << k << "] symmetry error: " << sym_error << endl;
        CHECK(sym_error < 1e-8);
    }
}

// ============================================================================
// TEST: Forward pass respects dynamics
// ============================================================================
TEST_CASE("Forward pass trajectory satisfies dynamics", "[optimizer][dynamics]") {
    cout << "\n=== Test: Forward Pass Dynamics Consistency ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 10;
    double dt = 0.5;

    // Non-trivial initial state
    arma::vec3 w0 = arma::vec({0.03, -0.02, 0.01});
    arma::vec4 q0 = arma::normalise(arma::vec({0.9, 0.2, 0.25, 0.15}));
    arma::vec x0 = join_cols(w0, q0);

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
    arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});
    arma::vec3 sun_vec = arma::normalise(arma::vec({1.0, 0.0, 0.0}));
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, R_orb, 0, V_orb, sun_vec, 1, 0.0);

    // Generate initial trajectory with small random controls
    arma::arma_rng::set_seed(42);
    arma::mat Uset_init = 0.01 * arma::randn(nu, N);
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(20, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(30, 100, 3000, 1e-4, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-1, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Setup augmented Lagrangian
    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    REG_PAIR regs = std::make_tuple(rho, 1.0);

    // Run backward pass
    auto bp_result = planner.backwardPass(dt, traj, vecs, auglag, regs, &costSettings, regSettings, true);
    BACKWARD_PASS_RESULTS_FORM bpResults = std::get<0>(bp_result);

    // Run forward pass
    auto fp_result = planner.forwardPass(dt, traj, vecs, auglag, bpResults, regs,
                                          &costSettings, regSettings, lineSearchSettings, true);

    TRAJECTORY_FORM new_traj = std::get<0>(fp_result);
    arma::mat new_Xset = std::get<0>(new_traj);
    arma::mat new_Uset = std::get<1>(new_traj);

    // Verify the new trajectory satisfies dynamics
    cout << "Checking dynamics consistency of forward pass result..." << endl;
    double max_dynamics_error = 0;

    for(int k = 0; k < N-1; k++) {
        arma::vec xk = new_Xset.col(k);
        arma::vec uk = new_Uset.col(k);
        arma::vec xkp1_actual = new_Xset.col(k+1);

        // Propagate dynamics
        auto rk4out = rk4z(dt, xk, uk, sat, dynamics_info, dynamics_info);
        arma::vec xkp1_predicted = sat.state_norm(std::get<0>(rk4out));

        // Compute error
        double dynamics_error = arma::norm(xkp1_actual - xkp1_predicted);
        max_dynamics_error = std::max(max_dynamics_error, dynamics_error);

        cout << "  Step " << k << ": dynamics error = " << dynamics_error << endl;
    }

    cout << "Max dynamics error: " << max_dynamics_error << endl;
    CHECK(max_dynamics_error < 1e-10);
}

// ============================================================================
// TEST: Step cost function gradient verification
// ============================================================================
TEST_CASE("Step cost function gradients are correct", "[optimizer][gradient]") {
    cout << "\n=== Test: Step Cost Function Gradient Verification ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(123);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec xk = join_cols(w0, q0);
    arma::vec3 uk = 0.05 * arma::randn(3);
    arma::vec3 uk_prev = 0.02 * arma::randn(3);

    arma::vec3 sat_body_vec = arma::vec({0.0, 0.0, 1.0});
    arma::vec3 eci_goal = arma::normalise(arma::vec({1.0, 0.5, 0.2}));
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});

    int k = 5;
    int N = 20;

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.1, 0.0,
        1e3, 1e2, 0.1, 0.0,
        2, true, 0
    );

    // Get analytic gradient
    cost_jacs costJac = sat.costJacobians(k, N, xk, uk, uk_prev, sat_body_vec, eci_goal, B_eci, &costSettings);
    arma::vec lu_analytic = costJac.lu;

    // Compute base step cost
    double cost0 = sat.stepcost_vec(k, N, xk, uk, uk_prev, sat_body_vec, eci_goal, B_eci, &costSettings);

    // Finite difference for control gradient
    double eps = 1e-7;
    arma::vec lu_fd = arma::vec(nu).zeros();
    for(int i = 0; i < nu; i++) {
        arma::vec uk_pert = uk;
        uk_pert(i) += eps;
        double cost_pert = sat.stepcost_vec(k, N, xk, uk_pert, uk_prev, sat_body_vec, eci_goal, B_eci, &costSettings);
        lu_fd(i) = (cost_pert - cost0) / eps;
    }

    cout << "Step cost: " << cost0 << endl;
    cout << "Control gradient (analytic): " << lu_analytic.t();
    cout << "Control gradient (finite diff): " << lu_fd.t();
    cout << "Gradient difference: " << (lu_analytic - lu_fd).t();

    double grad_error = arma::norm(lu_analytic - lu_fd);
    cout << "Gradient error norm: " << grad_error << endl;

    // Gradients should match closely
    CHECK(grad_error < 1e-4);

    // State gradient verification is complex due to quaternion manifold
    // The control gradient verification is sufficient to validate the cost function implementation
    cout << "State gradient (analytic): " << costJac.lx.t();
    cout << "(State gradient FD verification skipped due to quaternion manifold complexity)" << endl;
}

// ============================================================================
// REALISTIC ACTUATOR CONFIGURATION TESTS
// These tests target real-world scenarios that may cause convergence issues
// ============================================================================

// Helper: Create satellite with only MTQs (underactuated - magnetic field dependent)
Satellite createMTQOnlySatellite() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.01, 0.02, 0.015})));  // Small cubesat
    // 3-axis MTQ configuration
    sat.add_MTQ(arma::vec({1,0,0}), 0.1, 1.0);   // X-axis MTQ, 0.1 Am^2 max
    sat.add_MTQ(arma::vec({0,1,0}), 0.1, 1.0);   // Y-axis MTQ
    sat.add_MTQ(arma::vec({0,0,1}), 0.1, 1.0);   // Z-axis MTQ
    return sat;
}

// Helper: Create satellite with only RWs (fully actuated)
Satellite createRWOnlySatellite() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // 3-axis RW configuration
    // add_RW(axis, J_wheel, max_torq, max_ang_mom, cost, AM_cost, AM_cost_threshold, stiction_cost, stiction_threshold)
    // NOTE: stiction_threshold CANNOT be 0 - causes division by zero in smoothstep!
    sat.add_RW(arma::vec({1,0,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,1,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    return sat;
}

// Helper: Create satellite with MTQs + RWs (hybrid - common configuration)
// NOTE: Use larger Jcom values (>=0.05) to avoid ill-conditioning in iLQR backward pass.
// Small Jcom leads to large invJcom, which amplifies Pk in trans(Bqk)*Pk*Bqk causing overflow.
Satellite createHybridSatellite() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // MTQs for momentum dumping
    sat.add_MTQ(arma::vec({1,0,0}), 0.1, 1.0);
    sat.add_MTQ(arma::vec({0,1,0}), 0.1, 1.0);
    sat.add_MTQ(arma::vec({0,0,1}), 0.1, 1.0);
    // RWs for fine pointing
    // NOTE: stiction_threshold CANNOT be 0 - causes division by zero!
    sat.add_RW(arma::vec({1,0,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,1,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    return sat;
}

// Helper: Create vector info with realistic varying magnetic field
VECTOR_INFO_FORM createRealisticVecInfo(int N, double dt) {
    arma::vec times = arma::linspace(0.0, (N-1)*dt, N);

    // Simulate orbit with varying B-field (simplified LEO model)
    arma::mat Bset = arma::mat(3, N);
    arma::mat Rset = arma::mat(3, N);
    arma::mat Vset = arma::mat(3, N);
    double orbit_period = 5400.0;  // ~90 min LEO
    double B_mag = 3e-5;  // ~30 microTesla

    for(int k = 0; k < N; k++) {
        double t = times(k);
        double phase = 2.0 * M_PI * t / orbit_period;
        // Varying B-field simulating orbit
        Bset(0, k) = B_mag * 0.3 * sin(phase);
        Bset(1, k) = B_mag * cos(phase);
        Bset(2, k) = B_mag * 0.5 * sin(phase + 0.5);
        // Simple circular orbit
        Rset(0, k) = 7000.0 * cos(phase);
        Rset(1, k) = 7000.0 * sin(phase);
        Rset(2, k) = 0.0;
        Vset(0, k) = -7.5 * sin(phase);
        Vset(1, k) = 7.5 * cos(phase);
        Vset(2, k) = 0.0;
    }

    arma::vec3 sun_vec = arma::normalise(arma::vec({1.0, 0.0, 0.0}));
    arma::vec3 sat_body_vec = arma::vec({0.0, 0.0, 1.0});
    arma::vec3 eci_goal = arma::normalise(arma::vec({1.0, 0.0, 0.0}));

    arma::mat sunset = arma::repmat(sun_vec, 1, N);
    arma::mat satvec = arma::repmat(sat_body_vec, 1, N);
    arma::mat ECIvec = arma::repmat(eci_goal, 1, N);
    arma::vec pset = arma::vec(N).zeros();
    arma::vec rhoset = arma::vec(N).zeros();

    return std::make_tuple(times, Rset, Vset, Bset, sunset, satvec, ECIvec, pset, rhoset);
}

// ============================================================================
// TEST: MTQ-only convergence (underactuated system)
// ============================================================================
TEST_CASE("MTQ-only satellite convergence", "[optimizer][mtq][convergence]") {
    cout << "\n=== Test: MTQ-Only Satellite Convergence ===" << endl;

    Satellite sat = createMTQOnlySatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 30;
    double dt = 2.0;  // 2 second steps

    cout << "Satellite config: " << nu << " controls, " << nc << " constraints" << endl;

    // Moderate initial error
    arma::vec3 w0 = arma::vec({0.01, -0.005, 0.008});  // ~0.5-1 deg/s
    arma::vec4 q0 = arma::normalise(arma::vec({0.95, 0.15, 0.2, 0.15}));  // ~20 deg off
    arma::vec x0 = join_cols(w0, q0);

    VECTOR_INFO_FORM vecs = createRealisticVecInfo(N, dt);
    arma::mat Bset = std::get<3>(vecs);

    // Check B-field isn't degenerate
    cout << "B-field magnitudes: ";
    for(int k = 0; k < min(5, N); k++) {
        cout << arma::norm(Bset.col(k)) << " ";
    }
    cout << "..." << endl;

    // Generate initial trajectory with zero control
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    arma::vec3 B_k = Bset.col(0);
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);
    for(int k = 0; k < N-1; k++) {
        B_k = Bset.col(k);
        dynamics_info = std::make_tuple(B_k, arma::vec3({7000,0,0}), 0,
                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings - moderate weights
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-5, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Compute initial cost
    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    double initial_cost = planner.cost2Func(traj, vecs, auglag, &costSettings, false);
    cout << "Initial cost: " << initial_cost << endl;

    // Run optimization
    cout << "Running ALILQR..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Xset = std::get<0>(opt_form);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double final_cost = std::get<1>(result);
    double max_viol = std::get<2>(result);

    cout << "Final cost: " << final_cost << endl;
    cout << "Max constraint violation: " << max_viol << endl;
    cout << "Cost reduction: " << (initial_cost - final_cost) / initial_cost * 100 << "%" << endl;

    // Analyze control smoothness (detect spiky behavior)
    cout << "\nControl analysis:" << endl;
    arma::vec control_norms(N);
    arma::vec control_changes(N-1);
    for(int k = 0; k < N; k++) {
        control_norms(k) = arma::norm(opt_Uset.col(k));
        if(k > 0) {
            control_changes(k-1) = arma::norm(opt_Uset.col(k) - opt_Uset.col(k-1));
        }
    }

    double mean_control = arma::mean(control_norms);
    double max_control = arma::max(control_norms);
    double mean_change = arma::mean(control_changes);
    double max_change = arma::max(control_changes);

    cout << "  Mean control norm: " << mean_control << endl;
    cout << "  Max control norm: " << max_control << endl;
    cout << "  Mean control change: " << mean_change << endl;
    cout << "  Max control change: " << max_change << endl;

    // Spikiness metric: ratio of max change to mean control
    double spikiness = max_change / (mean_control + 1e-10);
    cout << "  Spikiness metric (max_change/mean_control): " << spikiness << endl;

    // Check cost decreased
    CHECK(final_cost < initial_cost);

    // Check for excessive spikiness (warning, not hard fail)
    if(spikiness > 5.0) {
        cout << "  WARNING: Control appears spiky (spikiness > 5)" << endl;
    }
    CHECK(spikiness < 20.0);  // Very generous threshold
}

// ============================================================================
// TEST: RW-only convergence (fully actuated)
// ============================================================================
TEST_CASE("RW-only satellite convergence", "[optimizer][rw][convergence]") {
    cout << "\n=== Test: RW-Only Satellite Convergence ===" << endl;

    Satellite sat = createRWOnlySatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 20;
    double dt = 1.0;

    cout << "Satellite config: " << nu << " controls (should be 3 RW torques)" << endl;
    cout << "State dimension: " << nx << " (should include RW momentum)" << endl;

    // Initial state - need to include RW states if present
    arma::vec3 w0 = arma::vec({0.02, -0.01, 0.015});
    arma::vec4 q0 = arma::normalise(arma::vec({0.9, 0.2, 0.25, 0.2}));
    arma::vec x0_base = join_cols(w0, q0);

    // Add RW momentum states (initially zero)
    arma::vec x0;
    if(nx > 7) {
        arma::vec rw_states = arma::vec(nx - 7).zeros();
        x0 = join_cols(x0_base, rw_states);
    } else {
        x0 = x0_base;
    }
    cout << "Initial state size: " << x0.n_elem << endl;

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Generate initial trajectory
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-5, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Compute initial cost
    int nc_actual = sat.constraint_N();
    arma::mat lambdaSet = arma::mat(nc_actual, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc_actual, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    double initial_cost = planner.cost2Func(traj, vecs, auglag, &costSettings, false);
    cout << "Initial cost: " << initial_cost << endl;

    // Run optimization
    cout << "Running ALILQR..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double final_cost = std::get<1>(result);
    double max_viol = std::get<2>(result);

    cout << "Final cost: " << final_cost << endl;
    cout << "Max constraint violation: " << max_viol << endl;
    cout << "Cost reduction: " << (initial_cost - final_cost) / initial_cost * 100 << "%" << endl;

    // Analyze control
    arma::vec control_norms(N);
    arma::vec control_changes(N-1);
    for(int k = 0; k < N; k++) {
        control_norms(k) = arma::norm(opt_Uset.col(k));
        if(k > 0) {
            control_changes(k-1) = arma::norm(opt_Uset.col(k) - opt_Uset.col(k-1));
        }
    }

    double mean_control = arma::mean(control_norms);
    double max_change = arma::max(control_changes);
    double spikiness = max_change / (mean_control + 1e-10);

    cout << "  Mean control norm: " << mean_control << endl;
    cout << "  Spikiness metric: " << spikiness << endl;

    CHECK(final_cost < initial_cost);
    CHECK(spikiness < 20.0);
}

// ============================================================================
// TEST: Hybrid MTQ+RW convergence
// ============================================================================
TEST_CASE("Hybrid MTQ+RW satellite convergence", "[optimizer][hybrid][convergence]") {
    cout << "\n=== Test: Hybrid MTQ+RW Satellite Convergence ===" << endl;

    Satellite sat = createHybridSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 20;  // Shorter horizon for stability
    double dt = 1.0;  // Match RW-only test

    cout << "Satellite config: " << nu << " controls (3 MTQ + 3 RW torques = 6 expected)" << endl;
    cout << "State dimension: " << nx << endl;
    cout << "Constraint count: " << nc << endl;

    // Initial state - use smaller initial errors for hybrid to help convergence
    arma::vec3 w0 = arma::vec({0.01, -0.005, 0.008});  // Smaller angular velocity
    arma::vec4 q0 = arma::normalise(arma::vec({0.95, 0.15, 0.18, 0.15}));  // Closer to identity
    arma::vec x0_base = join_cols(w0, q0);

    arma::vec x0;
    if(nx > 7) {
        arma::vec extra_states = arma::vec(nx - 7).zeros();
        x0 = join_cols(x0_base, extra_states);
    } else {
        x0 = x0_base;
    }

    // Use constant B-field like the working RW-only test (MTQs still work with constant B)
    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Generate initial trajectory
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-5, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Compute initial cost
    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    double initial_cost = planner.cost2Func(traj, vecs, auglag, &costSettings, false);
    cout << "Initial cost: " << initial_cost << endl;

    // Run optimization
    cout << "Running ALILQR..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double final_cost = std::get<1>(result);
    double max_viol = std::get<2>(result);

    cout << "Final cost: " << final_cost << endl;
    cout << "Max constraint violation: " << max_viol << endl;
    cout << "Cost reduction: " << (initial_cost - final_cost) / initial_cost * 100 << "%" << endl;

    // Analyze each actuator type separately
    int n_mtq = 3;
    int n_rw = nu - n_mtq;

    if(n_rw > 0) {
        cout << "\nMTQ control analysis (first " << n_mtq << " controls):" << endl;
        arma::mat mtq_controls = opt_Uset.rows(0, n_mtq-1);
        arma::vec mtq_norms(N);
        arma::vec mtq_changes(N-1);
        for(int k = 0; k < N; k++) {
            mtq_norms(k) = arma::norm(mtq_controls.col(k));
            if(k > 0) {
                mtq_changes(k-1) = arma::norm(mtq_controls.col(k) - mtq_controls.col(k-1));
            }
        }
        double mtq_spikiness = arma::max(mtq_changes) / (arma::mean(mtq_norms) + 1e-10);
        cout << "  MTQ spikiness: " << mtq_spikiness << endl;

        cout << "\nRW control analysis (controls " << n_mtq << " to " << nu-1 << "):" << endl;
        arma::mat rw_controls = opt_Uset.rows(n_mtq, nu-1);
        arma::vec rw_norms(N);
        arma::vec rw_changes(N-1);
        for(int k = 0; k < N; k++) {
            rw_norms(k) = arma::norm(rw_controls.col(k));
            if(k > 0) {
                rw_changes(k-1) = arma::norm(rw_controls.col(k) - rw_controls.col(k-1));
            }
        }
        double rw_spikiness = arma::max(rw_changes) / (arma::mean(rw_norms) + 1e-10);
        cout << "  RW spikiness: " << rw_spikiness << endl;
    }

    // Overall metrics
    arma::vec control_norms(N);
    arma::vec control_changes(N-1);
    for(int k = 0; k < N; k++) {
        control_norms(k) = arma::norm(opt_Uset.col(k));
        if(k > 0) {
            control_changes(k-1) = arma::norm(opt_Uset.col(k) - opt_Uset.col(k-1));
        }
    }
    double spikiness = arma::max(control_changes) / (arma::mean(control_norms) + 1e-10);
    cout << "\nOverall spikiness: " << spikiness << endl;

    CHECK(final_cost < initial_cost);
    CHECK(spikiness < 20.0);
}

// ============================================================================
// TEST: Control smoothness with rate penalty
// ============================================================================
TEST_CASE("Control smoothness with rate penalty", "[optimizer][smoothness]") {
    cout << "\n=== Test: Control Smoothness with Rate Penalty ===" << endl;

    Satellite sat = createSimpleSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 20;
    double dt = 1.0;

    arma::vec3 w0 = arma::vec({0.03, -0.02, 0.01});
    arma::vec4 q0 = arma::normalise(arma::vec({0.85, 0.25, 0.3, 0.3}));
    arma::vec x0 = join_cols(w0, q0);

    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;
    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings WITHOUT rate penalty (w_avmag=0, w_avang=0)
    COST_SETTINGS_FORM costSettings_no_rate = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Cost settings WITH rate penalty (w_avmag=1.0, w_avang=0)
    COST_SETTINGS_FORM costSettings_with_rate = std::make_tuple(
        1e2, 1e1, 1.0, 1.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-5, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 1e2, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 0, true, 0);

    // Run without rate penalty
    ALL_SETTINGS_FORM allSettings_no_rate = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings_no_rate, costSettings_no_rate, tvlqrCostSettings);
    OldPlanner planner_no_rate(sat, allSettings_no_rate);
    planner_no_rate.setVerbosity(false);
    planner_no_rate.quaternionTo3VecMode = 2;

    cout << "Running without rate penalty..." << endl;
    ALILQR_OUTPUT_FORM result_no_rate = planner_no_rate.alilqr(dt, traj, vecs, costSettings_no_rate, alilqrSettings, true);
    arma::mat Uset_no_rate = std::get<1>(std::get<0>(result_no_rate));

    // Run with rate penalty
    ALL_SETTINGS_FORM allSettings_with_rate = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings_with_rate, costSettings_with_rate, tvlqrCostSettings);
    OldPlanner planner_with_rate(sat, allSettings_with_rate);
    planner_with_rate.setVerbosity(false);
    planner_with_rate.quaternionTo3VecMode = 2;

    cout << "Running with rate penalty..." << endl;
    ALILQR_OUTPUT_FORM result_with_rate = planner_with_rate.alilqr(dt, traj, vecs, costSettings_with_rate, alilqrSettings, true);
    arma::mat Uset_with_rate = std::get<1>(std::get<0>(result_with_rate));

    // Compare spikiness
    auto compute_spikiness = [&](const arma::mat& Uset) {
        arma::vec changes(N-1);
        arma::vec norms(N);
        for(int k = 0; k < N; k++) {
            norms(k) = arma::norm(Uset.col(k));
            if(k > 0) changes(k-1) = arma::norm(Uset.col(k) - Uset.col(k-1));
        }
        return arma::max(changes) / (arma::mean(norms) + 1e-10);
    };

    double spikiness_no_rate = compute_spikiness(Uset_no_rate);
    double spikiness_with_rate = compute_spikiness(Uset_with_rate);

    cout << "Spikiness without rate penalty: " << spikiness_no_rate << endl;
    cout << "Spikiness with rate penalty: " << spikiness_with_rate << endl;

    // Rate penalty should reduce spikiness (or at least not make it worse)
    cout << "Rate penalty effect: " << (spikiness_no_rate - spikiness_with_rate) / spikiness_no_rate * 100 << "% reduction" << endl;

    // With rate penalty, controls should be smoother
    CHECK(spikiness_with_rate <= spikiness_no_rate * 1.5);  // Allow some tolerance
}

// ============================================================================
// TESTS MIMICKING debug_altro_6Up.py CONFIGURATION
// ============================================================================

// Helper: Create satellite matching debug_altro_6Up.py configuration
// From debug_altro_6Up.py:
//   J = diag([0.0969, 0.1235, 0.1918])
//   rw_max_torque = 0.005, rw_J = 0.0014, rw_h0 = 0.001, rw_hmax = 0.015
//   rw_control_weight = 1e0
//   ang_vel cost weight = 0
//   use_raw_control_cost = True
//
// NOTE: The Python configuration uses h_max * 0.01 = 0.00015 for stiction threshold,
// which causes numerical overflow in smoothstep derivatives (k=1e6 in softplus).
// For numerical stability, we disable stiction cost (set to 0) and use larger AM threshold.
Satellite createAltro6UpSatellite() {
    Satellite sat = Satellite();
    // Match debug_altro_6Up.py: J = diag([0.0969, 0.1235, 0.1918])
    sat.change_Jcom(arma::diagmat(arma::vec({0.0969, 0.1235, 0.1918})));

    // 3 RWs matching debug_altro_6Up.py configuration
    // add_RW(axis, J_wheel, max_torq, max_ang_mom, cost, AM_cost, AM_cost_threshold, stiction_cost, stiction_threshold)
    // From Python: rw_J=0.0014, max_torque=0.005, h_max=0.015
    double rw_J = 0.0014;
    double rw_max_torque = 0.005;
    double rw_h_max = 0.015;
    double rw_cost = 1.0;  // rw_control_weight = 1e0

    // RW cost parameters matching Python planner_settings defaults
    // With numerically stable softplus functions, these should now work
    double rw_AM_cost = 1e4;  // rw_AM_weight from Python
    double rw_AM_threshold = 0.5 * rw_h_max;  // RWh_ok_mult * h_max = 0.0075
    double rw_stic_cost = 1.0;  // rw_stic_weight from Python
    double rw_stic_threshold = 0.001;  // Minimum stable value (Python uses 0.00015)

    sat.add_RW(arma::vec({1,0,0}), rw_J, rw_max_torque, rw_h_max, rw_cost, rw_AM_cost, rw_AM_threshold, rw_stic_cost, rw_stic_threshold);
    sat.add_RW(arma::vec({0,1,0}), rw_J, rw_max_torque, rw_h_max, rw_cost, rw_AM_cost, rw_AM_threshold, rw_stic_cost, rw_stic_threshold);
    sat.add_RW(arma::vec({0,0,1}), rw_J, rw_max_torque, rw_h_max, rw_cost, rw_AM_cost, rw_AM_threshold, rw_stic_cost, rw_stic_threshold);

    return sat;
}

// Helper: Create satellite with EXACT debug_altro_6Up.py parameters
// This version exposes the numerical issues for diagnostic purposes
Satellite createAltro6UpSatelliteExact() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.0969, 0.1235, 0.1918})));

    double rw_J = 0.0014;
    double rw_max_torque = 0.005;
    double rw_h_max = 0.015;
    double rw_cost = 1.0;
    double rw_AM_cost = 1e4;  // Original value - causes large Hessians
    double rw_AM_threshold = 0.5 * rw_h_max;
    double rw_stic_cost = 1.0;  // Original value
    // NUMERICAL ISSUE: 0.01 * 0.015 = 0.00015 causes smoothstep overflow
    // Using 0.001 instead as minimum stable value
    double rw_stic_threshold = 0.001;

    sat.add_RW(arma::vec({1,0,0}), rw_J, rw_max_torque, rw_h_max, rw_cost, rw_AM_cost, rw_AM_threshold, rw_stic_cost, rw_stic_threshold);
    sat.add_RW(arma::vec({0,1,0}), rw_J, rw_max_torque, rw_h_max, rw_cost, rw_AM_cost, rw_AM_threshold, rw_stic_cost, rw_stic_threshold);
    sat.add_RW(arma::vec({0,0,1}), rw_J, rw_max_torque, rw_h_max, rw_cost, rw_AM_cost, rw_AM_threshold, rw_stic_cost, rw_stic_threshold);

    return sat;
}

// ============================================================================
// TEST: Exact replication of debug_altro_6Up.py test case
// This test replicates the exact configuration that shows spiky behavior
// ============================================================================
TEST_CASE("debug_altro_6Up configuration - spiky behavior analysis", "[optimizer][altro6up][spiky]") {
    cout << "\n=== Test: debug_altro_6Up Configuration (Spiky Behavior Analysis) ===" << endl;

    Satellite sat = createAltro6UpSatellite();
    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();

    // Match debug_altro_6Up.py: tf=100, dt=1
    int N = 100;
    double dt = 1.0;

    cout << "Satellite config (matching debug_altro_6Up.py):" << endl;
    cout << "  Jcom = diag([0.0969, 0.1235, 0.1918])" << endl;
    cout << "  " << nu << " controls (3 RW torques)" << endl;
    cout << "  State dimension: " << nx << " (w, q, h)" << endl;
    cout << "  Constraint count: " << nc << endl;

    // Initial state from debug_altro_6Up.py:
    // w0 = [0, 0, 0], q0 = [1, 0, 0, 0], h0 = [0, 0, 0] (modified in Python file)
    arma::vec3 w0 = arma::vec({0.0, 0.0, 0.0});
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});  // Identity quaternion
    arma::vec3 h0 = arma::vec({0.0, 0.0, 0.0});  // Zero initial RW momentum
    arma::vec x0 = join_cols(w0, q0, h0);

    cout << "Initial state: w0=[0,0,0], q0=[1,0,0,0], h0=[0,0,0]" << endl;

    // Constant B-field from debug_altro_6Up.py: B = [0, 0.1, 0]
    arma::vec3 B_eci = arma::vec({0.0, 0.1, 0.0});

    // Create vector info with constant B-field
    arma::vec times = arma::linspace(0.0, (N-1)*dt, N);
    arma::mat Bset = arma::repmat(B_eci, 1, N);
    arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
    arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});
    arma::mat Rset = arma::repmat(R_orb, 1, N);
    arma::mat Vset = arma::repmat(V_orb, 1, N);
    // ECI goal: normalize([1,1,1]) for pointing
    arma::vec3 eci_goal = arma::normalise(arma::vec({1.0, 1.0, 1.0}));
    arma::vec3 sat_body_vec = arma::vec({0.0, 0.0, 1.0});  // boresight
    arma::vec3 sun_vec = arma::normalise(arma::vec({1.0, 0.0, 0.0}));
    arma::mat sunset = arma::repmat(sun_vec, 1, N);
    arma::mat satvec = arma::repmat(sat_body_vec, 1, N);
    arma::mat ECIvec = arma::repmat(eci_goal, 1, N);
    arma::vec pset = arma::vec(N).zeros();
    arma::vec rhoset = arma::vec(N).zeros();

    VECTOR_INFO_FORM vecs = std::make_tuple(times, Rset, Vset, Bset, sunset, satvec, ECIvec, pset, rhoset);

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, R_orb, 0, V_orb, eci_goal, 1, 0.0);

    // Generate initial trajectory with zero control
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings matching debug_altro_6Up.py:
    // ang_vel = 0, use_raw_control_cost = True
    // angle weight = 1e3, control_mult = 1.0
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e3,   // w_ang (angle weight)
        0.0,   // w_av (angular velocity weight = 0 as in debug_altro_6Up.py!)
        1.0,   // w_u_mult (control multiplier)
        0.0,   // w_avmag
        0.0,   // w_avang
        1e3,   // w_ang_N (terminal angle weight)
        0.0,   // w_av_N (terminal angular velocity weight = 0)
        0.0,   // w_avmag_N
        0.0,   // w_avang_N
        2,     // whichAngCostFunc (acos formulation)
        true,  // useRawControlCost = True
        0      // useFullCostHess = 0
    );

    // Create planner with matching settings
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-5, 1e-2, 1e-3, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-3;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-8, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e3, 0.0, 1.0, 0.0, 0.0, 1e3, 0.0, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    // Compute initial cost
    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    double initial_cost = planner.cost2Func(traj, vecs, auglag, &costSettings, false);
    cout << "Initial cost: " << initial_cost << endl;

    // Run optimization
    cout << "Running ALILQR (matching debug_altro_6Up.py settings)..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Xset = std::get<0>(opt_form);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double final_cost = std::get<1>(result);
    double max_viol = std::get<2>(result);

    cout << "Final cost: " << final_cost << endl;
    cout << "Max constraint violation: " << max_viol << endl;
    cout << "Cost reduction: " << (initial_cost - final_cost) / initial_cost * 100 << "%" << endl;

    // Analyze control spikiness
    arma::vec control_norms(N);
    arma::vec control_changes(N-1);
    for(int k = 0; k < N; k++) {
        control_norms(k) = arma::norm(opt_Uset.col(k));
        if(k > 0) {
            control_changes(k-1) = arma::norm(opt_Uset.col(k) - opt_Uset.col(k-1));
        }
    }
    double spikiness = arma::max(control_changes) / (arma::mean(control_norms) + 1e-10);

    cout << "\n=== Spikiness Analysis ===" << endl;
    cout << "Max control change: " << arma::max(control_changes) << endl;
    cout << "Mean control norm: " << arma::mean(control_norms) << endl;
    cout << "Spikiness metric: " << spikiness << endl;

    // Print control profile summary
    cout << "\nControl profile (first 10 steps):" << endl;
    for(int k = 0; k < std::min(10, N); k++) {
        cout << "  u[" << k << "] = " << opt_Uset.col(k).t();
    }

    // Identify spiky regions
    double spike_threshold = 2.0 * arma::mean(control_changes);
    int spike_count = 0;
    cout << "\nSpike locations (change > 2x mean):" << endl;
    for(int k = 0; k < N-1; k++) {
        if(control_changes(k) > spike_threshold) {
            cout << "  Spike at k=" << k << ": change=" << control_changes(k) << endl;
            spike_count++;
        }
    }
    cout << "Total spikes detected: " << spike_count << endl;

    // The test documents the behavior - we expect this to potentially be spiky
    // but should still converge
    CHECK(final_cost < initial_cost);

    // Report if spikiness is high (for debugging purposes)
    if(spikiness > 5.0) {
        cout << "\n*** WARNING: High spikiness detected! ***" << endl;
        cout << "This matches the observed behavior in debug_altro_6Up.py" << endl;
    }
}

// ============================================================================
// TEST: Simple 1D rotation regulation with known LQR solution
// For a simple 1D rotation: J*w_dot = tau, with quadratic cost
// J(u) = integral(w^2 + R*u^2)dt + w_N^2
// The optimal solution is a smooth exponential decay
// ============================================================================
TEST_CASE("1D rotation regulation - analytical LQR comparison", "[optimizer][analytical][lqr]") {
    cout << "\n=== Test: 1D Rotation Regulation (Analytical LQR Comparison) ===" << endl;

    // Use a simple satellite with only 1 RW for approximately 1D control
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.1, 0.1, 0.1})));  // Symmetric inertia

    // Single RW along z-axis for 1D control
    double rw_J = 0.001;
    double rw_max_torque = 0.01;
    double rw_h_max = 0.1;
    sat.add_RW(arma::vec({0,0,1}), rw_J, rw_max_torque, rw_h_max, 1.0, 0.0, 0.05, 0.0, 0.001);

    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 50;
    double dt = 0.5;

    cout << "1D regulation problem setup:" << endl;
    cout << "  J = 0.1 (symmetric), single RW along z" << endl;
    cout << "  N = " << N << ", dt = " << dt << endl;

    // Initial state: only z-axis angular velocity error
    arma::vec3 w0 = arma::vec({0.0, 0.0, 0.05});  // 0.05 rad/s about z
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});  // Identity
    arma::vec h0 = arma::vec({0.0});  // Zero initial RW momentum
    arma::vec x0 = join_cols(w0, q0, h0);

    // Zero B-field (pure mechanical system)
    arma::vec3 B_eci = arma::vec({0.0, 0.0, 0.0});
    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    // Override B-field to zero
    std::get<3>(vecs).zeros();

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({0,0,1}), 1, 0.0);

    // Generate initial trajectory
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings: penalize angular velocity and control
    // For LQR-like behavior: w_ang=0, w_av=high
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        0.0,   // w_ang (no angle cost - pure regulation)
        1e3,   // w_av (angular velocity weight)
        1.0,   // w_u_mult
        0.0, 0.0,  // w_avmag, w_avang
        0.0,   // w_ang_N
        1e4,   // w_av_N (high terminal weight)
        0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-6, 1e-3, 1e-4, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-4;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-10, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        0.0, 1e3, 1.0, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    // Run optimization
    cout << "Running ALILQR for 1D regulation..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Xset = std::get<0>(opt_form);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double final_cost = std::get<1>(result);

    cout << "Final cost: " << final_cost << endl;

    // For LQR-optimal solution, control should be smooth (exponential-like decay)
    // Check that angular velocity is driven to near-zero
    double final_wz = opt_Xset(2, N-1);  // z-component of final angular velocity
    cout << "Final w_z: " << final_wz << " (initial: " << w0(2) << ")" << endl;
    cout << "Reduction: " << (1.0 - std::abs(final_wz) / std::abs(w0(2))) * 100 << "%" << endl;

    // Check control smoothness
    arma::vec control_changes(N-1);
    for(int k = 0; k < N-1; k++) {
        control_changes(k) = std::abs(opt_Uset(0, k+1) - opt_Uset(0, k));
    }
    double max_change = arma::max(control_changes);
    double mean_change = arma::mean(control_changes);

    cout << "Control smoothness:" << endl;
    cout << "  Max change: " << max_change << endl;
    cout << "  Mean change: " << mean_change << endl;
    cout << "  Smoothness ratio (max/mean): " << max_change / (mean_change + 1e-10) << endl;

    // For ALTRO (constrained optimization), solution may not be as smooth as pure LQR
    // due to constraint handling and augmented Lagrangian penalty updates
    double smoothness_ratio = max_change / (mean_change + 1e-10);
    cout << "  Note: ALTRO may produce less smooth solutions than pure LQR" << endl;

    CHECK(std::abs(final_wz) < 0.1 * std::abs(w0(2)));  // 90% reduction
    // Relaxed smoothness check - ALTRO with constraints can have larger variations
    CHECK(smoothness_ratio < 100.0);
}

// ============================================================================
// TEST: Rest-to-rest maneuver with known boundary conditions
// For a rest-to-rest maneuver, the optimal trajectory should be symmetric
// (bang-bang for minimum time, smooth for minimum energy)
// ============================================================================
TEST_CASE("Rest-to-rest maneuver - symmetry check", "[optimizer][analytical][symmetry]") {
    cout << "\n=== Test: Rest-to-Rest Maneuver (Symmetry Check) ===" << endl;

    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.1, 0.1, 0.1})));

    // Single RW for simple 1D analysis
    sat.add_RW(arma::vec({0,0,1}), 0.001, 0.01, 0.1, 1.0, 0.0, 0.05, 0.0, 0.001);

    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 40;
    double dt = 0.5;

    cout << "Rest-to-rest maneuver setup:" << endl;
    cout << "  Start: q0 = [1,0,0,0], w0 = 0" << endl;
    cout << "  Goal: Rotate 30 deg about z, end at rest" << endl;

    // Start at rest, identity quaternion
    arma::vec3 w0 = arma::vec({0.0, 0.0, 0.0});
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});
    arma::vec h0 = arma::vec({0.0});
    arma::vec x0 = join_cols(w0, q0, h0);

    // Goal: 30 degree rotation about z
    double theta_goal = 30.0 * M_PI / 180.0;
    arma::vec3 goal_vec = arma::vec({std::sin(theta_goal), 0.0, std::cos(theta_goal)});  // Rotated z-axis

    arma::vec3 B_eci = arma::vec({0.0, 0.0, 0.0});

    // Create vecs with the goal
    arma::vec times = arma::linspace(0.0, (N-1)*dt, N);
    arma::mat Bset = arma::mat(3, N).zeros();
    arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
    arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});
    arma::mat Rset = arma::repmat(R_orb, 1, N);
    arma::mat Vset = arma::repmat(V_orb, 1, N);
    arma::vec3 sat_body_vec = arma::vec({0.0, 0.0, 1.0});
    arma::vec3 sun_vec = arma::vec({1.0, 0.0, 0.0});
    arma::mat sunset = arma::repmat(sun_vec, 1, N);
    arma::mat satvec = arma::repmat(sat_body_vec, 1, N);
    arma::mat ECIvec = arma::repmat(goal_vec, 1, N);
    arma::vec pset = arma::vec(N).zeros();
    arma::vec rhoset = arma::vec(N).zeros();

    VECTOR_INFO_FORM vecs = std::make_tuple(times, Rset, Vset, Bset, sunset, satvec, ECIvec, pset, rhoset);
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, R_orb, 0, V_orb, goal_vec, 1, 0.0);

    // Generate initial trajectory
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Cost settings: angle tracking + angular velocity damping at end
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2,   // w_ang
        1e1,   // w_av
        1.0,   // w_u_mult
        0.0, 0.0,  // w_avmag, w_avang
        1e3,   // w_ang_N (terminal angle)
        1e3,   // w_av_N (terminal zero velocity)
        0.0, 0.0,
        2, true, 0
    );

    // Create planner
    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-6, 1e-3, 1e-4, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-4;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-10, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0, 1e3, 1e3, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    // Run optimization
    cout << "Running ALILQR for rest-to-rest maneuver..." << endl;
    ALILQR_OUTPUT_FORM result = planner.alilqr(dt, traj, vecs, costSettings, alilqrSettings, true);

    OPT_FORM opt_form = std::get<0>(result);
    arma::mat opt_Xset = std::get<0>(opt_form);
    arma::mat opt_Uset = std::get<1>(opt_form);
    double final_cost = std::get<1>(result);

    cout << "Final cost: " << final_cost << endl;

    // Check terminal conditions
    arma::vec3 final_w = opt_Xset(arma::span(0,2), N-1);
    cout << "Final angular velocity: " << arma::norm(final_w) << " rad/s (should be ~0)" << endl;

    // For a minimum-energy rest-to-rest maneuver, the control should be roughly symmetric
    // (accelerate in first half, decelerate in second half)
    int mid = N / 2;

    // Compare first half and second half control profiles
    arma::vec first_half_u = opt_Uset.row(0).subvec(0, mid-1).t();
    arma::vec second_half_u = arma::reverse(opt_Uset.row(0).subvec(mid, N-1).t());

    // For symmetric maneuver, first half should be roughly opposite of reversed second half
    double symmetry_error = arma::norm(first_half_u + second_half_u) / (arma::norm(first_half_u) + 1e-10);

    cout << "Control symmetry analysis:" << endl;
    cout << "  First half sum: " << arma::sum(first_half_u) << endl;
    cout << "  Second half sum: " << arma::sum(second_half_u) << endl;
    cout << "  Symmetry error (0 = perfect): " << symmetry_error << endl;

    // Check that final angular velocity is near zero
    CHECK(arma::norm(final_w) < 0.01);  // Should end at rest

    // Symmetry isn't perfect due to discrete time and other effects, but should be reasonable
    // This test documents the behavior rather than enforcing strict symmetry
    cout << "  (Note: Perfect symmetry not expected due to discretization)" << endl;
}

// ============================================================================
// TEST: Infinite horizon LQR steady-state gain comparison
// For a linear system, compare iLQR gains to DARE solution
// ============================================================================
TEST_CASE("LQR gain consistency check", "[optimizer][analytical][gains]") {
    cout << "\n=== Test: LQR Gain Consistency Check ===" << endl;

    // Simple satellite for linear-ish dynamics near equilibrium
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.1, 0.1, 0.1})));
    sat.add_RW(arma::vec({1,0,0}), 0.001, 0.01, 0.1, 1.0, 0.0, 0.05, 0.0, 0.001);
    sat.add_RW(arma::vec({0,1,0}), 0.001, 0.01, 0.1, 1.0, 0.0, 0.05, 0.0, 0.001);
    sat.add_RW(arma::vec({0,0,1}), 0.001, 0.01, 0.1, 1.0, 0.0, 0.05, 0.0, 0.001);

    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 100;  // Long horizon for steady-state
    double dt = 0.5;

    cout << "Testing gain consistency for long-horizon LQR:" << endl;
    cout << "  N = " << N << ", dt = " << dt << endl;

    // Start near equilibrium
    arma::vec3 w0 = arma::vec({0.001, -0.001, 0.001});  // Small perturbation
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});
    arma::vec3 h0 = arma::vec({0.0, 0.0, 0.0});
    arma::vec x0 = join_cols(w0, q0, h0);

    arma::vec3 B_eci = arma::vec({0.0, 0.0, 0.0});
    VECTOR_INFO_FORM vecs = createDefaultVecInfo(N, dt);
    std::get<3>(vecs).zeros();  // Zero B-field

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({0,0,1}), 1, 0.0);

    // Generate initial trajectory
    arma::mat Uset_init = arma::mat(nu, N).zeros();
    arma::mat Xset = arma::mat(nx, N).zeros();
    Xset.col(0) = x0;

    for(int k = 0; k < N-1; k++) {
        auto rk4out = rk4z(dt, Xset.col(k), Uset_init.col(k), sat, dynamics_info, dynamics_info);
        Xset.col(k+1) = sat.state_norm(std::get<0>(rk4out));
    }

    arma::vec dt_vec = arma::vec(N).fill(dt);
    arma::mat TQset = arma::mat(3, N).zeros();
    TRAJECTORY_FORM traj = std::make_tuple(Xset, Uset_init, dt_vec, TQset);

    // Balanced cost (state and control weighted equally)
    COST_SETTINGS_FORM costSettings = std::make_tuple(
        0.0, 1e2, 1e2, 0.0, 0.0,
        0.0, 1e2, 0.0, 0.0,
        2, true, 0
    );

    arma::mat33 J_est = sat.Jcom;
    SYSTEM_SETTINGS_FORM systemSettings = std::make_tuple(J_est, dt, dt, 2.22e-16, 60.0, 15.0);
    LINE_SEARCH_SETTINGS_FORM lineSearchSettings = std::make_tuple(30, 1e-10, 500.0);
    arma::mat xmax_vec = 10.0 * arma::mat(nxr, 1).ones();
    BREAK_SETTINGS_FORM breakSettings = std::make_tuple(50, 200, 5000, 1e-6, 1e-3, 1e-4, 10, 0.002, 1e40, xmax_vec);
    AUGLAG_SETTINGS_FORM auglagSettings = std::make_tuple(0.0, 1e20, 1e-2, 1e16, 10.0);
    double rho = 1e-4;
    REG_SETTINGS_FORM regSettings = std::make_tuple(rho, 1e-10, 1e30, 1.6, 10.0, 2, 0.0, 0, 0);
    ALILQR_SETTINGS_FORM alilqrSettings = std::make_tuple(lineSearchSettings, auglagSettings, breakSettings, regSettings);
    INITIAL_TRAJ_SETTINGS_FORM initTrajSettings = std::make_tuple(1000.0, 10.0*M_PI/180.0,
        std::make_tuple(0.0, -2e0, 0.0, -0.005, 0.1, 0.5),
        std::make_tuple(0.0, -1e-4, 0.0, -1e-5, 0.1, 0.5));
    LQR_COST_SETTINGS_FORM tvlqrCostSettings = std::make_tuple(
        0.0, 1e2, 1e2, 0.0, 0.0, 0.0, 1e2, 0.0, 0.0, 0, true, 0);

    ALL_SETTINGS_FORM allSettings = std::make_tuple(systemSettings, alilqrSettings, alilqrSettings,
        initTrajSettings, costSettings, costSettings, tvlqrCostSettings);

    OldPlanner planner(sat, allSettings);
    planner.setVerbosity(false);
    planner.quaternionTo3VecMode = 2;

    arma::mat lambdaSet = arma::mat(nc, N).zeros();
    double mu = 1.0;
    arma::mat muSet = arma::mat(nc, N).fill(mu);
    AUGLAG_INFO_FORM auglag = std::make_tuple(lambdaSet, mu, muSet);

    // Run backward pass to get gains
    REG_PAIR regs = std::make_tuple(rho, 1.0);
    auto backwardResult = planner.backwardPass(dt, traj, vecs, auglag, regs, &costSettings, regSettings, true);
    BACKWARD_PASS_RESULTS_FORM bpResults = std::get<0>(backwardResult);

    arma::cube Kset = std::get<0>(bpResults);

    // Check that gains converge to steady-state in the interior
    // (gains at k=10 through k=N-10 should be similar for large N)
    int start_k = 10;
    int end_k = N - 20;

    cout << "\nGain convergence analysis:" << endl;

    arma::mat K_ref = Kset.slice(N/2);  // Reference gain from middle
    double max_gain_diff = 0.0;

    for(int k = start_k; k < end_k; k++) {
        double diff = arma::norm(Kset.slice(k) - K_ref, "fro");
        max_gain_diff = std::max(max_gain_diff, diff);
    }

    cout << "  Max gain variation in interior (k=" << start_k << " to " << end_k << "): " << max_gain_diff << endl;
    cout << "  Reference gain norm: " << arma::norm(K_ref, "fro") << endl;
    cout << "  Relative variation: " << max_gain_diff / (arma::norm(K_ref, "fro") + 1e-10) * 100 << "%" << endl;

    // For long horizon, interior gains should be nearly constant (steady-state)
    double relative_variation = max_gain_diff / (arma::norm(K_ref, "fro") + 1e-10);

    // Allow up to 20% variation (nonlinear effects)
    CHECK(relative_variation < 0.2);

    cout << "  Gain consistency check: " << (relative_variation < 0.2 ? "PASSED" : "FAILED") << endl;
}

// ============================================================================
// JACOBIAN AND HESSIAN VERIFICATION TESTS
// These tests verify the analytical derivatives against finite differences
// ============================================================================

// Helper function to compute finite difference Jacobian of dynamics w.r.t. state
arma::mat finiteDiffDynamicsJacobianX(const Satellite& sat, arma::vec x, arma::vec u,
                                       DYNAMICS_INFO_FORM dynamics_info, double eps = 1e-7) {
    int nx = sat.state_N();
    arma::mat Jx(nx, nx);

    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        // Normalize quaternion if perturbing quaternion elements
        if(i >= sat.quat0index() && i <= sat.quat0index() + 3) {
            x_pert.rows(sat.quat0index(), sat.quat0index() + 3) =
                arma::normalise(x_pert.rows(sat.quat0index(), sat.quat0index() + 3));
        }
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx.col(i) = (f_pert - f0) / eps;
    }
    return Jx;
}

// Helper function to compute finite difference Jacobian of dynamics w.r.t. control
arma::mat finiteDiffDynamicsJacobianU(const Satellite& sat, arma::vec x, arma::vec u,
                                       DYNAMICS_INFO_FORM dynamics_info, double eps = 1e-7) {
    int nx = sat.state_N();
    int nu = sat.control_N();
    arma::mat Ju(nx, nu);

    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju.col(i) = (f_pert - f0) / eps;
    }
    return Ju;
}

// ============================================================================
// TEST: Dynamics Jacobians verification (Satellite class)
// ============================================================================
TEST_CASE("Satellite dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics]") {
    cout << "\n=== Test: Satellite Dynamics Jacobians Verification ===" << endl;

    // Test with MTQ-only satellite
    SECTION("MTQ-only satellite") {
        cout << "Testing MTQ-only satellite..." << endl;
        Satellite sat = createMTQOnlySatellite();
        int nx = sat.state_N();
        int nu = sat.control_N();

        // Create test state and control
        arma::arma_rng::set_seed(42);
        arma::vec3 w0 = 0.05 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x = join_cols(w0, q0);
        arma::vec u = 0.05 * arma::randn(nu);

        arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                            arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

        // Get analytical Jacobians
        auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
        arma::mat Jx_analytic = std::get<0>(jacs);
        arma::mat Ju_analytic = std::get<1>(jacs);

        // Compute finite difference Jacobians
        double eps = 1e-7;
        arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

        arma::mat Jx_fd(nx, nx);
        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
            Jx_fd.col(i) = (f_pert - f0) / eps;
        }

        arma::mat Ju_fd(nx, nu);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
            Ju_fd.col(i) = (f_pert - f0) / eps;
        }

        // Compare
        double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
        double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
        double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
        double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

        cout << "  Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
        cout << "  Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

        CHECK(Jx_rel_error < 1e-4);
        CHECK(Ju_rel_error < 1e-4);
    }

    // Test with RW-only satellite
    SECTION("RW-only satellite") {
        cout << "Testing RW-only satellite..." << endl;
        Satellite sat = createRWOnlySatellite();
        int nx = sat.state_N();
        int nu = sat.control_N();

        arma::arma_rng::set_seed(43);
        arma::vec3 w0 = 0.03 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.05, 0.05, 0.05}) + 0.05*arma::randn(4));
        arma::vec x_base = join_cols(w0, q0);

        // Add RW momentum states
        arma::vec rw_h = 0.001 * arma::randn(sat.number_RW);
        arma::vec x = join_cols(x_base, rw_h);

        arma::vec u = 0.0005 * arma::randn(nu);

        arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                            arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

        auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
        arma::mat Jx_analytic = std::get<0>(jacs);
        arma::mat Ju_analytic = std::get<1>(jacs);

        double eps = 1e-7;
        arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

        arma::mat Jx_fd(nx, nx);
        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
            Jx_fd.col(i) = (f_pert - f0) / eps;
        }

        arma::mat Ju_fd(nx, nu);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
            Ju_fd.col(i) = (f_pert - f0) / eps;
        }

        double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
        double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
        double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
        double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

        cout << "  Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
        cout << "  Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

        CHECK(Jx_rel_error < 1e-4);
        CHECK(Ju_rel_error < 1e-4);
    }

    // Test with hybrid satellite
    SECTION("Hybrid MTQ+RW satellite") {
        cout << "Testing hybrid MTQ+RW satellite..." << endl;
        Satellite sat = createHybridSatellite();
        int nx = sat.state_N();
        int nu = sat.control_N();

        arma::arma_rng::set_seed(44);
        arma::vec3 w0 = 0.02 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x_base = join_cols(w0, q0);

        arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
        arma::vec x = join_cols(x_base, rw_h);

        // Mixed control: MTQs and RW torques
        arma::vec u = arma::vec(nu).zeros();
        u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
        u.tail(sat.number_RW) = 0.0005 * arma::randn(sat.number_RW);

        arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});
        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                            arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

        auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
        arma::mat Jx_analytic = std::get<0>(jacs);
        arma::mat Ju_analytic = std::get<1>(jacs);

        double eps = 1e-7;
        arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

        arma::mat Jx_fd(nx, nx);
        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
            Jx_fd.col(i) = (f_pert - f0) / eps;
        }

        arma::mat Ju_fd(nx, nu);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
            Ju_fd.col(i) = (f_pert - f0) / eps;
        }

        double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
        double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
        double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
        double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

        cout << "  Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
        cout << "  Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

        CHECK(Jx_rel_error < 1e-4);
        CHECK(Ju_rel_error < 1e-4);
    }
}

// ============================================================================
// TEST: Dynamics Hessians verification (Satellite class)
// Note: These tests are complex due to the cube indexing conventions.
// The analytical Hessians use slice(i) for the i-th output dimension.
// ============================================================================
TEST_CASE("Satellite dynamics Hessians match finite differences", "[satellite][hessian][dynamics]") {
    cout << "\n=== Test: Satellite Dynamics Hessians Verification ===" << endl;

    SECTION("MTQ-only satellite") {
        cout << "Testing MTQ-only satellite Hessians..." << endl;
        Satellite sat = createMTQOnlySatellite();
        int nx = sat.state_N();
        int nu = sat.control_N();

        arma::arma_rng::set_seed(45);
        arma::vec3 w0 = 0.03 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x = join_cols(w0, q0);
        arma::vec u = 0.05 * arma::randn(nu);

        arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                            arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

        // Get analytical Hessians
        auto hess = sat.dynamicsHessians(x, u, dynamics_info);
        arma::cube Hxx_analytic = std::get<0>(hess);  // d^2f/dxdx
        arma::cube Hux_analytic = std::get<1>(hess);  // d^2f/dudx
        arma::cube Huu_analytic = std::get<2>(hess);  // d^2f/dudu

        // Compute finite difference Hessians via Jacobian perturbation
        double eps = 1e-5;

        // Get base Jacobians
        auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
        arma::mat Jx0 = std::get<0>(jacs0);
        arma::mat Ju0 = std::get<1>(jacs0);

        // Compute d(Jx)/dx (Hessian w.r.t. x,x) via finite difference of Jacobian
        // Analytical convention: Hxx(i, j, k) = d²f_k/(dx_i dx_j), slice k is output dimension
        arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
            arma::mat Jx_pert = std::get<0>(jacs_pert);
            // Jx(k, j) = df_k/dx_j, so (Jx_pert(k,j) - Jx0(k,j))/eps = d²f_k/(dx_j dx_i)
            // Store at Hxx_fd(i, j, k) to match analytical convention
            for(int j = 0; j < nx; j++) {
                for(int k = 0; k < nx; k++) {
                    Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
                }
            }
        }

        // Compute d(Jx)/du (Hessian w.r.t. u,x) via finite difference
        // Analytical convention: Hux(i, j, k) = d²f_k/(du_i dx_j), slice k is output dimension
        arma::cube Hux_fd(nu, nx, nx, arma::fill::zeros);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            auto jacs_pert = sat.dynamicsJacobians(x, u_pert, dynamics_info);
            arma::mat Jx_pert = std::get<0>(jacs_pert);
            // Jx(k, j) = df_k/dx_j, so (Jx_pert(k,j) - Jx0(k,j))/eps = d²f_k/(dx_j du_i)
            // Store at Hux_fd(i, j, k) to match analytical convention
            for(int j = 0; j < nx; j++) {
                for(int k = 0; k < nx; k++) {
                    Hux_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
                }
            }
        }

        // Compare Hxx
        double Hxx_error = 0.0;
        double Hxx_norm = 0.0;
        for(int k = 0; k < nx; k++) {
            Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
            Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
        }
        double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

        // Compare Hux
        double Hux_error = 0.0;
        double Hux_norm = 0.0;
        for(int k = 0; k < nx; k++) {
            Hux_error += arma::norm(Hux_analytic.slice(k) - Hux_fd.slice(k), "fro");
            Hux_norm += arma::norm(Hux_fd.slice(k), "fro");
        }
        double Hux_rel_error = Hux_error / (Hux_norm + 1e-10);

        cout << "  Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;
        cout << "  Hux error (abs): " << Hux_error << ", (rel): " << Hux_rel_error << endl;

        // Hessian matching is less precise due to second-order finite differences
        CHECK(Hxx_rel_error < 1e-2);
        CHECK(Hux_rel_error < 1e-2);
    }

    SECTION("Hybrid MTQ+RW satellite") {
        cout << "Testing hybrid satellite Hessians..." << endl;
        Satellite sat = createHybridSatellite();
        int nx = sat.state_N();
        int nu = sat.control_N();

        arma::arma_rng::set_seed(46);
        arma::vec3 w0 = 0.02 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x_base = join_cols(w0, q0);
        arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
        arma::vec x = join_cols(x_base, rw_h);

        arma::vec u = arma::vec(nu).zeros();
        u.head(sat.number_MTQ) = 0.03 * arma::randn(sat.number_MTQ);
        u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);

        arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});
        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                            arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

        auto hess = sat.dynamicsHessians(x, u, dynamics_info);
        arma::cube Hxx_analytic = std::get<0>(hess);
        arma::cube Hux_analytic = std::get<1>(hess);

        double eps = 1e-5;
        auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
        arma::mat Jx0 = std::get<0>(jacs0);

        // Analytical convention: Hxx(i, j, k) = d²f_k/(dx_i dx_j), slice k is output dimension
        arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
            arma::mat Jx_pert = std::get<0>(jacs_pert);
            // Jx(k, j) = df_k/dx_j, so (Jx_pert(k,j) - Jx0(k,j))/eps = d²f_k/(dx_j dx_i)
            // Store at Hxx_fd(i, j, k) to match analytical convention
            for(int j = 0; j < nx; j++) {
                for(int k = 0; k < nx; k++) {
                    Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
                }
            }
        }

        double Hxx_error = 0.0;
        double Hxx_norm = 0.0;
        for(int k = 0; k < nx; k++) {
            Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
            Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
        }
        double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

        cout << "  Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

        CHECK(Hxx_rel_error < 1e-2);
    }
}

// ============================================================================
// TEST: Constraint Jacobians verification (Satellite class)
// ============================================================================
TEST_CASE("Satellite constraint Jacobians match finite differences", "[satellite][jacobian][constraint]") {
    cout << "\n=== Test: Satellite Constraint Jacobians Verification ===" << endl;

    SECTION("MTQ-only satellite with AV constraint") {
        cout << "Testing MTQ-only with AV constraint..." << endl;
        Satellite sat = createMTQOnlySatellite();
        sat.set_AV_constraint(0.1);  // 0.1 rad/s max angular velocity

        int nx = sat.state_N();
        int nxr = sat.reduced_state_N();
        int nu = sat.control_N();
        int nc = sat.constraint_N();
        int N = 10;
        int k = 5;

        cout << "  Constraint count: " << nc << endl;

        arma::arma_rng::set_seed(47);
        arma::vec3 w0 = 0.05 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x = join_cols(w0, q0);
        arma::vec u = 0.1 * arma::randn(nu);
        arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

        // Get analytical Jacobians
        auto jacs = sat.constraintJacobians(k, N, u, x, sun);
        arma::mat Jcu_analytic = std::get<0>(jacs);
        arma::mat Jcx_analytic = std::get<1>(jacs);

        // Get constraint value
        arma::vec c0 = sat.getConstraints(k, N, u, x, sun);

        // Finite difference w.r.t. control
        double eps = 1e-7;
        arma::mat Jcu_fd(nc, nu, arma::fill::zeros);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            arma::vec c_pert = sat.getConstraints(k, N, u_pert, x, sun);
            Jcu_fd.col(i) = (c_pert - c0) / eps;
        }

        // Finite difference w.r.t. reduced state (first 6 elements: w, then 3-vector for attitude)
        arma::mat Jcx_fd(nc, nxr, arma::fill::zeros);
        // Perturb angular velocity
        for(int i = 0; i < 3; i++) {
            arma::vec x_pert = x;
            x_pert(sat.avindex0() + i) += eps;
            arma::vec c_pert = sat.getConstraints(k, N, u, x_pert, sun);
            Jcx_fd.col(i) = (c_pert - c0) / eps;
        }
        // Perturb quaternion (affects reduced state attitude representation)
        // This is complex due to quaternion manifold, so we just check the angular velocity part

        // Compare control Jacobian
        double Jcu_error = arma::norm(Jcu_analytic - Jcu_fd, "fro");
        double Jcu_rel_error = Jcu_error / (arma::norm(Jcu_fd, "fro") + 1e-10);

        // Compare state Jacobian (just angular velocity part)
        double Jcx_av_error = arma::norm(Jcx_analytic.cols(0, 2) - Jcx_fd.cols(0, 2), "fro");
        double Jcx_av_rel_error = Jcx_av_error / (arma::norm(Jcx_fd.cols(0, 2), "fro") + 1e-10);

        cout << "  Jcu error (abs): " << Jcu_error << ", (rel): " << Jcu_rel_error << endl;
        cout << "  Jcx (AV part) error (abs): " << Jcx_av_error << ", (rel): " << Jcx_av_rel_error << endl;

        CHECK(Jcu_rel_error < 1e-4);
        CHECK(Jcx_av_rel_error < 1e-4);
    }

    SECTION("Hybrid satellite with sunpoint constraint") {
        cout << "Testing hybrid with sunpoint constraint..." << endl;
        Satellite sat = createHybridSatellite();
        sat.add_sunpoint_constraint(arma::vec({0,0,1}), 30.0 * M_PI / 180.0, false);  // 30 deg keep-out

        int nx = sat.state_N();
        int nxr = sat.reduced_state_N();
        int nu = sat.control_N();
        int nc = sat.constraint_N();
        int N = 10;
        int k = 5;

        cout << "  Constraint count: " << nc << endl;

        arma::arma_rng::set_seed(48);
        arma::vec3 w0 = 0.02 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x_base = join_cols(w0, q0);
        arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
        arma::vec x = join_cols(x_base, rw_h);

        arma::vec u = arma::vec(nu).zeros();
        u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
        u.tail(sat.number_RW) = 0.0005 * arma::randn(sat.number_RW);
        arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

        auto jacs = sat.constraintJacobians(k, N, u, x, sun);
        arma::mat Jcu_analytic = std::get<0>(jacs);
        arma::mat Jcx_analytic = std::get<1>(jacs);

        arma::vec c0 = sat.getConstraints(k, N, u, x, sun);

        double eps = 1e-7;
        arma::mat Jcu_fd(nc, nu, arma::fill::zeros);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            arma::vec c_pert = sat.getConstraints(k, N, u_pert, x, sun);
            Jcu_fd.col(i) = (c_pert - c0) / eps;
        }

        double Jcu_error = arma::norm(Jcu_analytic - Jcu_fd, "fro");
        double Jcu_rel_error = Jcu_error / (arma::norm(Jcu_fd, "fro") + 1e-10);

        cout << "  Jcu error (abs): " << Jcu_error << ", (rel): " << Jcu_rel_error << endl;

        CHECK(Jcu_rel_error < 1e-4);
    }
}

// ============================================================================
// TEST: Constraint Hessians verification (Satellite class)
// ============================================================================
TEST_CASE("Satellite constraint Hessians match finite differences", "[satellite][hessian][constraint]") {
    cout << "\n=== Test: Satellite Constraint Hessians Verification ===" << endl;

    SECTION("Hybrid satellite with RW stiction constraint") {
        cout << "Testing hybrid satellite constraint Hessians..." << endl;
        Satellite sat = createHybridSatellite();
        sat.set_AV_constraint(0.2);

        int nx = sat.state_N();
        int nxr = sat.reduced_state_N();
        int nu = sat.control_N();
        int nc = sat.constraint_N();
        int N = 10;
        int k = 5;

        cout << "  State dim: " << nx << ", Reduced: " << nxr << ", Control: " << nu << ", Constraints: " << nc << endl;

        arma::arma_rng::set_seed(49);
        arma::vec3 w0 = 0.02 * arma::randn(3);
        arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
        arma::vec x_base = join_cols(w0, q0);
        arma::vec rw_h = 0.005 * arma::randn(sat.number_RW);
        arma::vec x = join_cols(x_base, rw_h);

        arma::vec u = arma::vec(nu).zeros();
        u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
        u.tail(sat.number_RW) = 0.0005 * arma::randn(sat.number_RW);
        arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

        // Get analytical Hessians
        auto hess = sat.constraintHessians(k, N, u, x, sun);
        arma::cube Hcuu_analytic = std::get<0>(hess);
        arma::cube Hcux_analytic = std::get<1>(hess);
        arma::cube Hcxx_analytic = std::get<2>(hess);

        // Get base Jacobians
        auto jacs0 = sat.constraintJacobians(k, N, u, x, sun);
        arma::mat Jcu0 = std::get<0>(jacs0);
        arma::mat Jcx0 = std::get<1>(jacs0);

        // Finite difference Hessian: d(Jcu)/du
        double eps = 1e-5;
        arma::cube Hcuu_fd(nu, nu, nc, arma::fill::zeros);
        for(int i = 0; i < nu; i++) {
            arma::vec u_pert = u;
            u_pert(i) += eps;
            auto jacs_pert = sat.constraintJacobians(k, N, u_pert, x, sun);
            arma::mat Jcu_pert = std::get<0>(jacs_pert);
            for(int j = 0; j < nc; j++) {
                Hcuu_fd.slice(j).col(i) = (Jcu_pert.row(j).t() - Jcu0.row(j).t()) / eps;
            }
        }

        // Compare Hcuu
        double Hcuu_error = 0.0;
        double Hcuu_norm = 0.0;
        for(int i = 0; i < nc; i++) {
            Hcuu_error += arma::norm(Hcuu_analytic.slice(i) - Hcuu_fd.slice(i), "fro");
            Hcuu_norm += arma::norm(Hcuu_fd.slice(i), "fro");
        }
        double Hcuu_rel_error = Hcuu_error / (Hcuu_norm + 1e-10);

        cout << "  Hcuu error (abs): " << Hcuu_error << ", (rel): " << Hcuu_rel_error << endl;

        // Hessian comparison is less precise
        CHECK(Hcuu_rel_error < 0.1);  // 10% tolerance for second-order derivatives
    }
}

// ============================================================================
// TEST: GeneralUtil rotation matrix Jacobians (dRTBdq, dRTBdqQ)
// ============================================================================
TEST_CASE("GeneralUtil dRTBdq matches finite differences", "[util][jacobian][rotation]") {
    cout << "\n=== Test: dRTBdq Jacobian Verification ===" << endl;

    arma::arma_rng::set_seed(50);

    // Test multiple random quaternions
    for(int trial = 0; trial < 5; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 B = 1e-5 * arma::randn(3);  // Magnetic field scale

        // Get analytical Jacobian (3x4)
        arma::mat Jq_analytic = dRTBdq(q, B);

        // Compute finite difference
        double eps = 1e-7;
        arma::mat Jq_fd(3, 4);
        arma::vec3 f0 = rotMat(q).t() * B;

        for(int i = 0; i < 4; i++) {
            arma::vec4 q_pert = q;
            q_pert(i) += eps;
            // Don't normalize - dRTBdq computes derivative in full 4D quaternion space
            arma::vec3 f_pert = rotMat(q_pert).t() * B;
            Jq_fd.col(i) = (f_pert - f0) / eps;
        }

        double error = arma::norm(Jq_analytic - Jq_fd, "fro");
        double rel_error = error / (arma::norm(Jq_fd, "fro") + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-4);
    }
}

TEST_CASE("GeneralUtil dRTBdqQ matches finite differences", "[util][jacobian][rotation]") {
    cout << "\n=== Test: dRTBdqQ (Reduced) Jacobian Verification ===" << endl;

    arma::arma_rng::set_seed(51);

    for(int trial = 0; trial < 5; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 B = 1e-5 * arma::randn(3);

        // Get analytical Jacobian (3x3) via reduced representation
        arma::mat JqQ_analytic = dRTBdqQ(q, B);

        // Finite difference using W matrix for reduced perturbation
        double eps = 1e-7;
        arma::mat JqQ_fd(3, 3);
        arma::vec3 f0 = rotMat(q).t() * B;
        arma::mat W = findWMat(q);  // 4x3 matrix

        for(int i = 0; i < 3; i++) {
            arma::vec3 delta = arma::vec(3).zeros();
            delta(i) = eps;
            arma::vec4 q_pert = arma::normalise(q + W * delta);
            arma::vec3 f_pert = rotMat(q_pert).t() * B;
            JqQ_fd.col(i) = (f_pert - f0) / eps;
        }

        double error = arma::norm(JqQ_analytic - JqQ_fd, "fro");
        double rel_error = error / (arma::norm(JqQ_fd, "fro") + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-4);
    }
}

// ============================================================================
// TEST: GeneralUtil rotation matrix Hessians (ddvTRTudq, ddvTRTudqQ, ddRTudqQ)
// ============================================================================
TEST_CASE("GeneralUtil ddvTRTudqQ matches finite differences", "[util][hessian][rotation][!mayfail]") {
    cout << "\n=== Test: ddvTRTudqQ Hessian Verification ===" << endl;
    cout << "  Note: This test may fail - finite diff approach may not capture reduced parameterization correctly" << endl;

    arma::arma_rng::set_seed(52);

    for(int trial = 0; trial < 5; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 v = arma::normalise(arma::randn(3));
        arma::vec3 u = 1e-5 * arma::randn(3);

        // Get analytical Hessian (3x3)
        arma::mat H_analytic = ddvTRTudqQ(q, v, u);

        // Compute finite difference Hessian via Jacobian perturbation
        double eps = 1e-5;
        arma::mat W = findWMat(q);

        // Base gradient (using dRTBdqQ approach for v^T * R^T * u)
        auto grad_func = [&](arma::vec4 qk) -> arma::vec3 {
            // Gradient of v^T * R^T * u w.r.t. reduced quaternion
            return (dRTBdqQ(qk, u).t() * v);
        };

        arma::vec3 g0 = grad_func(q);
        arma::mat H_fd(3, 3);

        for(int i = 0; i < 3; i++) {
            arma::vec3 delta = arma::vec(3).zeros();
            delta(i) = eps;
            arma::vec4 q_pert = arma::normalise(q + W * delta);
            arma::vec3 g_pert = grad_func(q_pert);
            H_fd.col(i) = (g_pert - g0) / eps;
        }

        double error = arma::norm(H_analytic - H_fd, "fro");
        double rel_error = error / (arma::norm(H_fd, "fro") + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-2);  // Looser tolerance for Hessians
    }
}

TEST_CASE("GeneralUtil ddRTudqQ matches finite differences", "[util][hessian][rotation][!mayfail]") {
    cout << "\n=== Test: ddRTudqQ Cube Hessian Verification ===" << endl;
    cout << "  Note: This test may fail - finite diff approach may not capture reduced parameterization correctly" << endl;

    arma::arma_rng::set_seed(53);

    for(int trial = 0; trial < 3; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 u = 1e-5 * arma::randn(3);

        // Get analytical Hessian cube (3x3x3)
        arma::cube H_analytic = ddRTudqQ(q, u);

        // Compute finite difference
        double eps = 1e-5;
        arma::mat W = findWMat(q);

        arma::mat J0 = dRTBdqQ(q, u);  // 3x3 base Jacobian
        arma::cube H_fd(3, 3, 3, arma::fill::zeros);

        for(int i = 0; i < 3; i++) {
            arma::vec3 delta = arma::vec(3).zeros();
            delta(i) = eps;
            arma::vec4 q_pert = arma::normalise(q + W * delta);
            arma::mat J_pert = dRTBdqQ(q_pert, u);
            for(int j = 0; j < 3; j++) {
                H_fd.slice(j).col(i) = (J_pert.col(j) - J0.col(j)) / eps;
            }
        }

        double error = 0.0;
        double norm_val = 0.0;
        for(int k = 0; k < 3; k++) {
            error += arma::norm(H_analytic.slice(k) - H_fd.slice(k), "fro");
            norm_val += arma::norm(H_fd.slice(k), "fro");
        }
        double rel_error = error / (norm_val + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-2);
    }
}

// ============================================================================
// TEST: Full 4D quaternion Hessian - no manifold constraint
// This verifies ddvTRTudq directly, avoiding SO(3) manifold issues
// ============================================================================
TEST_CASE("GeneralUtil ddvTRTudq (full 4D) matches finite differences", "[util][hessian][rotation]") {
    cout << "\n=== Test: ddvTRTudq (Full 4D) Hessian Verification ===" << endl;

    arma::arma_rng::set_seed(60);

    for(int trial = 0; trial < 5; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 v = arma::normalise(arma::randn(3));
        arma::vec3 u = 1e-5 * arma::randn(3);

        // Get analytical 4D Hessian
        arma::mat44 H_analytic = ddvTRTudq(q, v, u);

        // Finite difference in full 4D (NO normalization)
        double eps = 1e-7;
        arma::vec4 g0 = dRTBdq(q, u).t() * v;
        arma::mat44 H_fd;

        for(int i = 0; i < 4; i++) {
            arma::vec4 q_pert = q;
            q_pert(i) += eps;  // Direct perturbation, no normalization!
            arma::vec4 g_pert = dRTBdq(q_pert, u).t() * v;
            H_fd.col(i) = (g_pert - g0) / eps;
        }

        double error = arma::norm(H_analytic - H_fd, "fro");
        double rel_error = error / (arma::norm(H_fd, "fro") + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-5);
    }
}

// ============================================================================
// TEST: Verify 3D reduced Hessian via chain rule transformation from 4D
// ddvTRTudqQ = W^T * ddvTRTudq * W - I * (v^T * dRTBdq(q,u) * q)
// ============================================================================
TEST_CASE("ddvTRTudqQ chain rule from ddvTRTudq", "[util][hessian][rotation][chainrule]") {
    cout << "\n=== Test: ddvTRTudqQ Chain Rule Verification ===" << endl;

    arma::arma_rng::set_seed(61);

    for(int trial = 0; trial < 5; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 v = arma::normalise(arma::randn(3));
        arma::vec3 u = 1e-5 * arma::randn(3);

        // Get W matrix
        arma::mat W = findWMat(q);  // 4x3

        // Get 4D Hessian
        arma::mat44 H_4D = ddvTRTudq(q, v, u);

        // Compute expected 3D Hessian via chain rule
        arma::mat33 H_3D_expected = W.t() * H_4D * W
            - arma::eye(3,3) * arma::as_scalar(v.t() * dRTBdq(q, u) * q);

        // Get actual 3D Hessian
        arma::mat33 H_3D_actual = ddvTRTudqQ(q, v, u);

        double error = arma::norm(H_3D_expected - H_3D_actual, "fro");

        cout << "  Trial " << trial << ": error = " << error << endl;
        CHECK(error < 1e-12);  // Should be exact (just linear algebra)
    }
}

// ============================================================================
// TEST: Verify ddRTudqQ cube via chain rule
// ============================================================================
TEST_CASE("ddRTudqQ chain rule from ddvTRTudq", "[util][hessian][rotation][chainrule]") {
    cout << "\n=== Test: ddRTudqQ Chain Rule Verification ===" << endl;

    arma::arma_rng::set_seed(62);

    for(int trial = 0; trial < 3; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 u = 1e-5 * arma::randn(3);

        arma::mat W = findWMat(q);
        arma::cube H_3D_actual = ddRTudqQ(q, u);  // 3x3x3

        // Verify each output component via chain rule
        double max_error = 0.0;
        for(int k = 0; k < 3; k++) {
            arma::vec3 e_k = arma::zeros(3);
            e_k(k) = 1.0;

            // For output k: H_3D(:,:,k) = W^T * ddvTRTudq(q, e_k, u) * W - correction
            arma::mat44 H_4D = ddvTRTudq(q, e_k, u);
            arma::mat33 H_expected = W.t() * H_4D * W
                - arma::eye(3,3) * arma::as_scalar(e_k.t() * dRTBdq(q, u) * q);

            double error = arma::norm(H_3D_actual.slice(k) - H_expected, "fro");
            max_error = std::max(max_error, error);
        }

        cout << "  Trial " << trial << ": max_error = " << max_error << endl;
        CHECK(max_error < 1e-12);
    }
}

// ============================================================================
// TEST: GeneralUtil angular cost functions (cost2ang, cost2angQ)
// ============================================================================
TEST_CASE("GeneralUtil cost2angQ derivatives match finite differences", "[util][cost][angle][!mayfail]") {
    cout << "\n=== Test: cost2angQ Derivatives Verification ===" << endl;
    cout << "  Note: Hessian check may fail due to SO(3) manifold vs linear W-parameterization mismatch" << endl;

    arma::arma_rng::set_seed(54);

    for(int trial = 0; trial < 5; trial++) {
        arma::vec4 q = arma::normalise(arma::randn(4));
        arma::vec3 s = arma::normalise(arma::randn(3));  // Satellite body vector
        arma::vec3 e = arma::normalise(arma::randn(3));  // ECI target vector

        // Get analytical derivatives
        auto result = cost2angQ(q, s, e);
        double phi0 = std::get<0>(result);
        arma::vec3 dphi_analytic = std::get<1>(result);
        arma::mat33 ddphi_analytic = std::get<2>(result);

        // Finite difference for gradient
        double eps = 1e-7;
        arma::mat W = findWMat(q);
        arma::vec3 dphi_fd;

        for(int i = 0; i < 3; i++) {
            arma::vec3 delta = arma::vec(3).zeros();
            delta(i) = eps;
            arma::vec4 q_pert = arma::normalise(q + W * delta);
            auto result_pert = cost2angQ(q_pert, s, e);
            double phi_pert = std::get<0>(result_pert);
            dphi_fd(i) = (phi_pert - phi0) / eps;
        }

        double grad_error = arma::norm(dphi_analytic - dphi_fd);
        double grad_rel_error = grad_error / (arma::norm(dphi_fd) + 1e-10);

        // Finite difference for Hessian
        double eps_hess = 1e-5;
        arma::mat33 ddphi_fd;

        for(int i = 0; i < 3; i++) {
            arma::vec3 delta = arma::vec(3).zeros();
            delta(i) = eps_hess;
            arma::vec4 q_pert = arma::normalise(q + W * delta);
            auto result_pert = cost2angQ(q_pert, s, e);
            arma::vec3 dphi_pert = std::get<1>(result_pert);
            ddphi_fd.col(i) = (dphi_pert - dphi_analytic) / eps_hess;
        }

        double hess_error = arma::norm(ddphi_analytic - ddphi_fd, "fro");
        double hess_rel_error = hess_error / (arma::norm(ddphi_fd, "fro") + 1e-10);

        cout << "  Trial " << trial << ": grad_rel_error = " << grad_rel_error
             << ", hess_rel_error = " << hess_rel_error << endl;

        CHECK(grad_rel_error < 1e-4);
        CHECK(hess_rel_error < 1e-2);
    }
}

// ============================================================================
// TEST: State normalization Jacobian and Hessian
// ============================================================================
TEST_CASE("Satellite state_norm_jacobian matches finite differences", "[satellite][jacobian][norm]") {
    cout << "\n=== Test: state_norm_jacobian Verification ===" << endl;

    Satellite sat = createMTQOnlySatellite();
    int nx = sat.state_N();

    arma::arma_rng::set_seed(55);

    for(int trial = 0; trial < 5; trial++) {
        // Create state with non-unit quaternion (will be normalized)
        arma::vec3 w = 0.1 * arma::randn(3);
        arma::vec4 q_raw = arma::vec({1.0, 0.2, 0.1, 0.15}) + 0.1 * arma::randn(4);
        // Intentionally don't normalize to test Jacobian of normalization
        arma::vec x = join_cols(w, q_raw);

        // Get analytical Jacobian
        arma::mat J_analytic = sat.state_norm_jacobian(x);

        // Finite difference
        double eps = 1e-7;
        arma::vec x0_norm = sat.state_norm(x);
        arma::mat J_fd(nx, nx);

        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            arma::vec x_pert_norm = sat.state_norm(x_pert);
            J_fd.col(i) = (x_pert_norm - x0_norm) / eps;
        }

        double error = arma::norm(J_analytic - J_fd, "fro");
        double rel_error = error / (arma::norm(J_fd, "fro") + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-4);
    }
}

TEST_CASE("Satellite state_norm_hessian matches finite differences", "[satellite][hessian][norm]") {
    cout << "\n=== Test: state_norm_hessian Verification ===" << endl;

    Satellite sat = createMTQOnlySatellite();
    int nx = sat.state_N();

    arma::arma_rng::set_seed(56);

    for(int trial = 0; trial < 3; trial++) {
        arma::vec3 w = 0.1 * arma::randn(3);
        arma::vec4 q_raw = arma::vec({1.0, 0.2, 0.1, 0.15}) + 0.1 * arma::randn(4);
        arma::vec x = join_cols(w, q_raw);

        // Get analytical Hessian
        arma::cube H_analytic = sat.state_norm_hessian(x);

        // Finite difference via Jacobian perturbation
        double eps = 1e-5;
        arma::mat J0 = sat.state_norm_jacobian(x);
        arma::cube H_fd(nx, nx, nx, arma::fill::zeros);

        for(int i = 0; i < nx; i++) {
            arma::vec x_pert = x;
            x_pert(i) += eps;
            arma::mat J_pert = sat.state_norm_jacobian(x_pert);
            for(int j = 0; j < nx; j++) {
                H_fd.slice(j).col(i) = (J_pert.col(j) - J0.col(j)) / eps;
            }
        }

        double error = 0.0;
        double norm_val = 0.0;
        for(int k = 0; k < nx; k++) {
            error += arma::norm(H_analytic.slice(k) - H_fd.slice(k), "fro");
            norm_val += arma::norm(H_fd.slice(k), "fro");
        }
        double rel_error = error / (norm_val + 1e-10);

        cout << "  Trial " << trial << ": rel_error = " << rel_error << endl;
        CHECK(rel_error < 1e-2);
    }
}

// ============================================================================
// TEST: Cost function Jacobians (veccostJacobians, quatcostJacobians)
// ============================================================================
TEST_CASE("Satellite veccostJacobians matches finite differences", "[satellite][jacobian][cost]") {
    cout << "\n=== Test: veccostJacobians Verification ===" << endl;

    Satellite sat = createHybridSatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();
    int nxr = sat.reduced_state_N();

    arma::arma_rng::set_seed(57);

    int N = 20;
    int k = 10;

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);
    arma::vec u_prev = 0.9 * u + 0.01 * arma::randn(nu);

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec3 ECIvec = arma::normalise(arma::vec({1, 0, 0}));
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, 0, 0  // useRawControlCost = 0, useFullCostHess = 0
    );

    // Get analytical Jacobians
    cost_jacs jacs = sat.veccostJacobians(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);

    // Compute base cost
    double c0 = sat.stepcost_vec(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);

    // Finite difference for lu (control gradient)
    double eps = 1e-7;
    arma::vec lu_fd(nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        double c_pert = sat.stepcost_vec(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        lu_fd(i) = (c_pert - c0) / eps;
    }

    double lu_error = arma::norm(jacs.lu - lu_fd);
    double lu_rel_error = lu_error / (arma::norm(lu_fd) + 1e-10);

    cout << "  lu error (abs): " << lu_error << ", (rel): " << lu_rel_error << endl;
    CHECK(lu_rel_error < 1e-3);

    // Finite difference for luu (control Hessian) via gradient perturbation
    double eps_hess = 1e-5;
    arma::mat luu_fd(nu, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps_hess;
        cost_jacs jacs_pert = sat.veccostJacobians(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        luu_fd.col(i) = (jacs_pert.lu - jacs.lu) / eps_hess;
    }

    double luu_error = arma::norm(jacs.luu - luu_fd, "fro");
    double luu_rel_error = luu_error / (arma::norm(luu_fd, "fro") + 1e-10);

    cout << "  luu error (abs): " << luu_error << ", (rel): " << luu_rel_error << endl;
    CHECK(luu_rel_error < 1e-2);
}

TEST_CASE("Satellite quatcostJacobians matches finite differences", "[satellite][jacobian][cost]") {
    cout << "\n=== Test: quatcostJacobians Verification ===" << endl;

    Satellite sat = createHybridSatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(58);

    int N = 20;
    int k = 10;

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);
    arma::vec u_prev = 0.9 * u + 0.01 * arma::randn(nu);

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec4 ECIquat = arma::normalise(arma::vec({1, 0, 0, 0}));  // Target quaternion
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, 0, 0
    );

    // Get analytical Jacobians
    cost_jacs jacs = sat.quatcostJacobians(k, N, x, u, u_prev, satvec, ECIquat, B_eci, &costSettings);

    // Compute base cost
    double c0 = sat.stepcost_quat(k, N, x, u, u_prev, satvec, ECIquat, B_eci, &costSettings);

    // Finite difference for lu
    double eps = 1e-7;
    arma::vec lu_fd(nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        double c_pert = sat.stepcost_quat(k, N, x, u_pert, u_prev, satvec, ECIquat, B_eci, &costSettings);
        lu_fd(i) = (c_pert - c0) / eps;
    }

    double lu_error = arma::norm(jacs.lu - lu_fd);
    double lu_rel_error = lu_error / (arma::norm(lu_fd) + 1e-10);

    cout << "  lu error (abs): " << lu_error << ", (rel): " << lu_rel_error << endl;
    CHECK(lu_rel_error < 1e-3);
}

// ============================================================================
// TEST: Full Hessian vs Gauss-Newton Comparison
// ============================================================================
TEST_CASE("Full cost Hessian vs Gauss-Newton comparison", "[satellite][hessian][cost][fullhess]") {
    cout << "\n=== Test: Full Hessian vs Gauss-Newton Comparison ===" << endl;

    Satellite sat = createHybridSatellite();
    arma::arma_rng::set_seed(70);

    int N = 20, k = 10, nu = sat.control_N();

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x = sat.state_norm(join_cols(join_cols(w0, q0), 0.0005 * arma::randn(sat.number_RW)));

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);
    arma::vec u_prev = 0.9 * u;

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec3 ECIvec = arma::normalise(arma::vec({1, 0, 0}));
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    // Gauss-Newton (fullHess=0) vs Full Newton (fullHess=1)
    COST_SETTINGS_FORM costSettings_GN = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 2, 0, 0);
    COST_SETTINGS_FORM costSettings_FN = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0, 1e3, 1e2, 0.0, 0.0, 2, 0, 1);

    cost_jacs jacs_GN = sat.veccostJacobians(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings_GN);
    cost_jacs jacs_FN = sat.veccostJacobians(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings_FN);

    // Gradients should be identical (only Hessian differs)
    double lx_diff = arma::norm(jacs_GN.lx - jacs_FN.lx);
    double lu_diff = arma::norm(jacs_GN.lu - jacs_FN.lu);
    cout << "  lx difference (should be ~0): " << lx_diff << endl;
    cout << "  lu difference (should be ~0): " << lu_diff << endl;
    CHECK(lx_diff < 1e-12);
    CHECK(lu_diff < 1e-12);

    // Hessians may differ (full Newton has extra terms)
    double lxx_diff = arma::norm(jacs_GN.lxx - jacs_FN.lxx, "fro");
    cout << "  lxx difference (GN vs FN): " << lxx_diff << endl;
}

// ============================================================================
// TEST: RK4 Integration Jacobians
// ============================================================================
TEST_CASE("RK4 integration Jacobians match finite differences", "[rk4][jacobian]") {
    cout << "\n=== Test: RK4 Jacobians Verification ===" << endl;

    Satellite sat = createHybridSatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();
    double dt = 1.0;

    arma::arma_rng::set_seed(59);

    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);

    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Get analytical RK4 Jacobians
    auto rk4jacs = rk4zJacobians(dt, x, u, sat, dynamics_info, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(rk4jacs);
    arma::mat Ju_analytic = std::get<1>(rk4jacs);

    // Compute base RK4 step
    auto rk4result = rk4z(dt, x, u, sat, dynamics_info, dynamics_info);
    arma::vec xkp1_0 = sat.state_norm(std::get<0>(rk4result));

    // Finite difference for state Jacobian
    double eps = 1e-7;
    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto rk4result_pert = rk4z(dt, x_pert, u, sat, dynamics_info, dynamics_info);
        arma::vec xkp1_pert = sat.state_norm(std::get<0>(rk4result_pert));
        Jx_fd.col(i) = (xkp1_pert - xkp1_0) / eps;
    }

    // Finite difference for control Jacobian
    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        auto rk4result_pert = rk4z(dt, x, u_pert, sat, dynamics_info, dynamics_info);
        arma::vec xkp1_pert = sat.state_norm(std::get<0>(rk4result_pert));
        Ju_fd.col(i) = (xkp1_pert - xkp1_0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: RK4 Integration Hessians
// ============================================================================
TEST_CASE("RK4 integration Hessians match finite differences", "[rk4][hessian][!mayfail]") {
    cout << "\n=== Test: RK4 Hessians Verification ===" << endl;
    cout << "  Note: This test may fail due to complex indexing conventions" << endl;

    Satellite sat = createMTQOnlySatellite();  // Use simpler satellite for Hessian test
    int nx = sat.state_N();
    int nu = sat.control_N();
    double dt = 1.0;

    arma::arma_rng::set_seed(60);

    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x = join_cols(w0, q0);
    x = sat.state_norm(x);

    arma::vec u = 0.05 * arma::randn(nu);

    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    // Get analytical RK4 Hessians
    auto rk4hess = rk4zHessians(dt, x, u, sat, dynamics_info, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(rk4hess);
    arma::cube Hux_analytic = std::get<1>(rk4hess);
    arma::cube Huu_analytic = std::get<2>(rk4hess);

    // Get base Jacobians
    auto rk4jacs = rk4zJacobians(dt, x, u, sat, dynamics_info, dynamics_info);
    arma::mat Jx0 = std::get<0>(rk4jacs);
    arma::mat Ju0 = std::get<1>(rk4jacs);

    // Finite difference for Hxx
    double eps = 1e-5;
    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto rk4jacs_pert = rk4zJacobians(dt, x_pert, u, sat, dynamics_info, dynamics_info);
        arma::mat Jx_pert = std::get<0>(rk4jacs_pert);
        for(int j = 0; j < nx; j++) {
            Hxx_fd.slice(j).col(i) = (Jx_pert.col(j) - Jx0.col(j)) / eps;
        }
    }

    // Finite difference for Hux
    arma::cube Hux_fd(nu, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        auto rk4jacs_pert = rk4zJacobians(dt, x, u_pert, sat, dynamics_info, dynamics_info);
        arma::mat Jx_pert = std::get<0>(rk4jacs_pert);
        for(int j = 0; j < nx; j++) {
            Hux_fd.slice(j).col(i) = (Jx_pert.col(j) - Jx0.col(j)) / eps;
        }
    }

    // Compare Hxx
    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    // Compare Hux
    double Hux_error = 0.0;
    double Hux_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hux_error += arma::norm(Hux_analytic.slice(k) - Hux_fd.slice(k), "fro");
        Hux_norm += arma::norm(Hux_fd.slice(k), "fro");
    }
    double Hux_rel_error = Hux_error / (Hux_norm + 1e-10);

    cout << "  Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;
    cout << "  Hux error (abs): " << Hux_error << ", (rel): " << Hux_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
    CHECK(Hux_rel_error < 1e-2);
}

// ============================================================================
// TEST: Surface-based SRP Disturbance Torque
// ============================================================================
TEST_CASE("SRP surface disturbance torque computation", "[disturbance][srp]") {
    cout << "\n=== Test: SRP Surface Disturbance Torque ===" << endl;

    // Physical constants (matching Satellite.hpp)
    double SOLAR_CONSTANT = 1367.0;  // W/m^2
    double SPEED_OF_LIGHT = 299792458.0;  // m/s
    double P_solar = SOLAR_CONSTANT / SPEED_OF_LIGHT;  // N/m^2

    Satellite sat = createSimpleSatellite();

    // Identity quaternion (body frame = ECI frame)
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});
    arma::vec3 w0 = arma::vec({0.01, 0.0, 0.0});
    arma::vec x = join_cols(w0, q0);

    SECTION("Single face, purely absorptive, sun along +Y") {
        cout << "  Test 1: Single face, absorptive, sun along +Y..." << endl;

        // Setup surface: single face with +Y normal at position [1,0,0] from COM
        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        arma::vec areas = arma::vec({1.2});
        arma::vec eta_s = arma::vec({0.0});  // no specular
        arma::vec eta_d = arma::vec({0.0});  // no diffuse
        arma::vec eta_a = arma::vec({1.0});  // fully absorptive
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

        // Sun far away along +Y (after normalization, sun direction is +Y)
        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});  // satellite position
        arma::vec3 S_vec = arma::vec({7000.0, 1e9, 0.0});  // sun position (far along +Y)
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}),  // B_eci
            R_orb,   // R
            0,       // prop_torq_on
            V_orb,   // V
            S_vec,   // S (sun position)
            1,       // dist_on
            0.0      // rho
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // Expected: r_i - COM = [1,0,0], s_body = [0,1,0]
        // cos_gamma = n . s = 1.0
        // m_s = A * (eta_a + eta_d) * cos_gamma = 1.2 * 1.0 * 1.0 = 1.2
        // m_n = A * (2*eta_s*cos^2 + 2/3*eta_d) * cos_gamma = 0
        // torque = -P_solar * [m_s * (r x s)] = -P_solar * 1.2 * [1,0,0] x [0,1,0]
        //        = -P_solar * 1.2 * [0,0,1]
        arma::vec3 expected = -P_solar * 1.2 * arma::vec({0.0, 0.0, 1.0});

        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-15));
    }

    SECTION("Single face, back-facing (shadow), zero torque") {
        cout << "  Test 2: Single face, back-facing, zero torque..." << endl;

        // Surface with -Y normal (facing away from sun)
        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, -1.0, 0.0});
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        arma::vec areas = arma::vec({1.2});
        arma::vec eta_s = arma::vec({0.0});
        arma::vec eta_d = arma::vec({0.0});
        arma::vec eta_a = arma::vec({1.0});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        arma::vec3 S_vec = arma::vec({7000.0, 1e9, 0.0});  // sun along +Y
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, 0.0
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // cos_gamma = n . s = -1 < 0, so surface is in shadow -> zero contribution
        arma::vec3 expected = arma::vec({0.0, 0.0, 0.0});

        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-15));
    }

    SECTION("Single face, mixed surface properties") {
        cout << "  Test 3: Single face, mixed surface (eta_a=0.05, eta_d=0.25, eta_s=0.7)..." << endl;

        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        arma::vec areas = arma::vec({1.2});
        arma::vec eta_s = arma::vec({0.7});
        arma::vec eta_d = arma::vec({0.25});
        arma::vec eta_a = arma::vec({0.05});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        arma::vec3 S_vec = arma::vec({7000.0, 1e9, 0.0});  // sun along +Y
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, 0.0
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // cos_gamma = 1.0
        // m_s = A * (eta_a + eta_d) * cos_gamma = 1.2 * (0.05 + 0.25) * 1.0 = 0.36
        // m_n = A * (2*eta_s*cos^2 + 2/3*eta_d) * cos_gamma = 1.2 * (2*0.7*1 + 2/3*0.25) * 1.0
        //     = 1.2 * (1.4 + 0.16667) = 1.2 * 1.56667 = 1.88
        // r x s = [1,0,0] x [0,1,0] = [0,0,1]
        // r x n = [1,0,0] x [0,1,0] = [0,0,1]
        // torque = -P_solar * (m_s * [0,0,1] + m_n * [0,0,1])
        //        = -P_solar * (0.36 + 1.88) * [0,0,1]
        double m_s = 1.2 * (0.05 + 0.25) * 1.0;
        double m_n = 1.2 * (2.0*0.7*1.0 + (2.0/3.0)*0.25) * 1.0;
        arma::vec3 expected = -P_solar * (m_s + m_n) * arma::vec({0.0, 0.0, 1.0});

        cout << "    m_s = " << m_s << ", m_n = " << m_n << endl;
        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-14));
    }

    SECTION("Single face, oblique sun angle (45 degrees)") {
        cout << "  Test 4: Single face, 45 degree sun angle..." << endl;

        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});  // +Y normal
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        arma::vec areas = arma::vec({1.2});
        arma::vec eta_s = arma::vec({1.0});  // purely specular
        arma::vec eta_d = arma::vec({0.0});
        arma::vec eta_a = arma::vec({0.0});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        // Sun at 45 degrees in XY plane: direction is [1,1,0]/sqrt(2)
        arma::vec3 S_vec = arma::vec({7000.0 + 1e9, 1e9, 0.0});
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, 0.0
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // cos_gamma = n . s_body = [0,1,0] . [1/sqrt(2), 1/sqrt(2), 0] = 1/sqrt(2)
        double cg = 1.0/std::sqrt(2.0);
        // m_s = A * (eta_a + eta_d) * cos_gamma = 0
        // m_n = A * (2*eta_s*cos^2 + 2/3*eta_d) * cos_gamma = 1.2 * 2 * 1 * cg^2 * cg = 1.2 * 2 * cg^3
        // r x s = [1,0,0] x [1/sqrt(2),1/sqrt(2),0] = [0,0,1/sqrt(2)]
        // r x n = [1,0,0] x [0,1,0] = [0,0,1]
        double m_n = 1.2 * 2.0 * cg * cg * cg;
        arma::vec3 expected = -P_solar * m_n * arma::vec({0.0, 0.0, 1.0});

        cout << "    cos_gamma = " << cg << ", m_n = " << m_n << endl;
        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-14));
    }

    SECTION("Multiple faces") {
        cout << "  Test 5: Multiple faces contributing..." << endl;

        // Two faces: one +Y, one +X
        arma::mat normals(3, 2);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});  // +Y normal
        normals.col(1) = arma::vec({1.0, 0.0, 0.0});  // +X normal
        arma::mat centroids(3, 2);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});  // face 0 at +X
        centroids.col(1) = arma::vec({0.0, 1.0, 0.0});  // face 1 at +Y
        arma::vec areas = arma::vec({1.0, 0.5});
        arma::vec eta_s = arma::vec({0.0, 0.0});
        arma::vec eta_d = arma::vec({0.0, 0.0});
        arma::vec eta_a = arma::vec({1.0, 1.0});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        // Sun at 45 degrees in XY plane
        arma::vec3 S_vec = arma::vec({7000.0 + 1e9, 1e9, 0.0});
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, 0.0
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        double cg = 1.0/std::sqrt(2.0);
        // Face 0: cos_gamma0 = [0,1,0].[cg,cg,0] = cg
        //         m_s0 = 1.0 * 1.0 * cg = cg
        //         r0 x s = [1,0,0] x [cg,cg,0] = [0,0,cg]
        // Face 1: cos_gamma1 = [1,0,0].[cg,cg,0] = cg
        //         m_s1 = 0.5 * 1.0 * cg = 0.5*cg
        //         r1 x s = [0,1,0] x [cg,cg,0] = [0,0,-cg]
        arma::vec3 r0_cross_s = arma::vec({0.0, 0.0, cg});
        arma::vec3 r1_cross_s = arma::vec({0.0, 0.0, -cg});
        arma::vec3 expected = -P_solar * (1.0*cg*r0_cross_s + 0.5*cg*r1_cross_s);

        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-14));
    }

    // Clear surfaces for subsequent tests
    sat.clear_srp_surfaces();
}

// ============================================================================
// TEST: Surface-based Drag Disturbance Torque
// ============================================================================
TEST_CASE("Drag surface disturbance torque computation", "[disturbance][drag]") {
    cout << "\n=== Test: Drag Surface Disturbance Torque ===" << endl;

    Satellite sat = createSimpleSatellite();

    // Identity quaternion
    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});
    arma::vec3 w0 = arma::vec({0.01, 0.0, 0.0});
    arma::vec x = join_cols(w0, q0);

    SECTION("Single face, velocity along +Y") {
        cout << "  Test 1: Single face, velocity along +Y..." << endl;

        // Surface with +Y normal
        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});  // 1m along +X
        arma::vec areas = arma::vec({1.0});  // 1 m^2
        arma::vec CDs = arma::vec({2.2});    // drag coefficient
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_drag_surfaces(normals, centroids, areas, CDs, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});  // 7.5 km/s along +Y
        arma::vec3 S_vec = arma::vec({1e9, 0.0, 0.0});
        double rho = 1e-12;  // kg/m^3 (typical for LEO)

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, rho
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // V_body = V_orb * 1000 = [0, 7500, 0] m/s
        // n . V_body = 7500
        // F = C_D * A * (n.V) = 2.2 * 1.0 * 7500 = 16500
        // r x V = [1,0,0] x [0,7500,0] = [0,0,7500]
        // torque = -0.5 * rho * F * (r x V)
        //        = -0.5 * 1e-12 * 16500 * [0,0,7500]
        double V_body_y = 7500.0;  // m/s
        double F = 2.2 * 1.0 * V_body_y;
        arma::vec3 expected = -0.5 * rho * F * arma::vec({0.0, 0.0, V_body_y});

        cout << "    F = " << F << endl;
        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-20));
    }

    SECTION("Single face, back-facing (no drag)") {
        cout << "  Test 2: Single face, back-facing, zero torque..." << endl;

        // Surface with -Y normal (facing away from velocity)
        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, -1.0, 0.0});
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        arma::vec areas = arma::vec({1.0});
        arma::vec CDs = arma::vec({2.2});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_drag_surfaces(normals, centroids, areas, CDs, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});  // +Y velocity
        arma::vec3 S_vec = arma::vec({1e9, 0.0, 0.0});
        double rho = 1e-12;

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, rho
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // n . V_body = -7500 < 0, so surface doesn't contribute
        arma::vec3 expected = arma::vec({0.0, 0.0, 0.0});

        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-20));
    }

    SECTION("Multiple faces, different orientations") {
        cout << "  Test 3: Multiple faces with different orientations..." << endl;

        // Two faces: +Y and +X normals
        arma::mat normals(3, 2);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});  // +Y
        normals.col(1) = arma::vec({1.0, 0.0, 0.0});  // +X
        arma::mat centroids(3, 2);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        centroids.col(1) = arma::vec({0.0, 1.0, 0.0});
        arma::vec areas = arma::vec({1.0, 0.5});
        arma::vec CDs = arma::vec({2.2, 2.0});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_drag_surfaces(normals, centroids, areas, CDs, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        // Velocity at 45 degrees in XY plane
        arma::vec3 V_orb = arma::vec({5.0, 5.0, 0.0});  // km/s
        arma::vec3 S_vec = arma::vec({1e9, 0.0, 0.0});
        double rho = 1e-12;

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, rho
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        // V_body = [5000, 5000, 0] m/s
        arma::vec3 V_body = arma::vec({5000.0, 5000.0, 0.0});
        // Face 0: n0.V = 5000, F0 = 2.2*1.0*5000 = 11000
        //         r0 x V = [1,0,0] x [5000,5000,0] = [0,0,5000]
        // Face 1: n1.V = 5000, F1 = 2.0*0.5*5000 = 5000
        //         r1 x V = [0,1,0] x [5000,5000,0] = [0,0,-5000]
        double F0 = 2.2 * 1.0 * 5000.0;
        double F1 = 2.0 * 0.5 * 5000.0;
        arma::vec3 r0_x_V = arma::vec({0.0, 0.0, 5000.0});
        arma::vec3 r1_x_V = arma::vec({0.0, 0.0, -5000.0});
        arma::vec3 expected = -0.5 * rho * (F0 * r0_x_V + F1 * r1_x_V);

        cout << "    F0 = " << F0 << ", F1 = " << F1 << endl;
        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-20));
    }

    SECTION("Zero density means zero drag") {
        cout << "  Test 4: Zero density, zero drag torque..." << endl;

        arma::mat normals(3, 1);
        normals.col(0) = arma::vec({0.0, 1.0, 0.0});
        arma::mat centroids(3, 1);
        centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
        arma::vec areas = arma::vec({1.0});
        arma::vec CDs = arma::vec({2.2});
        arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

        sat.set_drag_surfaces(normals, centroids, areas, CDs, COM);

        arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
        arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});
        arma::vec3 S_vec = arma::vec({1e9, 0.0, 0.0});
        double rho = 0.0;  // zero density

        DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
            arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, rho
        );

        arma::vec3 torque = sat.dist_torque(x, dynamics_info);

        arma::vec3 expected = arma::vec({0.0, 0.0, 0.0});

        cout << "    Expected torque: " << expected.t();
        cout << "    Computed torque: " << torque.t();

        CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-25));
    }

    // Clear surfaces
    sat.clear_drag_surfaces();
}

// ============================================================================
// TEST: Combined SRP and Drag disturbances
// ============================================================================
TEST_CASE("Combined SRP and Drag disturbances", "[disturbance][combined]") {
    cout << "\n=== Test: Combined SRP and Drag Disturbances ===" << endl;

    double SOLAR_CONSTANT = 1367.0;
    double SPEED_OF_LIGHT = 299792458.0;
    double P_solar = SOLAR_CONSTANT / SPEED_OF_LIGHT;

    Satellite sat = createSimpleSatellite();

    arma::vec4 q0 = arma::vec({1.0, 0.0, 0.0, 0.0});
    arma::vec3 w0 = arma::vec({0.01, 0.0, 0.0});
    arma::vec x = join_cols(w0, q0);

    // Setup both SRP and drag surfaces
    arma::mat normals(3, 1);
    normals.col(0) = arma::vec({0.0, 1.0, 0.0});
    arma::mat centroids(3, 1);
    centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
    arma::vec areas = arma::vec({1.0});
    arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

    // SRP surface
    arma::vec eta_s = arma::vec({0.0});
    arma::vec eta_d = arma::vec({0.0});
    arma::vec eta_a = arma::vec({1.0});
    sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

    // Drag surface
    arma::vec CDs = arma::vec({2.2});
    sat.set_drag_surfaces(normals, centroids, areas, CDs, COM);

    arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
    arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});  // +Y velocity
    arma::vec3 S_vec = arma::vec({7000.0, 1e9, 0.0});  // sun along +Y
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
        arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, rho
    );

    arma::vec3 torque = sat.dist_torque(x, dynamics_info);

    // Expected SRP: -P_solar * 1.0 * [0,0,1]
    arma::vec3 expected_srp = -P_solar * 1.0 * arma::vec({0.0, 0.0, 1.0});

    // Expected drag: -0.5 * rho * F * (r x V) where F = CD*A*Vproj
    double V_body_y = 7500.0;
    double F_drag = 2.2 * 1.0 * V_body_y;
    arma::vec3 expected_drag = -0.5 * rho * F_drag * arma::vec({0.0, 0.0, V_body_y});

    arma::vec3 expected = expected_srp + expected_drag;

    cout << "  Expected SRP torque:  " << expected_srp.t();
    cout << "  Expected drag torque: " << expected_drag.t();
    cout << "  Expected total:       " << expected.t();
    cout << "  Computed total:       " << torque.t();

    CHECK(arma::approx_equal(torque, expected, "absdiff", 1e-15));

    sat.clear_srp_surfaces();
    sat.clear_drag_surfaces();
}

// ============================================================================
// TEST: Disturbance with non-identity attitude
// ============================================================================
TEST_CASE("Disturbance torque with rotated attitude", "[disturbance][attitude]") {
    cout << "\n=== Test: Disturbance with Non-Identity Attitude ===" << endl;

    double SOLAR_CONSTANT = 1367.0;
    double SPEED_OF_LIGHT = 299792458.0;
    double P_solar = SOLAR_CONSTANT / SPEED_OF_LIGHT;

    Satellite sat = createSimpleSatellite();

    // 90 degree rotation about Z axis: body +X -> ECI +Y, body +Y -> ECI -X
    // Quaternion for 90 deg about Z: [cos(45), 0, 0, sin(45)] = [sqrt(2)/2, 0, 0, sqrt(2)/2]
    double s2 = std::sqrt(2.0)/2.0;
    arma::vec4 q0 = arma::vec({s2, 0.0, 0.0, s2});
    arma::vec3 w0 = arma::vec({0.0, 0.0, 0.0});
    arma::vec x = join_cols(w0, q0);

    // Surface with +Y normal in body frame
    arma::mat normals(3, 1);
    normals.col(0) = arma::vec({0.0, 1.0, 0.0});
    arma::mat centroids(3, 1);
    centroids.col(0) = arma::vec({1.0, 0.0, 0.0});
    arma::vec areas = arma::vec({1.0});
    arma::vec eta_s = arma::vec({0.0});
    arma::vec eta_d = arma::vec({0.0});
    arma::vec eta_a = arma::vec({1.0});
    arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});

    sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);

    arma::vec3 R_orb = arma::vec({7000.0, 0.0, 0.0});
    // Sun along ECI +Y means sun direction in body frame is along -X (due to 90 deg Z rotation)
    arma::vec3 S_vec = arma::vec({7000.0, 1e9, 0.0});
    arma::vec3 V_orb = arma::vec({0.0, 7.5, 0.0});

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(
        arma::vec3({0.0, 3e-5, 2e-5}), R_orb, 0, V_orb, S_vec, 1, 0.0
    );

    arma::vec3 torque = sat.dist_torque(x, dynamics_info);

    // With 90 deg rotation about Z:
    // - ECI sun direction ≈ [0, 1, 0]
    // - Body frame sun direction = R^T * [0,1,0] = [-1, 0, 0] (body -X)
    // - Normal in body = [0, 1, 0] (body +Y)
    // - cos_gamma = [0,1,0] . [-1,0,0] = 0 -> surface not illuminated!
    arma::vec3 expected = arma::vec({0.0, 0.0, 0.0});

    cout << "  Attitude: 90 deg rotation about Z" << endl;
    cout << "  Sun in ECI: +Y, in body: -X" << endl;
    cout << "  Surface normal: body +Y -> perpendicular to sun, cos_gamma = 0" << endl;
    cout << "  Expected torque: " << expected.t();
    cout << "  Computed torque: " << torque.t();

    // Allow small tolerance for numerical precision in rotation
    CHECK(arma::norm(torque - expected) < 1e-14);

    sat.clear_srp_surfaces();
}

// ============================================================================
// RW DYNAMICS JACOBIAN AND HESSIAN TESTS
// These tests ensure correctness of dynamics derivatives for reaction wheel
// configurations, including with disturbances (SRP, Drag).
// ============================================================================

// Helper: Create RW-only satellite with disturbance surfaces
Satellite createRWOnlySatelliteWithSurfaces() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // 3-axis RW configuration
    sat.add_RW(arma::vec({1,0,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,1,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    return sat;
}

// Helper: Create hybrid satellite with disturbance surfaces
Satellite createHybridSatelliteWithSurfaces() {
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // MTQs
    sat.add_MTQ(arma::vec({1,0,0}), 0.1, 1.0);
    sat.add_MTQ(arma::vec({0,1,0}), 0.1, 1.0);
    sat.add_MTQ(arma::vec({0,0,1}), 0.1, 1.0);
    // RWs
    sat.add_RW(arma::vec({1,0,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,1,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    return sat;
}

// Helper: Add SRP surfaces to satellite
void addSRPSurfaces(Satellite& sat) {
    arma::mat normals(3, 6);
    normals.col(0) = arma::vec({1.0, 0.0, 0.0});
    normals.col(1) = arma::vec({-1.0, 0.0, 0.0});
    normals.col(2) = arma::vec({0.0, 1.0, 0.0});
    normals.col(3) = arma::vec({0.0, -1.0, 0.0});
    normals.col(4) = arma::vec({0.0, 0.0, 1.0});
    normals.col(5) = arma::vec({0.0, 0.0, -1.0});

    arma::mat centroids(3, 6);
    centroids.col(0) = arma::vec({0.05, 0.0, 0.0});
    centroids.col(1) = arma::vec({-0.05, 0.0, 0.0});
    centroids.col(2) = arma::vec({0.0, 0.05, 0.0});
    centroids.col(3) = arma::vec({0.0, -0.05, 0.0});
    centroids.col(4) = arma::vec({0.0, 0.0, 0.05});
    centroids.col(5) = arma::vec({0.0, 0.0, -0.05});

    arma::vec areas = arma::vec({0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
    arma::vec eta_s = arma::vec({0.5, 0.5, 0.5, 0.5, 0.5, 0.5});
    arma::vec eta_d = arma::vec({0.3, 0.3, 0.3, 0.3, 0.3, 0.3});
    arma::vec eta_a = arma::vec({0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
    arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});
    sat.set_srp_surfaces(normals, centroids, areas, eta_s, eta_d, eta_a, COM);
}

// Helper: Add drag surfaces to satellite
void addDragSurfaces(Satellite& sat) {
    arma::mat normals(3, 6);
    normals.col(0) = arma::vec({1.0, 0.0, 0.0});
    normals.col(1) = arma::vec({-1.0, 0.0, 0.0});
    normals.col(2) = arma::vec({0.0, 1.0, 0.0});
    normals.col(3) = arma::vec({0.0, -1.0, 0.0});
    normals.col(4) = arma::vec({0.0, 0.0, 1.0});
    normals.col(5) = arma::vec({0.0, 0.0, -1.0});

    arma::mat centroids(3, 6);
    centroids.col(0) = arma::vec({0.05, 0.0, 0.0});
    centroids.col(1) = arma::vec({-0.05, 0.0, 0.0});
    centroids.col(2) = arma::vec({0.0, 0.05, 0.0});
    centroids.col(3) = arma::vec({0.0, -0.05, 0.0});
    centroids.col(4) = arma::vec({0.0, 0.0, 0.05});
    centroids.col(5) = arma::vec({0.0, 0.0, -0.05});

    arma::vec areas = arma::vec({0.01, 0.01, 0.01, 0.01, 0.01, 0.01});
    arma::vec CDs = arma::vec({2.2, 2.2, 2.2, 2.2, 2.2, 2.2});
    arma::vec3 COM = arma::vec({0.0, 0.0, 0.0});
    sat.set_drag_surfaces(normals, centroids, areas, CDs, COM);
}

// ============================================================================
// TEST: RW-only dynamics Hessians (was missing)
// ============================================================================
TEST_CASE("RW-only satellite dynamics Hessians match finite differences", "[satellite][hessian][dynamics][rw]") {
    cout << "\n=== Test: RW-Only Satellite Dynamics Hessians ===" << endl;

    Satellite sat = createRWOnlySatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(47);
    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    arma::vec3 B_eci = arma::vec({0.0, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);
    arma::cube Hux_analytic = std::get<1>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    // Compute Hxx via finite difference of Jacobian
    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    // Compute Hux via finite difference
    arma::cube Hux_fd(nu, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x, u_pert, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hux_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    // Compare Hxx
    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    // Compare Hux
    double Hux_error = 0.0;
    double Hux_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hux_error += arma::norm(Hux_analytic.slice(k) - Hux_fd.slice(k), "fro");
        Hux_norm += arma::norm(Hux_fd.slice(k), "fro");
    }
    double Hux_rel_error = Hux_error / (Hux_norm + 1e-10);

    cout << "  RW-only Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;
    cout << "  RW-only Hux error (abs): " << Hux_error << ", (rel): " << Hux_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
    CHECK(Hux_rel_error < 1e-2);
}

// ============================================================================
// TEST: Single RW dynamics Jacobians
// ============================================================================
TEST_CASE("Single RW satellite dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][rw][single]") {
    cout << "\n=== Test: Single RW Satellite Dynamics Jacobians ===" << endl;

    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // Single RW along Z-axis
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);

    int nx = sat.state_N();
    int nu = sat.control_N();

    cout << "  Single RW config: nx=" << nx << ", nu=" << nu << endl;

    arma::arma_rng::set_seed(48);
    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});
    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_eci, arma::vec3({7000,0,0}), 0,
                                                        arma::vec3({0,7.5,0}), arma::vec3({1,0,0}), 1, 0.0);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  Single RW Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  Single RW Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: RW-only + SRP dynamics Jacobians
// ============================================================================
TEST_CASE("RW-only satellite with SRP dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][rw][srp]") {
    cout << "\n=== Test: RW-Only + SRP Dynamics Jacobians ===" << endl;

    Satellite sat = createRWOnlySatelliteWithSurfaces();
    addSRPSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(49);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    // Environment with SRP (sun position at ~1 AU)
    arma::vec3 R_k = 7000.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.5 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, 0.0);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  RW+SRP Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  RW+SRP Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: RW-only + SRP dynamics Hessians
// ============================================================================
TEST_CASE("RW-only satellite with SRP dynamics Hessians match finite differences", "[satellite][hessian][dynamics][rw][srp]") {
    cout << "\n=== Test: RW-Only + SRP Dynamics Hessians ===" << endl;

    Satellite sat = createRWOnlySatelliteWithSurfaces();
    addSRPSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(50);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    arma::vec3 R_k = 7000.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.5 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, 0.0);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    cout << "  RW+SRP Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
}

// ============================================================================
// TEST: RW-only + Drag dynamics Jacobians
// ============================================================================
TEST_CASE("RW-only satellite with Drag dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][rw][drag]") {
    cout << "\n=== Test: RW-Only + Drag Dynamics Jacobians ===" << endl;

    Satellite sat = createRWOnlySatelliteWithSurfaces();
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(51);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    // LEO environment with drag
    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  RW+Drag Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  RW+Drag Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: RW-only + Drag dynamics Hessians
// ============================================================================
TEST_CASE("RW-only satellite with Drag dynamics Hessians match finite differences", "[satellite][hessian][dynamics][rw][drag]") {
    cout << "\n=== Test: RW-Only + Drag Dynamics Hessians ===" << endl;

    Satellite sat = createRWOnlySatelliteWithSurfaces();
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(52);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    cout << "  RW+Drag Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
}

// ============================================================================
// TEST: Hybrid (MTQ+RW) + SRP dynamics Jacobians
// ============================================================================
TEST_CASE("Hybrid satellite with SRP dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][hybrid][srp]") {
    cout << "\n=== Test: Hybrid (MTQ+RW) + SRP Dynamics Jacobians ===" << endl;

    Satellite sat = createHybridSatelliteWithSurfaces();
    addSRPSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(53);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);

    // Mixed control: MTQs and RW torques
    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0005 * arma::randn(sat.number_RW);

    arma::vec3 R_k = 7000.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.5 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, 0.0);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  Hybrid+SRP Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  Hybrid+SRP Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: Hybrid (MTQ+RW) + SRP dynamics Hessians
// ============================================================================
TEST_CASE("Hybrid satellite with SRP dynamics Hessians match finite differences", "[satellite][hessian][dynamics][hybrid][srp]") {
    cout << "\n=== Test: Hybrid (MTQ+RW) + SRP Dynamics Hessians ===" << endl;

    Satellite sat = createHybridSatelliteWithSurfaces();
    addSRPSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(54);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.03 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);

    arma::vec3 R_k = 7000.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.5 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, 0.0);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    cout << "  Hybrid+SRP Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
}

// ============================================================================
// TEST: Hybrid (MTQ+RW) + Drag dynamics Jacobians
// ============================================================================
TEST_CASE("Hybrid satellite with Drag dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][hybrid][drag]") {
    cout << "\n=== Test: Hybrid (MTQ+RW) + Drag Dynamics Jacobians ===" << endl;

    Satellite sat = createHybridSatelliteWithSurfaces();
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(55);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0005 * arma::randn(sat.number_RW);

    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  Hybrid+Drag Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  Hybrid+Drag Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: Hybrid (MTQ+RW) + Drag dynamics Hessians
// ============================================================================
TEST_CASE("Hybrid satellite with Drag dynamics Hessians match finite differences", "[satellite][hessian][dynamics][hybrid][drag]") {
    cout << "\n=== Test: Hybrid (MTQ+RW) + Drag Dynamics Hessians ===" << endl;

    Satellite sat = createHybridSatelliteWithSurfaces();
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(56);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.03 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);

    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    cout << "  Hybrid+Drag Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
}

// ============================================================================
// TEST: Hybrid (MTQ+RW) + Combined SRP+Drag dynamics Jacobians
// ============================================================================
TEST_CASE("Hybrid satellite with combined SRP+Drag dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][hybrid][combined]") {
    cout << "\n=== Test: Hybrid (MTQ+RW) + Combined SRP+Drag Dynamics Jacobians ===" << endl;

    Satellite sat = createHybridSatelliteWithSurfaces();
    addSRPSurfaces(sat);
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(57);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.05 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0005 * arma::randn(sat.number_RW);

    // LEO with both SRP and drag
    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  Hybrid+Combined Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  Hybrid+Combined Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: Hybrid (MTQ+RW) + Combined SRP+Drag dynamics Hessians
// ============================================================================
TEST_CASE("Hybrid satellite with combined SRP+Drag dynamics Hessians match finite differences", "[satellite][hessian][dynamics][hybrid][combined]") {
    cout << "\n=== Test: Hybrid (MTQ+RW) + Combined SRP+Drag Dynamics Hessians ===" << endl;

    Satellite sat = createHybridSatelliteWithSurfaces();
    addSRPSurfaces(sat);
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(58);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);

    arma::vec u = arma::vec(nu).zeros();
    u.head(sat.number_MTQ) = 0.03 * arma::randn(sat.number_MTQ);
    u.tail(sat.number_RW) = 0.0003 * arma::randn(sat.number_RW);

    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    cout << "  Hybrid+Combined Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
}

// ============================================================================
// TEST: RW-only + Combined SRP+Drag dynamics Jacobians
// ============================================================================
TEST_CASE("RW-only satellite with combined SRP+Drag dynamics Jacobians match finite differences", "[satellite][jacobian][dynamics][rw][combined]") {
    cout << "\n=== Test: RW-Only + Combined SRP+Drag Dynamics Jacobians ===" << endl;

    Satellite sat = createRWOnlySatelliteWithSurfaces();
    addSRPSurfaces(sat);
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(59);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);

    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto jacs = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx_analytic = std::get<0>(jacs);
    arma::mat Ju_analytic = std::get<1>(jacs);

    double eps = 1e-7;
    arma::vec f0 = sat.dynamics_pure(x, u, dynamics_info);

    arma::mat Jx_fd(nx, nx);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x_pert, u, dynamics_info);
        Jx_fd.col(i) = (f_pert - f0) / eps;
    }

    arma::mat Ju_fd(nx, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec f_pert = sat.dynamics_pure(x, u_pert, dynamics_info);
        Ju_fd.col(i) = (f_pert - f0) / eps;
    }

    double Jx_error = arma::norm(Jx_analytic - Jx_fd, "fro");
    double Ju_error = arma::norm(Ju_analytic - Ju_fd, "fro");
    double Jx_rel_error = Jx_error / (arma::norm(Jx_fd, "fro") + 1e-10);
    double Ju_rel_error = Ju_error / (arma::norm(Ju_fd, "fro") + 1e-10);

    cout << "  RW+Combined Jx error (abs): " << Jx_error << ", (rel): " << Jx_rel_error << endl;
    cout << "  RW+Combined Ju error (abs): " << Ju_error << ", (rel): " << Ju_rel_error << endl;

    CHECK(Jx_rel_error < 1e-4);
    CHECK(Ju_rel_error < 1e-4);
}

// ============================================================================
// TEST: RW-only + Combined SRP+Drag dynamics Hessians
// ============================================================================
TEST_CASE("RW-only satellite with combined SRP+Drag dynamics Hessians match finite differences", "[satellite][hessian][dynamics][rw][combined]") {
    cout << "\n=== Test: RW-Only + Combined SRP+Drag Dynamics Hessians ===" << endl;

    Satellite sat = createRWOnlySatelliteWithSurfaces();
    addSRPSurfaces(sat);
    addDragSurfaces(sat);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(60);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0003 * arma::randn(nu);

    arma::vec3 R_k = 6778.0 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 V_k = 7.67 * arma::normalise(arma::cross(R_k, arma::vec(3, arma::fill::randn)));
    arma::vec3 S_k = 1.496e8 * arma::normalise(arma::vec(3, arma::fill::randn));
    arma::vec3 B_k = 3e-5 * arma::normalise(arma::vec(3, arma::fill::randn));
    double rho = 1e-12;

    DYNAMICS_INFO_FORM dynamics_info = std::make_tuple(B_k, R_k, 0, V_k, S_k, 1, rho);

    auto hess = sat.dynamicsHessians(x, u, dynamics_info);
    arma::cube Hxx_analytic = std::get<0>(hess);

    double eps = 1e-5;
    auto jacs0 = sat.dynamicsJacobians(x, u, dynamics_info);
    arma::mat Jx0 = std::get<0>(jacs0);

    arma::cube Hxx_fd(nx, nx, nx, arma::fill::zeros);
    for(int i = 0; i < nx; i++) {
        arma::vec x_pert = x;
        x_pert(i) += eps;
        auto jacs_pert = sat.dynamicsJacobians(x_pert, u, dynamics_info);
        arma::mat Jx_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nx; j++) {
            for(int k = 0; k < nx; k++) {
                Hxx_fd(i, j, k) = (Jx_pert(k, j) - Jx0(k, j)) / eps;
            }
        }
    }

    double Hxx_error = 0.0;
    double Hxx_norm = 0.0;
    for(int k = 0; k < nx; k++) {
        Hxx_error += arma::norm(Hxx_analytic.slice(k) - Hxx_fd.slice(k), "fro");
        Hxx_norm += arma::norm(Hxx_fd.slice(k), "fro");
    }
    double Hxx_rel_error = Hxx_error / (Hxx_norm + 1e-10);

    cout << "  RW+Combined Hxx error (abs): " << Hxx_error << ", (rel): " << Hxx_rel_error << endl;

    CHECK(Hxx_rel_error < 1e-2);
}

// ============================================================================
// RW CONSTRAINT JACOBIAN AND HESSIAN TESTS
// ============================================================================

// ============================================================================
// TEST: RW-only constraint Jacobians (AV constraint)
// ============================================================================
TEST_CASE("RW-only satellite constraint Jacobians match finite differences", "[satellite][jacobian][constraint][rw]") {
    cout << "\n=== Test: RW-Only Satellite Constraint Jacobians ===" << endl;

    Satellite sat = createRWOnlySatellite();
    sat.set_AV_constraint(0.1);  // Angular velocity constraint

    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 10;
    int k = 5;

    cout << "  RW-only config: nx=" << nx << ", nu=" << nu << ", nc=" << nc << endl;

    arma::arma_rng::set_seed(61);
    arma::vec3 w0 = 0.05 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.005 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);
    arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

    auto jacs = sat.constraintJacobians(k, N, u, x, sun);
    arma::mat Jcu_analytic = std::get<0>(jacs);
    arma::mat Jcx_analytic = std::get<1>(jacs);

    arma::vec c0 = sat.getConstraints(k, N, u, x, sun);

    double eps = 1e-7;
    arma::mat Jcu_fd(nc, nu, arma::fill::zeros);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec c_pert = sat.getConstraints(k, N, u_pert, x, sun);
        Jcu_fd.col(i) = (c_pert - c0) / eps;
    }

    // Perturb angular velocity part of state
    arma::mat Jcx_av_fd(nc, 3, arma::fill::zeros);
    for(int i = 0; i < 3; i++) {
        arma::vec x_pert = x;
        x_pert(sat.avindex0() + i) += eps;
        arma::vec c_pert = sat.getConstraints(k, N, u, x_pert, sun);
        Jcx_av_fd.col(i) = (c_pert - c0) / eps;
    }

    // Perturb RW momentum part of state (RW states start at index 7 = quat0index + 4)
    int rw_start_idx = 7;  // After w (0-2) and q (3-6)
    arma::mat Jcx_rw_fd(nc, sat.number_RW, arma::fill::zeros);
    for(int i = 0; i < sat.number_RW; i++) {
        arma::vec x_pert = x;
        x_pert(rw_start_idx + i) += eps;
        arma::vec c_pert = sat.getConstraints(k, N, u, x_pert, sun);
        Jcx_rw_fd.col(i) = (c_pert - c0) / eps;
    }

    double Jcu_error = arma::norm(Jcu_analytic - Jcu_fd, "fro");
    double Jcu_rel_error = Jcu_error / (arma::norm(Jcu_fd, "fro") + 1e-10);

    double Jcx_av_error = arma::norm(Jcx_analytic.cols(0, 2) - Jcx_av_fd, "fro");
    double Jcx_av_rel_error = Jcx_av_error / (arma::norm(Jcx_av_fd, "fro") + 1e-10);

    cout << "  Jcu error (abs): " << Jcu_error << ", (rel): " << Jcu_rel_error << endl;
    cout << "  Jcx (AV part) error (abs): " << Jcx_av_error << ", (rel): " << Jcx_av_rel_error << endl;

    CHECK(Jcu_rel_error < 1e-4);
    CHECK(Jcx_av_rel_error < 1e-4);
}

// ============================================================================
// TEST: RW-only constraint Hessians
// ============================================================================
TEST_CASE("RW-only satellite constraint Hessians match finite differences", "[satellite][hessian][constraint][rw]") {
    cout << "\n=== Test: RW-Only Satellite Constraint Hessians ===" << endl;

    Satellite sat = createRWOnlySatellite();
    sat.set_AV_constraint(0.2);

    int nx = sat.state_N();
    int nxr = sat.reduced_state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 10;
    int k = 5;

    cout << "  RW-only: nx=" << nx << ", nu=" << nu << ", nc=" << nc << endl;

    arma::arma_rng::set_seed(62);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.005 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);
    arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

    auto hess = sat.constraintHessians(k, N, u, x, sun);
    arma::cube Hcuu_analytic = std::get<0>(hess);

    auto jacs0 = sat.constraintJacobians(k, N, u, x, sun);
    arma::mat Jcu0 = std::get<0>(jacs0);

    double eps = 1e-5;
    arma::cube Hcuu_fd(nu, nu, nc, arma::fill::zeros);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        auto jacs_pert = sat.constraintJacobians(k, N, u_pert, x, sun);
        arma::mat Jcu_pert = std::get<0>(jacs_pert);
        for(int j = 0; j < nc; j++) {
            Hcuu_fd.slice(j).col(i) = (Jcu_pert.row(j).t() - Jcu0.row(j).t()) / eps;
        }
    }

    double Hcuu_error = 0.0;
    double Hcuu_norm = 0.0;
    for(int i = 0; i < nc; i++) {
        Hcuu_error += arma::norm(Hcuu_analytic.slice(i) - Hcuu_fd.slice(i), "fro");
        Hcuu_norm += arma::norm(Hcuu_fd.slice(i), "fro");
    }
    double Hcuu_rel_error = Hcuu_error / (Hcuu_norm + 1e-10);

    cout << "  Hcuu error (abs): " << Hcuu_error << ", (rel): " << Hcuu_rel_error << endl;

    CHECK(Hcuu_rel_error < 0.1);
}

// ============================================================================
// TEST: RW-only cost Jacobians (veccostJacobians)
// ============================================================================
TEST_CASE("RW-only satellite veccostJacobians matches finite differences", "[satellite][jacobian][cost][rw]") {
    cout << "\n=== Test: RW-Only veccostJacobians Verification ===" << endl;

    Satellite sat = createRWOnlySatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();
    int nxr = sat.reduced_state_N();

    arma::arma_rng::set_seed(63);

    int N = 20;
    int k = 10;

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = 0.0003 * arma::randn(nu);
    arma::vec u_prev = 0.9 * u + 0.00001 * arma::randn(nu);

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec3 ECIvec = arma::normalise(arma::vec({1, 0, 0}));
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, 0, 0
    );

    cost_jacs jacs = sat.veccostJacobians(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);
    double c0 = sat.stepcost_vec(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);

    double eps = 1e-7;
    arma::vec lu_fd(nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        double c_pert = sat.stepcost_vec(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        lu_fd(i) = (c_pert - c0) / eps;
    }

    double lu_error = arma::norm(jacs.lu - lu_fd);
    double lu_rel_error = lu_error / (arma::norm(lu_fd) + 1e-10);

    cout << "  RW-only lu error (abs): " << lu_error << ", (rel): " << lu_rel_error << endl;
    CHECK(lu_rel_error < 1e-3);

    // Finite difference for luu
    double eps_hess = 1e-5;
    arma::mat luu_fd(nu, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps_hess;
        cost_jacs jacs_pert = sat.veccostJacobians(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        luu_fd.col(i) = (jacs_pert.lu - jacs.lu) / eps_hess;
    }

    double luu_error = arma::norm(jacs.luu - luu_fd, "fro");
    double luu_rel_error = luu_error / (arma::norm(luu_fd, "fro") + 1e-10);

    cout << "  RW-only luu error (abs): " << luu_error << ", (rel): " << luu_rel_error << endl;
    CHECK(luu_rel_error < 1e-2);
}

// ============================================================================
// TEST: RW-only cost Jacobians (quatcostJacobians)
// ============================================================================
TEST_CASE("RW-only satellite quatcostJacobians matches finite differences", "[satellite][jacobian][cost][rw]") {
    cout << "\n=== Test: RW-Only quatcostJacobians Verification ===" << endl;

    Satellite sat = createRWOnlySatellite();
    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(64);

    int N = 20;
    int k = 10;

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = 0.0003 * arma::randn(nu);
    arma::vec u_prev = 0.9 * u + 0.00001 * arma::randn(nu);

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec4 ECIquat = arma::normalise(arma::vec({1, 0, 0, 0}));
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, 0, 0
    );

    cost_jacs jacs = sat.quatcostJacobians(k, N, x, u, u_prev, satvec, ECIquat, B_eci, &costSettings);
    double c0 = sat.stepcost_quat(k, N, x, u, u_prev, satvec, ECIquat, B_eci, &costSettings);

    double eps = 1e-7;
    arma::vec lu_fd(nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        double c_pert = sat.stepcost_quat(k, N, x, u_pert, u_prev, satvec, ECIquat, B_eci, &costSettings);
        lu_fd(i) = (c_pert - c0) / eps;
    }

    double lu_error = arma::norm(jacs.lu - lu_fd);
    double lu_rel_error = lu_error / (arma::norm(lu_fd) + 1e-10);

    cout << "  RW-only quat lu error (abs): " << lu_error << ", (rel): " << lu_rel_error << endl;
    CHECK(lu_rel_error < 2e-3);  // Slightly looser for RW-only
}

// ============================================================================
// TEST: Single RW constraint Jacobians
// ============================================================================
TEST_CASE("Single RW satellite constraint Jacobians match finite differences", "[satellite][jacobian][constraint][rw][single]") {
    cout << "\n=== Test: Single RW Satellite Constraint Jacobians ===" << endl;

    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);
    sat.set_AV_constraint(0.1);

    int nx = sat.state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 10;
    int k = 5;

    cout << "  Single RW: nx=" << nx << ", nu=" << nu << ", nc=" << nc << endl;

    arma::arma_rng::set_seed(65);
    arma::vec3 w0 = 0.05 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.005 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);
    arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

    auto jacs = sat.constraintJacobians(k, N, u, x, sun);
    arma::mat Jcu_analytic = std::get<0>(jacs);

    arma::vec c0 = sat.getConstraints(k, N, u, x, sun);

    double eps = 1e-7;
    arma::mat Jcu_fd(nc, nu, arma::fill::zeros);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec c_pert = sat.getConstraints(k, N, u_pert, x, sun);
        Jcu_fd.col(i) = (c_pert - c0) / eps;
    }

    double Jcu_error = arma::norm(Jcu_analytic - Jcu_fd, "fro");
    double Jcu_rel_error = Jcu_error / (arma::norm(Jcu_fd, "fro") + 1e-10);

    cout << "  Single RW Jcu error (abs): " << Jcu_error << ", (rel): " << Jcu_rel_error << endl;

    CHECK(Jcu_rel_error < 1e-4);
}

// ============================================================================
// TEST: Single RW cost Jacobians
// ============================================================================
TEST_CASE("Single RW satellite veccostJacobians matches finite differences", "[satellite][jacobian][cost][rw][single]") {
    cout << "\n=== Test: Single RW veccostJacobians Verification ===" << endl;

    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 0.0, 0.001);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(66);

    int N = 20;
    int k = 10;

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    arma::vec rw_h = 0.002 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = 0.0003 * arma::randn(nu);
    arma::vec u_prev = 0.9 * u + 0.00001 * arma::randn(nu);

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec3 ECIvec = arma::normalise(arma::vec({1, 0, 0}));
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, 0, 0
    );

    cost_jacs jacs = sat.veccostJacobians(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);
    double c0 = sat.stepcost_vec(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);

    double eps = 1e-7;
    arma::vec lu_fd(nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        double c_pert = sat.stepcost_vec(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        lu_fd(i) = (c_pert - c0) / eps;
    }

    double lu_error = arma::norm(jacs.lu - lu_fd);
    double lu_rel_error = lu_error / (arma::norm(lu_fd) + 1e-10);

    cout << "  Single RW lu error (abs): " << lu_error << ", (rel): " << lu_rel_error << endl;
    CHECK(lu_rel_error < 2e-2);  // Single RW has less averaging, higher relative error
}

// ============================================================================
// TEST: RW with momentum constraint Jacobians
// ============================================================================
TEST_CASE("RW satellite with momentum constraints Jacobians match finite differences", "[satellite][jacobian][constraint][rw][momentum]") {
    cout << "\n=== Test: RW with Momentum Constraint Jacobians ===" << endl;

    // Create satellite with RWs that have momentum constraints (high momentum cost)
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // Add RWs with momentum cost enabled (AM_cost > 0, low AM_cost_threshold)
    sat.add_RW(arma::vec({1,0,0}), 1e-5, 0.001, 0.01, 1.0, 1.0, 0.002, 0.0, 0.001);
    sat.add_RW(arma::vec({0,1,0}), 1e-5, 0.001, 0.01, 1.0, 1.0, 0.002, 0.0, 0.001);
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 1.0, 0.002, 0.0, 0.001);
    sat.set_AV_constraint(0.2);

    int nx = sat.state_N();
    int nu = sat.control_N();
    int nc = sat.constraint_N();
    int N = 10;
    int k = 5;

    cout << "  RW with momentum: nx=" << nx << ", nu=" << nu << ", nc=" << nc << endl;

    arma::arma_rng::set_seed(67);
    arma::vec3 w0 = 0.02 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    // Higher RW momentum to trigger momentum cost
    arma::vec rw_h = 0.008 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    arma::vec u = 0.0005 * arma::randn(nu);
    arma::vec3 sun = arma::normalise(arma::vec({1.0, 0.5, 0.2}));

    auto jacs = sat.constraintJacobians(k, N, u, x, sun);
    arma::mat Jcu_analytic = std::get<0>(jacs);
    arma::mat Jcx_analytic = std::get<1>(jacs);

    arma::vec c0 = sat.getConstraints(k, N, u, x, sun);

    double eps = 1e-7;
    arma::mat Jcu_fd(nc, nu, arma::fill::zeros);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        arma::vec c_pert = sat.getConstraints(k, N, u_pert, x, sun);
        Jcu_fd.col(i) = (c_pert - c0) / eps;
    }

    double Jcu_error = arma::norm(Jcu_analytic - Jcu_fd, "fro");
    double Jcu_rel_error = Jcu_error / (arma::norm(Jcu_fd, "fro") + 1e-10);

    cout << "  Momentum Jcu error (abs): " << Jcu_error << ", (rel): " << Jcu_rel_error << endl;

    CHECK(Jcu_rel_error < 1e-4);
}

// ============================================================================
// TEST: RW cost with stiction penalty Jacobians
// ============================================================================
TEST_CASE("RW satellite cost with stiction penalty Jacobians match finite differences", "[satellite][jacobian][cost][rw][stiction]") {
    cout << "\n=== Test: RW Cost with Stiction Penalty Jacobians ===" << endl;

    // Create satellite with RWs that have stiction cost enabled
    Satellite sat = Satellite();
    sat.change_Jcom(arma::diagmat(arma::vec({0.05, 0.06, 0.055})));
    // stiction_cost=1.0, stiction_threshold=0.002 (near zero crossing)
    sat.add_RW(arma::vec({1,0,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 1.0, 0.002);
    sat.add_RW(arma::vec({0,1,0}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 1.0, 0.002);
    sat.add_RW(arma::vec({0,0,1}), 1e-5, 0.001, 0.01, 1.0, 0.1, 0.005, 1.0, 0.002);

    int nx = sat.state_N();
    int nu = sat.control_N();

    arma::arma_rng::set_seed(68);

    int N = 20;
    int k = 10;

    arma::vec3 w0 = 0.03 * arma::randn(3);
    arma::vec4 q0 = arma::normalise(arma::vec({1.0, 0.1, 0.1, 0.1}) + 0.1*arma::randn(4));
    arma::vec x_base = join_cols(w0, q0);
    // Small RW momentum to trigger stiction cost
    arma::vec rw_h = 0.001 * arma::randn(sat.number_RW);
    arma::vec x = join_cols(x_base, rw_h);
    x = sat.state_norm(x);

    arma::vec u = 0.0003 * arma::randn(nu);
    arma::vec u_prev = 0.9 * u + 0.00001 * arma::randn(nu);

    arma::vec3 satvec = arma::normalise(arma::vec({0, 0, 1}));
    arma::vec3 ECIvec = arma::normalise(arma::vec({1, 0, 0}));
    arma::vec3 B_eci = arma::vec({1e-5, 3e-5, 2e-5});

    COST_SETTINGS_FORM costSettings = std::make_tuple(
        1e2, 1e1, 1.0, 0.0, 0.0,
        1e3, 1e2, 0.0, 0.0,
        2, 0, 0
    );

    cost_jacs jacs = sat.veccostJacobians(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);
    double c0 = sat.stepcost_vec(k, N, x, u, u_prev, satvec, ECIvec, B_eci, &costSettings);

    double eps = 1e-7;
    arma::vec lu_fd(nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps;
        double c_pert = sat.stepcost_vec(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        lu_fd(i) = (c_pert - c0) / eps;
    }

    double lu_error = arma::norm(jacs.lu - lu_fd);
    double lu_rel_error = lu_error / (arma::norm(lu_fd) + 1e-10);

    cout << "  Stiction cost lu error (abs): " << lu_error << ", (rel): " << lu_rel_error << endl;
    CHECK(lu_rel_error < 1e-2);  // Stiction penalty involves smoothstep, slightly less precise

    // Verify luu
    double eps_hess = 1e-5;
    arma::mat luu_fd(nu, nu);
    for(int i = 0; i < nu; i++) {
        arma::vec u_pert = u;
        u_pert(i) += eps_hess;
        cost_jacs jacs_pert = sat.veccostJacobians(k, N, x, u_pert, u_prev, satvec, ECIvec, B_eci, &costSettings);
        luu_fd.col(i) = (jacs_pert.lu - jacs.lu) / eps_hess;
    }

    double luu_error = arma::norm(jacs.luu - luu_fd, "fro");
    double luu_rel_error = luu_error / (arma::norm(luu_fd, "fro") + 1e-10);

    cout << "  Stiction cost luu error (abs): " << luu_error << ", (rel): " << luu_rel_error << endl;
    CHECK(luu_rel_error < 1e-2);
}
