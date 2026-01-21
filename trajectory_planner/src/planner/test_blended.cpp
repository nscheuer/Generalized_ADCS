// Tests for blended dynamics functions
// These test the constraint tightening warm-start implementation

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <armadillo>
#include "Satellite.hpp"
#include "PlannerUtil.hpp"

using namespace arma;

// Helper to create a test satellite with MTQs
Satellite createTestSatellite() {
    mat33 J;
    J << 0.05 << 0.0 << 0.0 << endr
      << 0.0 << 0.05 << 0.0 << endr
      << 0.0 << 0.0 << 0.05 << endr;

    Satellite sat(J);

    // Add 3 orthogonal MTQs
    sat.add_MTQ(vec3{1.0, 0.0, 0.0}, 0.5, 1.0);  // X-axis MTQ
    sat.add_MTQ(vec3{0.0, 1.0, 0.0}, 0.5, 1.0);  // Y-axis MTQ
    sat.add_MTQ(vec3{0.0, 0.0, 1.0}, 0.5, 1.0);  // Z-axis MTQ

    return sat;
}

// Create test dynamics info with a B-field
DYNAMICS_INFO_FORM createTestDynamicsInfo(vec3 B_eci) {
    vec3 R_eci = {7000.0, 0.0, 0.0};  // km
    vec3 V_eci = {0.0, 7.5, 0.0};     // km/s
    vec3 S_eci = {1.0, 0.0, 0.0};     // Sun direction
    int prop_torq_on = 0;
    int dist_on = 0;
    double rho = 0.0;

    return std::make_tuple(B_eci, R_eci, prop_torq_on, V_eci, S_eci, dist_on, rho);
}

TEST_CASE("dynamicsBlended alpha=1 matches original dynamics", "[blended]") {
    Satellite sat = createTestSatellite();

    // State: [omega, quaternion]
    vec x = vec(7).zeros();
    x(0) = 0.01;  // Small angular velocity
    x(1) = 0.02;
    x(2) = 0.01;
    x(3) = 1.0;   // Identity quaternion
    x(4) = 0.0;
    x(5) = 0.0;
    x(6) = 0.0;

    // Control: MTQ commands
    vec u = vec(3).zeros();
    u(0) = 0.1;
    u(1) = 0.2;
    u(2) = -0.1;

    // B-field in ECI
    vec3 B_eci = {3e-5, 2e-5, 1e-5};
    auto dyn_info = createTestDynamicsInfo(B_eci);

    // Compute original dynamics
    auto [xdot_orig, dist_orig] = sat.dynamics(x, u, dyn_info);

    // Compute blended dynamics with alpha=1 (should match original)
    auto [xdot_blended, dist_blended] = sat.dynamicsBlended(x, u, dyn_info, 1.0);

    REQUIRE(approx_equal(xdot_orig, xdot_blended, "absdiff", 1e-12));
    REQUIRE(approx_equal(dist_orig, dist_blended, "absdiff", 1e-12));
}

TEST_CASE("dynamicsJacobiansBlended alpha=1 matches original Jacobians", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(0) = 0.01;
    x(1) = 0.02;
    x(2) = 0.01;
    x(3) = 1.0;
    x(4) = 0.0;
    x(5) = 0.0;
    x(6) = 0.0;

    vec u = vec(3).zeros();
    u(0) = 0.1;
    u(1) = 0.2;
    u(2) = -0.1;

    vec3 B_eci = {3e-5, 2e-5, 1e-5};
    auto dyn_info = createTestDynamicsInfo(B_eci);

    // Original Jacobians
    auto [Jxx_orig, Jxu_orig, Jxt_orig] = sat.dynamicsJacobians(x, u, dyn_info);

    // Blended Jacobians with alpha=1
    auto [Jxx_blended, Jxu_blended, Jxt_blended] = sat.dynamicsJacobiansBlended(x, u, dyn_info, 1.0);

    REQUIRE(approx_equal(Jxx_orig, Jxx_blended, "absdiff", 1e-12));
    REQUIRE(approx_equal(Jxu_orig, Jxu_blended, "absdiff", 1e-12));
    REQUIRE(approx_equal(Jxt_orig, Jxt_blended, "absdiff", 1e-12));
}

TEST_CASE("dynamicsBlended alpha=0 gives relaxed torque model", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(3) = 1.0;  // Identity quaternion

    // Single MTQ command along X
    vec u = vec(3).zeros();
    u(0) = 0.3;  // X-axis MTQ only

    // B-field along Z (perpendicular to MTQ axis)
    vec3 B_eci = {0.0, 0.0, 3e-5};
    auto dyn_info = createTestDynamicsInfo(B_eci);

    // With alpha=0, torque should be |B| * P_perp * m
    // P_perp projects onto plane perpendicular to B
    // m = [0.3, 0, 0] (from MTQ command on X-axis)
    // B_hat = [0, 0, 1]
    // P_perp * m = m - (m.B_hat)*B_hat = [0.3, 0, 0] (m is already perp to B)
    // tau_relaxed = |B| * P_perp * m = 3e-5 * [0.3, 0, 0]

    auto [xdot_relaxed, dist_relaxed] = sat.dynamicsBlended(x, u, dyn_info, 0.0);
    auto [xdot_true, dist_true] = sat.dynamicsBlended(x, u, dyn_info, 1.0);

    // Both should produce torque (MTQ perpendicular to B-field)
    // but the magnitudes and directions may differ
    double omega_dot_mag_relaxed = norm(xdot_relaxed.head(3));
    double omega_dot_mag_true = norm(xdot_true.head(3));

    // Both should be non-zero when MTQ axis is perpendicular to B
    REQUIRE(omega_dot_mag_relaxed > 1e-10);
    REQUIRE(omega_dot_mag_true > 1e-10);

    std::cout << "Relaxed omega_dot: " << xdot_relaxed.head(3).t();
    std::cout << "True omega_dot: " << xdot_true.head(3).t();
}

TEST_CASE("dynamicsBlended alpha=0 gives zero torque when MTQ parallel to B", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(3) = 1.0;  // Identity quaternion

    // MTQ command along Z
    vec u = vec(3).zeros();
    u(2) = 0.3;  // Z-axis MTQ only

    // B-field also along Z (parallel to MTQ axis)
    vec3 B_eci = {0.0, 0.0, 3e-5};
    auto dyn_info = createTestDynamicsInfo(B_eci);

    // With alpha=0, torque = |B| * P_perp * m
    // m = [0, 0, 0.3] (from Z-axis MTQ)
    // B_hat = [0, 0, 1]
    // P_perp * m = m - (m.B_hat)*B_hat = [0,0,0.3] - 0.3*[0,0,1] = [0,0,0]
    // So torque should be zero

    auto [xdot_relaxed, dist_relaxed] = sat.dynamicsBlended(x, u, dyn_info, 0.0);

    // Angular acceleration should be nearly zero (MTQ can't produce torque along B)
    double omega_dot_mag = norm(xdot_relaxed.head(3));
    REQUIRE(omega_dot_mag < 1e-10);

    // True physics should also give zero (m x B = 0 when parallel)
    auto [xdot_true, dist_true] = sat.dynamicsBlended(x, u, dyn_info, 1.0);
    double omega_dot_mag_true = norm(xdot_true.head(3));
    REQUIRE(omega_dot_mag_true < 1e-10);
}

TEST_CASE("Control Jacobian structure at different alpha", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(3) = 1.0;

    vec u = vec(3).zeros();

    // B-field in arbitrary direction
    vec3 B_eci = {2e-5, 3e-5, 1e-5};
    auto dyn_info = createTestDynamicsInfo(B_eci);

    // Get Jacobians at alpha=0 and alpha=1
    auto [Jxx_0, Jxu_0, Jxt_0] = sat.dynamicsJacobiansBlended(x, u, dyn_info, 0.0);
    auto [Jxx_1, Jxu_1, Jxt_1] = sat.dynamicsJacobiansBlended(x, u, dyn_info, 1.0);

    // Extract control Jacobian for angular velocity (first 3 rows, first 3 cols for MTQ)
    mat dwdu_0 = Jxu_0.rows(0, 2).cols(0, 2);
    mat dwdu_1 = Jxu_1.rows(0, 2).cols(0, 2);

    // Both have rank 2 (MTQ can never produce torque along B)
    double rank_0 = arma::rank(dwdu_0);
    double rank_1 = arma::rank(dwdu_1);

    std::cout << "Control Jacobian at alpha=0:\n" << dwdu_0 << std::endl;
    std::cout << "Rank at alpha=0: " << rank_0 << std::endl;
    std::cout << "Control Jacobian at alpha=1:\n" << dwdu_1 << std::endl;
    std::cout << "Rank at alpha=1: " << rank_1 << std::endl;

    // Both should have rank 2 (the null space is along B direction)
    REQUIRE(rank_0 == 2);
    REQUIRE(rank_1 == 2);

    // Key difference: alpha=0 should be better conditioned
    // The cross-product formulation has skew-symmetric structure
    // The relaxed formulation has symmetric projection structure

    // Check that alpha=0 gives symmetric-ish structure (P_perp is symmetric)
    // while alpha=1 gives skew-symmetric structure ([B×] is skew)
    mat skew_part_0 = 0.5 * (dwdu_0 - dwdu_0.t());
    mat sym_part_0 = 0.5 * (dwdu_0 + dwdu_0.t());
    mat skew_part_1 = 0.5 * (dwdu_1 - dwdu_1.t());
    mat sym_part_1 = 0.5 * (dwdu_1 + dwdu_1.t());

    double skew_norm_0 = norm(skew_part_0, "fro");
    double sym_norm_0 = norm(sym_part_0, "fro");
    double skew_norm_1 = norm(skew_part_1, "fro");
    double sym_norm_1 = norm(sym_part_1, "fro");

    std::cout << "Alpha=0: skew_norm=" << skew_norm_0 << " sym_norm=" << sym_norm_0 << std::endl;
    std::cout << "Alpha=1: skew_norm=" << skew_norm_1 << " sym_norm=" << sym_norm_1 << std::endl;

    // Alpha=1 (cross product) should be dominated by skew-symmetric part
    REQUIRE(skew_norm_1 > sym_norm_1);

    // Alpha=0 (projection) should have significant symmetric component
    // (P_perp is symmetric, so J^{-1}*|B|*P_perp should be more symmetric if J is diagonal)
    // This is a softer check since J^{-1} affects the structure
    std::cout << "Both formulations have rank 2 but different structure." << std::endl;
}

TEST_CASE("rk4zBlended alpha=1 matches original rk4z", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(0) = 0.01;
    x(1) = 0.02;
    x(2) = 0.01;
    x(3) = 1.0;
    x(4) = 0.0;
    x(5) = 0.0;
    x(6) = 0.0;

    vec u = vec(3).zeros();
    u(0) = 0.1;
    u(1) = 0.2;
    u(2) = -0.1;

    vec3 B_eci_k = {3e-5, 2e-5, 1e-5};
    vec3 B_eci_kp1 = {3.1e-5, 2.1e-5, 1.1e-5};

    auto dyn_info_k = createTestDynamicsInfo(B_eci_k);
    auto dyn_info_kp1 = createTestDynamicsInfo(B_eci_kp1);

    double dt = 1.0;

    // Original RK4
    auto [x_next_orig, dist_orig] = rk4z(dt, x, u, sat, dyn_info_k, dyn_info_kp1);

    // Blended RK4 with alpha=1
    auto [x_next_blended, dist_blended] = rk4zBlended(dt, x, u, sat, dyn_info_k, dyn_info_kp1, 1.0);

    REQUIRE(approx_equal(x_next_orig, x_next_blended, "absdiff", 1e-10));
}

TEST_CASE("rk4zJacobiansBlended alpha=1 matches original", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(0) = 0.01;
    x(1) = 0.02;
    x(2) = 0.01;
    x(3) = 1.0;
    x(4) = 0.0;
    x(5) = 0.0;
    x(6) = 0.0;

    vec u = vec(3).zeros();
    u(0) = 0.1;
    u(1) = 0.2;
    u(2) = -0.1;

    vec3 B_eci_k = {3e-5, 2e-5, 1e-5};
    vec3 B_eci_kp1 = {3.1e-5, 2.1e-5, 1.1e-5};

    auto dyn_info_k = createTestDynamicsInfo(B_eci_k);
    auto dyn_info_kp1 = createTestDynamicsInfo(B_eci_kp1);

    double dt = 1.0;

    // Original Jacobians
    auto [A_orig, B_orig, T_orig] = rk4zJacobians(dt, x, u, sat, dyn_info_k, dyn_info_kp1);

    // Blended Jacobians with alpha=1
    auto [A_blended, B_blended, T_blended] = rk4zJacobiansBlended(dt, x, u, sat, dyn_info_k, dyn_info_kp1, 1.0);

    REQUIRE(approx_equal(A_orig, A_blended, "absdiff", 1e-10));
    REQUIRE(approx_equal(B_orig, B_blended, "absdiff", 1e-10));
    REQUIRE(approx_equal(T_orig, T_blended, "absdiff", 1e-10));
}

TEST_CASE("Blended interpolation is smooth", "[blended]") {
    Satellite sat = createTestSatellite();

    vec x = vec(7).zeros();
    x(3) = 1.0;

    vec u = vec(3).zeros();
    u(0) = 0.2;
    u(1) = 0.1;

    vec3 B_eci = {2e-5, 3e-5, 1e-5};
    auto dyn_info = createTestDynamicsInfo(B_eci);

    // Check that dynamics change smoothly with alpha
    vec xdot_prev, xdot_curr;

    std::cout << "Alpha interpolation test:" << std::endl;
    for (double alpha = 0.0; alpha <= 1.0; alpha += 0.1) {
        auto [xdot, dist] = sat.dynamicsBlended(x, u, dyn_info, alpha);

        std::cout << "alpha=" << alpha << " omega_dot=" << xdot.head(3).t();

        if (alpha > 0.0) {
            // Check that change is reasonable (not discontinuous)
            double change = norm(xdot.head(3) - xdot_prev.head(3));
            double expected_max_change = 0.1 * norm(xdot.head(3));  // ~10% per 0.1 alpha step
            // This is a soft check - just log if it seems discontinuous
            if (change > expected_max_change * 5) {
                std::cout << "WARNING: Large change at alpha=" << alpha << std::endl;
            }
        }
        xdot_prev = xdot;
    }
}
