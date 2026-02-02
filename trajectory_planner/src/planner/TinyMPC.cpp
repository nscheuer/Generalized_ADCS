#include "TinyMPC.hpp"
#include <chrono>
#include <algorithm>

using namespace arma;
using namespace std;

// ============================================================================
// State Layout: [omega(3), q(4), h_rw(n_rw)]
// Error Layout: [omega_err(3), theta_err(3), h_rw_err(n_rw)]  (reduced, 7D for 1RW)
// ============================================================================

// ============================================================================
// Constructors
// ============================================================================

TinyMPC::TinyMPC() : n(0), m(0), n_err(0) {}

TinyMPC::TinyMPC(Satellite sat_in, const TinyMPCSettings& settings_in)
    : sat(sat_in), settings(settings_in)
{
    n = sat.state_N();      // Full state dimension (8 for 1 RW)
    n_err = n - 1;          // Reduced error dimension (7 for 1 RW) - quaternion 4D -> 3D
    m = sat.control_N();

    // Initialize cost matrices for REDUCED error state
    Q = eye(n_err, n_err);
    R = eye(m, m);
    Qf = eye(n_err, n_err);

    // Set control bounds from satellite
    u_min = -join_cols(vec(sat.MTQ_max), vec(sat.RW_max_torq), vec(sat.magic_max_torq));
    u_max = join_cols(vec(sat.MTQ_max), vec(sat.RW_max_torq), vec(sat.magic_max_torq));

    // Initialize ADMM variables
    int N = settings.track_horizon;
    Z = zeros(m, N);
    Y = zeros(m, N);
}

// ============================================================================
// Configuration
// ============================================================================

void TinyMPC::setSettings(const TinyMPCSettings& settings_in) {
    settings = settings_in;

    // Resize ADMM variables if horizon changed
    int N = settings.track_horizon;
    if (Z.n_cols != (uword)N) {
        Z = zeros(m, N);
        Y = zeros(m, N);
        has_warm_start = false;
    }
}

TinyMPCSettings TinyMPC::getSettings() const {
    return settings;
}

void TinyMPC::setCostMatrices(const mat& Q_in, const mat& R_in, const mat& Qf_in) {
    Q = Q_in;
    R = R_in;
    Qf = Qf_in;
    riccati_computed = false;
}

void TinyMPC::setCostFromSettings(const COST_SETTINGS_FORM& cost_settings) {
    // Extract weights from cost settings tuple
    double angle_weight = get<0>(cost_settings);
    double angvel_weight = get<1>(cost_settings);
    double u_weight = get<2>(cost_settings);
    double angle_weight_N = get<6>(cost_settings);
    double angvel_weight_N = get<7>(cost_settings);

    // Build Q matrix for REDUCED error state (n_err x n_err)
    // Error state: [omega_err(3), theta_err(3), h_rw_err(n_rw)]
    Q = zeros(n_err, n_err);
    // Angular velocity error (indices 0-2)
    Q(0,0) = angvel_weight;
    Q(1,1) = angvel_weight;
    Q(2,2) = angvel_weight;
    // Attitude error - 3D (indices 3-5)
    Q(3,3) = angle_weight;
    Q(4,4) = angle_weight;
    Q(5,5) = angle_weight;
    // Reaction wheel momentum (indices 6+)
    for (int i = 6; i < n_err; i++) {
        Q(i,i) = angvel_weight * 0.1;  // Lower weight on RW speeds
    }

    // Build R matrix (control cost)
    R = u_weight * eye(m, m);

    // Build Qf matrix (terminal cost) - same structure as Q
    Qf = zeros(n_err, n_err);
    Qf(0,0) = angvel_weight_N;
    Qf(1,1) = angvel_weight_N;
    Qf(2,2) = angvel_weight_N;
    Qf(3,3) = angle_weight_N;
    Qf(4,4) = angle_weight_N;
    Qf(5,5) = angle_weight_N;
    for (int i = 6; i < n_err; i++) {
        Qf(i,i) = angvel_weight_N * 0.1;
    }

    riccati_computed = false;
}

// ============================================================================
// Reference Trajectory Loading
// ============================================================================

void TinyMPC::loadReferenceTrajectory(const TrajectorySegment& traj_segment) {
    ref_traj = traj_segment;
    has_reference = true;
    riccati_computed = false;
}

void TinyMPC::loadReferenceFromALTRO(const OPT_FORM& altro_opt, const vec& times, double dt) {
    ref_traj.X_ref = get<0>(altro_opt);
    ref_traj.U_ref = get<1>(altro_opt);
    ref_traj.times = times;
    ref_traj.dt_ref = dt;

    // Extract K gains if available (stored as flattened matrix in OPT_FORM)
    mat K_flat = get<3>(altro_opt);
    if (K_flat.n_elem > 0) {
        int N_ref = ref_traj.U_ref.n_cols;
        ref_traj.K_ref = cube(m, n_err, N_ref);
        for (int k = 0; k < N_ref && k * n_err < (int)K_flat.n_cols; k++) {
            ref_traj.K_ref.slice(k) = K_flat.cols(k * n_err, (k + 1) * n_err - 1);
        }
    }

    has_reference = true;
    riccati_computed = false;
}

bool TinyMPC::hasValidReference() const {
    return has_reference && ref_traj.X_ref.n_cols > 0;
}

pair<double, double> TinyMPC::getReferenceTimeRange() const {
    if (!has_reference || ref_traj.times.n_elem == 0) {
        return {0.0, 0.0};
    }
    return {ref_traj.times(0), ref_traj.times(ref_traj.times.n_elem - 1)};
}

// ============================================================================
// Reference Interpolation
// ============================================================================

void TinyMPC::interpolateReference(double t, vec& x_ref, vec& u_ref, mat* K_ref) const {
    if (!has_reference || ref_traj.times.n_elem == 0) {
        x_ref = zeros(n);
        u_ref = zeros(m);
        return;
    }

    // Find bracketing indices
    double t_start = ref_traj.times(0);
    double t_end = ref_traj.times(ref_traj.times.n_elem - 1);

    // Clamp to valid range
    t = std::clamp(t, t_start, t_end);

    // Find index using time
    double t_rel = t - t_start;
    double idx_float = t_rel / ref_traj.dt_ref;
    int idx = (int)floor(idx_float);
    double alpha = idx_float - idx;

    // Clamp index
    int N_ref = ref_traj.X_ref.n_cols - 1;
    idx = std::min(idx, N_ref - 1);
    idx = std::max(idx, 0);

    // Linear interpolation for states
    if (alpha < 1e-10 || idx >= N_ref) {
        x_ref = ref_traj.X_ref.col(idx);
    } else {
        vec x0 = ref_traj.X_ref.col(idx);
        vec x1 = ref_traj.X_ref.col(idx + 1);

        // For quaternion part (indices 3-6), use SLERP
        vec4 q0 = x0.subvec(3, 6);
        vec4 q1 = x1.subvec(3, 6);

        // Ensure quaternions are in same hemisphere
        if (dot(q0, q1) < 0) {
            q1 = -q1;
        }

        // Simple linear interpolation (LERP) with renormalization
        vec4 q_interp = (1.0 - alpha) * q0 + alpha * q1;
        q_interp = normalise(q_interp);

        // Linear interpolation for rest of state
        x_ref = (1.0 - alpha) * x0 + alpha * x1;
        x_ref.subvec(3, 6) = q_interp;
    }

    // Linear interpolation for controls
    int idx_u = std::min(idx, (int)ref_traj.U_ref.n_cols - 1);
    if (alpha < 1e-10 || idx_u >= (int)ref_traj.U_ref.n_cols - 1) {
        u_ref = ref_traj.U_ref.col(idx_u);
    } else {
        u_ref = (1.0 - alpha) * ref_traj.U_ref.col(idx_u) +
                alpha * ref_traj.U_ref.col(idx_u + 1);
    }

    // Interpolate K gain if requested and available
    if (K_ref != nullptr && ref_traj.K_ref.n_slices > 0) {
        int idx_k = std::min(idx, (int)ref_traj.K_ref.n_slices - 1);
        *K_ref = ref_traj.K_ref.slice(idx_k);
    }
}

pair<vec, vec> TinyMPC::getReference(double t) const {
    vec x_ref, u_ref;
    interpolateReference(t, x_ref, u_ref);
    return {x_ref, u_ref};
}

mat TinyMPC::getLQRGain(double t) const {
    if (!has_reference || ref_traj.K_ref.n_slices == 0) {
        return zeros(m, n_err);
    }

    vec x_ref, u_ref;
    mat K_ref;
    interpolateReference(t, x_ref, u_ref, &K_ref);
    return K_ref;
}

// ============================================================================
// Local Reference Building
// ============================================================================

void TinyMPC::buildLocalReference(double t_start, mat& X_ref_local, mat& U_ref_local) {
    int N = settings.track_horizon;
    X_ref_local = zeros(n, N + 1);
    U_ref_local = zeros(m, N);

    for (int k = 0; k <= N; k++) {
        double t = t_start + k * settings.track_dt;
        vec x_ref, u_ref;
        interpolateReference(t, x_ref, u_ref);
        X_ref_local.col(k) = x_ref;
        if (k < N) {
            U_ref_local.col(k) = u_ref;
        }
    }
}

// ============================================================================
// Dynamics Linearization (for reduced error state)
// ============================================================================

void TinyMPC::linearizeDynamics(const vec& x_op, const vec& u_op,
                                 const vec3& B_field, const DYNAMICS_INFO_FORM& dynamics_info) {
    // Get Jacobians from satellite model (full state: 8D for 1 RW)
    // Layout: [omega(3), q(4), h_rw(n_rw)]
    // Jacobian is (n x n) where rows/cols 0-2 = omega, 3-6 = quaternion, 7+ = h_rw
    auto [dxdot_dx, dxdot_du, dxdot_dtorq] =
        sat.dynamicsJacobians(x_op, u_op, dynamics_info);

    // Discretize using forward Euler (full state)
    double dt = settings.track_dt;
    mat A_full = eye(n, n) + dt * dxdot_dx;
    mat B_full = dt * dxdot_du;

    // Now reduce to error state: map 8D full state to 7D error state
    // Full state: [omega(3), q(4), h_rw(n_rw)]
    // Error state: [omega_err(3), theta_err(3), h_rw_err(n_rw)]
    //
    // The linearization uses E = [I_3  0_3x4  0; 0_3x1 2*I_3 0; 0 0 I] to map
    // from quaternion-based state to error state (taking 2 * vector part of q).
    // 
    // For small errors: dtheta ≈ 2 * q_vec (for unit quaternion near identity)

    A_lin = zeros(n_err, n_err);
    B_lin = zeros(n_err, m);
    int n_rw = n - 7;

    // Build reduction matrix E (n_err x n) that maps full state to error state
    // E = [I_3    0_3x4        0_3xn_rw ]
    //     [0_3x3  [0|2*I_3]    0_3xn_rw ]  <- takes 2*vector part of quaternion
    //     [0      0            I_n_rw   ]
    mat E = zeros(n_err, n);
    E.submat(0, 0, 2, 2) = eye(3, 3);              // omega -> omega_err
    E.submat(3, 4, 5, 6) = 2.0 * eye(3, 3);        // q_vec -> theta_err (2x)
    if (n_rw > 0) {
        E.submat(6, 7, n_err-1, n-1) = eye(n_rw, n_rw);  // h_rw -> h_rw_err
    }

    // Reduced A: A_err = E * A_full * E^+
    // For our structure, E^+ (pseudoinverse) maps error back to full state
    // E^+ = [I_3    0_3x3        0     ]
    //       [0_1x3  0_1x3        0     ]  <- scalar part of quaternion (ignored)
    //       [0_3x3  0.5*I_3      0     ]  <- theta_err -> q_vec (0.5x)
    //       [0      0            I_n_rw]
    mat Eplus = zeros(n, n_err);
    Eplus.submat(0, 0, 2, 2) = eye(3, 3);           // omega_err -> omega
    Eplus.submat(4, 3, 6, 5) = 0.5 * eye(3, 3);    // theta_err -> q_vec (0.5x)
    if (n_rw > 0) {
        Eplus.submat(7, 6, n-1, n_err-1) = eye(n_rw, n_rw);  // h_rw_err -> h_rw
    }

    A_lin = E * A_full * Eplus;
    B_lin = E * B_full;

    // Affine term (not used in basic formulation)
    c_lin = zeros(n_err);
}

// ============================================================================
// Riccati Recursion (for reduced error state)
// ============================================================================

void TinyMPC::solveRiccati() {
    if (A_lin.n_rows == 0) return;

    int N = settings.track_horizon;
    P_riccati.resize(N + 1);
    K_riccati.resize(N);

    // Terminal cost
    P_riccati[N] = Qf;

    // Backward recursion
    for (int k = N - 1; k >= 0; k--) {
        mat P_next = P_riccati[k + 1];

        // Standard discrete-time Riccati recursion
        mat BtP = B_lin.t() * P_next;
        mat BtPB = BtP * B_lin;
        mat BtPA = BtP * A_lin;

        // Add regularization for numerical stability
        mat R_reg = R + 1e-6 * eye(m, m);

        K_riccati[k] = arma::solve(R_reg + BtPB, BtPA);
        P_riccati[k] = Q + A_lin.t() * P_next * A_lin - A_lin.t() * P_next * B_lin * K_riccati[k];

        // Ensure P is symmetric
        P_riccati[k] = 0.5 * (P_riccati[k] + P_riccati[k].t());
    }

    riccati_computed = true;
}

// ============================================================================
// Quaternion-aware State Error (returns REDUCED 7D error)
// ============================================================================

vec TinyMPC::computeStateError(const vec& x, const vec& x_ref) const {
    // Full state: [omega(3), q(4), h_rw(n_rw)]
    // Returns reduced error: [omega_err(3), theta_err(3), h_rw_err(n_rw)]

    vec error = zeros(n_err);

    // Angular velocity error (indices 0-2)
    error.subvec(0, 2) = x.subvec(0, 2) - x_ref.subvec(0, 2);

    // Quaternion error -> 3D attitude error (indices 3-5)
    vec4 q = x.subvec(3, 6);
    vec4 q_ref = x_ref.subvec(3, 6);

    // Ensure quaternions are normalized
    q = normalise(q);
    vec4 q_ref_norm = normalise(q_ref);

    // Quaternion inverse (conjugate for unit quaternion): q^-1 = [s, -v]
    vec4 q_ref_inv = {q_ref_norm(0), -q_ref_norm(1), -q_ref_norm(2), -q_ref_norm(3)};

    // Quaternion multiplication: q_err = q_ref_inv * q
    double s1 = q_ref_inv(0), s2 = q(0);
    vec3 v1 = {q_ref_inv(1), q_ref_inv(2), q_ref_inv(3)};
    vec3 v2 = {q(1), q(2), q(3)};

    double s_err = s1 * s2 - dot(v1, v2);
    vec3 v_err = s1 * v2 + s2 * v1 + cross(v1, v2);

    // Ensure q_err is in positive hemisphere (s > 0)
    if (s_err < 0) {
        v_err = -v_err;
    }

    // Convert to 3D attitude error: theta_err = 2 * vec(q_err)
    error.subvec(3, 5) = 2.0 * v_err;

    // RW momentum error (indices 6+)
    int n_rw = n - 7;
    if (n_rw > 0) {
        error.subvec(6, n_err - 1) = x.subvec(7, n - 1) - x_ref.subvec(7, n - 1);
    }

    return error;
}

// ============================================================================
// ADMM Solver Components (using reduced error state)
// ============================================================================

void TinyMPC::admm_x_update(mat& X, mat& U,
                            const vec& x0,
                            const mat& X_ref, const mat& U_ref) {
    // Solve unconstrained LQR with ADMM terms
    // X is full state (8D), but K_riccati operates on reduced error (7D)

    int N = settings.track_horizon;
    double rho = settings.rho;

    // Forward pass using linearized dynamics around reference
    X.col(0) = x0;
    for (int k = 0; k < N; k++) {
        vec x_err = computeStateError(X.col(k), X_ref.col(k));
        vec u_riccati = U_ref.col(k) - K_riccati[k] * x_err;
        vec u_admm = u_riccati + rho / (rho + R(0,0)) * (Z.col(k) - Y.col(k) - u_riccati);
        U.col(k) = u_admm;
        
        // Propagate: x_{k+1} ≈ x_ref_{k+1} + A*(x_k - x_ref_k) + B*(u_k - u_ref_k)
        vec dx_next = A_lin * x_err + B_lin * (U.col(k) - U_ref.col(k));
        
        // Map reduced error back to full state
        X.col(k + 1) = X_ref.col(k + 1);
        X.col(k + 1).subvec(0, 2) += dx_next.subvec(0, 2);  // omega
        // For quaternion, use small angle: q ≈ q_ref * [1; 0.5*dtheta]
        vec3 dtheta = 0.5 * dx_next.subvec(3, 5);
        vec4 q_ref = X_ref.col(k + 1).subvec(3, 6);
        vec4 dq = {1.0, dtheta(0), dtheta(1), dtheta(2)};
        // q_new = q_ref * dq (quaternion multiplication)
        double s1 = q_ref(0), s2 = dq(0);
        vec3 v1 = {q_ref(1), q_ref(2), q_ref(3)};
        vec3 v2 = {dq(1), dq(2), dq(3)};
        double s_new = s1 * s2 - dot(v1, v2);
        vec3 v_new = s1 * v2 + s2 * v1 + cross(v1, v2);
        vec4 q_new = {s_new, v_new(0), v_new(1), v_new(2)};
        X.col(k + 1).subvec(3, 6) = normalise(q_new);
        // RW momentum
        int n_rw = n - 7;
        if (n_rw > 0) {
            X.col(k + 1).subvec(7, n - 1) += dx_next.subvec(6, n_err - 1);
        }
    }
}

void TinyMPC::admm_z_update(const mat& U) {
    // z = project(u + y) onto control bounds
    int N = settings.track_horizon;
    for (int k = 0; k < N; k++) {
        Z.col(k) = projectControl(U.col(k) + Y.col(k));
    }
}

void TinyMPC::admm_y_update(const mat& U) {
    // y = y + u - z
    Y = Y + U - Z;
}

vec TinyMPC::projectControl(const vec& u) {
    vec u_proj = u;
    for (uword i = 0; i < u.n_elem; i++) {
        u_proj(i) = std::clamp(u(i), u_min(i), u_max(i));
    }
    return u_proj;
}

pair<double, double> TinyMPC::computeResiduals(const mat& U) {
    // Primal residual: ||u - z||
    double primal_res = norm(U - Z, "fro");

    // Dual residual: rho * ||z - z_prev||
    static mat Z_prev;
    double dual_res = 0.0;
    if (Z_prev.n_elem == Z.n_elem) {
        dual_res = settings.rho * norm(Z - Z_prev, "fro");
    }
    Z_prev = Z;

    return {primal_res, dual_res};
}

bool TinyMPC::checkConvergence(double primal_res, double dual_res) {
    int N = settings.track_horizon;
    double eps_pri = settings.abs_tol * sqrt(N * m) +
                     settings.rel_tol * std::max(norm(Z, "fro"), 1.0);
    double eps_dual = settings.abs_tol * sqrt(N * m) +
                      settings.rel_tol * settings.rho * norm(Y, "fro");

    return (primal_res < eps_pri) && (dual_res < eps_dual);
}

void TinyMPC::updateRho(double primal_res, double dual_res) {
    if (!settings.adaptive_rho) return;

    double ratio = primal_res / (dual_res + 1e-10);
    if (ratio > 10.0) {
        settings.rho = std::min(settings.rho * 2.0, settings.rho_max);
    } else if (ratio < 0.1) {
        settings.rho = std::max(settings.rho / 2.0, settings.rho_min);
    }
}

// ============================================================================
// Main Solve Function
// ============================================================================

TinyMPCResult TinyMPC::solve(const vec& x_current, double t_current,
                              const vec3& B_field, const vec3& sun_vec,
                              const DYNAMICS_INFO_FORM& dynamics_info) {
    auto start_time = chrono::high_resolution_clock::now();

    TinyMPCResult result;
    result.converged = false;
    result.iterations = 0;

    if (!hasValidReference()) {
        result.u_opt = zeros(m);
        result.solve_time_ms = 0.0;
        return result;
    }

    int N = settings.track_horizon;

    // Build local reference trajectory
    mat X_ref_local, U_ref_local;
    buildLocalReference(t_current, X_ref_local, U_ref_local);

    // Get reference at current time for linearization
    vec x_ref, u_ref;
    interpolateReference(t_current, x_ref, u_ref);

    // Linearize dynamics about reference
    linearizeDynamics(x_ref, u_ref, B_field, dynamics_info);

    // Solve Riccati equation
    solveRiccati();

    // Initialize trajectories (full state dimension)
    mat X = zeros(n, N + 1);
    mat U = zeros(m, N);

    // Use warm start if available
    if (has_warm_start && X_warm.n_cols == (uword)(N + 1)) {
        X = X_warm;
        U = U_warm;
    } else {
        X = X_ref_local;
        U = U_ref_local;
        Z = U;
        Y.zeros();
    }

    // ADMM iterations
    double primal_res = 0.0, dual_res = 0.0;
    for (int iter = 0; iter < settings.max_iter; iter++) {
        // x-update
        admm_x_update(X, U, x_current, X_ref_local, U_ref_local);

        // z-update
        admm_z_update(U);

        // y-update
        admm_y_update(U);

        result.iterations = iter + 1;

        // Check convergence
        if ((iter + 1) % settings.check_interval == 0) {
            tie(primal_res, dual_res) = computeResiduals(U);

            if (checkConvergence(primal_res, dual_res)) {
                result.converged = true;
                break;
            }

            updateRho(primal_res, dual_res);
        }
    }

    // Final residuals
    tie(primal_res, dual_res) = computeResiduals(U);
    result.primal_residual = primal_res;
    result.dual_residual = dual_res;

    // Extract optimal control
    result.u_opt = Z.col(0);
    result.X_pred = X;
    result.U_pred = U;

    // Compute tracking error
    result.tracking_error = norm(computeStateError(x_current, X_ref_local.col(0)));

    // Store for warm start
    X_warm = X;
    U_warm = U;
    has_warm_start = true;

    auto end_time = chrono::high_resolution_clock::now();
    result.solve_time_ms = chrono::duration<double, milli>(end_time - start_time).count();

    if (settings.verbose >= 1) {
        cout << "TinyMPC: " << result.iterations << " iters, "
             << result.solve_time_ms << " ms, "
             << (result.converged ? "converged" : "max_iter")
             << ", err=" << result.tracking_error << endl;
    }

    return result;
}

// ============================================================================
// Warm Start and Reset
// ============================================================================

void TinyMPC::warmStart(const mat& X_prev, const mat& U_prev) {
    int N = settings.track_horizon;

    if (X_prev.n_cols >= 2) {
        X_warm = zeros(n, N + 1);
        int copy_cols = std::min((int)X_prev.n_cols - 1, N + 1);
        X_warm.head_cols(copy_cols) = X_prev.tail_cols(copy_cols);
        if (copy_cols <= N) {
            for (int k = copy_cols; k <= N; k++) {
                X_warm.col(k) = X_warm.col(copy_cols - 1);
            }
        }
    }

    if (U_prev.n_cols >= 2) {
        U_warm = zeros(m, N);
        int copy_cols = std::min((int)U_prev.n_cols - 1, N);
        U_warm.head_cols(copy_cols) = U_prev.tail_cols(copy_cols);
        if (copy_cols < N) {
            for (int k = copy_cols; k < N; k++) {
                U_warm.col(k) = U_warm.col(copy_cols - 1);
            }
        }
    }

    has_warm_start = true;
}

void TinyMPC::reset() {
    int N = settings.track_horizon;
    Z = zeros(m, N);
    Y = zeros(m, N);
    has_warm_start = false;
    riccati_computed = false;
}
