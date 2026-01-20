#include "TinyMPC.hpp"
#include <chrono>
#include <algorithm>

using namespace arma;
using namespace std;

// ============================================================================
// Constructors
// ============================================================================

TinyMPC::TinyMPC() : n(0), m(0) {}

TinyMPC::TinyMPC(Satellite sat_in, const TinyMPCSettings& settings_in)
    : sat(sat_in), settings(settings_in)
{
    n = sat.state_N();
    m = sat.control_N();

    // Initialize cost matrices to identity (will be overwritten)
    Q = eye(n, n);
    R = eye(m, m);
    Qf = eye(n, n);

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

    // Build Q matrix (state cost)
    Q = zeros(n, n);
    // Quaternion part (indices 0-3) - use angle weight
    Q(0,0) = angle_weight;
    Q(1,1) = angle_weight;
    Q(2,2) = angle_weight;
    Q(3,3) = angle_weight;
    // Angular velocity part (indices 4-6)
    Q(4,4) = angvel_weight;
    Q(5,5) = angvel_weight;
    Q(6,6) = angvel_weight;
    // Reaction wheel speeds (indices 7+)
    for (int i = 7; i < n; i++) {
        Q(i,i) = angvel_weight * 0.1;  // Lower weight on RW speeds
    }

    // Build R matrix (control cost)
    R = u_weight * eye(m, m);

    // Build Qf matrix (terminal cost)
    Qf = zeros(n, n);
    Qf(0,0) = angle_weight_N;
    Qf(1,1) = angle_weight_N;
    Qf(2,2) = angle_weight_N;
    Qf(3,3) = angle_weight_N;
    Qf(4,4) = angvel_weight_N;
    Qf(5,5) = angvel_weight_N;
    Qf(6,6) = angvel_weight_N;
    for (int i = 7; i < n; i++) {
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
        ref_traj.K_ref = cube(m, n, N_ref);
        for (int k = 0; k < N_ref && k * n < (int)K_flat.n_cols; k++) {
            ref_traj.K_ref.slice(k) = K_flat.cols(k * n, (k + 1) * n - 1);
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
    t = clamp(t, t_start, t_end);

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

        // For quaternion part, use SLERP
        vec4 q0 = x0.head(4);
        vec4 q1 = x1.head(4);

        // Ensure quaternions are in same hemisphere
        if (dot(q0, q1) < 0) {
            q1 = -q1;
        }

        // Simple linear interpolation (LERP) with renormalization
        // For small alpha this is close to SLERP
        vec4 q_interp = (1.0 - alpha) * q0 + alpha * q1;
        q_interp = normalise(q_interp);

        // Linear interpolation for rest of state
        x_ref = (1.0 - alpha) * x0 + alpha * x1;
        x_ref.head(4) = q_interp;
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
        return zeros(m, n);
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
// Dynamics Linearization
// ============================================================================

void TinyMPC::linearizeDynamics(const vec& x_op, const vec& u_op,
                                 const vec3& B_field, const DYNAMICS_INFO_FORM& dynamics_info) {
    // Get Jacobians from satellite model
    // Returns: (dxdot_dx, dxdot_du, dxdot_dtorq)
    auto [dxdot_dx, dxdot_du, dxdot_dtorq] =
        sat.dynamicsJacobians(x_op, u_op, dynamics_info);

    // Get xdot for affine term computation (dynamics returns (xdot, disturbance))
    auto [xdot, dist] = sat.dynamics(x_op, u_op, dynamics_info);

    // Discretize using forward Euler (simple but effective for small dt)
    // x_{k+1} = x_k + dt * f(x_k, u_k)
    // A = I + dt * df/dx
    // B = dt * df/du
    // c = dt * (f(x_op, u_op) - df/dx * x_op - df/du * u_op)

    double dt = settings.track_dt;
    A_lin = eye(n, n) + dt * dxdot_dx;
    B_lin = dt * dxdot_du;
    c_lin = dt * (xdot - dxdot_dx * x_op - dxdot_du * u_op);
}

// ============================================================================
// Riccati Recursion
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
        // K = (R + B'PB)^{-1} B'PA
        // P = Q + A'PA - A'PB*K

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
// Quaternion-aware State Error
// ============================================================================

vec TinyMPC::computeStateError(const vec& x, const vec& x_ref) const {
    vec error = x - x_ref;

    // For quaternion error, use proper quaternion difference
    // e_q = q_ref^{-1} * q  (quaternion multiplication)
    // Then convert to rotation vector or use small-angle approximation

    vec4 q = x.head(4);
    vec4 q_ref = x_ref.head(4);

    // Ensure quaternions are normalized
    q = normalise(q);
    vec4 q_ref_norm = normalise(q_ref);

    // Quaternion inverse (conjugate for unit quaternion)
    vec4 q_ref_inv = {q_ref_norm(0), -q_ref_norm(1), -q_ref_norm(2), -q_ref_norm(3)};

    // Quaternion multiplication: q_err = q_ref_inv * q
    // q1 * q2 = [s1*s2 - v1.v2, s1*v2 + s2*v1 + v1 x v2]
    double s1 = q_ref_inv(0), s2 = q(0);
    vec3 v1 = {q_ref_inv(1), q_ref_inv(2), q_ref_inv(3)};
    vec3 v2 = {q(1), q(2), q(3)};

    double s_err = s1 * s2 - dot(v1, v2);
    vec3 v_err = s1 * v2 + s2 * v1 + cross(v1, v2);

    // Convert to error quaternion
    vec4 q_err = {s_err, v_err(0), v_err(1), v_err(2)};

    // Ensure q_err is in positive hemisphere (s > 0) for uniqueness
    if (q_err(0) < 0) {
        q_err = -q_err;
    }

    // Use vector part as error (small angle approximation: theta ≈ 2 * v)
    error.head(4) = q_err;

    return error;
}

// ============================================================================
// ADMM Solver Components
// ============================================================================

void TinyMPC::admm_x_update(mat& X, mat& U,
                            const vec& x0,
                            const mat& X_ref, const mat& U_ref) {
    // Solve unconstrained LQR with modified cost including ADMM terms
    // min sum_{k=0}^{N-1} [ (x_k - x_ref_k)' Q (x_k - x_ref_k) +
    //                       (u_k - u_ref_k)' R (u_k - u_ref_k) +
    //                       rho/2 ||u_k - z_k + y_k||^2 ]
    //     + (x_N - x_ref_N)' Qf (x_N - x_ref_N)
    // s.t. x_{k+1} = A x_k + B u_k + c

    int N = settings.track_horizon;
    double rho = settings.rho;

    // Forward pass with current gains to get trajectory
    X.col(0) = x0;
    for (int k = 0; k < N; k++) {
        // Compute optimal control using Riccati gain
        vec x_err = computeStateError(X.col(k), X_ref.col(k));

        // Modified control law including ADMM terms
        // u* = u_ref - K*(x - x_ref) + adjustment for z,y
        vec u_riccati = U_ref.col(k) - K_riccati[k] * x_err;

        // ADMM adjustment: add term to push u toward z - y
        vec u_admm = u_riccati + rho / (rho + R(0,0)) * (Z.col(k) - Y.col(k) - u_riccati);

        U.col(k) = u_admm;

        // Propagate dynamics
        X.col(k + 1) = A_lin * X.col(k) + B_lin * U.col(k) + c_lin;
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
        u_proj(i) = clamp(u(i), u_min(i), u_max(i));
    }
    return u_proj;
}

pair<double, double> TinyMPC::computeResiduals(const mat& U) {
    // Primal residual: ||u - z||
    double primal_res = norm(U - Z, "fro");

    // Dual residual: rho * ||z - z_prev||
    // For simplicity, use ||z|| as proxy
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

    // Standard ADMM rho adaptation
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

    // Linearize dynamics about reference (or current state for better accuracy)
    linearizeDynamics(x_ref, u_ref, B_field, dynamics_info);

    // Solve Riccati equation for unconstrained LQR gains
    solveRiccati();

    // Initialize trajectories
    mat X = zeros(n, N + 1);
    mat U = zeros(m, N);

    // Use warm start if available
    if (has_warm_start && X_warm.n_cols == (uword)(N + 1)) {
        X = X_warm;
        U = U_warm;
    } else {
        // Initialize with reference
        X = X_ref_local;
        U = U_ref_local;
        Z = U;
        Y.zeros();
    }

    // ADMM iterations
    double primal_res = 0.0, dual_res = 0.0;
    for (int iter = 0; iter < settings.max_iter; iter++) {
        // x-update: solve unconstrained LQR
        admm_x_update(X, U, x_current, X_ref_local, U_ref_local);

        // z-update: project onto constraints
        admm_z_update(U);

        // y-update: dual variable
        admm_y_update(U);

        result.iterations = iter + 1;

        // Check convergence periodically
        if ((iter + 1) % settings.check_interval == 0) {
            tie(primal_res, dual_res) = computeResiduals(U);

            if (checkConvergence(primal_res, dual_res)) {
                result.converged = true;
                break;
            }

            // Adapt rho
            updateRho(primal_res, dual_res);
        }
    }

    // Final residuals
    tie(primal_res, dual_res) = computeResiduals(U);
    result.primal_residual = primal_res;
    result.dual_residual = dual_res;

    // Extract optimal control (use projected value)
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
    // Shift solution by one timestep
    int N = settings.track_horizon;

    if (X_prev.n_cols >= 2) {
        X_warm = zeros(n, N + 1);
        int copy_cols = std::min((int)X_prev.n_cols - 1, N + 1);
        X_warm.head_cols(copy_cols) = X_prev.tail_cols(copy_cols);
        // Extrapolate last column
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
        // Extrapolate last column
        if (copy_cols < N) {
            for (int k = copy_cols; k < N; k++) {
                U_warm.col(k) = U_warm.col(copy_cols - 1);
            }
        }
    }

    // Also shift Z and Y
    if (Z.n_cols > 1) {
        Z.head_cols(N - 1) = Z.tail_cols(N - 1);
    }
    if (Y.n_cols > 1) {
        Y.head_cols(N - 1) = Y.tail_cols(N - 1);
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
