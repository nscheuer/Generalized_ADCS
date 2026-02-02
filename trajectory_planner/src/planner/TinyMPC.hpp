#ifndef TPR_TINYMPC_HPP
#define TPR_TINYMPC_HPP

#include <armadillo>
#include "Satellite.hpp"
#include "PlannerUtil.hpp"

/**
 * TinyMPC - Lightweight tracking MPC controller for spacecraft attitude control
 *
 * Used as the "Track" component in Plan & Track architecture:
 * - ALTRO generates optimal reference trajectory (Plan)
 * - TinyMPC tracks the reference trajectory in real-time (Track)
 *
 * Uses ADMM (Alternating Direction Method of Multipliers) to efficiently solve
 * the tracking QP with actuator constraints. Runs at higher frequency than
 * ALTRO for responsive disturbance rejection.
 */

// TinyMPC solver settings
struct TinyMPCSettings {
    int max_iter = 50;            // Maximum ADMM iterations
    double abs_tol = 1e-4;        // Absolute tolerance for convergence
    double rel_tol = 1e-4;        // Relative tolerance for convergence
    double rho = 1.0;             // ADMM penalty parameter
    double rho_min = 0.1;         // Minimum rho
    double rho_max = 10.0;        // Maximum rho
    bool adaptive_rho = true;     // Enable adaptive rho adjustment
    int check_interval = 10;      // Iterations between convergence checks
    int verbose = 0;              // Verbosity level

    // Tracking-specific settings
    int track_horizon = 10;       // MPC horizon for tracking (shorter than ALTRO)
    double track_dt = 0.1;        // Tracking timestep (can be faster than ALTRO dt)
};

// Result from TinyMPC solve
struct TinyMPCResult {
    arma::vec u_opt;              // Optimal control to apply
    arma::mat X_pred;             // Predicted state trajectory
    arma::mat U_pred;             // Predicted control trajectory
    int iterations;               // Number of ADMM iterations used
    double solve_time_ms;         // Solve time in milliseconds
    bool converged;               // Whether solver converged
    double primal_residual;       // Final primal residual
    double dual_residual;         // Final dual residual
    double tracking_error;        // State tracking error norm
};

// Reference trajectory segment (from ALTRO)
struct TrajectorySegment {
    arma::mat X_ref;              // Reference states (n x N+1)
    arma::mat U_ref;              // Reference controls (m x N)
    arma::cube K_ref;             // LQR gains from ALTRO (m x n x N), optional
    arma::vec times;              // Time stamps for reference points
    double dt_ref;                // Reference trajectory timestep
};

class TinyMPC {
public:
    TinyMPC();
    TinyMPC(Satellite sat_in, const TinyMPCSettings& settings = TinyMPCSettings());

    // Configure solver settings
    void setSettings(const TinyMPCSettings& settings);
    TinyMPCSettings getSettings() const;

    // Set tracking cost matrices (deviation from reference)
    // Q: state tracking error cost
    // R: control deviation cost
    // Qf: terminal tracking error cost
    void setCostMatrices(const arma::mat& Q, const arma::mat& R, const arma::mat& Qf);

    // Set default cost matrices based on satellite and cost settings
    void setCostFromSettings(const COST_SETTINGS_FORM& cost_settings);

    // Load reference trajectory from ALTRO solution
    void loadReferenceTrajectory(const TrajectorySegment& traj_segment);

    // Load reference from OPT_FORM (ALTRO output format)
    void loadReferenceFromALTRO(const OPT_FORM& altro_opt, const arma::vec& times, double dt);

    // Solve tracking MPC at current time
    // x_current: current measured/estimated state
    // t_current: current time (used to interpolate reference)
    // B_field: current magnetic field
    // sun_vec: current sun vector (for constraints)
    // dynamics_info: current dynamics information
    TinyMPCResult solve(const arma::vec& x_current,
                        double t_current,
                        const arma::vec3& B_field,
                        const arma::vec3& sun_vec,
                        const DYNAMICS_INFO_FORM& dynamics_info);

    // Get interpolated reference at time t
    std::pair<arma::vec, arma::vec> getReference(double t) const;

    // Get LQR gain at time t (if available from ALTRO)
    arma::mat getLQRGain(double t) const;

    // Check if reference trajectory is loaded and valid
    bool hasValidReference() const;

    // Get time range of loaded reference
    std::pair<double, double> getReferenceTimeRange() const;

    // Get state/control dimensions
    int getStateDim() const { return n; }
    int getControlDim() const { return m; }

    // Warm start from previous solution
    void warmStart(const arma::mat& X_prev, const arma::mat& U_prev);

    // Reset solver state (clear warm start, dual variables)
    void reset();

private:
    Satellite sat;
    int n;                        // Full state dimension (e.g., 8 for 1 RW)
    int n_err;                    // Reduced error state dimension (e.g., 7 - quaternion 4D -> 3D)
    int m;                        // Control dimension

    TinyMPCSettings settings;

    // Cost matrices for tracking (reduced error state dimension)
    arma::mat Q;                  // State tracking error cost (n_err x n_err)
    arma::mat R;                  // Control deviation cost (m x m)
    arma::mat Qf;                 // Terminal tracking error cost (n_err x n_err)

    // Control bounds (from satellite)
    arma::vec u_min;
    arma::vec u_max;

    // Reference trajectory
    TrajectorySegment ref_traj;
    bool has_reference = false;

    // Linearized dynamics at current operating point
    arma::mat A_lin;              // Linearized state matrix
    arma::mat B_lin;              // Linearized control matrix
    arma::vec c_lin;              // Affine term

    // ADMM variables (for warm starting)
    arma::mat Z;                  // Slack variable for controls
    arma::mat Y;                  // Dual variable
    arma::mat X_warm;             // Warm start states
    arma::mat U_warm;             // Warm start controls
    bool has_warm_start = false;

    // Precomputed Riccati solution for steady-state tracking
    std::vector<arma::mat> P_riccati;
    std::vector<arma::mat> K_riccati;
    bool riccati_computed = false;

    // Internal methods

    // Interpolate reference trajectory at time t
    void interpolateReference(double t, arma::vec& x_ref, arma::vec& u_ref, arma::mat* K_ref = nullptr) const;

    // Build local reference trajectory for MPC horizon
    void buildLocalReference(double t_start, arma::mat& X_ref_local, arma::mat& U_ref_local);

    // Linearize dynamics about reference point
    void linearizeDynamics(const arma::vec& x_op, const arma::vec& u_op,
                          const arma::vec3& B_field, const DYNAMICS_INFO_FORM& dynamics_info);

    // Solve Riccati equation for LQR gains
    void solveRiccati();

    // ADMM solver core
    void admm_x_update(arma::mat& X, arma::mat& U,
                       const arma::vec& x0,
                       const arma::mat& X_ref, const arma::mat& U_ref);
    void admm_z_update(const arma::mat& U);
    void admm_y_update(const arma::mat& U);

    // Convergence checking
    std::pair<double, double> computeResiduals(const arma::mat& U);
    bool checkConvergence(double primal_res, double dual_res);
    void updateRho(double primal_res, double dual_res);

    // Project control onto bounds
    arma::vec projectControl(const arma::vec& u);

    // Quaternion-aware state error computation
    arma::vec computeStateError(const arma::vec& x, const arma::vec& x_ref) const;
};

#endif // TPR_TINYMPC_HPP
