#ifndef TPR_PYTINYMPC_HPP
#define TPR_PYTINYMPC_HPP

/**
 * PyTinyMPC - Python bindings for TinyMPC tracking controller
 *
 * This wrapper exposes the C++ TinyMPC class to Python via pybind11,
 * handling numpy<->armadillo conversions for arrays and matrices.
 *
 * Usage from Python:
 *   import trajectory_planner.build.pytinympc as pytinympc
 *   mpc = pytinympc.TinyMPCController(satellite, settings_tuple)
 *   mpc.loadReferenceFromALTRO(X_ref, U_ref, K_ref, times, dt)
 *   result = mpc.solve(x_current, t_current, B_field, sun_vec, dynamics_info)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "TinyMPC.hpp"
#include "Satellite.hpp"
#include "../ArmaNumpy.hpp"

namespace py = pybind11;

// Python-friendly settings tuple:
// (max_iter, abs_tol, rel_tol, rho, rho_min, rho_max, adaptive_rho, check_interval,
//  track_horizon, track_dt, verbose)
#define TINYMPC_SETTINGS_PY_FORM std::tuple<int, double, double, double, double, double, bool, int, int, double, int>

// Python-friendly dynamics info tuple (same as ALTRO):
// (B_field, R_eci, prop_torq_on, V_eci, sun_vec, dist_on)
#define DYNAMICS_INFO_PY_FORM_TINY std::tuple<py::array_t<double>, py::array_t<double>, int, py::array_t<double>, py::array_t<double>, int>

// Python-friendly result tuple:
// (u_opt, X_pred, U_pred, iterations, solve_time_ms, converged, tracking_error)
#define TINYMPC_RESULT_PY_FORM std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, int, double, bool, double>


/**
 * Convert Python settings tuple to C++ TinyMPCSettings struct
 */
inline TinyMPCSettings settingsPy2Cpp(TINYMPC_SETTINGS_PY_FORM py_settings) {
    TinyMPCSettings settings;
    settings.max_iter = std::get<0>(py_settings);
    settings.abs_tol = std::get<1>(py_settings);
    settings.rel_tol = std::get<2>(py_settings);
    settings.rho = std::get<3>(py_settings);
    settings.rho_min = std::get<4>(py_settings);
    settings.rho_max = std::get<5>(py_settings);
    settings.adaptive_rho = std::get<6>(py_settings);
    settings.check_interval = std::get<7>(py_settings);
    settings.track_horizon = std::get<8>(py_settings);
    settings.track_dt = std::get<9>(py_settings);
    settings.verbose = std::get<10>(py_settings);
    return settings;
}

/**
 * Convert C++ TinyMPCSettings to Python tuple
 */
inline TINYMPC_SETTINGS_PY_FORM settingsCpp2Py(const TinyMPCSettings& settings) {
    return std::make_tuple(
        settings.max_iter,
        settings.abs_tol,
        settings.rel_tol,
        settings.rho,
        settings.rho_min,
        settings.rho_max,
        settings.adaptive_rho,
        settings.check_interval,
        settings.track_horizon,
        settings.track_dt,
        settings.verbose
    );
}


/**
 * PyTinyMPC - Python wrapper for TinyMPC C++ class
 *
 * Handles all numpy<->armadillo conversions and exposes a clean Python API.
 */
class PyTinyMPC {
public:
    /**
     * Construct with satellite only (uses default settings)
     */
    PyTinyMPC(Satellite sat);

    /**
     * Construct with satellite and settings tuple
     */
    PyTinyMPC(Satellite sat, TINYMPC_SETTINGS_PY_FORM settings_py);

    /**
     * Update solver settings
     */
    void setSettings(TINYMPC_SETTINGS_PY_FORM settings_py);

    /**
     * Get current settings as Python tuple
     */
    TINYMPC_SETTINGS_PY_FORM getSettings() const;

    /**
     * Set cost matrices for tracking
     *
     * @param Q_py  State tracking error cost (n x n)
     * @param R_py  Control deviation cost (m x m)
     * @param Qf_py Terminal tracking error cost (n x n)
     */
    void setCostMatrices(
        py::array_t<double> Q_py,
        py::array_t<double> R_py,
        py::array_t<double> Qf_py
    );

    /**
     * Load reference trajectory from ALTRO output
     *
     * @param X_ref_py  Reference states (n x N+1) or (N+1 x n)
     * @param U_ref_py  Reference controls (m x N) or (N x m)
     * @param K_ref_py  Optional LQR gains (flattened or empty)
     * @param times_py  Time stamps (N+1,)
     * @param dt        Trajectory timestep
     */
    void loadReferenceFromALTRO(
        py::array_t<double> X_ref_py,
        py::array_t<double> U_ref_py,
        py::array_t<double> K_ref_py,
        py::array_t<double> times_py,
        double dt
    );

    /**
     * Solve tracking MPC at current state
     *
     * @param x_current_py  Current state vector (n,)
     * @param t_current     Current time (J2000 centuries)
     * @param B_field_py    Magnetic field in ECI (3,)
     * @param sun_vec_py    Sun vector in ECI (3,)
     * @param dynamics_info_py  Dynamics info tuple
     *
     * @return Tuple: (u_opt, X_pred, U_pred, iterations, solve_time_ms, converged, tracking_error)
     */
    TINYMPC_RESULT_PY_FORM solve(
        py::array_t<double> x_current_py,
        double t_current,
        py::array_t<double> B_field_py,
        py::array_t<double> sun_vec_py,
        DYNAMICS_INFO_PY_FORM_TINY dynamics_info_py
    );

    /**
     * Get interpolated reference at time t
     *
     * @return Tuple: (x_ref, u_ref)
     */
    py::tuple getReference(double t) const;

    /**
     * Check if reference trajectory is loaded
     */
    bool hasValidReference() const;

    /**
     * Get time range of loaded reference
     *
     * @return Tuple: (t_start, t_end)
     */
    py::tuple getReferenceTimeRange() const;

    /**
     * Warm start from previous solution
     */
    void warmStart(py::array_t<double> X_prev_py, py::array_t<double> U_prev_py);

    /**
     * Reset solver state (clear warm start and ADMM variables)
     */
    void reset();

    /**
     * Get state dimension
     */
    int getStateDim() const { return mpc.getStateDim(); }

    /**
     * Get control dimension
     */
    int getControlDim() const { return mpc.getControlDim(); }

private:
    TinyMPC mpc;
};

#endif // TPR_PYTINYMPC_HPP
