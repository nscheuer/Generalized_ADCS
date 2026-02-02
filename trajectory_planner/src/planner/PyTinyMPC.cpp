/**
 * PyTinyMPC.cpp - Python bindings for TinyMPC tracking controller
 *
 * Implementation of the PyTinyMPC wrapper class and pybind11 module registration.
 */

#include "PyTinyMPC.hpp"
#include "PlannerUtil.hpp"
#include <stdexcept>

namespace py = pybind11;
using namespace arma;
using namespace std;


// ============================================================================
// PyTinyMPC Implementation
// ============================================================================

PyTinyMPC::PyTinyMPC(Satellite sat)
    : mpc(sat, TinyMPCSettings())
{
}

PyTinyMPC::PyTinyMPC(Satellite sat, TINYMPC_SETTINGS_PY_FORM settings_py)
    : mpc(sat, settingsPy2Cpp(settings_py))
{
}

void PyTinyMPC::setSettings(TINYMPC_SETTINGS_PY_FORM settings_py) {
    mpc.setSettings(settingsPy2Cpp(settings_py));
}

TINYMPC_SETTINGS_PY_FORM PyTinyMPC::getSettings() const {
    return settingsCpp2Py(mpc.getSettings());
}

void PyTinyMPC::setCostMatrices(
    py::array_t<double> Q_py,
    py::array_t<double> R_py,
    py::array_t<double> Qf_py
) {
    mat Q = numpyToArmaMatrix(Q_py);
    mat R = numpyToArmaMatrix(R_py);
    mat Qf = numpyToArmaMatrix(Qf_py);
    mpc.setCostMatrices(Q, R, Qf);
}

void PyTinyMPC::loadReferenceFromALTRO(
    py::array_t<double> X_ref_py,
    py::array_t<double> U_ref_py,
    py::array_t<double> K_ref_py,
    py::array_t<double> times_py,
    double dt
) {
    mat X_ref = numpyToArmaMatrix(X_ref_py);
    mat U_ref = numpyToArmaMatrix(U_ref_py);
    vec times = numpyToArmaVector(times_py);

    // Build TrajectorySegment
    TrajectorySegment traj_seg;

    // Normalize to (n, N+1) format - state dimension is first
    int n = mpc.getStateDim();
    int m = mpc.getControlDim();

    if (X_ref.n_rows != (uword)n) {
        // Transpose if needed
        X_ref = X_ref.t();
    }
    if (U_ref.n_rows != (uword)m) {
        U_ref = U_ref.t();
    }

    traj_seg.X_ref = X_ref;
    traj_seg.U_ref = U_ref;
    traj_seg.times = times;
    traj_seg.dt_ref = dt;

    // Handle optional K_ref (may be empty or flattened)
    py::buffer_info K_buf = K_ref_py.request();
    if (K_buf.size > 0) {
        mat K_flat = numpyToArmaMatrix(K_ref_py);
        // K gains are stored flattened as (m*n_err, N_K) where n_err = n-1
        // N_K is typically N-1 or N (depending on whether terminal gain exists)
        int n_err = n - 1;  // Reduced quaternion representation
        if (K_flat.n_rows == (uword)(m * n_err)) {
            int N_K = K_flat.n_cols;  // Use actual K columns, not U columns
            // Reshape into cube (m, n_err, N_K)
            traj_seg.K_ref = cube(m, n_err, N_K);
            for (int k = 0; k < N_K; k++) {
                traj_seg.K_ref.slice(k) = reshape(K_flat.col(k), m, n_err);
            }
        }
    }

    mpc.loadReferenceTrajectory(traj_seg);
}

TINYMPC_RESULT_PY_FORM PyTinyMPC::solve(
    py::array_t<double> x_current_py,
    double t_current,
    py::array_t<double> B_field_py,
    py::array_t<double> sun_vec_py,
    DYNAMICS_INFO_PY_FORM_TINY dynamics_info_py
) {
    // Convert inputs
    vec x_current = numpyToArmaVector(x_current_py);
    vec3 B_field = numpyToArmaVector(B_field_py);
    vec3 sun_vec = numpyToArmaVector(sun_vec_py);

    // Unpack dynamics info tuple
    vec3 B_dyn = numpyToArmaVector(std::get<0>(dynamics_info_py));
    vec3 R_eci = numpyToArmaVector(std::get<1>(dynamics_info_py));
    int prop_torq_on = std::get<2>(dynamics_info_py);
    vec3 V_eci = numpyToArmaVector(std::get<3>(dynamics_info_py));
    vec3 S_eci = numpyToArmaVector(std::get<4>(dynamics_info_py));
    int dist_on = std::get<5>(dynamics_info_py);

    DYNAMICS_INFO_FORM dynamics_info = make_tuple(B_dyn, R_eci, prop_torq_on, V_eci, S_eci, dist_on);

    // Solve
    TinyMPCResult result = mpc.solve(x_current, t_current, B_field, sun_vec, dynamics_info);

    // Convert outputs
    py::array_t<double> u_opt_py = armaVectorToNumpy(result.u_opt);
    py::array_t<double> X_pred_py = armaMatrixToNumpy(result.X_pred);
    py::array_t<double> U_pred_py = armaMatrixToNumpy(result.U_pred);

    return std::make_tuple(
        u_opt_py,
        X_pred_py,
        U_pred_py,
        result.iterations,
        result.solve_time_ms,
        result.converged,
        result.tracking_error
    );
}

py::tuple PyTinyMPC::getReference(double t) const {
    auto [x_ref, u_ref] = mpc.getReference(t);
    return py::make_tuple(
        armaVectorToNumpy(x_ref),
        armaVectorToNumpy(u_ref)
    );
}

bool PyTinyMPC::hasValidReference() const {
    return mpc.hasValidReference();
}

py::tuple PyTinyMPC::getReferenceTimeRange() const {
    auto [t_start, t_end] = mpc.getReferenceTimeRange();
    return py::make_tuple(t_start, t_end);
}

void PyTinyMPC::warmStart(py::array_t<double> X_prev_py, py::array_t<double> U_prev_py) {
    mat X_prev = numpyToArmaMatrix(X_prev_py);
    mat U_prev = numpyToArmaMatrix(U_prev_py);
    mpc.warmStart(X_prev, U_prev);
}

void PyTinyMPC::reset() {
    mpc.reset();
}


// ============================================================================
// Pybind11 Module Registration
// ============================================================================

PYBIND11_MODULE(pytinympc, m) {
    m.doc() = "TinyMPC tracking controller for spacecraft attitude control";

    // Import pysat first to ensure Satellite type binding is registered
    py::module_::import("trajectory_planner.build.pysat");

    py::class_<PyTinyMPC>(m, "TinyMPCController")
        .def(py::init<Satellite>(),
             py::arg("satellite"),
             "Construct TinyMPC with satellite and default settings")

        .def(py::init<Satellite, TINYMPC_SETTINGS_PY_FORM>(),
             py::arg("satellite"),
             py::arg("settings"),
             "Construct TinyMPC with satellite and settings tuple")

        .def("setSettings", &PyTinyMPC::setSettings,
             py::arg("settings"),
             "Update solver settings")

        .def("getSettings", &PyTinyMPC::getSettings,
             "Get current settings as tuple")

        .def("setCostMatrices", &PyTinyMPC::setCostMatrices,
             py::arg("Q"), py::arg("R"), py::arg("Qf"),
             "Set tracking cost matrices (Q: state, R: control, Qf: terminal)")

        .def("loadReferenceFromALTRO", &PyTinyMPC::loadReferenceFromALTRO,
             py::arg("X_ref"), py::arg("U_ref"), py::arg("K_ref"),
             py::arg("times"), py::arg("dt"),
             "Load reference trajectory from ALTRO output")

        .def("solve", &PyTinyMPC::solve,
             py::arg("x_current"), py::arg("t_current"),
             py::arg("B_field"), py::arg("sun_vec"), py::arg("dynamics_info"),
             "Solve tracking MPC. Returns (u_opt, X_pred, U_pred, iters, time_ms, converged, error)")

        .def("getReference", &PyTinyMPC::getReference,
             py::arg("t"),
             "Get interpolated reference at time t. Returns (x_ref, u_ref)")

        .def("hasValidReference", &PyTinyMPC::hasValidReference,
             "Check if reference trajectory is loaded")

        .def("getReferenceTimeRange", &PyTinyMPC::getReferenceTimeRange,
             "Get time range of loaded reference. Returns (t_start, t_end)")

        .def("warmStart", &PyTinyMPC::warmStart,
             py::arg("X_prev"), py::arg("U_prev"),
             "Warm start from previous solution")

        .def("reset", &PyTinyMPC::reset,
             "Reset solver state")

        .def("getStateDim", &PyTinyMPC::getStateDim,
             "Get state dimension")

        .def("getControlDim", &PyTinyMPC::getControlDim,
             "Get control dimension");
}
