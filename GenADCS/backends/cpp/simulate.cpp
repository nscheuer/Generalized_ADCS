#include "simulate.hpp"
#include <pybind11/numpy.h>
#include <string>
namespace py = pybind11;
namespace omni::simulation {
SimulationResult simulate(py::object satellite, py::object environment,
                          py::object dynamics, py::object state,
                          py::object solver, const std::string& backend) {
    if (backend != "C") throw py::value_error("backend must be C");
    py::object solver_module = py::module_::import("GenADCS.backends.cpp.solver._solver");
    if (py::isinstance(solver, solver_module.attr("Solver"))) solver = solver.attr("coupled");
    if (solver.is_none() || !py::isinstance(solver, solver_module.attr("CoupledSolver")))
        throw py::type_error("solver must be a CoupledSolver or a Solver containing one");
    py::object native_states = py::module_::import("GenADCS.backends.cpp._state");
    py::object python_states = py::module_::import("GenADCS.backends.python.state");
    const bool native_state = py::isinstance(state, native_states.attr("State"));
    if (!native_state && !py::isinstance(state, python_states.attr("State")))
        throw py::type_error("state must be an GenADCS State");
    py::object result = solver.attr("propagator").attr("integrate")(
        state, satellite, environment, dynamics, solver.attr("tf"),
        py::arg("progress") = solver.attr("progress"),
        py::arg("progress_chunk_steps") = solver.attr("progress_chunk_steps"));
    py::tuple trajectory = result.cast<py::tuple>();
    if (trajectory.size() != 2) throw py::value_error("propagator.integrate must return times and state vectors");
    py::array_t<double, py::array::c_style | py::array::forcecast> vectors(trajectory[1]);
    if (vectors.ndim() != 2 || vectors.shape(0) < 1 || vectors.shape(1) < 13)
        throw py::value_error("state vectors must have at least one row and 13 columns");
    py::object numpy = py::module_::import("numpy");
    py::object row = py::reinterpret_borrow<py::object>(vectors[py::int_(-1)]);
    py::object attitude_data = numpy.attr("array")(row[py::slice(0, 4, 1)], py::arg("copy") = true);
    py::object rates = numpy.attr("array")(row[py::slice(4, 7, 1)], py::arg("copy") = true);
    py::object position = numpy.attr("array")(row[py::slice(7, 10, 1)], py::arg("copy") = true);
    py::object velocity = numpy.attr("array")(row[py::slice(10, 13, 1)], py::arg("copy") = true);
    py::object momentum = numpy.attr("array")(row[py::slice(13, vectors.shape(1), 1)], py::arg("copy") = true);
    py::object state_module = native_state ? native_states : python_states;
    py::object initial_orbit = state.attr("orbit");
    double final_jd = initial_orbit.attr("julian_date").cast<double>() + solver.attr("tf").cast<double>() / 86400.0;
    py::object attitude = state_module.attr("AttitudeState")(attitude_data, rates);
    py::object orbit = state_module.attr("OrbitalState").attr("from_cartesian")(
        position, velocity, final_jd, initial_orbit.attr("ephemeris"));
    py::object internal = state_module.attr("WheelMomentumState")(py::arg("wheel_momentum") = momentum);
    py::object final_state = state_module.attr("State")(attitude, orbit, internal);
    return {py::reinterpret_borrow<py::object>(trajectory[0]),
            py::reinterpret_borrow<py::object>(trajectory[1]), final_state};
}
}
