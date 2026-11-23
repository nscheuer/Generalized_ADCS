#include "simulate.hpp"
#include <pybind11/numpy.h>
#include <string>
#include <cmath>
namespace py = pybind11;
namespace omni::simulation {
py::object simulate(py::object satellite, py::object environment,
                          py::object dynamics, py::object state,
                          py::object solver, const std::string& backend,
                          py::object resultsformat) {
    if (backend != "C") throw py::value_error("backend must be C");
    py::object solver_module = py::module_::import("GenADCS.backends.cpp.solver._solver");
    if (py::isinstance(solver, solver_module.attr("Solver"))) solver = solver.attr("coupled");
    if (solver.is_none() || !py::isinstance(solver, solver_module.attr("CoupledSolver")))
        throw py::type_error("solver must be a CoupledSolver or a Solver containing one");
    py::object format = resultsformat.is_none()
        ? py::module_::import("GenADCS.backends.shared.simulationresult.resultsformat").attr("ResultsFormat")()
        : resultsformat;
    std::size_t n = format.attr("N").cast<std::size_t>();
    py::object native_states = py::module_::import("GenADCS.backends.cpp._state");
    py::object python_states = py::module_::import("GenADCS.backends.python.state");
    const bool native_state = py::isinstance(state, native_states.attr("State"));
    if (!native_state && !py::isinstance(state, python_states.attr("State")))
        throw py::type_error("state must be an GenADCS State");
    py::object result = solver.attr("propagator").attr("integrate")(
        state, satellite, environment, dynamics, solver.attr("tf"),
        py::arg("progress") = solver.attr("progress"),
        py::arg("progress_chunk_steps") = solver.attr("progress_chunk_steps"),
        py::arg("record_every") = n);
    py::tuple trajectory = result.cast<py::tuple>();
    if (trajectory.size() != 2) throw py::value_error("propagator.integrate must return times and state vectors");
    py::array_t<double, py::array::c_style | py::array::forcecast> vectors(trajectory[1]);
    if (vectors.ndim() != 2 || vectors.shape(0) < 1 || vectors.shape(1) < 13)
        throw py::value_error("state vectors must have at least one row and 13 columns");
    py::object initial_orbit = state.attr("orbit");
    double epoch_jd = initial_orbit.attr("julian_date").cast<double>();
    py::object times = py::reinterpret_borrow<py::object>(trajectory[0]);
    double interval = solver.attr("propagator").attr("dt").cast<double>() * n;
    double last_time = times[py::int_(-1)].cast<double>();
    bool on_grid = std::abs(last_time / interval - std::round(last_time / interval)) < 1e-9;
    py::ssize_t end = vectors.shape(0) - (on_grid ? 0 : 1);
    py::slice rows(0, end, 1);
    py::object sampled_vectors = py::reinterpret_borrow<py::object>(
        vectors[py::make_tuple(rows, py::slice(0, vectors.shape(1), 1))]);
    return py::module_::import("GenADCS.backends.shared.simulationresult.record")
        .attr("record_samples")(times[rows], sampled_vectors, format,
                                 satellite, environment, dynamics, epoch_jd, backend);
}
}
