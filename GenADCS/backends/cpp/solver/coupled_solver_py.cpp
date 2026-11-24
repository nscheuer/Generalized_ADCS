#include "coupled_solver.hpp"

#include <pybind11/pybind11.h>

#include <limits>
#include <utility>

namespace py = pybind11;

namespace
{
struct PythonPropagator
{
    py::object value;

    double dt() const
    {
        return py::cast<double>(value.attr("dt"));
    }

    py::object step(py::object state, py::object satellite, py::object environment, py::object dynamics) const
    {
        return value.attr("step")(std::move(state), std::move(satellite), std::move(environment), std::move(dynamics));
    }
};

using BoundCoupledSolver = omni::solver::CoupledSolver<PythonPropagator>;

int read_chunk_steps(const py::handle value)
{
    if (!PyLong_CheckExact(value.ptr()))
    {
        throw py::type_error("progress_chunk_steps must be a positive integer");
    }
    const long long chunk_steps = py::cast<long long>(value);
    if (chunk_steps < 1 || chunk_steps > std::numeric_limits<int>::max())
    {
        throw py::value_error("progress_chunk_steps must be a positive integer");
    }
    return static_cast<int>(chunk_steps);
}

double read_final_time(const py::handle value)
{
    return py::module_::import("builtins").attr("float")(value).cast<double>();
}
} // namespace

void bind_coupled_solver(py::module_ &module)
{
    py::class_<BoundCoupledSolver>(module, "CoupledSolver")
        .def(py::init([](py::object propagator, py::object tf, bool progress, py::object progress_chunk_steps) {
                 return BoundCoupledSolver(PythonPropagator{std::move(propagator)}, read_final_time(tf), progress,
                                           read_chunk_steps(progress_chunk_steps));
             }),
             py::arg("propagator"), py::arg("tf"), py::arg("progress") = false, py::arg("progress_chunk_steps") = 500)
        .def_property(
            "propagator", [](const BoundCoupledSolver &self) { return self.propagator.value; },
            [](BoundCoupledSolver &self, py::object value) { self.propagator.value = std::move(value); })
        .def_readwrite("tf", &BoundCoupledSolver::tf)
        .def_readwrite("progress", &BoundCoupledSolver::progress)
        .def_readwrite("progress_chunk_steps", &BoundCoupledSolver::progress_chunk_steps)
        .def_property_readonly("dt", &BoundCoupledSolver::dt)
        .def("step", [](BoundCoupledSolver &self, py::object state, py::object satellite, py::object environment,
                        py::object dynamics) {
            return self.step(std::move(state), std::move(satellite), std::move(environment), std::move(dynamics));
        });
}
