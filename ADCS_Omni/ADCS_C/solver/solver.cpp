#include "solver.hpp"
namespace py=pybind11;
namespace omni::solver {
Solver::Solver(py::object d,py::object o,py::object a,py::object c):dynamics(d),orbital(o),attitude(a),coupled(c){if((c.is_none()&&(o.is_none()||a.is_none()))||(!c.is_none()&&(!o.is_none()||!a.is_none())))throw py::value_error("provide a coupled solver or both orbital and attitude solvers");}
}
