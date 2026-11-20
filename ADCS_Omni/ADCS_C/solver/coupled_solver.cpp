#include "coupled_solver.hpp"
#include <cmath>
namespace py=pybind11;
namespace omni::solver {
CoupledSolver::CoupledSolver(py::object p,double t,bool prog,int chunk):propagator(p),tf(t),progress(prog),progress_chunk_steps(chunk){if(!std::isfinite(tf)||tf<0)throw py::value_error("tf must be finite and nonnegative seconds");if(chunk<=0)throw py::value_error("progress_chunk_steps must be a positive integer");}
double CoupledSolver::dt()const{return propagator.attr("dt").cast<double>();}
py::object CoupledSolver::step(py::object state,py::object sat,py::object env,py::object dyn)const{return propagator.attr("step")(state,sat,env,dyn);}
}
