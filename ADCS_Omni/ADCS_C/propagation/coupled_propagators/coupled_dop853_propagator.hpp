#pragma once
#include "_coupled_dynamics.hpp"
#include <tuple>
namespace omni::propagation { std::tuple<Vector,Vector,double> dop853_attempt(const Vector&,const Vector&,double,double,const pybind11::object&,double,const Vector&); }
