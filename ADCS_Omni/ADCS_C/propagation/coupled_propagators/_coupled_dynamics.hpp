#pragma once
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
namespace omni::propagation {
using Vector=Eigen::VectorXd;
Vector derivative(const Vector& y,double jd,const pybind11::object& runtime);
void project(Vector& y,const pybind11::object& runtime);
}
