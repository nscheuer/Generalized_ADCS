#include "projected_area.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using omni::dynamics::M;
using omni::dynamics::Q;
using omni::dynamics::V;

void bind_drag_model_projected_area(py::module_ &m)
{
    m.def("projected_area_drag_force", [](const V &v, const Q &q, double rho, const Eigen::MatrixXd &n,
                                          const Eigen::VectorXd &a, const Eigen::VectorXd &cd) {
        if (n.rows() != a.size() || cd.size() != a.size())
            throw py::value_error("panel arrays must have matching row counts");
        return omni::dynamics::drag(v, q, rho, n, a, cd, Eigen::MatrixXd(), false);
    });
    m.def("projected_area_drag_torque",
          [](const V &v, const Q &q, double rho, const Eigen::MatrixXd &n, const Eigen::MatrixXd &arms,
             const Eigen::VectorXd &a, const Eigen::VectorXd &cd) {
              if (n.rows() != a.size() || arms.rows() != a.size() || cd.size() != a.size())
                  throw py::value_error("panel arrays must have matching row counts");
              return omni::dynamics::drag(v, q, rho, n, a, cd, arms, true);
          });
}
