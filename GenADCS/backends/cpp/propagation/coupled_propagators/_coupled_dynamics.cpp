#include "_coupled_dynamics.hpp"
#include "../../dynamics/drag_model/projected_area.hpp"
#include "../../dynamics/gravity_gradient_model/static.hpp"
#include "../../dynamics/residual_dipole_model/static.hpp"
#include "../../dynamics/srp_model/projected_area.hpp"
#include "../../dynamics/third_body_model/lunisolar.hpp"
#include "../../environment/atmosphere/atmospheric_model.hpp"
#include "../../environment/gravity/j_gravity_model.hpp"
#include "../../helpers/math/time/conversions.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace omni::propagation {
using V = Eigen::Vector3d;
using Q = Eigen::Vector4d;
using M = Eigen::Matrix3d;
void Runtime::initialize_models() {
  if (srp_force || srp_torque || third_body)
    ephemeris = std::make_unique<omni::environment::DE421>();
  if (residual_dipole) {
    if (magnetic_kind == 1)
      igrf = std::make_unique<omni::environment::IGRF14>();
    else if (magnetic_kind == 2)
      chaos =
          std::make_unique<omni::environment::CHAOS8>(magnetic_include_static);
    else
      throw std::invalid_argument("unsupported magnetic model");
  }
}
void project(Vector &y, const Runtime &r) {
  for (Eigen::Index i = 0; i < r.wheel_max_momentum.size(); ++i)
    y[13 + i] = std::clamp(y[13 + i], -r.wheel_max_momentum[i],
                           r.wheel_max_momentum[i]);
}
Vector derivative(const Vector &y, double jd, const Runtime &r) {

  Q q = y.segment<4>(0);
  q.normalize();
  const V omega = y.segment<3>(4), pos = y.segment<3>(7),
          vel = y.segment<3>(10);
  double density = 0, illumination = 0;
  V relative = vel, sun = V::Zero(), moon = V::Zero(), field = V::Zero();
  if (r.drag_force || r.drag_torque) {
    std::array<double, 7> history;
    history.fill(-1.);
    density = omni::environment::nrlmsise_density(pos, jd, 150., 150., 4., true,
                                                  history);
    relative -= omni::environment::co_rotating_atmosphere_velocity(pos, jd);
  }
  if (r.srp_force || r.srp_torque || r.third_body) {
    const double tdb = omni::time::utc_jd_to_tdb_jd(jd);
    sun = 1000. * r.ephemeris->position(0, tdb);
    if (r.third_body)
      moon = 1000. * r.ephemeris->position(2, tdb);
    if (r.srp_force || r.srp_torque)
      illumination = std::get<0>(
          omni::environment::eclipse_state(pos, sun, r.radius, 696340000.));
  }
  if (r.residual_dipole)
    field = r.igrf ? r.igrf->field_inertial_tesla(pos, jd)
                   : r.chaos->field_inertial_tesla(pos, jd);
  V accel = omni::environment::gravity_acceleration_inertial_m_s2(
        pos, jd, r.mu, r.radius, r.zonals),
    force = V::Zero(), torque = V::Zero();
  if (r.drag_force)
    force += omni::dynamics::drag(relative, q, density, r.normals, r.areas,
                                  r.drag_coefficients, r.arms, false);
  if (r.drag_torque)
    torque += omni::dynamics::drag(relative, q, density, r.normals, r.areas,
                                   r.drag_coefficients, r.arms, true);
  if (r.srp_force)
    force += omni::dynamics::srp(pos, sun, q, r.normals, r.areas,
                                 r.optical_coefficients, r.arms, illumination,
                                 false);
  if (r.srp_torque)
    torque +=
        omni::dynamics::srp(pos, sun, q, r.normals, r.areas,
                            r.optical_coefficients, r.arms, illumination, true);
  if (r.third_body)
    accel += omni::dynamics::lunisolar(pos, sun, moon, 1.32712440018e20,
                                       4.9048695e12);
  if (r.gravity_gradient)
    torque += omni::dynamics::gravity_gradient(pos, q, r.inertia, r.mu);
  if (r.residual_dipole)
    torque += omni::dynamics::residual_dipole(r.dipole, field, q);
  for (Eigen::Index i = 0; i < r.wheel_commands.size(); ++i)
    torque += r.wheel_axes.row(i).transpose() * r.wheel_commands[i];
  V h = r.inertia * omega;
  for (Eigen::Index i = 0; i < r.wheel_commands.size(); ++i)
    h += r.wheel_axes.row(i).transpose() * y[13 + i];
  V omegadot =
      (r.wheel_commands.size() ? r.inverse_bus_inertia : r.inverse_inertia) *
      (torque - omega.cross(h));
  Vector dy = Vector::Zero(y.size());
  dy[0] = 0.5 * q.tail<3>().dot(omega);
  dy[1] = -0.5 * (q[0] * omega[0] + omega[1] * q[3] - omega[2] * q[2]);
  dy[2] = -0.5 * (q[0] * omega[1] + omega[2] * q[1] - omega[0] * q[3]);
  dy[3] = -0.5 * (q[0] * omega[2] + omega[0] * q[2] - omega[1] * q[1]);
  dy.segment<3>(4) = omegadot;
  dy.segment<3>(7) = vel;
  dy.segment<3>(10) = accel + force / r.mass;
  for (Eigen::Index i = 0; i < r.wheel_commands.size(); ++i) {
    double rate = -r.wheel_commands[i] -
                  r.wheel_inertias[i] * r.wheel_axes.row(i).dot(omegadot);
    if ((y[13 + i] >= r.wheel_max_momentum[i] && rate > 0) ||
        (y[13 + i] <= -r.wheel_max_momentum[i] && rate < 0))
      rate = 0;
    dy[13 + i] = rate;
  }
  return dy;
}
} // namespace omni::propagation
