#include "kernels.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
namespace omni::hardware::sensors::kernels {
namespace {
V finish(V value, const V &bias, const M &noise_cholesky, double saturation,
         bool with_bias, bool with_noise, const V *noise_draw) {
  Eigen::Map<Eigen::Vector3d> measured(value.data());
  if (with_bias)
    measured += Eigen::Map<const Eigen::Vector3d>(bias.data());
  if (with_noise && noise_draw) {
    using Matrix = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    measured += Eigen::Map<const Matrix>(noise_cholesky[0].data()) *
                Eigen::Map<const Eigen::Vector3d>(noise_draw->data());
  }
  for (double &component : value)
    component = std::min(std::max(component, -saturation), saturation);
  return value;
}
} // namespace
V rate_measurement(const V &angular_velocity_body, const Q &orientation,
                   const Q &misalignment, const V &bias,
                   const M &noise_cholesky, double saturation, bool with_bias,
                   bool with_noise, const V *noise_draw) {
  return finish(
      rotate_body_to_sensor(angular_velocity_body, orientation, misalignment),
      bias, noise_cholesky, saturation, with_bias, with_noise, noise_draw);
}
V vector_measurement(const V &vector_inertial, const Q &attitude,
                     const Q &orientation, const Q &misalignment, const V &bias,
                     const M &noise_cholesky, double saturation, bool with_bias,
                     bool with_noise, const V *noise_draw) {
  return finish(rotate_inertial_to_sensor(vector_inertial, attitude,
                                          orientation, misalignment),
                bias, noise_cholesky, saturation, with_bias, with_noise,
                noise_draw);
}
M magnetometer_information(const V &field, const Q &attitude,
                           const Q &orientation, const Q &misalignment,
                           const M &noise, const M &field_covariance) {
  const M mounting = quaternion_matrix(multiply(misalignment, orientation));
  const V sensor_field =
      rotate_inertial_to_sensor(field, attitude, orientation, misalignment);
  const M cross = skew(sensor_field);
  const M inverse =
      inverse_from_cholesky(cholesky(add(noise, field_covariance)));
  return multiply(
      transpose(mounting),
      multiply(transpose(cross), multiply(inverse, multiply(cross, mounting))));
}
V accelerometer_measurement(const V &acceleration, const V &gravity,
                            const Q &attitude, const Q &orientation,
                            const Q &misalignment, const V &bias,
                            const M &noise_cholesky, double saturation,
                            bool with_bias, bool with_noise,
                            const V *noise_draw) {
  Eigen::Vector3d force =
      Eigen::Map<const Eigen::Vector3d>(acceleration.data()) -
      Eigen::Map<const Eigen::Vector3d>(gravity.data());
  const V specific_force{{force[0], force[1], force[2]}};
  return finish(rotate_inertial_to_sensor(specific_force, attitude, orientation,
                                          misalignment),
                bias, noise_cholesky, saturation, with_bias, with_noise,
                noise_draw);
}
double scalar_sun_measurement(const V &sun, const Q &attitude,
                              const Q &orientation, const Q &misalignment,
                              const V &normal, double sensitivity, double bias,
                              double noise_std, double saturation,
                              bool with_bias, bool with_noise, double draw) {
  const V sensor_sun =
      rotate_inertial_to_sensor(sun, attitude, orientation, misalignment);
  const double projection = std::max(
      0.0, Eigen::Map<const Eigen::Vector3d>(normal.data())
               .dot(Eigen::Map<const Eigen::Vector3d>(sensor_sun.data())));
  double value = sensitivity * projection;
  if (with_bias)
    value += bias;
  if (with_noise)
    value += noise_std * draw;
  return std::min(std::max(value, 0.0), saturation);
}
bool coarse_sun_available(const V &sun, const Q &attitude, const Q &orientation,
                          const Q &misalignment, const V &normal, double fov,
                          bool enabled, bool visible) {
  if (!enabled || !visible)
    return false;
  const V body = multiply(quaternion_matrix(normalize(attitude)), sun);
  const M mounting = quaternion_matrix(multiply(misalignment, orientation));
  const V normal_body = multiply(transpose(mounting), normal);
  return Eigen::Map<const Eigen::Vector3d>(normal_body.data())
             .dot(Eigen::Map<const Eigen::Vector3d>(body.data())) >=
         std::cos(fov);
}
namespace {
V coarse_jacobian(const V &sun, const Q &attitude, const Q &orientation,
                  const Q &misalignment, const V &normal, double sensitivity,
                  const V &delta, int kind) {
  const V body = multiply(quaternion_matrix(normalize(attitude)), sun);
  const M mounting = quaternion_matrix(multiply(misalignment, orientation));
  const V normal_body = multiply(transpose(mounting), normal);
  M jac{};
  if (kind == 1)
    jac = attitude_vector_jacobian_mrp(body, delta);
  else
    jac = attitude_vector_jacobian_angle(body, delta);
  const V row = multiply(transpose(jac), normal_body);
  return {{sensitivity * row[0], sensitivity * row[1], sensitivity * row[2]}};
}
} // namespace
V coarse_sun_attitude_jacobian(const V &sun, const Q &attitude,
                               const Q &orientation, const Q &misalignment,
                               const V &normal, double sensitivity,
                               const Q &delta, int) {
  const V body = multiply(quaternion_matrix(normalize(attitude)), sun);
  const Q q = normalize(delta);
  const V rotated = multiply(quaternion_matrix(q), body);
  const M right = scale(
      -2.0,
      multiply(skew(rotated),
               add(scale(q[0], identity()),
                   add(skew(V{{q[1], q[2], q[3]}}),
                       scale(1.0 / q[0], outer(V{{q[1], q[2], q[3]}},
                                               V{{q[1], q[2], q[3]}}))))));
  const V normal_body = multiply(
      transpose(quaternion_matrix(multiply(misalignment, orientation))),
      normal);
  const V row = multiply(transpose(right), normal_body);
  return {{sensitivity * row[0], sensitivity * row[1], sensitivity * row[2]}};
}
V coarse_sun_attitude_jacobian(const V &sun, const Q &attitude,
                               const Q &orientation, const Q &misalignment,
                               const V &normal, double sensitivity,
                               const V &delta, int kind) {
  return coarse_jacobian(sun, attitude, orientation, misalignment, normal,
                         sensitivity, delta, kind);
}
M coarse_sun_information(const V &sun, const Q &attitude, const Q &orientation,
                         const Q &misalignment, const V &normal,
                         double sensitivity, double variance) {
  const V body = multiply(quaternion_matrix(normalize(attitude)), sun);
  const V normal_body = multiply(
      transpose(quaternion_matrix(multiply(misalignment, orientation))),
      normal);
  Eigen::Map<const Eigen::Vector3d> n(normal_body.data()), s(body.data());
  Eigen::Vector3d d = n.cross(s);
  const Eigen::Matrix3d information =
      (sensitivity * sensitivity / variance) * (d * d.transpose());
  M result{};
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(result[0].data()) =
      information;
  return result;
}
std::array<double, 2> fine_direction_measurement(
    const V &sun, const Q &attitude, const Q &orientation,
    const Q &misalignment, const std::array<double, 2> &bias,
    const std::array<std::array<double, 2>, 2> &noise_cholesky, bool with_bias,
    bool with_noise, const std::array<double, 2> *draw) {
  const V s =
      rotate_inertial_to_sensor(sun, attitude, orientation, misalignment);
  std::array<double, 2> out{{std::atan2(s[0], s[2]), std::atan2(s[1], s[2])}};
  Eigen::Map<Eigen::Vector2d> y(out.data());
  if (with_bias)
    y += Eigen::Map<const Eigen::Vector2d>(bias.data());
  if (with_noise && draw) {
    using Mat = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
    y += Eigen::Map<const Mat>(noise_cholesky[0].data()) *
         Eigen::Map<const Eigen::Vector2d>(draw->data());
  }
  return out;
}
bool fine_direction_available(const V &sun, const Q &attitude,
                              const Q &orientation, const Q &misalignment,
                              const std::array<double, 2> &fov, bool enabled,
                              bool visible) {
  if (!enabled || !visible)
    return false;
  auto angles = fine_direction_measurement(
      sun, attitude, orientation, misalignment, {{0, 0}},
      {{{{0, 0}}, {{0, 0}}}}, false, false, nullptr);
  return std::abs(angles[0]) <= fov[0] && std::abs(angles[1]) <= fov[1];
}
namespace {
using J = std::array<std::array<double, 3>, 2>;
J fine_jacobian(const V &sun, const Q &attitude, const Q &orientation,
                const Q &misalignment, const V &delta, int kind) {
  const V body = multiply(quaternion_matrix(normalize(attitude)), sun);
  const M mount = quaternion_matrix(multiply(misalignment, orientation));
  Q dq{};
  M direction{};
  if (kind == 1) {
    double r = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2],
           den = 1 + r;
    dq = {{(1 - r) / den, 2 * delta[0] / den, 2 * delta[1] / den,
           2 * delta[2] / den}};
    direction = attitude_vector_jacobian_mrp(body, delta);
  } else if (kind == 2) {
    dq = rotation_vector_quaternion(delta);
    direction = attitude_vector_jacobian_angle(body, delta);
  } else
    return {};
  V mounted = multiply(mount, multiply(quaternion_matrix(dq), body));
  double x = mounted[0], y = mounted[1], z = mounted[2], xd = x * x + z * z,
         yd = y * y + z * z;
  Eigen::Matrix<double, 2, 3> d;
  d << z / xd, 0, -x / xd, 0, z / yd, -y / yd;
  Eigen::Matrix<double, 2, 3> result =
      d *
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          mount[0].data()) *
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          direction[0].data());
  J out{};
  Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>(out[0].data()) =
      result;
  return out;
}
} // namespace
std::array<std::array<double, 3>, 2>
fine_sun_attitude_jacobian(const V &sun, const Q &attitude,
                           const Q &orientation, const Q &misalignment,
                           const Q &delta, int) {
  const V body = multiply(quaternion_matrix(normalize(attitude)), sun);
  const Q q = normalize(delta);
  const M mount = quaternion_matrix(multiply(misalignment, orientation));
  const V rotated = multiply(quaternion_matrix(q), body), v{{q[1], q[2], q[3]}};
  const M right =
      scale(-2., multiply(skew(rotated),
                          add(scale(q[0], identity()),
                              add(skew(v), scale(1. / q[0], outer(v, v))))));
  V mounted = multiply(mount, rotated);
  double x = mounted[0], y = mounted[1], z = mounted[2], xd = x * x + z * z,
         yd = y * y + z * z;
  Eigen::Matrix<double, 2, 3> d;
  d << z / xd, 0, -x / xd, 0, z / yd, -y / yd;
  Eigen::Matrix<double, 2, 3> result =
      d *
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          mount[0].data()) *
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          right[0].data());
  std::array<std::array<double, 3>, 2> out{};
  Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>(out[0].data()) =
      result;
  return out;
}
std::array<std::array<double, 3>, 2>
fine_sun_attitude_jacobian(const V &sun, const Q &attitude,
                           const Q &orientation, const Q &misalignment,
                           const V &delta, int kind) {
  return fine_jacobian(sun, attitude, orientation, misalignment, delta, kind);
}
M fine_sun_fisher_information(
    const std::array<double, 2> &angles,
    const std::array<std::array<double, 2>, 2> &covariance,
    const Q &orientation, const Q &misalignment) {
  double u = std::tan(angles[0]), v = std::tan(angles[1]);
  Eigen::Matrix<double, 2, 3> k;
  k << -u * v / (1 + u * u), 1., -v / (1 + u * u), -1., u * v / (1 + v * v),
      u / (1 + v * v);
  using C = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
  Eigen::Matrix2d cov = Eigen::Map<const C>(covariance[0].data());
  Eigen::Matrix3d mount =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          quaternion_matrix(multiply(misalignment, orientation))[0].data());
  Eigen::Matrix3d info =
      mount.transpose() * k.transpose() * cov.inverse() * k * mount;
  M out{};
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(out[0].data()) =
      info;
  return out;
}
} // namespace omni::hardware::sensors::kernels
