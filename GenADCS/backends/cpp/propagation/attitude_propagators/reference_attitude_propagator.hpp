#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace omni::propagation {

enum class ReferenceAttitudeMode { Nadir = 0, AlongTrack = 1, Sun = 2 };

struct ReferenceAttitude {
  Eigen::Vector4d quaternion;
  Eigen::Vector3d angular_velocity;
};

class ReferenceAttitudePropagator {
public:
  ReferenceAttitudePropagator(double dt, ReferenceAttitudeMode mode);
  double dt() const noexcept;
  ReferenceAttitudeMode mode() const noexcept;

  // Returns body-from-inertial [w, x, y, z] and body-frame angular rate.
  // Sun vectors are in the same inertial frame and length units as position.
  ReferenceAttitude attitude(const Eigen::Vector3d &position,
                            const Eigen::Vector3d &velocity, double mu,
                            const Eigen::Vector3d &sun_position = Eigen::Vector3d::Zero(),
                            const Eigen::Vector3d &sun_velocity = Eigen::Vector3d::Zero()) const;
  void apply(Eigen::Ref<Eigen::MatrixXd> states,
             const std::vector<double> &times, double initial_julian_date,
             double mu) const;

  static ReferenceAttitudeMode parse_mode(const std::string &mode);

private:
  double dt_;
  ReferenceAttitudeMode mode_;
};

} // namespace omni::propagation
