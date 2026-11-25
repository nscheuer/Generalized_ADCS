#include "sensor_sampling.hpp"
#include "../../environment/gravity/j_gravity_model.hpp"
#include "../../helpers/math/time/conversions.hpp"
#include <cmath>
#include <limits>
#include <type_traits>
namespace omni::propagation {
namespace {
using namespace omni::hardware::sensors;
using V = std::array<double, 3>;
V array(const Eigen::Vector3d &v) { return {v[0], v[1], v[2]}; }
const char *names[] = {"gyroscope",
                       "magnetometer",
                       "accelerometer",
                       "coarse_sun_sensor",
                       "fine_sun_sensor",
                       "fine_earth_sensor",
                       "gnss",
                       "autonomous_star_tracker"};
const int widths[] = {3, 3, 3, 1, 2, 2, 6, 4};
} // namespace
std::map<std::string, Eigen::MatrixXd>
sample_sensors(const Vector &times, const Eigen::MatrixXd &states, double epoch,
               double interval, Runtime &r,
               std::vector<omni::hardware::Sensor> sensors,
               std::uint64_t seed) {
  if (!(interval > 0) || states.rows() != times.size() || states.cols() < 13) {
    throw std::invalid_argument("invalid sensor sample trajectory or interval");
  }
  std::array<int, 8> counts{};
  std::vector<int> offsets;
  for (auto &sensor : sensors) {
    auto kind = sensor.index();
    offsets.push_back(counts[kind]);
    counts[kind] += widths[kind];
  }
  bool need_sun = counts[3] || counts[4] || counts[7];
  if (need_sun && !r.ephemeris) {
    r.ephemeris = std::make_unique<omni::environment::DE421>();
  }
  if (counts[1] && !r.igrf && !r.chaos) {
    if (r.magnetic_kind == 1) {
      r.igrf = std::make_unique<omni::environment::IGRF14>();
    } else if (r.magnetic_kind == 2) {
      r.chaos = std::make_unique<omni::environment::CHAOS8>(
          r.magnetic_include_static);
    } else {
      throw std::invalid_argument("magnetometer requires a magnetic model");
    }
  }
  std::map<std::string, Eigen::MatrixXd> result;
  for (int k = 0; k < 8; ++k) {
    for (auto suffix : {"_clean", "_reading"}) {
      result[std::string(names[k]) + suffix] = Eigen::MatrixXd::Constant(
          times.size(), counts[k], std::numeric_limits<double>::quiet_NaN());
    }
  }
  for (Eigen::Index row = 0; row < times.size(); ++row) {
    double tick = std::round(times[row] / interval);
    if (std::abs(times[row] / interval - tick) > 1e-9) {
      continue;
    }
    Vector y = states.row(row).transpose();
    double jd = epoch + times[row] / 86400.;
    std::array<double, 4> q{y[0], y[1], y[2], y[3]};
    V w = array(y.segment<3>(4)), pos = array(y.segment<3>(7)),
      vel = array(y.segment<3>(10));
    V earth{-pos[0], -pos[1], -pos[2]}, sun{}, field{}, acceleration{},
        gravity{};
    bool visible = false;
    if (need_sun) {
      Eigen::Vector3d sp =
          1000. * r.ephemeris->position(0, omni::time::utc_jd_to_tdb_jd(jd));
      sun = array(sp - y.segment<3>(7));
      visible = std::get<0>(omni::environment::eclipse_state(
                    y.segment<3>(7), sp, r.radius, 696340000.)) > 0;
    }
    if (counts[1]) {
      field =
          array(r.igrf ? r.igrf->field_inertial_tesla(y.segment<3>(7), jd)
                       : r.chaos->field_inertial_tesla(y.segment<3>(7), jd));
    }
    if (counts[2]) {
      acceleration = array(derivative(y, jd, r).segment<3>(10));
      gravity = array(omni::environment::gravity_acceleration_inertial_m_s2(
          y.segment<3>(7), jd, r.mu, r.radius, r.zonals));
    }
    for (std::size_t i = 0; i < sensors.size(); ++i) {
      std::visit(
          [&](auto &s) {
            using T = std::decay_t<decltype(s)>;
            if (!s.enabled) {
              return;
            }
            s.seed = seed + i;
            s.random_tick = static_cast<std::uint64_t>(tick);
            auto store = [&](const auto &a, const auto &b) {
              auto &clean =
                  result[std::string(names[sensors[i].index()]) + "_clean"];
              auto &reading =
                  result[std::string(names[sensors[i].index()]) + "_reading"];
              for (std::size_t j = 0; j < a.size(); ++j) {
                clean(row, offsets[i] + j) = a[j];
                reading(row, offsets[i] + j) = b[j];
              }
            };
            if constexpr (std::is_same_v<T, Gyroscope>) {
              store(s.noiseless_biasfree_measurement(w),
                    s.noisy_biased_measurement(w));
            } else if constexpr (std::is_same_v<T, Magnetometer>) {
              store(s.noiseless_biasfree_measurement(field, q),
                    s.noisy_biased_measurement(field, q));
            } else if constexpr (std::is_same_v<T, Accelerometer>) {
              store(s.noiseless_biasfree_measurement(acceleration, gravity, q),
                    s.noisy_biased_measurement(acceleration, gravity, q));
            } else if constexpr (std::is_same_v<T, GNSS>) {
              store(s.noiseless_biasfree_measurement(pos, vel),
                    s.noisy_biased_measurement(pos, vel));
            } else if constexpr (std::is_same_v<T, AutonomousStarTracker>) {
              if (s.availability(q, earth, sun)) {
                store(s.noiseless_biasfree_measurement(q),
                      s.noisy_biased_measurement(q));
              }
            } else if constexpr (std::is_same_v<T, CoarseSunSensor>) {
              if (s.availability(sun, q, visible)) {
                store(
                    std::array<double, 1>{
                        s.noiseless_biasfree_measurement(sun, q)},
                    std::array<double, 1>{s.noisy_biased_measurement(sun, q)});
              }
            } else {
              const auto &direction =
                  std::is_same_v<T, FineEarthSensor> ? earth : sun;
              if (s.availability(direction, q,
                                 std::is_same_v<T, FineEarthSensor> ||
                                     visible)) {
                store(s.noiseless_biasfree_measurement(direction, q),
                      s.noisy_biased_measurement(direction, q));
              }
            }
          },
          sensors[i]);
    }
  }
  return result;
}
} // namespace omni::propagation
