#include "conversions.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error("failed: " #x);                                 \
  } while (false)
using namespace omni::frames;

int main() {
  constexpr double jd = 2451545.0;
  FrameContext context;
  context.jd_ut1 = jd;
  context.jd_tt = jd;
  context.position_gcrf = Vector3(7e6, 0, 0);
  context.velocity_gcrf = Vector3(0, 7500, 0);
  context.quaternion_body_from_gcrf = Quaternion(1, 0, 0, 0);
  for (Frame target :
       {Frame::Gcrf, Frame::Ecef, Frame::Teme, Frame::Lvlh, Frame::Body}) {
    const auto matrix = frame_rotation(Frame::Gcrf, target, context);
    CHECK((matrix * matrix.transpose()).isApprox(Matrix3::Identity(), 2e-12));
    CHECK(std::abs(matrix.determinant() - 1.0) < 2e-12);
  }
  CHECK(frame_rotation(Frame::Gcrf, Frame::Gcrf).isApprox(Matrix3::Identity()));
  CHECK(ecef_to_gcrf_matrix(jd, jd).isApprox(
      gcrf_to_ecef_matrix(jd, jd).transpose(), 1e-14));
  CHECK(teme_to_gcrf_matrix(jd).isApprox(gcrf_to_teme_matrix(jd).transpose(),
                                         1e-14));
  CHECK(ecef_to_teme_matrix(jd).isApprox(teme_to_ecef_matrix(jd).transpose(),
                                         1e-14));

  const auto lvlh =
      gcrf_to_lvlh_matrix(*context.position_gcrf, *context.velocity_gcrf);
  CHECK(lvlh.isApprox(Matrix3::Identity(), 1e-14));
  Vector3 zero = Vector3::Zero();
  bool rejected = false;
  try {
    (void)gcrf_to_lvlh_matrix(zero, *context.velocity_gcrf);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    (void)frame_rotation(Frame::Gcrf, Frame::Ecef);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    (void)parse_frame("invalid");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    FrameContext missing;
    missing.jd_ut1 = jd;
    (void)frame_rotation(Frame::Gcrf, Frame::Teme, missing);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    FrameContext missing;
    (void)frame_rotation(Frame::Gcrf, Frame::Lvlh, missing);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    FrameContext missing;
    (void)frame_rotation(Frame::Gcrf, Frame::Body, missing);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  rejected = false;
  try {
    (void)gcrf_to_ecef_matrix(std::numeric_limits<double>::quiet_NaN(), jd);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  Vector3 parallel;
  parallel << 2, 0, 0;
  rejected = false;
  try {
    (void)gcrf_to_lvlh_matrix(*context.position_gcrf, parallel);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);

  Vector3 x;
  x << 1, -2, 3;
  const auto converted = convert_vector(x, Frame::Gcrf, Frame::Ecef, context);
  CHECK(convert_vector(converted, Frame::Ecef, Frame::Gcrf, context)
            .isApprox(x, 2e-12));
  const auto equator = ecef_to_geodetic(Vector3(6378137.0, 0, 0));
  CHECK(std::abs(equator[0]) < 1e-10 && std::abs(equator[1]) < 1e-10 &&
        std::abs(equator[2]) < 1e-9);
  const auto pole = ecef_to_geodetic(Vector3(0, 0, 6356752.314245));
  CHECK(std::abs(pole[0] - 90.0) < 1e-8);
  CHECK(std::abs(pole[2]) < 1e-6);
  const auto south_pole = ecef_to_geodetic(Vector3(0, 0, -6356752.314245));
  CHECK(std::abs(south_pole[0] + 90.0) < 1e-8);
  CHECK(std::abs(ecef_to_geodetic(Vector3(0, -6378137.0, 0))[1] + 90.0) <
        1e-10);
  CHECK(std::abs(ecef_to_geodetic(Vector3(-6378137.0, 0, 0))[1] - 180.0) <
        1e-10);
  rejected = false;
  Vector3 nonfinite = Vector3::Zero();
  nonfinite[0] = std::numeric_limits<double>::infinity();
  try {
    (void)ecef_to_geodetic(nonfinite);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
}
