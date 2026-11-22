#include "conversions.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace omni::frames {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kJ2000 = 2451545.0;
constexpr double kDaysPerCentury = 36525.0;
constexpr double kSecondsPerDay = 86400.0;
constexpr double kArcsecondsToRadians = kPi / (180.0 * 3600.0);
constexpr double kDegreesToRadians = kPi / 180.0;
constexpr double kWgs84SemiMajorAxisM = 6378137.0;
constexpr double kWgs84EccentricitySquared = 6.6943799901413165e-3;

void require_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

Matrix3 r1(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Matrix3 result;
    result << 1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine;
    return result;
}

Matrix3 r2(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Matrix3 result;
    result << cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine;
    return result;
}

Matrix3 r3(double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    Matrix3 result;
    result << cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0;
    return result;
}

Matrix3 precession_gcrf_to_mod(double jd_tt) {
    const double t = (jd_tt - kJ2000) / kDaysPerCentury;
    const double t_squared = t * t;
    const double t_cubed = t_squared * t;
    const double zeta = (2306.2181 * t + 0.30188 * t_squared + 0.017998 * t_cubed) * kArcsecondsToRadians;
    const double z = (2306.2181 * t + 1.09468 * t_squared + 0.018203 * t_cubed) * kArcsecondsToRadians;
    const double theta = (2004.3109 * t - 0.42665 * t_squared - 0.041833 * t_cubed) * kArcsecondsToRadians;
    return r3(-z) * r2(theta) * r3(-zeta);
}

struct NutationTerms {
    double dpsi;
    double deps;
    double eps;
};

NutationTerms nutation_4term(double jd_tt) {
    const double t = (jd_tt - kJ2000) / kDaysPerCentury;
    const double t_squared = t * t;
    const double t_cubed = t_squared * t;
    const double omega = (125.04452 - 1934.136261 * t) * kDegreesToRadians;
    const double mean_sun = (280.4665 + 36000.7698 * t) * kDegreesToRadians;
    const double mean_moon = (218.3165 + 481267.8813 * t) * kDegreesToRadians;
    const double dpsi = (-17.20 * std::sin(omega) - 1.32 * std::sin(2.0 * mean_sun)
                         - 0.23 * std::sin(2.0 * mean_moon) + 0.21 * std::sin(2.0 * omega))
                        * kArcsecondsToRadians;
    const double deps = (9.20 * std::cos(omega) + 0.57 * std::cos(2.0 * mean_sun)
                         + 0.10 * std::cos(2.0 * mean_moon) - 0.09 * std::cos(2.0 * omega))
                        * kArcsecondsToRadians;
    const double eps = (23.43929111 - 0.013004167 * t - 1.64e-7 * t_squared
                        + 5.04e-7 * t_cubed) * kDegreesToRadians;
    return NutationTerms{dpsi, deps, eps};
}

Matrix3 gcrf_to_teme_internal(double jd_tt) {
    const Matrix3 precession = precession_gcrf_to_mod(jd_tt);
    const NutationTerms terms = nutation_4term(jd_tt);
    const Matrix3 nutation = r1(-(terms.eps + terms.deps)) * r3(-terms.dpsi) * r1(terms.eps);
    const double equation_equinoxes = terms.dpsi * std::cos(terms.eps);
    return r3(-equation_equinoxes) * nutation * precession;
}

double gmst82(double jd_ut1) {
    const double t = (jd_ut1 - kJ2000) / kDaysPerCentury;
    const double t_squared = t * t;
    const double t_cubed = t_squared * t;
    const double seconds = 67310.54841
        + ((876600.0 * 3600.0) + 8640184.812866) * t
        + 0.093104 * t_squared - 6.2e-6 * t_cubed;
    const double turns = (seconds / kSecondsPerDay)
                         - std::floor(seconds / kSecondsPerDay);
    return turns * (2.0 * kPi);
}

Matrix3 teme_to_ecef_internal(double jd_ut1, double xp, double yp) {
    return r2(-xp) * r1(-yp) * r3(-gmst82(jd_ut1));
}

Matrix3 gcrf_to_frame_matrix(Frame frame, const FrameContext& context) {
    if (frame == Frame::Gcrf) {
        return Matrix3::Identity();
    }
    if ((frame == Frame::Ecef) || (frame == Frame::Teme)) {
        if (!context.jd_ut1.has_value()) {
            throw std::invalid_argument("jd_ut1 is required for ECEF or TEME conversions");
        }
        if (!context.jd_tt.has_value()) {
            throw std::invalid_argument("jd_tt is required for ECEF or TEME conversions");
        }
    }
    if (frame == Frame::Ecef) {
        return gcrf_to_ecef_matrix(context.jd_ut1.value(), context.jd_tt.value(), context.xp, context.yp);
    }
    if (frame == Frame::Teme) {
        return gcrf_to_teme_matrix(context.jd_tt.value());
    }
    if (frame == Frame::Lvlh) {
        if (!context.position_gcrf.has_value() || !context.velocity_gcrf.has_value()) {
            throw std::invalid_argument("position_gcrf and velocity_gcrf are required for LVLH conversions");
        }
        return gcrf_to_lvlh_matrix(context.position_gcrf.value(), context.velocity_gcrf.value());
    }
    if (!context.quaternion_body_from_gcrf.has_value()) {
        throw std::invalid_argument("q_body_from_gcrf is required for body-frame conversions");
    }
    return inertial_to_body_matrix(context.quaternion_body_from_gcrf.value());
}
}

Matrix3 gcrf_to_lvlh_matrix(const Vector3& position_gcrf, const Vector3& velocity_gcrf) {
    if (!position_gcrf.allFinite() || !velocity_gcrf.allFinite()) {
        throw std::invalid_argument("position_gcrf and velocity_gcrf must be finite");
    }
    const double position_norm = position_gcrf.norm();
    const Vector3 angular_momentum = position_gcrf.cross(velocity_gcrf);
    const double angular_momentum_norm = angular_momentum.norm();
    if ((position_norm == 0.0) || (angular_momentum_norm == 0.0)) {
        throw std::invalid_argument("LVLH requires nonzero position and angular momentum");
    }

    const Vector3 x_axis = position_gcrf / position_norm;
    const Vector3 z_axis = angular_momentum / angular_momentum_norm;
    Vector3 y_axis = z_axis.cross(x_axis);
    const double y_norm = y_axis.norm();
    if (y_norm == 0.0) {
        throw std::invalid_argument("LVLH axes are degenerate");
    }
    y_axis /= y_norm;

    Matrix3 result;
    result.row(0) = x_axis.transpose();
    result.row(1) = y_axis.transpose();
    result.row(2) = z_axis.transpose();
    return result;
}

Matrix3 gcrf_to_ecef_matrix(double jd_ut1, double jd_tt, double xp, double yp) {
    require_finite(jd_ut1, "jd_ut1");
    require_finite(jd_tt, "jd_tt");
    require_finite(xp, "xp");
    require_finite(yp, "yp");
    return teme_to_ecef_internal(jd_ut1, xp, yp) * gcrf_to_teme_internal(jd_tt);
}

Matrix3 ecef_to_gcrf_matrix(double jd_ut1, double jd_tt, double xp, double yp) {
    return gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp, yp).transpose();
}

Matrix3 gcrf_to_teme_matrix(double jd_tt) {
    require_finite(jd_tt, "jd_tt");
    return gcrf_to_teme_internal(jd_tt);
}

Matrix3 teme_to_gcrf_matrix(double jd_tt) {
    return gcrf_to_teme_matrix(jd_tt).transpose();
}

Matrix3 teme_to_ecef_matrix(double jd_ut1, double xp, double yp) {
    require_finite(jd_ut1, "jd_ut1");
    require_finite(xp, "xp");
    require_finite(yp, "yp");
    return teme_to_ecef_internal(jd_ut1, xp, yp);
}

Matrix3 ecef_to_teme_matrix(double jd_ut1, double xp, double yp) {
    return teme_to_ecef_matrix(jd_ut1, xp, yp).transpose();
}

Matrix3 frame_rotation(Frame source, Frame target, const FrameContext& context) {
    if (source == target) {
        return Matrix3::Identity();
    }
    const Matrix3 target_from_gcrf = gcrf_to_frame_matrix(target, context);
    const Matrix3 source_from_gcrf = gcrf_to_frame_matrix(source, context);
    return target_from_gcrf * source_from_gcrf.transpose();
}

Vector3 convert_vector(const Vector3& vector, Frame source, Frame target, const FrameContext& context) {
    if (!vector.allFinite()) {
        throw std::invalid_argument("vector must be finite");
    }
    return frame_rotation(source, target, context) * vector;
}

Vector3 ecef_to_geodetic(const Vector3& position_ecef_m) {
    if (!position_ecef_m.allFinite()) {
        throw std::invalid_argument("position_ecef_m must be finite");
    }
    const double x = position_ecef_m[0];
    const double y = position_ecef_m[1];
    const double z = position_ecef_m[2];
    const double p = std::hypot(x, y);
    const double longitude = std::atan2(y, x);
    double latitude = 0.0;
    double altitude = 0.0;
    if (p == 0.0) {
        latitude = (z >= 0.0) ? (kPi / 2.0) : (-kPi / 2.0);
        altitude = std::abs(z) - kWgs84SemiMajorAxisM * std::sqrt(1.0 - kWgs84EccentricitySquared);
    } else {
        latitude = std::atan2(z, p * (1.0 - kWgs84EccentricitySquared));
        for (std::size_t iteration = 0U; iteration < 8U; ++iteration) {
            const double sine_latitude = std::sin(latitude);
            const double prime_vertical_radius = kWgs84SemiMajorAxisM
                / std::sqrt(1.0 - kWgs84EccentricitySquared * sine_latitude * sine_latitude);
            latitude = std::atan2(z + kWgs84EccentricitySquared * prime_vertical_radius * sine_latitude, p);
        }
        const double sine_latitude = std::sin(latitude);
        const double prime_vertical_radius = kWgs84SemiMajorAxisM
            / std::sqrt(1.0 - kWgs84EccentricitySquared * sine_latitude * sine_latitude);
        altitude = (p / std::cos(latitude)) - prime_vertical_radius;
    }
    return Vector3(latitude / kDegreesToRadians,
                   longitude / kDegreesToRadians,
                   altitude / 1000.0);
}

Vector3 gcrf_position_to_geodetic(const Vector3& position_gcrf_m,
                                  double jd_ut1, double jd_tt, double xp, double yp) {
    const Vector3 position_ecef = gcrf_to_ecef_matrix(jd_ut1, jd_tt, xp, yp) * position_gcrf_m;
    return ecef_to_geodetic(position_ecef);
}

Frame parse_frame(const std::string& name) {
    std::string normalized_name = name;
    std::transform(normalized_name.begin(), normalized_name.end(), normalized_name.begin(),
                   [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
    if (normalized_name == "GCRF") return Frame::Gcrf;
    if (normalized_name == "ECEF") return Frame::Ecef;
    if (normalized_name == "BODY") return Frame::Body;
    if (normalized_name == "LVLH") return Frame::Lvlh;
    if (normalized_name == "TEME") return Frame::Teme;
    throw std::invalid_argument("unsupported frame " + name + "; expected GCRF, ECEF, BODY, LVLH, or TEME");
}
}
