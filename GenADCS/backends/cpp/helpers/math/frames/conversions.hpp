#ifndef ADCS_OMNI_FRAMES_CONVERSIONS_HPP
#define ADCS_OMNI_FRAMES_CONVERSIONS_HPP

#include "frames.hpp"

#include <optional>
#include <string>

namespace omni::frames
{
enum class Frame
{
    Gcrf,
    Ecef,
    Body,
    Lvlh,
    Teme
};

struct FrameContext
{
    std::optional<double> jd_ut1;
    std::optional<double> jd_tt;
    double xp = 0.0;
    double yp = 0.0;
    std::optional<Vector3> position_gcrf;
    std::optional<Vector3> velocity_gcrf;
    std::optional<Quaternion> quaternion_body_from_gcrf;
};

Matrix3 gcrf_to_lvlh_matrix(const Vector3 &position_gcrf, const Vector3 &velocity_gcrf);
Matrix3 gcrf_to_ecef_matrix(double jd_ut1, double jd_tt, double xp = 0.0, double yp = 0.0);
Matrix3 ecef_to_gcrf_matrix(double jd_ut1, double jd_tt, double xp = 0.0, double yp = 0.0);
Matrix3 gcrf_to_teme_matrix(double jd_tt);
Matrix3 teme_to_gcrf_matrix(double jd_tt);
Matrix3 teme_to_ecef_matrix(double jd_ut1, double xp = 0.0, double yp = 0.0);
Matrix3 ecef_to_teme_matrix(double jd_ut1, double xp = 0.0, double yp = 0.0);
Matrix3 frame_rotation(Frame source, Frame target, const FrameContext &context = {});
Vector3 convert_vector(const Vector3 &vector, Frame source, Frame target, const FrameContext &context = {});
Vector3 ecef_to_geodetic(const Vector3 &position_ecef_m);
Vector3 gcrf_position_to_geodetic(const Vector3 &position_gcrf_m, double jd_ut1, double jd_tt, double xp = 0.0,
                                  double yp = 0.0);
Frame parse_frame(const std::string &name);
} // namespace omni::frames

#endif // ADCS_OMNI_FRAMES_CONVERSIONS_HPP
