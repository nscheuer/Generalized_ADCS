#ifndef ADCS_OMNI_ENVIRONMENT_NATIVE_MODELS_HPP
#define ADCS_OMNI_ENVIRONMENT_NATIVE_MODELS_HPP

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace omni::environment
{

double nrlmsise_density_geodetic(int day_of_year, double seconds_utc, double altitude_km, double latitude_deg,
                                 double longitude_deg, double f107, double f107a, double ap, bool effective_density,
                                 const std::array<double, 7> &ap_history);
double nrlmsise_density(const Eigen::Vector3d &position_inertial_m, double epoch_jd, double f107, double f107a,
                        double ap, bool effective_density, const std::array<double, 7> &ap_history);

class DE421
{
  public:
    explicit DE421(const std::filesystem::path &filepath = {});
    Eigen::Vector3d position_eci(const std::string &target, double epoch_jd,
                                 const std::string &observer = "earth") const;
    Eigen::Vector3d velocity_eci(const std::string &target, double epoch_jd,
                                 const std::string &observer = "earth") const;
    std::pair<Eigen::Vector3d, Eigen::Vector3d> state_eci(const std::string &target, double epoch_jd,
                                                          const std::string &observer = "earth") const;
    Eigen::Vector3d position(int body_id, double epoch_jd) const;
    Eigen::Vector3d velocity(int body_id, double epoch_jd) const;
    std::vector<std::string> available_bodies() const;
    std::filesystem::path filepath() const
    {
        return filepath_;
    }

  private:
    struct Segment
    {
        int target;
        int center;
        int type;
        std::size_t offset;
        std::size_t size;
        double start;
        double end;
    };
    struct State
    {
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
    };
    std::filesystem::path filepath_;
    std::vector<Segment> segments_;
    std::vector<double> data_;
    State segment_state(int target, double et) const;
    State barycentric_state(int target, double et) const;
    int body_target(const std::string &name) const;
};

class MagneticFieldModel
{
  public:
    virtual ~MagneticFieldModel() = default;
    virtual Eigen::Vector3d field_inertial_tesla(const Eigen::Vector3d &position_inertial_m, double epoch_jd) const = 0;
    Eigen::Vector3d field(const Eigen::Vector3d &position_inertial_m, double epoch_jd) const;
};
class IGRF14 final : public MagneticFieldModel
{
  public:
    IGRF14();
    Eigen::Vector3d field_inertial_tesla(const Eigen::Vector3d &, double) const override;

  private:
    std::vector<double> times_;
    std::vector<Eigen::MatrixXd> g_, h_;
};
class CHAOS8 final : public MagneticFieldModel
{
  public:
    explicit CHAOS8(bool include_static = true);
    bool include_static() const
    {
        return include_static_;
    }
    Eigen::Vector3d field_inertial_tesla(const Eigen::Vector3d &, double) const override;

  private:
    bool include_static_;
    std::vector<double> times_;
    std::vector<Eigen::MatrixXd> g_, h_;
    Eigen::MatrixXd static_g_, static_h_;
};
class WMM final : public MagneticFieldModel
{
  public:
    Eigen::Vector3d field_inertial_tesla(const Eigen::Vector3d &, double) const override;
};

std::tuple<double, bool, bool> eclipse_state(const Eigen::Vector3d &spacecraft_position_m,
                                             const Eigen::Vector3d &sun_position_m, double occulting_body_radius_m,
                                             double sun_radius_m);
std::filesystem::path environment_data_path(const std::filesystem::path &relative);
Eigen::MatrixXd schmidt_table(int maximum_degree);
Eigen::Vector3d magnetic_field_with_coefficients(const Eigen::Vector3d &position_inertial_m, double epoch_jd,
                                                 const Eigen::MatrixXd &g, const Eigen::MatrixXd &h,
                                                 int maximum_degree);
std::tuple<std::vector<double>, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> load_shc(
    const std::filesystem::path &path, int minimum_degree, int maximum_degree);

} // namespace omni::environment
#endif
