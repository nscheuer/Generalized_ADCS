#ifndef ADCS_OMNI_ENVIRONMENT_J_GRAVITY_MODEL_HPP
#define ADCS_OMNI_ENVIRONMENT_J_GRAVITY_MODEL_HPP
#include <vector>
#include <Eigen/Core>
namespace omni::environment {
Eigen::Vector3d gravity_acceleration_inertial_m_s2(const Eigen::Vector3d& position, double epoch_jd, double mu, double reference_radius, const Eigen::VectorXd& coefficients);
class JGravityModel {
public:
    explicit JGravityModel(std::vector<int> degrees = {2, 3, 4, 5, 6}, const Eigen::VectorXd& coefficients = Eigen::VectorXd());
    virtual ~JGravityModel() noexcept = default;
    Eigen::Vector3d evaluate(const Eigen::Vector3d& position, double epoch_jd = 0.0) const;
    Eigen::Vector3d acceleration(const Eigen::Vector3d& position, double epoch_jd = 0.0) const;
    const std::vector<int>& degrees() const noexcept;
    const Eigen::VectorXd& zonal_coefficients() const noexcept;
    virtual int degree() const noexcept;
protected:
    explicit JGravityModel(int single_degree);
private:
    std::vector<int> degrees_;
    Eigen::VectorXd coefficients_;
};
class J2 final : public JGravityModel { public: J2(); int degree() const noexcept override; };
class J3 final : public JGravityModel { public: J3(); int degree() const noexcept override; };
class J4 final : public JGravityModel { public: J4(); int degree() const noexcept override; };
class J5 final : public JGravityModel { public: J5(); int degree() const noexcept override; };
class J6 final : public JGravityModel { public: J6(); int degree() const noexcept override; };
}
#endif
