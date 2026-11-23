#include "check.hpp"
#include "environment/gravity/j_gravity_model.hpp"
#include <Eigen/Core>
#include <stdexcept>
#include <vector>

using namespace omni::environment;

int main() {
    bool invalid_degree = false;
    try { (void)JGravityModel(std::vector<int>{1}); }
    catch (const std::invalid_argument&) { invalid_degree = true; }
    CHECK(invalid_degree);
    bool invalid_coefficients = false;
    try { Eigen::VectorXd short_coefficients = Eigen::VectorXd::Ones(4); (void)JGravityModel(std::vector<int>{2}, short_coefficients); }
    catch (const std::invalid_argument&) { invalid_coefficients = true; }
    CHECK(invalid_coefficients);
    bool centre = false;
    try { (void)J2().evaluate(Eigen::Vector3d::Zero()); }
    catch (const std::invalid_argument&) { centre = true; }
    CHECK(centre);
    return 0;
}
