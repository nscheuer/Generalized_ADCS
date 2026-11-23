#include "check.hpp"
#include "environment/gravity/j_gravity_model.hpp"

using namespace omni::environment;

int main() {
    const J4 model;
    const JGravityModel point_mass(std::vector<int>{});
    const Eigen::Vector3d p(7e6, 0.0, 1e6);
    const auto acceleration = model.evaluate(p);
    const double perturbation = (acceleration - point_mass.evaluate(p)).norm();
    CHECK(acceleration.allFinite() && perturbation > 1e-6 && perturbation < 1e-2);
    return 0;
}
