#include "environment/environment.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    } while (false)

using namespace omni::environment;

int main()
{
    const Environment defaults;
    CHECK(defaults.planetary_body != nullptr);
    CHECK(defaults.planetary_body->name == "Earth");
    CHECK(defaults.atmosphere == nullptr);
    CHECK(defaults.magnetic_field == nullptr);
    CHECK(defaults.j_gravity == nullptr);
    CHECK(defaults.ephemeris == nullptr);

    auto body = std::make_shared<PlanetaryBody>("test body", 12.0, 34.0);
    auto atmosphere = std::make_shared<AtmosphericModel>();
    auto magnetic = std::make_shared<WMM>();
    auto gravity = std::make_shared<J2>();
    auto ephemeris = std::make_shared<DE421>();
    const Environment configured(body, atmosphere, magnetic, gravity, ephemeris);
    CHECK(configured.planetary_body == body);
    CHECK(configured.planetary_body->name == "test body");
    CHECK(configured.atmosphere == atmosphere);
    CHECK(configured.magnetic_field == magnetic);
    CHECK(configured.j_gravity == gravity);
    CHECK(configured.ephemeris == ephemeris);

    const Environment null_body(nullptr, atmosphere, magnetic, gravity, ephemeris);
    CHECK(null_body.planetary_body != nullptr);
    CHECK(null_body.planetary_body->name == "Earth");
}
