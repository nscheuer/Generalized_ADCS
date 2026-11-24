#pragma once

#include <string>

namespace omni::config
{

class SimulationConfig
{
  public:
    explicit SimulationConfig(std::string backend = "Python");

    const std::string &backend() const noexcept
    {
        return backend_;
    }

  private:
    std::string backend_;
};

} // namespace omni::config
