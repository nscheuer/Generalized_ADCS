#ifndef ADCS_OMNI_STATE_VALIDATION_HPP
#define ADCS_OMNI_STATE_VALIDATION_HPP

#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace omni::detail
{
template <typename Derived> void require_finite(const Eigen::MatrixBase<Derived> &values, const char *field_name)
{
    if (!values.allFinite())
    {
        throw std::invalid_argument(std::string(field_name) + " must be finite");
    }
}
} // namespace omni::detail

#endif // ADCS_OMNI_STATE_VALIDATION_HPP
