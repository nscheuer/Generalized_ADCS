#ifndef ADCS_OMNI_ATTITUDE_STATE_HPP
#define ADCS_OMNI_ATTITUDE_STATE_HPP

#include "state_types.hpp"

namespace omni {
struct AttitudeState {
    Quaternion orientation;
    Vector3 rates;

    AttitudeState(const Quaternion& orientation_value, const Vector3& rates_value);
    Quaternion quaternion_array() const;
};
}

#endif  // ADCS_OMNI_ATTITUDE_STATE_HPP
