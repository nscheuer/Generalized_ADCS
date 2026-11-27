#ifndef ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_OBC_SUBSYSTEM_HPP
#define ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_OBC_SUBSYSTEM_HPP

#include "fdir.hpp"
#include "state_machine.hpp"

#include <memory>
#include <vector>

namespace omni::hardware::obc {

struct OBCSubsystem {
  OBCSubsystem(std::shared_ptr<StateMachine> state_machine = nullptr,
               std::shared_ptr<FDIR> fdir = nullptr, int N = 1,
               std::vector<TimelineEntry> timeline = {});

  std::shared_ptr<StateMachine> state_machine;
  std::shared_ptr<FDIR> fdir;
  int N = 1;
  std::vector<TimelineEntry> timeline;
};

} // namespace omni::hardware::obc

#endif
