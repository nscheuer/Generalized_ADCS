#include "_nrlmsise00_core.hpp"
namespace omni::environment {
void run_nrlmsise00(nrlmsise_input &input, nrlmsise_flags &flags,
                    nrlmsise_output &output, bool effective) {
  if (effective)
    gtd7d(&input, &flags, &output);
  else
    gtd7(&input, &flags, &output);
}
} // namespace omni::environment
