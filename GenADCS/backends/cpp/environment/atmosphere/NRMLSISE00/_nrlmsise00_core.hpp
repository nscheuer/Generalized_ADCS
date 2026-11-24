#pragma once
extern "C"
{
#include "nrlmsise00_core.h"
}
namespace omni::environment
{
void run_nrlmsise00(nrlmsise_input &, nrlmsise_flags &, nrlmsise_output &, bool);
}
