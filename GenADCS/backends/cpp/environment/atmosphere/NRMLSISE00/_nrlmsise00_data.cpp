#include "_nrlmsise00_data.hpp"
// Preserve the upstream coefficient table exactly and compile it as C++.
extern "C"
{
#include "nrlmsise00_data.c"
}
