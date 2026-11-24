#pragma once
#include <stdexcept>
#define CHECK(x)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(x))                                                                                                      \
            throw std::runtime_error("check failed: " #x);                                                             \
    } while (false)
