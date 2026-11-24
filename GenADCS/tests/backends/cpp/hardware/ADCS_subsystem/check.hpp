#pragma once

#include <stdexcept>
#include <string>

#define CHECK(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + " failed: " #condition); \
} while (false)
