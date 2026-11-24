#pragma once
#include <stdexcept>
#include <string>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error("check failed: " #condition);                                                     \
    } while (false)

template <class Error, class F> void expect_throw(F &&action)
{
    bool caught = false;
    try
    {
        action();
    }
    catch (const Error &)
    {
        caught = true;
    }
    CHECK(caught);
}
