/*
   Minimal in-house test harness.

   Register cases with MALA_TEST, assert with CHECK/CHECK_CLOSE/CHECK_THROWS.
   Run them all from a single main via runAll().
*/

#pragma once

#include <print>
#include <functional>
#include <string>
#include <vector>
#include <cmath> // IWYU pragma: keep


namespace Mala::Test
{

namespace Color
{
    inline constexpr auto reset  = "\033[0m";
    inline constexpr auto red    = "\033[31m";
    inline constexpr auto green  = "\033[32m";
    inline constexpr auto blue   = "\033[34m";
} // Color

struct Case
{
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry()
{
    static std::vector<Case> cases;
    return cases;
}

inline int& failureCount()
{
    static int count = 0;
    return count;
}

struct Registrar
{
    Registrar(char const* name, std::function<void()> fn)
    {
        registry().push_back({name, std::move(fn)});
    }
};

inline void reportFail(char const* file, int line, char const* expr)
{
    ++failureCount();
    std::println("  {}[FAIL]{} {}:{}: {}", Color::red, Color::reset, file, line, expr);
}

inline int runAll()
{
    int passed = 0;
    int failed = 0;
    for (auto const& testCase : registry()) {
        int const before = failureCount();
        std::println("{}[ RUN  ]{} {}", Color::blue, Color::reset, testCase.name);
        testCase.fn();
        if (failureCount() == before) {
            std::println("{}[  OK  ]{} {}", Color::green, Color::reset, testCase.name);
            ++passed;
        } else {
            std::println("{}[ FAIL ]{} {}", Color::red, Color::reset, testCase.name);
            ++failed;
        }
    }
    std::println("\n{} {}passed{}, {} {}failed{}, {} {}total{}",
                 passed, Color::green, Color::reset,
                 failed, Color::red, Color::reset,
                 registry().size(), Color::blue, Color::reset);
    return failed == 0 ? 0 : 1;
}

} // namespace Mala::Test

#define MALA_TEST(name)                                                 \
    static void name();                                                 \
    static ::Mala::Test::Registrar registrar_##name(#name, name);       \
    static void name()

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            ::Mala::Test::reportFail(__FILE__, __LINE__, #cond);        \
        }                                                               \
    } while (0)

#define CHECK_CLOSE(a, b, margin)                                       \
    do {                                                                \
        double const lhs = (a);                                         \
        double const rhs = (b);                                         \
        if (std::fabs(lhs - rhs) > (margin)) {                          \
            ::Mala::Test::reportFail(__FILE__, __LINE__, #a " ~= " #b); \
        }                                                               \
    } while (0)

#define CHECK_THROWS(expr)                                                          \
    do {                                                                            \
        bool threw = false;                                                         \
        try {                                                                       \
            (void)(expr);                                                           \
        } catch (...) {                                                             \
            threw = true;                                                           \
        }                                                                           \
        if (!threw) {                                                               \
            ::Mala::Test::reportFail(__FILE__, __LINE__, "expected throw: " #expr); \
        }                                                                           \
    } while (0)
