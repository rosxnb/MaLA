/*
   Minimal in-house test harness.

   Register cases with MALA_TEST, assert with CHECK/CHECK_CLOSE.
   Run them all from a single main via runAll().
*/

#pragma once

#include <print>
#include <functional>
#include <string>
#include <vector>


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
    std::println("  [FAIL] {}:{}: {}", file, line, expr);
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
    std::println("\n{} passed, {} failed, {} total", passed, failed, registry().size());
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

#define CHECK_CLOSE(a, b, tol)                                          \
    do {                                                                \
        double const lhs = (a);                                         \
        double const rhs = (b);                                         \
        if (std::fabs(lhs - rhs) > (tol)) {                             \
            ::Mala::Test::reportFail(__FILE__, __LINE__, #a " ~= " #b); \
        }                                                               \
    } while (0)
