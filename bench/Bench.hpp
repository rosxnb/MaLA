/*
   Mininal Benchmark timing helpers.

   Reports GB/s for bandwidth-bound kernels and
   GFLOP/s for compute-bound ones.
*/

#pragma once

#include <chrono>
#include <utility>
#include <print>
#include <string_view>

namespace Mala::Bench
{

template <typename Callable>
double timeSeconds(int iters, Callable&& body)
{
    using Clock = std::chrono::steady_clock;
    std::forward<Callable>(body)(); // warmup

    auto const start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        body();
    }
    auto const stop = Clock::now();

    return std::chrono::duration<double>(stop - start).count() / iters;
}

inline void
reportBandwidth(std::string_view name, double secs, double bytes)
{
    std::println("{:<24} {:8.3f} ms   {:8.2f} GB/s", name, secs * 1e3, bytes / secs / 1e9);
}

inline void
reportGflops(std::string_view name, double secs, double flops)
{
    std::println("{:<24} {:8.3f} ms   {:8.2f} GFLOP/s", name, secs * 1e3, flops / secs / 1e9);
}

} // namespace Mala::Bench
