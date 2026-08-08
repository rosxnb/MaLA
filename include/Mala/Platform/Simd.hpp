/*
   Portable SIMD wrapper.

   Defines a full API surface (load/store/arith/fma/min/max/reduce) so an
   AVX2/NEON specialization can be dropped in later without chaning callers.

   Current version: ships only the SCALAR FALLBACK.
   Scalar fallback is correct on every target, but isn't fast on any.
*/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>


namespace Mala::Platform
{

template <typename T, size_t N>
struct Simd
{
    std::array<T, N> v{};

    static constexpr size_t size() noexcept
    {
        return N;
    }

    static Simd load(T const* p) noexcept
    {
        Simd s;
        for (size_t i = 0; i < N; ++i) {
            s.v[i] = p[i];
        }
        return s;
    }

    static Simd loadu(T const* p) noexcept // no alignment requirement in scalar mode
    {
        return load(p);
    }

    static Simd broadcast(T x) noexcept
    {
        Simd s;
        s.v.fill(x);
        return s;
    }

    void store(T* p) const noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            p[i] = v[i];
        }
    }

    void storeu(T* p) const noexcept
    {
        store(p);
    }

    friend Simd operator+(Simd a, Simd b) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            a.v[i] += b.v[i];
        }
        return a;
    }

    friend Simd operator-(Simd a, Simd b) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            a.v[i] -= b.v[i];
        }
        return a;
    }

    friend Simd operator/(Simd a, Simd b) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            a.v[i] /= b.v[i];
        }
        return a;
    }

    // Fused multiply-add: a * b + c, lane-wise
    friend Simd fma(Simd a, Simd b, Simd c) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            c.v[i] += a.v[i] * b.v[i];
        }
        return c;
    }

    friend Simd min(Simd a, Simd b) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            a.v[i] = std::min(a.v[i], b.v[i]);
        }
        return a;
    }

    friend Simd max(Simd a, Simd b) noexcept
    {
        for (size_t i = 0; i < N; ++i) {
            a.v[i] = std::max(a.v[i], b.v[i]);
        }
        return a;
    }

    // Horizontal sum of all lanes
    T reduceAdd() const noexcept
    {
        T s {};
        for (size_t i = 0; i < N; ++i) {
            s += v[i];
        }
        return s;
    }
};

// Canonical widths (lane counts match AVX2 registers; scalar impl for now).
using F32x8 = Simd<float, 8>;
using F64x4 = Simd<double, 4>;

} // namespace Mala::Platform
