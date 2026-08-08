#include <MalaTest.hpp>

#include <Mala/Platform/Alloc.hpp>
#include <Mala/Platform/Cpu.hpp>
#include <Mala/Platform/Simd.hpp>
#include <Mala/Platform/ThreadPool.hpp>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace Mala;

MALA_TEST(allocIsAligned)
{
    void* p = Platform::alignedAlloc(64, 1000);
    CHECK(p != nullptr);
    CHECK((reinterpret_cast<std::uintptr_t>(p) % 64 == 0));
    Platform::alignedFree(p);
}

MALA_TEST(simdScalarOps)
{
    float a[8];
    float b[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f;
    }
    auto const va = Platform::F32x8::loadu(a);
    auto const vb = Platform::F32x8::loadu(b);
    auto const vc = fma(va, vb, va); // a * 2 + a = 3a
    float c[8];
    vc.storeu(c);
    for (int i = 0; i < 8; ++i) {
        CHECK_CLOSE(c[i], 3.0f * static_cast<float>(i), 1e-6);
    }
    CHECK_CLOSE(va.reduceAdd(), 28.0, 1e-6); // 0 + 1 + ... + 7
}

MALA_TEST(cpuDetectRuns)
{
    auto const& features = Platform::cpu();
    std::println("cpu: {}\n  sse42={}  avx={}  avx2={}\n  fma={}  avx512f={}  neon={}\n",
                 features.name,
                 features.sse42, features.avx, features.avx2,
                 features.fma, features.avx512f, features.neon);
    CHECK(true); // detection must simply not crash
}

MALA_TEST(parallelForSumsCorrectly)
{
    size_t const n = 100000;
    std::vector<int> values(n);
    std::iota(values.begin(), values.end(), 1);

    std::atomic<long long> total{0};
    Platform::parallelFor(0, n, 1024, [&](size_t i0, size_t i1) {
        long long local = 0;
        for (size_t i = i0; i < i1; ++i) {
            local += values[i];
        }
        total.fetch_add(local);
    });

    long long const expected =
        static_cast<long long>(n) * static_cast<long long>(n + 1) / 2;
    CHECK(total.load() == expected);
}
