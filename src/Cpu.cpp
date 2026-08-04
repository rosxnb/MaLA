#include <Mala/Platform/Cpu.hpp>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif


namespace Mala::Platform
{

namespace
{

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

void cpuidCount(unsigned leaf, unsigned sub, unsigned regs[4]) noexcept
{
#if defined(_MSC_VER)
    int out[4];
    __cpuidex(out, static_cast<int>(leaf), static_cast<int>(sub));
    regs[0] = static_cast<unsigned>(out[0]);
    regs[1] = static_cast<unsigned>(out[1]);
    regs[2] = static_cast<unsigned>(out[2]);
    regs[3] = static_cast<unsigned>(out[3]);
#else
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#endif
}

CpuFeatures detectImpl() noexcept
{
    CpuFeatures features {};
    unsigned regs[4] {0};

    // Leaf 1: legacy feature flags in ECX.
    cpuidCount(1, 0, regs);
    features.sse42 = (regs[2] & (1u << 20)) != 0;
    features.fma   = (regs[2] & (1u << 12)) != 0;
    features.avx   = (regs[2] & (1u << 28)) != 0;

    // Leaf 7, sub-leaf 0: extended features in EBX
    cpuidCount(7, 0, regs);
    features.avx2    = (regs[1] & (1u << 5)) != 0;
    features.avx512f = (regs[1] & (1u << 16)) != 0;

    features.name = "x86-64";
    return features;
}

#else

CpuFeatures detectImpl() noexcept
{
    CpuFeatures features {};
#if defined(__ARM_NEON) || defined(__aarch64__)
    features.neon = true;
#endif
    features.name = "generic";
    return features;
}

#endif

} // namespace


CpuFeatures const& cpu()
{
    static CpuFeatures const features = detectImpl();
    return features;
}

} // namespace Mala::Platform
