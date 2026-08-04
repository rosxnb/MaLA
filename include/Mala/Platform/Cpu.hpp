/*
   Runtime CPU feature detection.

   The GEMM dispatcher reads these to pick a microkernel.
   Also lets us report what the machine can do.
   Detection runs once and is cached.
*/

#pragma once

#include <string_view>


namespace Mala::Platform
{

struct CpuFeatures
{
    bool sse42      = false;
    bool avx        = false;
    bool avx2       = false;
    bool fma        = false;
    bool avx512f    = false;
    bool neon       = false;
    
    std::string_view name = "unknown";
};

// Returns the (cached) detected feature set for the running CPU
CpuFeatures const& cpu();

} // namespace Mala::Platform
