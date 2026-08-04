/*
   ––––– library-wide compile-time configuration and contants ––––––––––––
*/

#pragma once

#include <cstddef>


namespace Mala
{

inline constexpr int versionMajor = 0;
inline constexpr int versionMinor = 0;
inline constexpr int versionPatch = 1;

// SIMD/allocation alignment.
// 64 B satisties AVX2 (32 B) and future AVX-512 (64 B) and
// aligns every buffer to a cache line.
inline constexpr size_t alignBytes = 64;

// Cache-line size for padding per-thread state to avoid false sharing.
// Hardcoded rather than std::hardware_destructive_interference_size to avoid the libstdc++
// -Winterference-size ABI warning
inline constexpr size_t cacheLineBytes = 64;

} // namespace Mala
