/*
   Portable Aligned Allocation.

   std::aligned_alloc required size to be a multiple of alignment, and MSVC provides
   _aligned_malloc/_aligned_free instead.
*/

#pragma once

#include <cstddef>
#include <cstdlib>

#include <Mala/Config.hpp>

namespace Mala::Platform
{

[[nodiscard]] inline void*
alignedAlloc(size_t align, size_t size) noexcept
{
    // Round up to a multiple
    // size_t const padded = ((size + align - 1) / align) * align;

    // Since align is always power of 2
    size_t const padded = (size + align - 1) & ~(align - 1);

#if defined(_MSC_VER)
    return _aligned_malloc(padded, align);
#else
    return std::aligned_alloc(align, padded);
#endif
}

[[nodiscard]] inline void*
alignedAlloc(size_t size) noexcept
{
    return alignedAlloc(alignBytes, size);
}

inline void
alignedFree(void* p) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}


} // namespace Mala::Platform
