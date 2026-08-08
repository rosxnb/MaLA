/*
    Shape Metadata: a small inline vector of dimensions/strides plus the
    pure functions over them. Strides are signed so flips and reversed
    slices are metadata-only.
*/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <sstream>

#include <Mala/Core/Error.hpp>


namespace Mala
{

using Dim = int64_t;

class Dims
{
public:
    static constexpr size_t kMaxRank = 8;
    using value_type = Dim;

    Dims() = default;

    Dims(std::initializer_list<Dim> values)
    {
        if (values.size() > kMaxRank)
            throw ShapeError("Dims: rank exceeds kMaxRank");

        for (Dim value: values) {
            data_[rank_++] = value;
        }
    }

    // A zero-filled Dims of the given rank.
    static Dims withRank(size_t rank)
    {
        if (rank > kMaxRank)
            throw ShapeError("Dims: rank exceeds kMaxRank");

        Dims dims;
        dims.rank_ = rank;
        return dims;
    }

    size_t rank() const noexcept
    {
        return rank_;
    }

    size_t empty() const noexcept
    {
        return rank_ == 0;
    }

    Dim operator[](size_t i) const noexcept
    {
        return data_[i];
    }

    Dim& operator[](size_t i) noexcept
    {
        return data_[i];
    }

    void push(Dim value)
    {
        if (rank_ >= kMaxRank)
            throw ShapeError("Dims: ranks exceeds kMaxRank");

        data_[rank_++] = value;
    }

    Dim const* begin() const noexcept
    {
        return data_.data();
    }

    Dim const* end() const noexcept
    {
        return data_.data() + rank_;
    }

    bool operator==(Dims const& other) const noexcept
    {
        if (rank_ != other.rank_)
            return false;

        for (size_t i = 0; i < rank_; ++i) {
            if (data_[i] != other.data_[i]) {
                return false;
            }
        }

        return true;
    }

private:
    std::array<Dim, kMaxRank> data_ {};
    size_t                    rank_ = 0;
};

// Product of extents; an empty (rank-0) shape is scalar with 1 element.
inline Dim numElements(Dims const& shape) noexcept
{
    Dim count = 1;
    for (size_t i = 0; i < shape.rank(); ++i) {
        count *= shape[i];
    }
    return count;
}

// Row-major (C-order) strides for a shape.
inline Dims contiguousStrides(Dims const& shape)
{
    Dims strides = Dims::withRank(shape.rank());
    Dim acc = 1;
    for (size_t i = shape.rank(); i-- > 0;) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

// Row-major contiguity, ignoring size-1 axes (whose stride is irrelevant),
// matching the PyTorch definition.
inline bool isContiguous(Dims const& shape, Dims const& strides) noexcept
{
    Dim expected = 1;
    for (size_t i = shape.rank(); i-- > 0;) {
        if (shape[i] == 0)
            return true;

        if (shape[i] != 1) {
            if (strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
    }
    return true;
}

// NumPy broadcasting: align right; each axis pair must be equal or contain a 1.
inline Dims broadcastShapes(Dims const& a, Dims const& b)
{
    size_t const ra = a.rank();
    size_t const rb = b.rank();
    size_t const r  = ra > rb ? ra : rb;

    Dims out = Dims::withRank(r);
    for (size_t i = 0; i < r; ++i) {
        Dim const da = i < r - ra ? Dim{1} : a[i - (r - ra)];
        Dim const db = i < r - rb ? Dim{1} : b[i - (r - rb)];

        if (da == db || da == 1 || db == 1) {
            out[i] = da == 1 ? db : da;
        } else {
            throw ShapeError("broadcastShapes: incompatible shapes");
        }
    }
    return out;
}

inline std::string toString(Dims const& dims)
{
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < dims.rank(); ++i) {
        if (i != 0) {
            ss << ", ";
        }
        ss << dims[i];
    }
    ss << ']';
    return ss.str();
}

} // namespace Mala
