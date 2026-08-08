/*
    Tensor<T>: strided metadata over a shared storage.
    Copies are shallow (they share storage).
    Clone performs deep-copies.
    Views (transpose/reshape/slice/expand/...) are metadata-only.
*/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mdspan>
#include <random>
#include <string>
#include <utility>

#include <Mala/Core/Error.hpp>
#include <Mala/Core/Shape.hpp>
#include <Mala/Core/Storage.hpp>

namespace Mala
{

template <typename T>
class Tensor
{
public:
    using value_type = T;

    Tensor() = default;

    // –– Facotries ––––––––––––––––––––––––––––––
    // Owned, contiguous, uninitialized storage
    static Tensor make(Dims shape)
    {
        Dim const count = numElements(shape);
        size_t const bytes = static_cast<size_t>(count < 0 ? 0 : count) * sizeof(T);

        Tensor tensor;
        tensor.shape_ = shape;
        tensor.strides_ = contiguousStrides(shape);
        tensor.storage_ = Storage::allocate(bytes);
        tensor.data_ = reinterpret_cast<T*>(tensor.storage_->data());
        tensor.contiguous_ = true;
        return tensor;
    }

    static Tensor full(Dims shape, T value)
    {
        Tensor tensor = make(shape);
        Dim const count = numElements(shape);
        for (Dim i = 0; i < count; ++i) {
            tensor.data_[i] = value;
        }
        return tensor;
    }

    static Tensor zeros(Dims shape)
    {
        return full(shape, T{});
    }

    static Tensor ones(Dims shape)
    {
        return full(shape, T{1});
    }

    // 1-D ramp [start, stop) with the given step.
    static Tensor arange(T start, T stop, T step = T{1})
    {
        Dim count = 0;
        for (T value = start;
             step > T{0} ? value < stop : value > stop;
             value += step)
        {
            ++count;
        }

        Tensor tensor = make(Dims{count});
        T value = start;
        for (Dim i = 0; i < count; ++i) {
            tensor.data_[i] = value;
            value += step;
        }
        return tensor;
    }

    // Standard-normal samples. Deterministic given a seed (temporary std RND; a
    // custorm deterministic generator moves into Platform in later implementation).
    static Tensor randn(Dims shape, uint64_t seed = 0x853c49e6748fea9bULL)
    {
        Tensor tensor = make(shape);
        std::mt19937_64 engine(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        Dim const count = numElements(shape);
        for (Dim i = 0; i < count; ++i) {
            tensor.data_[i] = static_cast<T>(dist(engine));
        }
        return tensor;
    }

    // Copy external data into owned storage.
    static Tensor fromData(T const* src, Dims shape)
    {
        Tensor tensor = make(shape);
        Dim const count = numElements(shape);
        if (count > 0) {
            std::memcpy(tensor.data_, src, static_cast<size_t>(count) * sizeof(T));
        }
        return tensor;
    }

    // Zero-copy view over external memory (caller keeps it alive).
    static Tensor fromBlob(T* src, Dims shape)
    {
        Dim const count = numElements(shape);

        Tensor tensor;
        tensor.shape_ = shape;
        tensor.strides_ = contiguousStrides(shape);
        tensor.storage_ = Storage::wrap(src, static_cast<size_t>(count) * sizeof(T));
        tensor.data_ = src;
        tensor.contiguous_ = true;
        return tensor;
    }

    // –– Metadata –––––––––––––––––––––––––––––––––––––––––––––––––
    Dims const& shape() const noexcept
    {
        return shape_;
    }

    Dims const& strides() const noexcept
    {
        return strides_;
    }

    std::size_t rank() const noexcept
    {
        return shape_.rank();
    }

    Dim numel() const noexcept
    {
        return numElements(shape_);
    }

    bool isContiguousLayout() const noexcept
    {
        return contiguous_;
    }

    T* data() noexcept
    {
        return data_;
    }

    T const* data() const noexcept
    {
        return data_;
    }

    std::shared_ptr<Storage> const& storage() const noexcept
    {
        return storage_;
    }

    // –– Element access (unchecked) ––––––––––––––––––––––––––––––––––––––
    template <typename... Idx>
    T& operator()(Idx... idx) noexcept
    {
        return data_[flatOffset(std::array<Dim, sizeof...(Idx)>{static_cast<Dim>(idx)...})];
    }

    template <typename... Idx>
    T const& operator()(Idx... idx) const noexcept
    {
        return data_[flatOffset(std::array<Dim, sizeof...(Idx)>{static_cast<Dim>(idx)...})];
    }

    // –– Views (share storage) –––––––––––––––––––––––––––––––––––––––––––––
    Tensor transpose(std::size_t i, std::size_t j) const
    {
        if (i >= rank() || j >= rank()) {
            throw ShapeError("transpose: axis out of range");
        }
        Tensor tensor = *this;
        std::swap(tensor.shape_[i], tensor.shape_[j]);
        std::swap(tensor.strides_[i], tensor.strides_[j]);
        tensor.refreshContiguity();
        return tensor;
    }

    Tensor permute(std::initializer_list<std::size_t> order) const
    {
        if (order.size() != rank()) {
            throw ShapeError("permute: order size must equal rank");
        }
        Tensor tensor = *this;
        std::array<bool, Dims::kMaxRank> seen{};
        std::size_t position = 0;
        for (std::size_t axis : order) {
            if (axis >= rank() || seen[axis]) {
                throw ShapeError("permute: invalid permutation");
            }
            seen[axis] = true;
            tensor.shape_[position] = shape_[axis];
            tensor.strides_[position] = strides_[axis];
            ++position;
        }
        tensor.refreshContiguity();
        return tensor;
    }

    // Reinterpret shape; clones first if the current layout is not contiguous.
    Tensor reshape(Dims newShape) const
    {
        if (numElements(newShape) != numel()) {
            throw ShapeError("reshape: element count mismatch");
        }
        Tensor tensor = contiguous_ ? *this : clone();
        tensor.shape_ = newShape;
        tensor.strides_ = contiguousStrides(newShape);
        tensor.contiguous_ = true;
        return tensor;
    }

    // Like reshape but never copies: requires a contiguous layout.
    Tensor view(Dims newShape) const
    {
        if (!contiguous_) {
            throw ShapeError("view: tensor is not contiguous (use reshape)");
        }
        if (numElements(newShape) != numel()) {
            throw ShapeError("view: element count mismatch");
        }
        Tensor tensor = *this;
        tensor.shape_ = newShape;
        tensor.strides_ = contiguousStrides(newShape);
        tensor.contiguous_ = true;
        return tensor;
    }

    Tensor slice(std::size_t dim, Dim start, Dim stop, Dim step = 1) const
    {
        if (dim >= rank()) {
            throw ShapeError("slice: dim out of range");
        }
        if (step <= 0) {
            throw ShapeError("slice: step must be positive");
        }
        Dim const extent = shape_[dim];
        if (start < 0 || stop > extent || start > stop) {
            throw ShapeError("slice: range out of bounds");
        }
        Tensor tensor = *this;
        tensor.data_ = tensor.data_ + start * strides_[dim];
        tensor.shape_[dim] = (stop - start + step - 1) / step;
        tensor.strides_[dim] = strides_[dim] * step;
        tensor.refreshContiguity();
        return tensor;
    }

    Tensor narrow(std::size_t dim, Dim start, Dim length) const
    {
        return slice(dim, start, start + length, 1);
    }

    Tensor squeeze(std::size_t dim) const
    {
        if (dim >= rank()) {
            throw ShapeError("squeeze: dim out of range");
        }
        if (shape_[dim] != 1) {
            throw ShapeError("squeeze: dimension is not size 1");
        }
        Tensor tensor = *this;
        tensor.shape_ = removeAxis(shape_, dim);
        tensor.strides_ = removeAxis(strides_, dim);
        tensor.refreshContiguity();
        return tensor;
    }

    // Drop every size-1 axis.
    Tensor squeeze() const
    {
        Tensor tensor = *this;
        Dims newShape;
        Dims newStrides;
        for (std::size_t i = 0; i < rank(); ++i) {
            if (shape_[i] != 1) {
                newShape.push(shape_[i]);
                newStrides.push(strides_[i]);
            }
        }
        tensor.shape_ = newShape;
        tensor.strides_ = newStrides;
        tensor.refreshContiguity();
        return tensor;
    }

    Tensor unsqueeze(std::size_t dim) const
    {
        if (dim > rank()) {
            throw ShapeError("unsqueeze: dim out of range");
        }
        Dim insertedStride = 1;
        if (dim < rank()) {
            insertedStride = strides_[dim] * shape_[dim]; // keeps contiguity when contiguous
        }
        Tensor tensor = *this;
        tensor.shape_ = insertAxis(shape_, dim, 1);
        tensor.strides_ = insertAxis(strides_, dim, insertedStride);
        tensor.refreshContiguity();
        return tensor;
    }

    Tensor flip(std::size_t dim) const
    {
        if (dim >= rank()) {
            throw ShapeError("flip: dim out of range");
        }
        Tensor tensor = *this;
        Dim const extent = shape_[dim];
        if (extent > 0) {
            tensor.data_ = tensor.data_ + (extent - 1) * strides_[dim];
            tensor.strides_[dim] = -strides_[dim];
        }
        tensor.refreshContiguity();
        return tensor;
    }

    // Broadcast (0-stride expand) to a target shape; never copies.
    Tensor broadcastTo(Dims target) const
    {
        std::size_t const r = target.rank();
        std::size_t const cur = rank();
        if (cur > r) {
            throw ShapeError("broadcastTo: target rank smaller than tensor rank");
        }
        Tensor tensor = *this;
        Dims newStrides = Dims::withRank(r);
        for (std::size_t i = 0; i < r; ++i) {
            if (i < r - cur) {
                newStrides[i] = 0;
            } else {
                std::size_t const src = i - (r - cur);
                if (shape_[src] == target[i]) {
                    newStrides[i] = strides_[src];
                } else if (shape_[src] == 1) {
                    newStrides[i] = 0;
                } else {
                    throw ShapeError("broadcastTo: incompatible dimension");
                }
            }
        }
        tensor.shape_ = target;
        tensor.strides_ = newStrides;
        tensor.refreshContiguity();
        return tensor;
    }

    Tensor expand(Dims target) const
    {
        return broadcastTo(target);
    }

    // –– Copies –––––––––––––––––––––––––––––––––––––––––––
    // Deep copy into contiguous owned storage (gathers through the current strides).
    Tensor clone() const
    {
        Tensor tensor = make(shape_);
        if (numel() > 0) {
            copyInto(tensor.data_);
        }
        return tensor;
    }

    Tensor contiguous() const
    {
        return contiguous_ ? *this : clone();
    }

    // –– Mdspan bridge ––––––––––––––––––––––––––––––––––––––––
    // A standard, bounds-aware strided view for fixed-rank kernels. Requires non-negative
    // strides (i.e. not a flipped view).
    template <std::size_t Rank>
    auto asMdspan()
    {
        if (rank() != Rank) {
            throw ShapeError("asMdspan: rank mismatch");
        }
        using Extents = std::dextents<std::size_t, Rank>;
        std::array<std::size_t, Rank> extents{};
        std::array<std::size_t, Rank> strides{};
        for (std::size_t i = 0; i < Rank; ++i) {
            extents[i] = static_cast<std::size_t>(shape_[i]);
            strides[i] = static_cast<std::size_t>(strides_[i]);
        }
        std::layout_stride::mapping<Extents> mapping(Extents(extents), strides);
        return std::mdspan<T, Extents, std::layout_stride>(data_, mapping);
    }

    // –– Formatting ––––––––––––––––––––––––––––––––––––––––––––––––––
    std::string toString() const
    {
        std::string out = "Tensor(shape=" + Mala::toString(shape_) +
                          ", strides=" + Mala::toString(strides_) + ")\n";
        Dims index = Dims::withRank(rank());
        appendData(out, 0, index);
        return out;
    }

private:
    template <size_t R>
    Dim flatOffset(std::array<Dim, R> const& index) const noexcept
    {
        Dim offset = 0;
        for (size_t i = 0; i < R; ++i) {
            offset += index[i] * strides_[i];
        }
        return offset;
    }

    void refreshContiguity() noexcept
    {
        contiguous_ = isContiguous(shape_, strides_);
    }

    void copyInto(T* dst) const
    {
        Dim const count = numel();
        if (contiguous_) {
            std::memcpy(dst, data_, static_cast<std::size_t>(count) * sizeof(T));
            return;
        }
        Dims index = Dims::withRank(rank());
        for (Dim linear = 0; linear < count; ++linear) {
            Dim offset = 0;
            for (std::size_t d = 0; d < rank(); ++d) {
                offset += index[d] * strides_[d];
            }
            dst[linear] = data_[offset];
            for (std::size_t d = rank(); d-- > 0;) {
                if (++index[d] < shape_[d]) {
                    break;
                }
                index[d] = 0;
            }
        }
    }

    void appendData(std::string& out, std::size_t d, Dims& index) const
    {
        if (rank() == 0) {
            out += std::to_string(data_[0]);
            return;
        }
        out += "[";
        for (Dim i = 0; i < shape_[d]; ++i) {
            index[d] = i;
            if (d + 1 == rank()) {
                Dim offset = 0;
                for (std::size_t k = 0; k < rank(); ++k) {
                    offset += index[k] * strides_[k];
                }
                out += std::to_string(data_[offset]);
            } else {
                appendData(out, d + 1, index);
            }
            if (i + 1 < shape_[d]) {
                out += ", ";
            }
        }
        out += "]";
    }

    static Dims removeAxis(Dims const& dims, std::size_t axis)
    {
        Dims out;
        for (std::size_t i = 0; i < dims.rank(); ++i) {
            if (i != axis) {
                out.push(dims[i]);
            }
        }
        return out;
    }

    static Dims insertAxis(Dims const& dims, std::size_t axis, Dim value)
    {
        Dims out;
        for (std::size_t i = 0; i < axis; ++i) {
            out.push(dims[i]);
        }
        out.push(value);
        for (std::size_t i = axis; i < dims.rank(); ++i) {
            out.push(dims[i]);
        }
        return out;
    }

    std::shared_ptr<Storage> storage_    {};
    T*                       data_       {nullptr};
    Dims                     shape_      {};
    Dims                     strides_    {};
    bool                     contiguous_ {true};
};

} // namespace Mala
