/*
   Storage owns a contiguous, over-aligned, tail-padded block of bytes.
   Tensors hold a shared_ptr<Storage> and share it across views/reshapes.
*/

#pragma once

#include <cstddef>
#include <memory>
#include <new>

#include <Mala/Config.hpp>
#include <Mala/Platform/Alloc.hpp>

namespace Mala
{

class Storage
{
public:
    Storage() = default;

    Storage(Storage const&)            = delete;
    Storage& operator=(Storage const&) = delete;

    ~Storage()
    {
        if (owns_ && data_ != nullptr) {
            Platform::alignedFree(data_);
        }
    }

    // Allocate `nbytes` (rounded up + 64 B aligned),
    // owned and freed by this Storage.
    static std::shared_ptr<Storage> allocate(size_t nbytes)
    {
        auto storage = std::make_shared<Storage>();
        storage->data_ = static_cast<std::byte*>(Platform::alignedAlloc(nbytes));
        if (storage->data_ == nullptr && nbytes > 0) {
            throw std::bad_alloc{};
        }
        storage->nbytes_ = nbytes;
        storage->owns_   = true;
        return storage;
    }

    // Wrap external memory WITHOUT taking ownershop (truw zero-copy from_blob).
    // The caller must keep the buffer alive for the Storage's lifetime.
    static std::shared_ptr<Storage> wrap(void* data, size_t nbytes)
    {
        auto storage = std::make_shared<Storage>();
        storage->data_ = static_cast<std::byte*>(data);
        storage->nbytes_ = nbytes;
        storage->owns_ = false;
        return storage;
    }

    std::byte* data() noexcept
    {
        return data_;
    }

    std::byte const* data() const noexcept
    {
        return data_;
    }

    size_t nbytes() const noexcept
    {
        return nbytes_;
    }

    bool ownsData() const noexcept
    {
        return owns_;
    }

private:
    std::byte*  data_   = nullptr;
    size_t      nbytes_ = 0;
    bool        owns_   = false;
};

} // namespace Mala
