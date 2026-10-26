#pragma once

#include "../../GPU/Buffer/Buffer.h"
#include "../../GPU/Device/Device.h"

#include <cstdint>
#include <memory>

namespace eacp::ML::Compute
{
struct StorageShelf;

// Work recorded again and again - a sampling step, a codec chunk - asks for the
// same temporaries each time round, and a fresh GPU buffer costs more than its
// allocation: Metal zeroes its pages and makes them resident before the first
// command buffer that uses it can run, which is a quarter of a Stable Audio
// DiT step, and on D3D12 each one is a CreateCommittedResource.
//
// While a StorageReuse is alive on a Device, the storage of every
// Tensor::uninitializedF32 is kept when its last Tensor goes, and handed to a
// later one of the same size - but only after recycle() has said the GPU is
// done with it. Call recycle() once everything submitted so far has finished,
// as it has when CommandBuffer::commit() returns; storage given back while a
// command buffer is still being recorded is then never reused inside it.
// recycle() on a Device with no StorageReuse does nothing, so a library can
// call it after every submission whether or not its caller opened one.
//
// What is kept is held to bytesKeptAtMost - the working set of the largest job
// the caller runs is the natural figure: within it repeated work finds every
// size it asks for, and past it a request that finds nothing of its size frees
// the storage recycled longest ago, so a change of sizes trades the old storage
// for the new rather than adding to it.
//
// Scopes open at once on one Device share what they keep, and the last to end
// frees it, so a process with none open holds none of it. Used from the
// Device's thread.
class StorageReuse
{
public:
    StorageReuse(GPU::Device& device, std::int64_t bytesKeptAtMost);
    ~StorageReuse();

    StorageReuse(const StorageReuse&) = delete;
    StorageReuse& operator=(const StorageReuse&) = delete;

    static void recycle(GPU::Device& device);

    // The storage held for reuse, ready or waiting for recycle().
    static std::int64_t heldBytes(GPU::Device& device);

private:
    std::shared_ptr<StorageShelf> shelf;
};

namespace Detail
{
// A fresh buffer, or a kept one of the same size, for uninitializedF32.
std::shared_ptr<const GPU::Buffer> storageOf(std::int64_t bytes,
                                             GPU::Device& device);
} // namespace Detail
} // namespace eacp::ML::Compute
