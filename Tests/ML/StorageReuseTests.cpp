#include <NanoTest/NanoTest.h>

#include <eacp/GPU/Device/Device.h>
#include <eacp/ML/Tensor/StorageReuse.h>
#include <eacp/ML/Tensor/Tensor.h>

#include <optional>
#include <vector>

using namespace nano;
using namespace eacp;
using namespace eacp::GPU;
using namespace eacp::ML::Compute;

namespace
{
constexpr auto ample = std::int64_t {64} << 20;
} // namespace

auto tStorageReuseReusesAfterRecycle = test("StorageReuse/reusesAfterRecycle") = []
{
    auto device = Device {};

    if (!device.isValid())
        return;

    auto reuse = StorageReuse {device, ample};
    auto first = std::optional<Tensor> {Tensor::uninitializedF32({64, 32}, device)};
    auto native = first->range().buffer->nativeBuffer();

    first.reset();
    check(StorageReuse::heldBytes(device) == 64 * 32 * 4);

    StorageReuse::recycle(device);
    auto second = Tensor::uninitializedF32({32, 64}, device);

    check(second.range().buffer->nativeBuffer() == native);
    check(StorageReuse::heldBytes(device) == 0);
};

auto tStorageReuseWaitsForRecycle = test("StorageReuse/waitsForRecycle") = []
{
    auto device = Device {};

    if (!device.isValid())
        return;

    auto reuse = StorageReuse {device, ample};
    auto first = std::optional<Tensor> {Tensor::uninitializedF32({256}, device)};
    auto native = first->range().buffer->nativeBuffer();

    first.reset();
    auto second = Tensor::uninitializedF32({256}, device);

    check(second.range().buffer->nativeBuffer() != native);
};

auto tStorageReuseHoldsToItsCeiling =
    test("StorageReuse/pastItsCeilingTradesOldSizesForNew") = []
{
    auto device = Device {};

    if (!device.isValid())
        return;

    auto reuse = StorageReuse {device, 1024 * 4};

    {
        auto a = Tensor::uninitializedF32({512}, device);
        auto b = Tensor::uninitializedF32({256}, device);
    }

    StorageReuse::recycle(device);
    check(StorageReuse::heldBytes(device) == 768 * 4);

    {
        auto c = Tensor::uninitializedF32({384}, device);
        check(StorageReuse::heldBytes(device) == 512 * 4);
    }

    StorageReuse::recycle(device);
    check(StorageReuse::heldBytes(device) == (512 + 384) * 4);

    {
        auto big = Tensor::uninitializedF32({4096}, device);
    }

    StorageReuse::recycle(device);
    check(StorageReuse::heldBytes(device) == 0);
};

auto tStorageReuseFreesWhenScopeEnds = test("StorageReuse/freesWhenScopeEnds") = []
{
    auto device = Device {};

    if (!device.isValid())
        return;

    auto outlives = std::optional<Tensor> {};

    {
        auto reuse = StorageReuse {device, ample};
        outlives.emplace(Tensor::uninitializedF32({128}, device));
        auto gone = Tensor::uninitializedF32({128}, device);
    }

    check(StorageReuse::heldBytes(device) == 0);
    outlives.reset();
    check(StorageReuse::heldBytes(device) == 0);

    StorageReuse::recycle(device);
    auto fresh = Tensor::uninitializedF32({128}, device);
    check(fresh.count() == 128);
};

auto tStorageReuseScopesShare = test("StorageReuse/overlappingScopesShare") = []
{
    auto device = Device {};

    if (!device.isValid())
        return;

    auto outer = std::optional<StorageReuse> {std::in_place, device, ample};
    auto inner = std::optional<StorageReuse> {std::in_place, device, ample};

    {
        auto gone = Tensor::uninitializedF32({64}, device);
    }

    outer.reset();
    check(StorageReuse::heldBytes(device) == 64 * 4);

    inner.reset();
    check(StorageReuse::heldBytes(device) == 0);
};

auto tStorageReuseKeepsRepeatedWork =
    test("StorageReuse/keepsWhatRepeatedWorkAsksFor") = []
{
    auto device = Device {};

    if (!device.isValid())
        return;

    auto reuse = StorageReuse {device, ample};
    auto natives = std::vector<void*> {};

    for (auto round = 0; round < 3; ++round)
    {
        {
            auto a = Tensor::uninitializedF32({128}, device);
            auto b = Tensor::uninitializedF32({256}, device);
            auto c = Tensor::uninitializedF32({512}, device);

            if (round == 0)
                natives = {a.range().buffer->nativeBuffer(),
                           b.range().buffer->nativeBuffer(),
                           c.range().buffer->nativeBuffer()};
            else
                check(natives
                      == std::vector<void*> {a.range().buffer->nativeBuffer(),
                                             b.range().buffer->nativeBuffer(),
                                             c.range().buffer->nativeBuffer()});
        }

        StorageReuse::recycle(device);
    }

    check(StorageReuse::heldBytes(device) == (128 + 256 + 512) * 4);
};
