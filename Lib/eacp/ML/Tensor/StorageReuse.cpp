#include "StorageReuse.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace eacp::ML::Compute
{
struct StorageShelf
{
    struct Kept
    {
        std::uint64_t recycledAt = 0;
        GPU::Buffer buffer;
    };

    void giveBack(GPU::Buffer buffer)
    {
        auto lock = std::scoped_lock {mutex};
        waiting.push_back(std::move(buffer));
    }

    void recycle()
    {
        auto unused = std::vector<GPU::Buffer> {};
        auto lock = std::scoped_lock {mutex};
        ++recycles;

        for (auto& buffer: waiting)
        {
            auto bytes = buffer.size();
            readyBytes += bytes;
            ready.emplace(bytes, Kept {recycles, std::move(buffer)});
        }

        waiting.clear();
        freeOldestUntil(ceiling, unused);
    }

    // A kept buffer of exactly this size, if there is one. If not, the caller
    // is about to allocate, and if that would take what is kept past the
    // ceiling, the storage recycled longest ago is freed first: a change of
    // sizes - a longer clip, another model - then trades the old storage for
    // the new rather than holding both.
    std::optional<GPU::Buffer> take(std::int64_t bytes)
    {
        auto unused = std::vector<GPU::Buffer> {};
        auto lock = std::scoped_lock {mutex};

        if (auto kept = ready.find(bytes); kept != ready.end())
        {
            auto buffer = std::move(kept->second.buffer);
            readyBytes -= kept->first;
            ready.erase(kept);
            return buffer;
        }

        freeOldestUntil(ceiling - bytes, unused);
        return std::nullopt;
    }

    void freeOldestUntil(std::int64_t bytes, std::vector<GPU::Buffer>& unused)
    {
        while (readyBytes > std::max(bytes, std::int64_t {0}) && !ready.empty())
        {
            auto oldest = std::min_element(
                ready.begin(),
                ready.end(),
                [](const auto& a, const auto& b)
                { return a.second.recycledAt < b.second.recycledAt; });

            readyBytes -= oldest->first;
            unused.push_back(std::move(oldest->second.buffer));
            ready.erase(oldest);
        }
    }

    std::mutex mutex;
    std::int64_t ceiling = 0;
    std::int64_t readyBytes = 0;
    std::uint64_t recycles = 0;
    std::multimap<std::int64_t, Kept> ready;
    std::vector<GPU::Buffer> waiting;
};

namespace
{
struct ActiveShelf
{
    std::weak_ptr<StorageShelf> shelf;
};

std::shared_ptr<StorageShelf> activeShelf(GPU::Device& device)
{
    return device.perDevice<ActiveShelf>().shelf.lock();
}
} // namespace

StorageReuse::StorageReuse(GPU::Device& device, std::int64_t bytesKeptAtMost)
    : shelf(activeShelf(device))
{
    if (shelf == nullptr)
    {
        shelf = std::make_shared<StorageShelf>();
        device.perDevice<ActiveShelf>().shelf = shelf;
    }

    auto lock = std::scoped_lock {shelf->mutex};
    shelf->ceiling = std::max(shelf->ceiling, bytesKeptAtMost);
}

StorageReuse::~StorageReuse() = default;

void StorageReuse::recycle(GPU::Device& device)
{
    auto shelf = activeShelf(device);

    if (shelf == nullptr)
        return;

    shelf->recycle();
}

std::int64_t StorageReuse::heldBytes(GPU::Device& device)
{
    auto shelf = activeShelf(device);

    if (shelf == nullptr)
        return 0;

    auto lock = std::scoped_lock {shelf->mutex};
    auto total = shelf->readyBytes;

    for (const auto& buffer: shelf->waiting)
        total += buffer.size();

    return total;
}

std::shared_ptr<const GPU::Buffer> Detail::storageOf(std::int64_t bytes,
                                                     GPU::Device& device)
{
    auto shelf = activeShelf(device);

    if (shelf == nullptr)
        return std::make_shared<const GPU::Buffer>(
            device.makeBuffer(bytes, GPU::BufferUsage::Storage));

    auto buffer = shelf->take(bytes);

    if (!buffer.has_value())
        buffer.emplace(device.makeBuffer(bytes, GPU::BufferUsage::Storage));

    // Handed back through a weak link, so storage that outlives its scope is
    // simply freed.
    auto* held = new GPU::Buffer {std::move(*buffer)};

    return std::shared_ptr<const GPU::Buffer> {
        held,
        [link = std::weak_ptr<StorageShelf> {shelf}](const GPU::Buffer* gone)
        {
            auto owned =
                std::unique_ptr<GPU::Buffer> {const_cast<GPU::Buffer*>(gone)};

            if (auto kept = link.lock())
                kept->giveBack(std::move(*owned));
        }};
}
} // namespace eacp::ML::Compute
