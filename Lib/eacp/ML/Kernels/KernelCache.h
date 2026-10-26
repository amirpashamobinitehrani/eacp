#pragma once

#include "../../GPU/Codegen/ComputeProgram.h"
#include "../../GPU/Device/Device.h"

#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace eacp::ML::Compute
{
namespace Detail
{
struct KernelStore
{
    std::mutex mutex;
    std::map<std::pair<std::type_index, std::vector<int>>,
             std::unique_ptr<GPU::ComputeProgram>>
        kernels;
};
} // namespace Detail

// The one prepared Kernel of each variant on a Device, built and prepared the
// first time it is asked for: GPUWidgets' sharedKernel, kept per Device rather
// than on Device::shared() alone, since a model may run on a Device of its
// own, and per variant, since one kernel type can build several programs. The
// constructor arguments are what tell variants apart, so they are integers or
// enums.
//
// Shared by every caller, which is safe for the reason it is in GPUWidgets: a
// dispatch sets every uniform it reads immediately before issuing it, and a
// Device is used from one thread.
template <typename Kernel, typename... Args>
Kernel& sharedKernel(GPU::Device& device, Args... args)
{
    static_assert(((std::is_integral_v<Args> || std::is_enum_v<Args>) && ...),
                  "kernel variants are keyed by integer or enum arguments");

    auto& store = device.perDevice<Detail::KernelStore>();
    auto lock = std::scoped_lock {store.mutex};
    auto& slot =
        store.kernels[{std::type_index {typeid(Kernel)}, {static_cast<int>(args)...}}];

    if (slot == nullptr)
    {
        auto kernel = std::make_unique<Kernel>(args...);
        kernel->prepare(device);
        slot = std::move(kernel);
    }

    return static_cast<Kernel&>(*slot);
}
} // namespace eacp::ML::Compute
