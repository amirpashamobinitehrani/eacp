#pragma once

#include "HotReload.h"
#include "HotReloadLoop.h"
#include "ReloadableEntry.h"
#include "../Utils/Logging.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// Included directly, never through the Core.h umbrella: the constructor reads
// the mode and the module from the app's own compile definitions.

namespace eacp::Plugins
{
struct ReloadableOptions
{
    std::function<void(std::string_view line)> log = [](std::string_view line)
    { LOG(line); };

    std::function<void()> onReloaded = [] {};
    bool callOnReloadedAtCreation = true;

    std::function<void()> onStateChanged = [] {};
    std::function<void(const BuildFailure& failure)> onBuildFailed =
        [](const BuildFailure&) {};
    std::function<void(const std::string& error)> onLoadFailed =
        [](const std::string&) {};
    std::function<void(std::string_view chunk)> onBuildOutput =
        [](std::string_view) {};

    Files::SourceWatcher::Options sources;
    std::optional<HotReloadLoop::Options> loop;
};

namespace Detail
{
// What the app's call site decides: the compiled-in table, the mode and the
// module. Tests build one by hand.
struct ReloadableSetup
{
    ReloadableTable compiledIn;
    bool hotReload = false;
    HotReloadTarget target;
};
} // namespace Detail

// The functions EACP_RELOADABLE lists, compiled into the app as generation 0
// and, under EACP_HOT_RELOAD, rebuilt and swapped in from the module on every
// save. Each is found by its type: create<T>() calls the T*() factory and
// function<F>() returns the F. See Docs/HotReload.md.
//
// Lifetime: onReloaded runs once at the end of construction and again after
// every reload, so declare the Reloadable after everything onReloaded uses.
// On a reload, every object create<T>() made must be replaced or destroyed
// inside onReloaded, while the previous image is still mapped. Destroying the
// Reloadable leaves its image mapped for the rest of the loop turn, so members
// destroyed after it may still hold its objects.
class Reloadable
{
public:
    template <bool hotReload = isHotReload(), auto compiledIn = &eacpReloadableTable>
    explicit Reloadable(const ReloadableOptions& optionsToUse = {})
        : Reloadable(
              Detail::ReloadableSetup {compiledIn(), hotReload, hotReloadTarget()},
              optionsToUse)
    {
    }

    Reloadable(const Detail::ReloadableSetup& setup,
               const ReloadableOptions& optionsToUse);
    ~Reloadable();

    Reloadable(const Reloadable&) = delete;
    Reloadable& operator=(const Reloadable&) = delete;
    Reloadable(Reloadable&&) = delete;
    Reloadable& operator=(Reloadable&&) = delete;

    template <typename T>
    OwningPointer<T> create()
    {
        static_assert(std::has_virtual_destructor_v<T>,
                      "A module-made object is deleted through T, so T needs a "
                      "virtual destructor");

        using Factory = T* (*)();
        auto signature = Detail::signatureOf<T*()>();
        auto compiled = reinterpret_cast<Factory>(compiledInFunction(signature));

        if (!isModuleLoaded())
            return fromCompiledIn(compiled);

        try
        {
            auto loaded = reinterpret_cast<Factory>(currentFunction(signature));

            if (auto* made = loaded())
                return OwningPointer<T> {made};

            logFallback("factory returned nothing");
        }
        catch (const std::exception& e)
        {
            logFallback(std::string {"factory threw: "} + e.what());
        }
        catch (...)
        {
            logFallback("factory threw");
        }

        return fromCompiledIn(compiled);
    }

    template <typename F>
    F* function()
    {
        static_assert(std::is_function_v<F>, "function<F>() takes a function type");
        return reinterpret_cast<F*>(currentFunction(Detail::signatureOf<F>()));
    }

    template <typename F, typename... Args>
    decltype(auto) call(Args&&... args)
    {
        return function<F>()(std::forward<Args>(args)...);
    }

    [[nodiscard]] std::string describeState() const;
    [[nodiscard]] int getGeneration() const;

private:
    template <typename T>
    static OwningPointer<T> fromCompiledIn(T* (*factory)())
    {
        if (auto* made = factory())
            return OwningPointer<T> {made};

        throw std::logic_error("EACP_RELOADABLE: the compiled-in factory "
                               "returned nothing");
    }

    bool isModuleLoaded() const;
    ReloadableEntry::Erased compiledInFunction(std::string_view signature) const;
    ReloadableEntry::Erased currentFunction(std::string_view signature) const;
    void logFallback(const std::string& reason) const;

    HotReloadLoop::Options loopOptions(const HotReloadTarget& target) const;
    std::string refusalOf(const DynamicLibrary& image) const;
    void startLoop(const HotReloadTarget& target);
    void reloaded();

    ReloadableOptions options;
    ReloadableTable compiledIn;
    ReloadableTable loaded;
    std::optional<HotReloadLoop> loop;
};
} // namespace eacp::Plugins
