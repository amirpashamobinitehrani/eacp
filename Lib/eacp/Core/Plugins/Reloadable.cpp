#include "Reloadable.h"

namespace eacp::Plugins
{
namespace
{
using ReloadableTableFunction = ReloadableTable (*)();

std::string readableReloadableSignature(std::string_view signature)
{
    constexpr auto marker = std::string_view {"Signature = "};
    auto start = signature.find(marker);

    if (start != std::string_view::npos)
    {
        signature.remove_prefix(start + marker.size());

        if (signature.ends_with(']'))
            signature.remove_suffix(1);
    }

    return std::string {signature};
}

ReloadableEntry::Erased findReloadable(const ReloadableTable& table,
                                       std::string_view signature)
{
    for (auto index = 0; index < table.count; ++index)
    {
        const auto& entry = table.entries[index];

        if (entry.function != nullptr && signature == entry.signature)
            return entry.function;
    }

    return nullptr;
}

std::string joinedSignatures(const Vector<std::string>& names)
{
    auto text = std::string {};

    for (const auto& name: names)
        text += (text.empty() ? "" : ", ") + name;

    return text;
}

Vector<std::string> signaturesMissingFrom(const ReloadableTable& table,
                                          const ReloadableTable& reference)
{
    auto missing = Vector<std::string> {};

    for (auto index = 0; index < reference.count; ++index)
    {
        const auto& entry = reference.entries[index];

        if (findReloadable(table, entry.signature) == nullptr)
            missing.add(readableReloadableSignature(entry.signature));
    }

    return missing;
}

std::string reloadRefusalFor(const ReloadableTable& compiledIn,
                             const ReloadableTable& module)
{
    auto missing = signaturesMissingFrom(module, compiledIn);

    if (missing.empty())
        return {};

    auto extra = signaturesMissingFrom(compiledIn, module);

    if (extra.empty())
        return "the module lacks " + joinedSignatures(missing);

    return "signature mismatch: the module lacks " + joinedSignatures(missing)
           + " and has " + joinedSignatures(extra);
}

HotReloadLoop::Options
    reloadableTargetOptions(const HotReloadTarget& target,
                            const Files::SourceWatcher::Options& sources)
{
    auto options = hotReloadOptions(target);
    auto watched = sources;

    if (watched.directory.empty())
        watched.directory = options.sources.directory;

    options.sources = watched;
    return options;
}
} // namespace

Reloadable::Reloadable(const Detail::ReloadableSetup& setup,
                       const ReloadableOptions& optionsToUse)
    : options(optionsToUse)
    , compiledIn(setup.compiledIn)
{
    if (setup.hotReload)
        startLoop(setup.target);

    if (options.callOnReloadedAtCreation)
        options.onReloaded();
}

Reloadable::~Reloadable() = default;

std::string Reloadable::describeState() const
{
    return loop ? loop->describeState() : "static";
}

int Reloadable::getGeneration() const
{
    return loop ? loop->getGeneration() : 0;
}

bool Reloadable::isModuleLoaded() const
{
    return getGeneration() > 0;
}

ReloadableEntry::Erased
    Reloadable::compiledInFunction(std::string_view signature) const
{
    if (auto function = findReloadable(compiledIn, signature))
        return function;

    throw std::logic_error("EACP_RELOADABLE lists no "
                           + readableReloadableSignature(signature));
}

ReloadableEntry::Erased Reloadable::currentFunction(std::string_view signature) const
{
    auto compiled = compiledInFunction(signature);

    if (!isModuleLoaded())
        return compiled;

    auto function = findReloadable(loaded, signature);
    return function != nullptr ? function : compiled;
}

void Reloadable::logFallback(const std::string& reason) const
{
    options.log("HotReload: " + reason + ", using the compiled-in copy");
}

std::string Reloadable::refusalOf(const DynamicLibrary& image) const
{
    auto table = image.findFunction<ReloadableTableFunction>(reloadableTableSymbol);

    if (table == nullptr)
        return std::string {"no "} + reloadableTableSymbol;

    return reloadRefusalFor(compiledIn, table());
}

HotReloadLoop::Options Reloadable::loopOptions(const HotReloadTarget& target) const
{
    auto loopOptions = options.loop
                           ? *options.loop
                           : reloadableTargetOptions(target, options.sources);
    auto validateAlso = loopOptions.validate;

    auto validate = [this, validateAlso](const DynamicLibrary& image)
    {
        auto refusal = refusalOf(image);
        return refusal.empty() ? validateAlso(image) : refusal;
    };

    loopOptions.requiredSymbol = reloadableTableSymbol;
    loopOptions.validate = validate;
    return loopOptions;
}

void Reloadable::startLoop(const HotReloadTarget& target)
{
    auto& running = loop.emplace(loopOptions(target));

    auto stateChanged = [this] { options.onStateChanged(); };

    auto buildStarted = [this]
    {
        options.log("HotReload: building " + loop->getLibrary().getBuiltPath().stem()
                    + "…");
    };

    auto buildOutput = [this](std::string_view chunk)
    { options.onBuildOutput(chunk); };

    auto buildFailed = [this](const BuildFailure& failure)
    {
        options.log("HotReload: build failed: " + failure.summary);
        options.onBuildFailed(failure);
    };

    auto loadFailed = [this](const std::string& error)
    {
        options.log("HotReload: load failed: " + error);
        options.onLoadFailed(error);
    };

    auto swapped = [this](ReloadableLibrary&) { reloaded(); };

    running.onStateChanged = stateChanged;
    running.onBuildStarted = buildStarted;
    running.onBuildOutput = buildOutput;
    running.onBuildFailed = buildFailed;
    running.onLoadFailed = loadFailed;
    running.onReloaded = swapped;

    options.log("HotReload: watching " + running.getSourceDirectory().str());
}

void Reloadable::reloaded()
{
    auto table = loop->getLibrary().findFunction<ReloadableTableFunction>(
        reloadableTableSymbol);
    loaded = table();

    options.log("HotReload: generation " + std::to_string(getGeneration()));
    options.onReloaded();
}
} // namespace eacp::Plugins
