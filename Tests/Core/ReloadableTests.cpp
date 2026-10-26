#include "HotReloadSandbox.h"
#include "ReloadableTestObject.h"
#include <eacp/Core/Plugins/Reloadable.h>

#include <array>
#include <optional>
#include <stdexcept>

using namespace nano;

namespace ReloadableTests
{
using eacp::FilePath;
using eacp::OwningPointer;
using eacp::Plugins::Reloadable;
using eacp::Plugins::ReloadableEntry;
using eacp::Plugins::ReloadableOptions;
using eacp::Plugins::ReloadableTable;
using eacp::Plugins::Detail::reloadableEntries;
using eacp::Plugins::Detail::ReloadableSetup;
using eacp::Time::MS;

using AnswerFunction = int();
using Object = ReloadableTestObject;

const auto answerOne = FilePath {EACP_RELOADABLE_FIXTURE_A};
const auto answerTwo = FilePath {EACP_RELOADABLE_FIXTURE_B};
const auto answerLong = FilePath {EACP_RELOADABLE_FIXTURE_MISMATCH};
const auto objectOne = FilePath {EACP_RELOADABLE_FIXTURE_OBJECT_A};
const auto objectTwo = FilePath {EACP_RELOADABLE_FIXTURE_OBJECT_B};
const auto noObject = FilePath {EACP_RELOADABLE_FIXTURE_NULL};
const auto throwingObject = FilePath {EACP_RELOADABLE_FIXTURE_THROW};
const auto answerAndObject = FilePath {EACP_RELOADABLE_FIXTURE_PAIR};
const auto noTable = FilePath {EACP_RELOAD_FIXTURE_C};

int compiledInAnswer()
{
    return 0;
}

struct CompiledInObject final : Object
{
    ~CompiledInObject() override { markDestroyed(); }

    int value() const override { return -1; }
};

Object* makeCompiledInObject()
{
    return new CompiledInObject;
}

const auto answerEntries = reloadableEntries(&compiledInAnswer);
const auto objectEntries = reloadableEntries(&makeCompiledInObject);
const auto pairEntries = reloadableEntries(&compiledInAnswer, &makeCompiledInObject);

template <std::size_t size>
ReloadableTable tableOf(const std::array<ReloadableEntry, size>& entries)
{
    return {entries.data(), static_cast<int>(size)};
}

struct Log
{
    std::function<void(std::string_view line)> sink()
    {
        auto add = [this](std::string_view line) { lines.add(std::string {line}); };
        return add;
    }

    bool contains(std::string_view wanted) const
    {
        for (const auto& line: lines)
            if (line == wanted)
                return true;

        return false;
    }

    int finished() const
    {
        auto count = 0;

        for (const auto& line: lines)
            for (auto prefix: {"HotReload: generation ",
                               "HotReload: build failed: ",
                               "HotReload: load failed: "})
                if (line.starts_with(prefix))
                    ++count;

        return count;
    }

    bool waitForFinished(int count) const
    {
        auto reached = [this, count] { return finished() >= count; };
        return eacp::Threads::runEventLoopUntil(reached, MS {10000});
    }

    std::string line(int index) const
    {
        return index < lines.size() ? lines[index] : std::string {};
    }

    eacp::Vector<std::string> lines;
};

ReloadableOptions
    optionsFor(const HotReloadSandbox& sandbox, const std::string& script, Log& log)
{
    auto options = ReloadableOptions {};
    options.loop = sandbox.loopOptions(script);
    options.log = log.sink();
    return options;
}

ReloadableSetup staticSetup(const ReloadableTable& table)
{
    return {table, false, {}};
}

ReloadableSetup devSetup(const ReloadableTable& table)
{
    return {table, true, {}};
}

std::string thrownBy(const std::function<void()>& action)
{
    try
    {
        action();
    }
    catch (const std::logic_error& e)
    {
        return e.what();
    }

    return {};
}

auto tStatic = test("Reloadable/staticModeUsesTheCompiledInTable") = []
{
    auto log = Log {};
    auto options = ReloadableOptions {};
    options.log = log.sink();

    auto reloadable = Reloadable {staticSetup(tableOf(pairEntries)), options};

    check(reloadable.function<AnswerFunction>() == &compiledInAnswer);
    check(reloadable.call<AnswerFunction>() == 0);
    check(reloadable.create<Object>()->value() == -1);
    check(reloadable.describeState() == "static");
    check(reloadable.getGeneration() == 0);
    check(log.lines.empty());
};

auto tUnknownType = test("Reloadable/anUnlistedTypeThrows") = []
{
    auto reloadable = Reloadable {staticSetup(tableOf(answerEntries)), {}};

    auto createObject = [&reloadable] { reloadable.create<Object>(); };
    auto fetchLong = [&reloadable] { reloadable.function<long()>(); };

    check(thrownBy(createObject).find("ReloadableTestObject *()")
          != std::string::npos);
    check(thrownBy(fetchLong).find("long ()") != std::string::npos);
};

auto tSwapsFunction = test("Reloadable/aFunctionSwapsAcrossGenerations") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto reloads = 0;
    auto options =
        optionsFor(sandbox, sandbox.copyInTurnScript(answerOne, answerTwo), log);
    auto countReload = [&reloads] { ++reloads; };
    options.onReloaded = countReload;

    auto reloadable = Reloadable {devSetup(tableOf(answerEntries)), options};
    check(reloads == 1);
    check(reloadable.call<AnswerFunction>() == 0);
    check(reloadable.describeState() == "generation 0");

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));
    check(reloadable.function<AnswerFunction>()() == 1);
    check(reloadable.getGeneration() == 1);

    sandbox.touchSource("abc");
    check(log.waitForFinished(2));
    check(reloadable.call<AnswerFunction>() == 2);
    check(reloadable.getGeneration() == 2);
    check(reloadable.describeState() == "generation 2");
    check(reloads == 3);
};

auto tCalledAtCreation = test("Reloadable/onReloadedRunsOnceAtCreation") = []
{
    auto reloads = 0;
    auto countReload = [&reloads] { ++reloads; };
    auto options = ReloadableOptions {};
    options.onReloaded = countReload;

    auto staticReloadable =
        Reloadable {staticSetup(tableOf(answerEntries)), options};
    check(reloads == 1);

    auto sandbox = HotReloadSandbox {};
    auto log = Log {};
    auto devOptions = optionsFor(sandbox, sandbox.copyScript(answerOne), log);
    devOptions.onReloaded = countReload;

    auto devReloadable = Reloadable {devSetup(tableOf(answerEntries)), devOptions};
    check(reloads == 2);
    check(devReloadable.getGeneration() == 0);
    check(log.lines.size() == 1);
};

auto tNotCalledAtCreation =
    test("Reloadable/onReloadedCanBeLeftUntilTheFirstReload") = []
{
    auto reloads = 0;
    auto countReload = [&reloads] { ++reloads; };
    auto options = ReloadableOptions {};
    options.onReloaded = countReload;
    options.callOnReloadedAtCreation = false;

    auto reloadable = Reloadable {staticSetup(tableOf(answerEntries)), options};

    check(reloads == 0);
    check(reloadable.call<AnswerFunction>() == 0);
};

// The app's pattern: the holder of the objects first, the Reloadable last, and
// onReloaded the one place objects are made.
struct ObjectOwner
{
    ObjectOwner(const ReloadableTable& table, const ReloadableOptions& base)
        : reloadable {devSetup(table), withReplacement(base)}
    {
    }

    ReloadableOptions withReplacement(const ReloadableOptions& base)
    {
        auto options = base;
        auto replace = [this] { replaceObject(); };
        options.onReloaded = replace;
        return options;
    }

    void replaceObject()
    {
        auto oldDestroyed = false;

        if (current != nullptr)
        {
            current->destroyed = &oldDestroyed;
            oldValues.add(current->value());
        }

        current = reloadable.create<Object>();

        if (oldDestroyed)
            ++destroyedInsideTheHook;

        newValues.add(current->value());
    }

    OwningPointer<Object> current;
    eacp::Vector<int> oldValues;
    eacp::Vector<int> newValues;
    int destroyedInsideTheHook = 0;
    Reloadable reloadable;
};

auto tIndependentObjects = test("Reloadable/createMakesIndependentObjects") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto reloadable =
        Reloadable {devSetup(tableOf(objectEntries)),
                    optionsFor(sandbox, sandbox.copyScript(objectOne), log)};

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));

    auto first = reloadable.create<Object>();
    auto second = reloadable.create<Object>();
    auto firstDestroyed = false;
    auto secondDestroyed = false;
    first->destroyed = &firstDestroyed;
    second->destroyed = &secondDestroyed;

    check(first.get() != second.get());
    check(first->value() == 1);
    check(second->value() == 1);

    first.reset();
    check(firstDestroyed);
    check(!secondDestroyed);
    check(second->value() == 1);
};

auto tReplacedInsideHook =
    test("Reloadable/objectsReplacedInOnReloadedDieWhileTheOldImageIsMapped") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto owner = ObjectOwner {
        tableOf(objectEntries),
        optionsFor(sandbox, sandbox.copyInTurnScript(objectOne, objectTwo), log)};

    check(owner.newValues.size() == 1);
    check(owner.newValues[0] == -1);
    check(owner.oldValues.empty());

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));
    sandbox.touchSource("abc");
    check(log.waitForFinished(2));

    check(owner.oldValues.size() == 2);
    check(owner.oldValues[0] == -1);
    check(owner.oldValues[1] == 1);
    check(owner.newValues.size() == 3);
    check(owner.newValues[1] == 1);
    check(owner.newValues[2] == 2);
    check(owner.destroyedInsideTheHook == 2);
    check(owner.current->value() == 2);
};

auto tOutlivesTheReloadable =
    test("Reloadable/anObjectDestroyedAfterTheReloadableIsSafe") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto owner = std::optional<ObjectOwner> {};
    owner.emplace(tableOf(objectEntries),
                  optionsFor(sandbox, sandbox.copyScript(objectOne), log));

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));
    check(owner->current->value() == 1);

    auto destroyed = false;
    owner->current->destroyed = &destroyed;

    auto destroyOwner = [&owner]
    {
        check(eacp::Threads::isEventLoopRunning());
        owner.reset();
    };

    eacp::Threads::runEventLoopFor(MS {20}, destroyOwner);

    check(destroyed);
};

auto tTwoFactories = test("Reloadable/oneModuleProvidesTwoFactoriesByType") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto reloadable =
        Reloadable {devSetup(tableOf(pairEntries)),
                    optionsFor(sandbox, sandbox.copyScript(answerAndObject), log)};

    check(reloadable.call<AnswerFunction>() == 0);
    check(reloadable.create<Object>()->value() == -1);

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));

    check(reloadable.getGeneration() == 1);
    check(reloadable.call<AnswerFunction>() == 7);
    check(reloadable.create<Object>()->value() == 7);
};

auto tMissingFactory = test("Reloadable/aModuleMissingAFactoryIsRefused") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto reloadable =
        Reloadable {devSetup(tableOf(pairEntries)),
                    optionsFor(sandbox, sandbox.copyScript(answerOne), log)};

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));

    check(reloadable.getGeneration() == 0);
    check(reloadable.call<AnswerFunction>() == 0);
    check(reloadable.describeState()
          == "load failed: the module lacks ReloadableTestObject *()");
};

auto tMismatch = test("Reloadable/aSignatureMismatchIsRefused") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto loadFailures = eacp::Vector<std::string> {};
    auto options =
        optionsFor(sandbox, sandbox.copyInTurnScript(answerOne, answerLong), log);
    auto keepFailure = [&loadFailures](const std::string& error)
    { loadFailures.add(error); };
    options.onLoadFailed = keepFailure;

    auto reloadable = Reloadable {devSetup(tableOf(answerEntries)), options};

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));
    sandbox.touchSource("abc");
    check(log.waitForFinished(2));

    check(reloadable.call<AnswerFunction>() == 1);
    check(reloadable.getGeneration() == 1);
    check(loadFailures.size() == 1);
    check(reloadable.describeState().starts_with("load failed: signature mismatch"));
    check(loadFailures[0]
          == "signature mismatch: the module lacks int () and has long ()");
};

auto tFallback = test("Reloadable/aNullOrThrowingFactoryFallsBack") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto script = sandbox.copyInTurnScript(noObject, throwingObject);
    auto reloadable = Reloadable {devSetup(tableOf(objectEntries)),
                                  optionsFor(sandbox, script, log)};

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));
    check(reloadable.create<Object>()->value() == -1);
    check(log.contains(
        "HotReload: factory returned nothing, using the compiled-in copy"));

    sandbox.touchSource("abc");
    check(log.waitForFinished(2));
    check(reloadable.getGeneration() == 2);
    check(reloadable.create<Object>()->value() == -1);
    check(log.contains(
        "HotReload: factory threw: no object today, using the compiled-in copy"));
};

auto tDefaultLog = test("Reloadable/theDefaultLogLines") = []
{
    auto sandbox = HotReloadSandbox {};
    sandbox.touchSource("a");

    auto log = Log {};
    auto reloadable = Reloadable {
        devSetup(tableOf(answerEntries)),
        optionsFor(sandbox, sandbox.copyInTurnScript(answerOne, noTable), log)};

    check(log.line(0) == "HotReload: watching " + sandbox.sources().str());

    sandbox.touchSource("ab");
    check(log.waitForFinished(1));
    sandbox.touchSource("abc");
    check(log.waitForFinished(2));

    auto building = "HotReload: building " + sandbox.modulePath().stem() + "…";

    check(log.lines.size() == 5);
    check(log.line(1) == building);
    check(log.line(2) == "HotReload: generation 1");
    check(log.line(3) == building);
    check(log.line(4) == "HotReload: load failed: no eacpReloadableTable");
    check(reloadable.call<AnswerFunction>() == 1);
};
} // namespace ReloadableTests
