# Hot reload

Hot reload rebuilds part of a running app when you save a source file, and
swaps the new code in without restarting the app. A layout is the usual case:
you change a margin, save, and a second later the window moves its widgets. A
view is the other: you change a shader, or swap one view class for another,
and the running window shows the new one.

The code that reloads is a short list of functions, which you name with
`EACP_RELOADABLE`, and the app reaches them through `Plugins::Reloadable`.
The functions are always compiled into the app. In dev mode they are also
built as a small module, which the app rebuilds and reloads on every save. In
static mode, which covers every release build and every platform hot reload
does not run on, the app calls its compiled-in copies and nothing else.

## Using it: `Reloadable` and `EACP_RELOADABLE`

### Listing the functions

Put the functions in a source file of their own, in an anonymous namespace.
List them once with `EACP_RELOADABLE`, at global scope:

```cpp
// Layout.cpp
#include "LayoutABI.h"
#include <eacp/Core/Plugins/ReloadableEntry.h>

namespace
{
void demoLayout(const LayoutTargets& targets, const eacp::UI::Rect& bounds)
{
    auto area = bounds.inset(16.f);
    targets.title.setBounds(area.removeFromTop(36.f));
    // ...
}
} // namespace

EACP_RELOADABLE(demoLayout)
```

The app finds each function by its type, so each type may appear only once:
`EACP_RELOADABLE(makeDemoView, demoLayout)` is fine, but two factories of the
same type are a compile error. `EACP_RELOADABLE` may also appear only once
per app.

### Calling a function

Declare the function's type in a header that both sides include:

```cpp
// LayoutABI.h
using LayoutFunction = void(const LayoutTargets& targets,
                            const eacp::UI::Rect& bounds);
```

Then hold a `Reloadable` in the app and fetch the function by its type
whenever you call it:

```cpp
// Main.cpp
#include <eacp/Core/Plugins/Reloadable.h>

eacp::Plugins::Reloadable reloadable;

void applyLayout() { reloadable.call<LayoutFunction>(targets, getLocalBounds()); }
```

That is the minimal form. It needs no options, no symbol names and no
`if constexpr`. In dev mode it watches the source directory, rebuilds the
module on save, checks it and swaps the module in, logging each step through
`LOG`. In static mode it calls the compiled-in copy.

- `function<F>()` returns the current generation's `F*`, or the compiled-in
  one.
- `call<F>(args...)` calls it.

Fetch the function each time you use it. A pointer kept across a reload
points into a module that is about to be unloaded.

### Making objects

For an object, list a factory that returns a raw `T*`:

```cpp
// Views.cpp
#include <eacp/Core/Plugins/ReloadableEntry.h>
#include <eacp/GPU/GPU.h>

namespace
{
struct PlasmaView final : eacp::GPU::GPUView { /* ... */ };

eacp::Graphics::View* makeDemoView()
{
    return new PlasmaView;
}
} // namespace

EACP_RELOADABLE(makeDemoView)
```

`create<T>()` calls the current generation's `T*()` factory and returns the
object in an `OwningPointer<T>`. The caller owns it. `Reloadable` owns no
objects, so one `Reloadable` can make any number of them, of one type or of
several. `T` must have a virtual destructor, and a `static_assert` checks
this; see "Ownership" below for why.

If the module's factory returns null or throws, `create<T>()` uses the
compiled-in factory instead and logs the fallback. A type that
`EACP_RELOADABLE` does not list throws `std::logic_error` naming the
signature; so does a compiled-in factory that returns null.

### The lifetime rule

`onReloaded()` runs once at the end of the `Reloadable`'s constructor, in
static mode as well as dev mode, and again after every reload. Your setup
therefore lives in one place: the hook that makes the objects at start-up is
the same hook that replaces them on a reload.
`ReloadableOptions::callOnReloadedAtCreation` (default `true`) turns the
first call off.

- **Declare the `Reloadable` last**, after every member its `onReloaded`
  uses. Its constructor calls the hook, so those members must already exist.
- **On a reload, replace or destroy every object** made by `create<T>()`
  inside `onReloaded`. That is the one moment when the new module is loaded
  and the old one, which made those objects, is still loaded.
  `Plugins::unload` unloads the old module one loop turn later, and an object
  still alive then would run its destructor from code that is no longer
  there.
- **Destroying the `Reloadable` is safe for members destroyed after it.**
  Members are destroyed in reverse order, so the holders declared before the
  `Reloadable` die after it. Its module is not unloaded there and then: it
  goes one loop turn later, or, when no loop is running (as at the end of
  `Apps::run`), it stays loaded until the process exits.

### Views: `Graphics::OptionalView`

`Graphics::OptionalView<T = View>` (`Graphics/View/OptionalView.h`) is a
general-purpose view that holds at most one child, and it is the natural home
for a reloaded view:

- `set(content)` destroys the previous child, adds the new one as a subview
  and makes it fill the `OptionalView`.
- `clear()` destroys the child.
- `get()`, `operator*`, `operator->`, `explicit operator bool` and
  `hasContent()` read it.
- `resized()` keeps the child filling the view.

A destroyed `View` removes itself from its parent (every platform's
`View::~View` calls `removeFromParent()`), so the old view leaves the tree on
its own and nothing has to be detached. `Apps/GPU/HotReload` is the whole
pattern:

```cpp
struct DemoRoot final : Graphics::View
{
    DemoRoot() { addSubview(content); }

    Plugins::ReloadableOptions reloadOptions()
    {
        auto options = Plugins::ReloadableOptions {};
        auto reinstall = [this]
        { content.set(reloadable.create<Graphics::View>()); };
        options.onReloaded = reinstall;
        return options;
    }

    void resized() override { content.setBounds(getLocalBounds()); }

    Graphics::OptionalView<> content;
    Plugins::Reloadable reloadable {reloadOptions()};
};
```

`reinstall` is the only place a view is made. It runs once when `reloadable`
is constructed, which is why `reloadable` comes after `content`, and again
on every reload. There, `set` makes the new view and then destroys the old
one, while the old module is still loaded. When `DemoRoot` is destroyed,
`reloadable` goes first and `content` after it, and the module that made the
view stays loaded until then.

### Options and hooks

`Reloadable` takes an optional `Plugins::ReloadableOptions`. Every member has
a working default, so you set only the members you need.

| Member | Default | Called |
| --- | --- | --- |
| `log(std::string_view)` | `LOG` | for each line below |
| `onReloaded()` | no-op | once at construction, then after each new module is loaded, while the old one still is |
| `callOnReloadedAtCreation` | `true` | whether `onReloaded()` also runs at construction |
| `onStateChanged()` | no-op | whenever `describeState()` changes |
| `onBuildFailed(const BuildFailure&)` | no-op | after a failed build |
| `onLoadFailed(const std::string&)` | no-op | after a module was refused |
| `onBuildOutput(std::string_view)` | no-op | for each chunk of build output |
| `sources` | the target's | watcher settings: pattern, poll interval, settle time |
| `loop` | from the target | a whole `HotReloadLoop::Options`, replacing the target's |

`Reloadable` always writes to `log` first and then calls the hook. Setting a
hook therefore never silences the log line; to silence logging, set `log` to a
no-op. The lines are:

```
HotReload: watching <source directory>
HotReload: building <module>…
HotReload: build failed: <summary>
HotReload: load failed: <reason>
HotReload: generation <N>
HotReload: factory returned nothing, using the compiled-in copy
HotReload: factory threw: <what>, using the compiled-in copy
```

Static mode logs nothing and calls no hooks.

`sources` applies to the options taken from the app's target. If its
`directory` is empty, the target's source directory is kept. `loop` replaces
everything the target would give. Tests and custom builds use it to point
`Reloadable` at their own build command and module. Whichever is used,
`Reloadable` sets `requiredSymbol` and `validate` itself; a `validate` you set
in `loop` runs after its own check.

`describeState()` returns the loop's status line (`generation N`, `building…`,
`build failed: …` or `load failed: …`), or `static` in static mode.
`getGeneration()` returns 0 until the first reload and goes up by one with
each reload. A `Reloadable` can be neither copied nor moved.

### What happens on a reload

When a build succeeds, the new module is staged and opened. Before it replaces
the current one, two checks run:

1. The module must export the table `EACP_RELOADABLE` defines.
2. The table must provide every signature that the compiled-in table has.

If either check fails, the module is refused. The previous generation stays,
and the reason becomes `load failed: …`, naming what is wrong:

- a missing factory: `the module lacks ReloadableTestObject *()`
- a changed type: changing `int answer()` to `long answer()` gives
  `signature mismatch: the module lacks int () and has long ()`

Without these checks, the app would call a function through the wrong type.

If both checks pass:

1. The new module becomes the current generation, and
   `HotReload: generation N` is logged.
2. `function<F>()` and `create<T>()` now reach the new module.
3. `onReloaded()` is called. This is where the app replaces its objects.
4. The old module is unloaded one loop turn later.

A failed build, a module that does not load, a missing table and a refused
table all keep the previous generation. Each one is logged, passed to its
hook and shown by `describeState()`. Nothing builds until the first save.

### Ownership and the virtual destructor

A reloadable object is created by code in the module, and its vtable and
destructor live there too. Its owner deletes it through `T*`. A virtual
destructor sends that delete through the object's own vtable, so the deleting
destructor that runs is the module's: the matching destructor body and
`operator delete` are called. Without a virtual destructor, the app would run
its own idea of `~T` on an object it did not create. The `static_assert` in
`create<T>()` stops that at compile time.

## The module rules

The module is built from the same sources as the app's compiled-in copy, but
it links nothing. Its references to eacp stay undefined and are resolved when
it loads, against the eacp the app exports. Both therefore share one eacp,
built by one compiler, and C++ types can cross between them. The rules follow
from that.

- **Only the table is exported.** `EACP_RELOADABLE` defines
  `eacpReloadableTable`, an `extern "C"` function marked
  `EACP_HOT_RELOAD_EXPORT`, which returns every listed function with its
  signature. Everything else in the module's sources stays in an anonymous
  namespace. A function with external linkage would bind to the app's
  generation-0 copy of itself, so editing it would have no effect.
- **One per app.** `eacp_add_hot_reload` describes one module, and the table
  has a fixed name. A second `EACP_RELOADABLE` in the same app fails to link
  with a duplicate symbol, which is intended. List every function in one
  `EACP_RELOADABLE`.
- **One function per type.** Lookup is by type, so `EACP_RELOADABLE`
  `static_assert`s that the listed functions' types are pairwise distinct.
- **The module owns nothing.** It installs no callbacks, starts no threads or
  timers, keeps no statics the app can see and registers nothing with the
  app. Its code is unloaded one loop turn after the next reload, and anything
  still pointing into it would then point at unloaded code.
- **Objects die inside `onReloaded`.** Anything the module creates is handed
  to the app, which destroys or replaces it inside the next `onReloaded` at
  the latest.

The table is a C++ struct, `ReloadableTable` (a pointer and a count of
`ReloadableEntry {function, signature}`), returned from a C-linkage function.
This is safe because the module and the app share a compiler.
`ReloadableEntry.h` turns off clang's `-Wreturn-type-c-linkage` around the
entry, and `eacp_add_hot_reload` sets `EACP_HOT_RELOAD_ABI` on the app and
its module, which turns that warning off for any other exported entry point
the app writes. The entries live in a local static of that non-inline
function, so the app's copy and the module's copy are never merged by the
loader.

## CMake: `eacp_add_hot_reload` and `EACP_HOT_RELOAD`

```cmake
eacp_add_app(HotReload Main.cpp)
target_link_libraries(HotReload PRIVATE eacp-ui)
eacp_add_hot_reload(HotReload MODULE HotReloadLayout SOURCES Layout.cpp
        [HOT_RELOAD ON|OFF])
```

`eacp_add_hot_reload` (`CMake/EacpPlugin.cmake`) always adds `SOURCES` to the
app. That copy is generation 0 and the fallback. Whether the app is in dev
mode or static mode is decided like this:

- `HOT_RELOAD ON` is always dev mode, and `HOT_RELOAD OFF` is always static,
  whatever the option says.
- With no `HOT_RELOAD`, the app follows the `EACP_HOT_RELOAD` cache option
  (default `OFF`).
- iOS, Android and Windows are always static, whatever either says.

Both in-tree examples pass `HOT_RELOAD ON`, so on macOS and Linux they are in
dev mode in every build directory, and in static mode on Windows, iOS and
Android:

- `Apps/UI/HotReload` (`HotReload`, module `HotReloadLayout`) reloads a layout
  function.
- `Apps/GPU/HotReload` (`HotReloadGPU`, module `HotReloadGPUView`) reloads a
  factory for a `GPUView` defined in the module. The app holds the view in an
  `OptionalView` and replaces it inside `onReloaded`.

A configure prints `<app>: hot reload ON` or `<app>: hot reload OFF`.

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DEACP_UNITY_BUILD=OFF \
      -DEACP_HOT_RELOAD=ON
```

**Static mode** defines `EACP_HOT_RELOAD=0` on the app and builds no module,
no watcher and no build runner. `Reloadable` uses the compiled-in table, and
`HotReloadLoop` is never constructed.

**Dev mode** also does the following:

- It builds the sources as `MODULE`, a plugin library, and makes the app
  depend on it.
- It defines `EACP_HOT_RELOAD=1` and the module's path, the CMake executable,
  the build directory, the configuration, the module's target name and the
  source directory on the app. `Plugins::hotReloadTarget()` reads them back.
- It gives the app `ENABLE_EXPORTS`: `-export_dynamic` on Apple, and
  `--export-dynamic` on Linux, which keeps the exports alive under the
  libraries' Release IPO.
- It links every static library in the app's link closure whole-archive, so
  the module can call eacp code the app itself never calls.
- It links the module with `-undefined dynamic_lookup` on Apple, which is
  already a Linux module's default.
- It compiles every eacp library in that closure at default visibility. That
  setting belongs to the library targets, so it applies to the whole build
  directory. `eacp_add_plugin` modules built there re-export eacp and lose
  their isolation, so do not mix the two in one build directory.

A save rebuilds the module alone (`cmake --build <dir> --config <config>
--target <module>`), never the running app.

`Plugins/HotReload.h` holds the mode (`Plugins::isHotReload()`) and
`hotReloadTarget()`. It and `Plugins/Reloadable.h` are included directly,
never through the `Core.h` umbrella, because what they contain depends on the
app's own compile definitions. `Reloadable` itself is a plain class built
into eacp-core, which never sees those definitions or the app's table. Only
its public constructor is a template, `template <bool hotReload =
isHotReload(), auto compiledIn = &eacpReloadableTable>`, so the mode, the
module and the compiled-in table are read at the app's own call site and
handed to the class as a `Detail::ReloadableSetup`. Tests build that setup by
hand: their own table, `true` or `false`, and a `loop` in the options.

## Platform limits

Hot reload runs on macOS and Linux. Everywhere else it is static:

- **Windows.** `WINDOWS_EXPORT_ALL_SYMBOLS` never exports a static library's
  symbols, so a module has no eacp of the app's to bind to. A module that
  linked the executable instead would relink the locked, running exe on every
  save. `EACP_HOT_RELOAD_EXPORT` is empty there.
- **iOS and Android.** The app runs on a device or simulator with no CMake to
  run and no source tree to watch.

## The lower layers

`Reloadable` is built from four pieces in `eacp-core`. Each can be used on its
own, for a module with its own entry points or a build that is not CMake's.
All of them run on the main thread only.

### `Plugins::HotReloadLoop` (`Plugins/HotReloadLoop.h`)

A `Files::SourceWatcher`, a `Processes::BuildRunner` and a `ReloadableLibrary`
combined into one object. It is built from `Options`, or from a
`HotReloadTarget` and a required symbol. `hotReloadOptions(target)` returns
the `Options` a target describes, and `hotReloadBuildCommand(target)`
(`Plugins/HotReloadTarget.h`) returns the build command alone.

- Each settled burst of changes starts a build. A change during a build
  causes exactly one more run, reported as a build of its own.
- A successful build reloads the module and calls `onReloaded(library)`. That
  happens only when the module exports `Options::requiredSymbol` and
  `Options::validate` accepts it, so `onReloaded` can look its entry point up
  with no fallback.
- A failed build or load keeps the generation and calls `onBuildFailed` or
  `onLoadFailed`.
- `onBuildStarted`, `onBuildOutput` and `onStateChanged` report the rest.
- `getState()` and `describeState()` (`generation N`, `building…`,
  `build failed: …`, `load failed: …`) drive a status line.
- The module's directory and the staging directory are left out of the watch
  when they lie inside the sources.
- Nothing builds until the first change. `rebuild()` starts a build by hand.
- No callback may destroy the loop.

### `Plugins::BuildFailure` and `buildFailureFrom`

These hold the build's output, whether it launched, whether it exited, its
exit code and a one-line `summary`. To pick the summary line,
`buildFailureFrom` strips ANSI colour from the output and takes the first
match from this list, in order:

1. a compiler error with a file location (clang, GCC or MSVC)
2. any `error:` line that is not the driver's or the linker's
3. a `CMake Error`
4. ninja's `FAILED:` line
5. the last line of output

With no output at all, the summary names the cause instead: the build did
not start, it was cancelled, or it exited with the given code.

### `Plugins::ReloadableLibrary` (`Plugins/ReloadableLibrary.h`)

A `DynamicLibrary` over a module file that is rebuilt while the app runs.

- **Staged copies.** `reload()` copies the built file to
  `<stem>-<gen>-<pid>-<serial>` in a staging directory and opens that copy,
  never the built file itself. A new path for each load avoids the registry
  handing back the old image, and keeps the built file free for the linker to
  overwrite. Two running copies of an app never remove each other's staged
  files.
- **The old image.** It goes to `Plugins::unload`, which unloads it one loop
  turn later when a loop is running. Every pointer that came from it is
  invalid after the next successful reload.
- **Destruction.** The current image is not unloaded in the destructor
  either. It goes one loop turn later, or, with no loop running, stays
  loaded until the process exits. Objects it made can therefore outlive the
  library for the rest of the current turn.
- **Refusals.** A module without `requiredSymbol` is refused with
  `lastError()` set to `no <symbol>`. A module that `validate` answers with a
  non-empty string is refused with that string as `lastError()`. Either way,
  the current generation stays loaded.
- **Accessors.** `getGeneration()`, `getLoadedPath()`, `findFunction` and
  `findSymbol`.

### `Processes::BuildRunner` (`Process/BuildRunner.h`)

Runs one command again on demand, on a background thread, with stderr merged
into stdout. `onOutput` and `onFinished` are called on the main thread. A
`run()` during a run asks for exactly one more run. A cancelled run still
finishes, with `exited == false`. Cancelling kills only the direct child, not
ninja under cmake.

### `Files::SourceWatcher` (`Utils/SourceWatcher.h`)

A `Files::FileWatcher` over one source tree. It is always recursive and
matches `Files::sourceFilePattern` (C, C++ and Objective-C sources and
headers) unless its `Options` name another pattern. The rest of
`FileWatcherOptions` goes in `Options::watcher`: poll interval, settle time
and excluded directories. A directory that does not exist yet is watched until
it appears.
