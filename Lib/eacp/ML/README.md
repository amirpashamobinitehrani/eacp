# ML

A Core ML backend for tensor-level compute, beside the compute kernels the
shader EDSL already runs on Metal, D3D12, Vulkan and the CPU. A net written
against eacp keeps its `ComputeProgram` kernels on every platform; this module
is the second way to run the same maths on Apple hardware, as a fixed graph of
whole-tensor ops that Core ML compiles ahead of time and places on the CPU,
the GPU or the Apple Neural Engine. The net that proved it is
[WhisperEACP](https://github.com/eyalamirmusic/WhisperEACP), whose own tests
check its tiny.en encoder against the kernel path; `Tests/ML` builds the
encoder's graph at its real sizes and runs small programs against fp32 scalar
references.

It is not a fourth backend of `GPU::ComputePass`. A kernel body is one
thread's view — `threadId()`, `shared<>`, `barrier()`, atomics — and Core ML
has no thread. What the two share is the value layer: an elementwise body
here is written with the EDSL's `GPU::Float` and lowered to MIL rather than to
MSL.

## Three targets

| | |
| --- | --- |
| `eacp-ml-graph` | The graph builder and the writers that turn it into an `.mlpackage`: a MIL program, a hand-rolled protobuf encoder and the fp16 weight blob. Bytes in, bytes out, verified byte for byte against coremltools, with no Core ML under it — so it builds and `MLGraphTests` runs on macOS, Windows and Linux. Links `eacp-core` and `eacp-gpu-codegen` only |
| `eacp-ml-kernels` | Tensors, a safetensors loader and the transformer kernels written in the compute EDSL, over `eacp-gpu` wherever it builds — see "Tensors and kernels" below. Its `Tensor` and `DType` share `eacp::ML` with `eacp-ml-graph`'s, so no target links both. Links `eacp-gpu` |
| `eacp-ml` | The runner, Apple-only behind `EACP_HAS_COREML` (`EACP_HAS_GPU` and Apple; a PUBLIC define on the target): compiles a package through a cache, loads it on a chosen set of compute units, predicts synchronously or asynchronously, and hands tensors across to `GPU::Buffer`. Links `eacp-ml-graph`, `eacp-gpu`, Core ML, CoreVideo and Accelerate |

`ML.h` includes the graph and writers everywhere and the runner only where
`EACP_HAS_COREML` is 1, so a caller reaching `ML::Model` on a platform without
it fails to compile rather than to link. Everything is in `eacp::ML`.

## Building a graph

`Graph` records ops as it goes. A failing op returns an invalid `Tensor` and
records why: `isValid()` and `errors()` say so, and `build()` of a graph with
errors is an empty `Package`.

```cpp
#include <eacp/ML/ML.h>

using namespace eacp;

ML::Package projectionAndSoftmax(int rows, const Vector<float>& weights)
{
    auto graph = ML::Graph {};
    auto x = graph.input("x", {rows, 384}, ML::DType::float16);
    auto weight = graph.halfConstant("weight", {384, 384}, weights);
    auto zeros = Vector<float> {};
    zeros.resize(384, 0.f);
    auto bias = graph.halfConstant("bias", {384}, zeros);

    graph.output(graph.softmax(graph.linear(x, weight, bias), -1), "y");

    return graph.build();
}
```

An input is a name, a `Shape` and a `DType`; the four-argument `input` takes
a default shape and a list of enumerated shapes, which is how a decoder step
takes a growing context without a recompile. A `constant` is bytes of any
type, `halfConstant` packs floats to fp16 (`MIL/Half.h` is the conversion),
and `scalar` is one float. Constants that reach an output go into the weight
blob rather than the program text.

The ops are what a transformer needs: `linear` with or without a bias,
`matmul` with either operand transposed, `transpose`, `reshape`, `softmax`,
`sum`, `max` and `argmax` over an axis, `layerNorm` over a set of axes,
`conv`, `gather`,
`concat`, `slice` and `sliceLike`, `gelu`, `cast`, and
`scaledDotProductAttention` either causal or with a run-time mask. The mask
is a float tensor that is lowered to a bool one (`allowed > 0.5`), because
Core ML's fp16 attention ignores a float mask past 32 positions. Anything
elementwise that has no op of its own is `apply`: one, two or three operand
tensors and a body written over `GPU::Float`, exactly as a shader would write
it. The arithmetic, comparisons, `select`, `clamp`, `abs`, `min`/`max`,
`sqrt`/`rsqrt`, `exp`/`log`/`pow`, `tanh`, `erf` and `floor` lower; a call
with no MIL lowering, `sin` say, fails the op with a message naming it.

```cpp
auto body = [](const GPU::Float& value, const GPU::Float& factor)
{ return value * factor + 0.5f; };

auto y = graph.apply(x, scale, body);
```

`toText()` prints the MIL program for a diff against coremltools' output,
`specification()` is the typed form the writers consume, and `build()` is a
`Package` — the model bytes and the weight blob — which `Package::write`
lays out as an `.mlpackage` directory on disk for anything else to open.

## Running it

```cpp
auto model = ML::Model {};
auto options = ML::Options {};
options.units = ML::ComputeUnits::cpuAndNeuralEngine;

if (auto result = model.load(package, options); !result)
    LOG(result.error);

auto input = ML::MultiArray::create({rows, 384}, ML::DType::float16);
auto output = ML::MultiArray::create({rows, 384}, ML::DType::float16);
input.fromFloats(values);

auto inputs = ML::Inputs {};
inputs["x"] = input;
auto outputs = ML::Outputs {};
outputs["y"] = output;

auto result = model.predict(inputs, outputs);
```

`load` takes a `Package`, or a path to an `.mlpackage` (compiled through the
cache) or an `.mlmodelc` (loaded where it lies). `ComputeUnits` is `all`,
`cpuAndNeuralEngine`, `cpuAndGPU` or `cpu`; `units()`, `inputs()` and
`outputs()` report what loaded, and `computePlan()` is Core ML's own account
of where each op was placed and what it costs, which `Apps/ML/Projection`
prints per op. `isSupported()`, `hasComputePlan()` and `hasNeuralEngine()`
answer before anything is loaded.

Placement is Core ML's and nothing here falls back for it: `cpuAndNeuralEngine`
on a machine with no engine is the CPU, and `all` is the GPU wherever there is
one, engine or not. One placement is refused rather than allowed. Before macOS
27 the first prediction of a model with enumerated input shapes placed on the
CPU traps in BNNS, a SIGTRAP nothing catches, so where the CPU is certain -
`isCpuOnly(units)`: `cpu`, or `cpuAndNeuralEngine` with no engine - and
`enumeratedShapesRunOnTheCpu()` is false, `load` fails with a message instead
of loading; a caller that wants Core ML there chooses units that reach the GPU,
and one that wants the CPU builds the model fixed at one shape, which does not
trap.

An output array passed in `Outputs` is bound, so Core ML writes straight into
memory the caller keeps; an output left out is allocated and returned. A
`MultiArray` is a handle: copying one shares its storage. An fp16 array is an
IOSurface-backed pixel buffer, the one form the Neural Engine takes without
converting, zeroed on creation and row-padded to the surface's alignment
(`rowStride()`); fp32 and int32 arrays are plain memory. `toFloats` and
`fromFloats` are the host copies, `copyTo`/`copyFrom` move a tensor to or from
a `GPU::Buffer` packed or at an offset and row stride, converting fp16 to and
from fp32 on the way, and `copyRows` writes one step's rows into a fixed cache
tensor — a key/value cache between decoder steps.

`loadAsync` and `predictAsync` return a `Threads::Async`; jobs run in call
order on the model's own serial queue and resolve on the main thread, so both
are called from there. A `Prediction` carries the outputs plus
`queueWaitSeconds` and `predictSeconds`, taken on the queue. The blocking
forms run on the caller's thread, pump no event loop, and one prediction runs
at a time per model. Destroying the model abandons the Asyncs it handed out.

## The compile cache

Compiling a package is the slow step — a first load of the Whisper encoder
onto the Neural Engine after its key changed took 13.5 s, a hit afterwards
tens of milliseconds — and Core ML keys its own engine cache on the compiled
model's directory. So a compiled model is kept at
`<cacheDirectory>/<hash>.mlmodelc`, where the hash covers the program bytes,
the weights' identity and the OS build, and a hit is never recompiled.
`Options::weightsName` and `weightsVersion` name the weights so a large blob
is not hashed on every run; empty hashes the blob. A fresh compile moves into
place by atomic rename, so two processes racing to the same key both end up
loading one copy; a hit that fails to load twice is treated as damaged, moved
aside and compiled again. `wasCacheHit()` and `compiledPath()` say what
happened.

`defaultCacheDirectory()` is `CoreML` at the company level of the per-user
cache root — `FilePath::appCacheDirectory(company, "")`, the parent of every
app folder of that vendor — so all of one vendor's binaries share their
compiles with nothing named; the entries are keyed by content, so sharing is
safe. When the embedded AppInfo names no company there is no level to share
under, and the default is `CoreML` in the app's own cache folder.
`defaultCacheDirectory(company, app)` is the same mapping from names, and
`Options::cacheDirectory` overrides both.

A shared directory would only grow, so it is kept to a budget. Every use of a
cached model touches a `<hash>.used` stamp beside it — a stamp rather than the
model directory's own time, so the directory Core ML keys its engine cache on
is never written to after it is installed — and a loaded model touches it
again every ten minutes it predicts. After a miss has installed its compile,
models are evicted least recently used first until the directory is within
`Options::cacheBudgetBytes`: 2 GiB by default (a product's compiled models
run from tens to hundreds of megabytes, so that holds its working set and the
versions an update has just replaced, while bounding what a long-lived shared
directory can reach), and 0 never evicts. A model a live `Model` of this
process holds is never evicted, nor one whose stamp is under an hour old, so
another process that has just taken a hit keeps its directory. Eviction
renames the model to a `.trash` name with `RENAME_EXCL` before removing it,
the path the temporaries take, so two processes evicting at once cannot both
remove one model and a crash leaves only a `.trash` the next miss sweeps.
Eviction never fails a load: what goes wrong there is logged.

## Design record

This is the record of the EDSL-CoreML plan, kept here now that the plan is
finished; the phase-by-phase narrative is in git history under `plan.md`
(last at eacp `f97a5efe`). The net that proved it is
[WhisperEACP](https://github.com/eyalamirmusic/WhisperEACP): its encoder was
the first thing to run both ways, its tests say the two ways agree, and its
benchmark says what the second way was worth.

### What this is not

It is not a fourth backend of `GPU::ComputePass`. A `ComputeProgram` body is
one thread's view (`threadId()`, `shared<>`, `barrier()`, atomics, a loop
with a data-dependent exit); Core ML has no thread and runs a fixed graph of
whole-tensor ops compiled ahead of time on whichever of the CPU, GPU and
Neural Engine it decides. The kernels stay what they are and serve every
platform; the shared code is the net above them. It is not Core AI either,
which only consumes `.aimodel` bundles the Python packages produce, from
Swift; Core ML's ML Program is a public protobuf an ObjC++ process can write
and compile, which is why it is the target.

### Decisions

- **Precision.** The engine is fp16 end to end, so a Core ML net never
  matches the fp32 kernel path bit for bit. Tests compare at a measured
  tolerance per compute-unit setting, and the consumer's transcript oracle is
  the acceptance test.
- **Shapes.** The engine wants static shapes. A net over a variable context
  declares it as enumerated shapes, one compiled member each; a context
  outside the set cannot be served by a larger member and trimmed, since
  attention sees every position it is given.
- **Layout.** The engine wants channels on the second axis; getting into and
  out of that is the Core ML program's own business, so what crosses the seam
  is a straight copy of frame-major rows.
- **Weights.** Every tiny.en value is exactly representable in fp16, so the
  blob is fp16 with no weight error at half the size, and `linear` and `conv`
  take the safetensors layouts as they stand. A model is built and compiled
  once per machine per model version, and the cache key says so.
- **Threads.** Predictions run on the model's own serial queue;
  `predictAsync()` resolves on the message thread through `Threads::callAsync`
  and the blocking `predict()` runs on the caller's thread under the same
  mutex. A `Model` may be destroyed on any thread; what it handed out is
  abandoned on the main thread.
- **Placement is asserted, not assumed.** A test that wants the engine reads
  the plan; a build that cannot get it fails under `EACP_REQUIRE_ANE=1` and
  skips otherwise. Core ML places by size and op mix: a lone layer norm,
  attention or elementwise program stays on the CPU under every setting, a
  448-row `linear` and `softmax` too, and only the 1500-row one went to the
  engine.
- **Masks are bool.** Core ML's fp16 attention ignores an additive float
  mask from 32 positions up on every device, so the causal and the
  `allowed`-tensor forms of `scaledDotProductAttention` both lower to a bool
  `attn_mask` through `greater`.
- **The decoder stays on the kernels.** A Core ML decode step was built at
  tiny.en's real sizes and measured under every setting (the table below);
  none fits in the Metal step's time before its hop and its copies, and a
  resident-cache bound says `MLState` would not change that. `argmax`, the
  masked attention and `MultiArray::copyRows` came out of that measurement
  and stay, for a consumer that wants a Core ML decoder for an idle GPU
  rather than for speed.

### What was measured

All on an M5 Max, macOS 27.0, unless stated. Small programs, fp16 in and out
against an fp32 reference, max abs error and the device Core ML chose:

| program | CPU | CPU and GPU | CPU and engine, and all |
| --- | --- | --- | --- |
| linear + softmax, `[1500, 384]` | 1.4e-3, CPU | 1.9e-4, GPU | 3.1e-4, engine |
| linear, then layer norm | 3.7e-2, CPU | 5.0e-3, GPU | 9.4e-3, engine |
| elementwise, layer norm alone, attention `[1, 128, 64]` | CPU | CPU | CPU |

The seeded tiny.en encoder (`WhisperEncoder.h`, 18 enumerated contexts; the
`MLEncoder` suite that ran it is in git history), max abs at 1500 / 448 / 576
rows and the median prediction:

| setting | placed | max abs | 1500 rows | 448 rows |
| --- | --- | --- | --- | --- |
| CPU | CPU | 4.9e-2 / 3.2e-2 / 3.5e-2 | 18.6–19.0 ms | 4.5–4.6 ms |
| CPU and GPU | GPU | 6.4e-3 / 5.8e-3 / 6.5e-3 | 3.1–5.1 ms | 1.6–2.2 ms |
| CPU and engine | engine | 2.2e-2 / 2.3e-2 / 2.3e-2 | 11.3–11.4 ms | 1.4–1.5 ms |
| all | GPU | as CPU and GPU | 2.4–6.2 ms | 1.8–2.9 ms |

With tiny.en's own weights the engine's max abs at the full window is 0.26,
ten times the seeded number, and the gap is there before the first layer, so
it is fp16 arithmetic throughout rather than layer norm drifting; the
maximum is a handful of elements (99.9th percentile 0.056, mean 0.0076), so
the consumer's tests hold the tail and the mean as well as the maximum. In
WhisperEACP's Release benchmark over jfk.wav the encoder on the engine
predicted the full window in 11.3 ms against the Metal kernels' 5.0 ms whole
encode, and under `all`, where Core ML puts every op on the GPU, in 1.5 ms:
Apple's GPU lowering of the same program is about three times faster than
the kernels, the engine is slower at every context and scales more steeply
with it, and what the engine buys is an idle GPU and main thread, not wall
time. The engine compile of the enumerated model is 13.5 s once per cache
directory (a fixed program about a second), a warm load 13–165 ms, the seam
copies 0.13–0.18 ms and the mel's own command buffer 0.36–0.41 ms at every
context. The engine is shared with the system: under load a live stream's
engine runs went from 26.6 ms to 61–106 ms while the kernel column moved by
1–2 ms.

The tiny.en decode step (`WhisperDecoderStep.h` and the `MLDecoderStep`
suite, both in git history; 196 ops, the KV cache as inputs, 12 MB a
prediction), median of 25 predictions after five warm-ups,
against a Metal step of 332–409 µs all in:

| setting | placed | max abs, logits | predict, prefix 1 / 447 | caches resident |
| --- | --- | --- | --- | --- |
| CPU | CPU | 7.9e-2 | 0.63–0.68 / 0.63–0.66 ms | 0.63–0.67 ms |
| CPU and GPU | GPU | 8.5e-3 | 2.57–2.63 / 1.85–2.69 ms | 1.10–1.59 ms |
| CPU and engine | engine, `gather`s on the CPU | 2.8e-2 | 0.76–1.18 / 0.76–1.15 ms | 0.76–0.77 ms |
| all | GPU | as CPU and GPU | 1.48–1.94 / 1.13–1.29 ms | 1.05–1.07 ms |

The hop from `predictAsync()` to its resolve is within the noise of the
prediction once `EventLoop::runFor` on macOS stopped waking a frame late
(`EventLoop/waitFor/returnsOnResolveNotAFrameLater`).

### What Core ML does that the code guards against

- A '.' in a MIL identifier and a damaged `model.mil` in a compiled model
  each crash the process inside `makeProgramWithMemoryLayout`; names are made
  identifiers and a damaged hit is retried once, then discarded and
  recompiled.
- `log` and `rsqrt` need an explicit `epsilon`, though documented optional.
- An output that folds away (an input cut to its own shape) fails the load
  with execution-plan error -6, so `sliceLike(x, x)` is `x`.
- `MLComputePlan` is the program's, not a member's, and reading it under an
  engine setting costs the engine compile again the first time; a probe per
  context is a program fixed at that context.
- Before macOS 27 an enumerated model placed on the CPU traps in BNNS; see
  the refusal under **Running it**.
- GitHub's macOS runners have no Neural Engine and place every op on the CPU
  under every setting, so placement is asserted only on a developer's Mac
  and `build.yml` must not set `EACP_REQUIRE_ANE`.

### Open

- `MLState` (MIL `read_state` and `coreml_update_state`) is not built: the
  resident-cache bound above says it buys nothing for a step whose caches
  cross as inputs.
- A graph with an enumerated input and a fixed one beside it falls to the CPU
  (E5RT refuses the fixed input's strides), Core ML allows one enumerated
  input, `Graph::slice` refuses a range on an axis an enumerated input left
  unknown, and no `gather` is placed on the engine: WhisperEACP's plan lists
  these as what a fused mel-and-encoder model would need.
- EA's `Span` refuses a temporary container outright, where `std::span`
  admits one for a const element type, so `fromFloats(Vector<float> {...})`
  names the vector first; the refinement belongs in cpp_data_structures.

## Tensors and kernels

Tensors, a safetensors loader and the kernels a transformer is built from —
linear layers, norms, activations, RoPE, attention — written in the compute
EDSL of `eacp-gpu` (see `Lib/eacp/GPU/README.md`, "Compute"). Each kernel is a
`GPU::ComputeProgram` with its uniforms public, and each has a free function
beside it (`linear`, `applyRoPE`, `attention`, …) that allocates the result,
binds the kernel from the device's cache and records the dispatch into the pass
it is handed. Nothing commits: a whole layer, or a whole stack of them, goes
into one command buffer.

### Binding a tensor

A `Tensor` is a shape and a dtype over bytes of a GPU buffer, and those bytes
need not start at the buffer's beginning: `byteOffset()` says where they do,
and several tensors can share one buffer (see "Weights from a safetensors
file"). A kernel binds a tensor by its range, so assigning one to a buffer
uniform binds exactly its bytes wherever they lie:

```cpp
kernel.input = input;      // input.range(): {buffer, byteOffset, byteCount}
kernel.output = result;
```

There is no `buffer()` to bind instead, because binding the whole buffer is
the one mistake an offset makes easy. A `TensorView`'s `range()` is the range
of the tensor it views, and its `rowStride` and `columnOffset` count from there.

### Weights from a safetensors file

`SafetensorsFile::open` maps the file; `loadF32` is the one call per tensor:

```cpp
auto file = SafetensorsFile::open(path);
auto weight = file->loadF32("layers.0.attn.to_qkv.weight");
```

On a device that can adopt host memory (Metal), every F32 tensor is a range of
a GPU buffer over the mapping. `open` cuts the file into segments of
neighbouring tensors of up to 256 MB, and a segment becomes a buffer on the
first load from it. Nothing is copied, each buffer asks for residency in the
background (see "A buffer over memory you already have" in
`Lib/eacp/GPU/README.md`), and the buffers hold the mapping, so the tensors
outlive the `SafetensorsFile` they came from. Segments nobody loads from are
never wired: Stable Audio 3 medium's codec encoder, the last 1.7 GB of its
9.2 GB checkpoint, is never touched. And a command buffer that reads one small
tensor waits for its own segment's residency rather than the whole file's, so
the prompt encoder, which reads the conditioner's padding embedding, no longer
waits half a second behind the DiT. A tensor whose offset is
off `Device::storageBufferOffsetAlignment()` is copied into a buffer of its own,
as is every tensor on a device that cannot adopt memory, and F16 and BF16 are
converted to F32 on the host - the caller writes the same line in every case.
`loadCounts()` says how many of each there were. `readF32` is the host copy, in
F32 whatever the storage, for weights a loader transforms before upload.

### Tensor ops

`Kernels/TensorOps.h` is the arithmetic and plumbing between the layers, one
function each, so a model's own files hold only what is its own:

```cpp
auto x1 = add(pass, x, attentionOut);
auto step = scaleAndAdd(pass, x, 1.f, velocity, -t);
auto seq = concatRows(pass, {memoryTokens, x});
auto tail = sliceRows(pass, seq, memoryTokens.rows(), latentRows);
auto flat = reshape(std::move(attentionOut), {rows, heads * headDim});
```

`add`, `subtract`, `multiply` and `scaleAndAdd` are element by element;
`zeros`, `fill`, `sliceRows`, `sliceColumns`, `concatRows`, `copyRowsInto` and
`padRowsWithZeros` move rows and columns about; `reshape` hands the same buffer
back under another shape without a dispatch.

### Norms per head

`rmsNorm`, `layerNorm` and `dynamicTanh` normalise each row. The QK norm a
transformer applies before attention normalises each *head* instead — every
`headDim` values of a `rows x (heads * headDim)` query or key, with one
`headDim`-long gamma — and `rmsNormPerHead` and `dynamicTanhPerHead` are that,
with the same kernels and the input's shape back:

```cpp
auto q = rmsNormPerHead(pass, query, qNormGamma, headDim, epsilon);
```

### Column views

A fused projection puts q, k and v side by side in one `rows x 3·dim` tensor.
`qkv.columns(first, count)` is a `TensorView` of some of those columns, read
where they lie: `rmsNormPerHead`, `dynamicTanhPerHead` and the value of
`attention`, `bandedAttention` and `attendWithScores` all take one, so
splitting a projection costs no dispatch at all.

```cpp
auto qkv = linear(pass, x, qkvWeight);
auto q = rmsNormPerHead(pass, qkv.columns(0, dim), qNorm, headDim, eps);
auto k = rmsNormPerHead(pass, qkv.columns(dim, dim), kNorm, headDim, eps);
auto out = attention(pass, q, k, qkv.columns(2 * dim, dim), heads, headDim);
```

A `Tensor` converts to a view of the whole of itself, so callers holding a
tensor change nothing. The kernels only change where each value is read from;
every sum runs in the order it did over a copy, and gives the same bits.

### RoPE over segments

`applyRoPE` rotates each row by its position. Several independent sequences
stacked into one tensor — every chunk of a chunked decoder, a batch of prompts
of one length — want each row rotated by its position *within its sequence*,
and the overload that takes a segment length does exactly that:

```cpp
auto q = applyRoPE(pass, query, invFreq, heads, headDim, chunkRows);
```

Row `r` takes position `r % chunkRows`, so every chunk goes through in one
dispatch and comes out bit for bit as it would have alone. A segment length of
0 is the whole tensor, which is the overload without one. It pairs with
`bandedAttention`'s `AttentionBand::segmentRows`: the same number keeps the
chunks from seeing each other.

### Attention, and each probability computed once

`attention` is three kernels: the scores, the row stats, the weighted sum. The
row stats find each (row, head)'s peak, then turn its scores into
probabilities *in place* — `exp(score - peak)`, written back over the score it
came from — and sum them. The weighted sum reads those probabilities back. Each
`exp` is evaluated once, where it used to be evaluated again by every one of
the `headDim` threads that weigh a value by it: 65 per score before, 1 now.

It gives the same bits as recomputing: the one `exp` has the same input the
weighted sum's used to, and the weighted sum still adds columns in ascending
order. `bandedAttention` does the same over its window.

The softmax and weighted sum are also a function of their own, for a model
whose scores are not `attention`'s — a soft cap, a bias, another scale:

```cpp
auto scores = Tensor::uninitializedF32({rows, heads, cols});
// ... a score kernel of the model's own writes them ...
auto output = attendWithScores(pass, scores, value, heads, headDim);
```

`scores` comes back holding the unnormalised probabilities; the output is
`rows x heads x headDim`.

The row stats write the probability and then add it to the sum, and they are
written the way the C++ would be — `auto p = exp(scores[i] - peak);`, stored,
then added. That `p` is one value, evaluated once before the store, is the
EDSL's rule rather than this kernel's care: see "A handle is a value" in
`Lib/eacp/GPU/README.md`.

## Tests and examples

`MLGraphTests` (88 `MLGraph/` cases: shapes, text, protobuf, blob, package,
`apply`, and the encoder graph at Whisper's real sizes) runs on every lane but
iOS, and where `eacp-ml` exists every package it builds is also compiled and
loaded by Core ML on the CPU and GPU. `MLTests` (53 cases over `MLMultiArray`,
`MLAsync`, `MLCache`, `MLPlacement` and `MLPrograms`) needs Core ML;
`EACP_REQUIRE_ANE=1` makes it assert Neural Engine placement, which CI leaves
unset because its macOS runners have none. Every case runs in about half a
second or less: the Whisper-sized encoder and decode-step suites took 30 s and
more each, mostly in the Neural Engine's compile, and were removed rather than
shrunk, because a reduced encoder behaves differently on the engine (Core ML
refuses its output backings as dynamic). `Apps/ML/Projection` builds the
graph above, loads it under `--units all|cpu-ane|cpu-gpu|cpu`, prints the
compute plan and times the prediction; `Apps/ML/Spike` is the raw
Objective-C++ spike the module grew out of, kept as a reference for the file
formats.
