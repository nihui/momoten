# momoten

momoten is an experimental implementation of a restricted subset of the Vulkan API over OpenCL.

## Experimental and non-conformant implementation

momoten implements a restricted, buffer-oriented subset of the Vulkan API over OpenCL. It is not a complete Vulkan implementation, has not passed the Khronos Vulkan Conformance Test Suite, and is not a Khronos conformant product.

This project is intended for research, testing, and development only. It is not affiliated with, endorsed by, or certified by the Khronos Group.

Applications can submit compute commands and Vulkan Shader SPIR-V from this restricted API subset; `libmomoten` translates compute entry points to OpenCL C and replays the command stream on a vendor OpenCL device.

The implementation is independent of ncnn. ncnn is one consumer: its simplevk loader can open `libmomoten` without changing shader-management code.

## Repository layout

```text
include/momoten/       build-tree translator API and version metadata
src/translator/         SPIR-V optimization and OpenCL C translation
src/backend/            OpenCL loader, profiles, memory sync, replay and program build
src/*.cpp               internal impl_* Vulkan modules split by responsibility
src/vulkan_procs.inc    canonical direct-export and proc-registry declarations
src/entrypoints.cpp     guarded C ABI wrappers and proc availability policy
src/source_manifest.txt canonical modular/amalgamation source order
third_party/             pinned build dependencies (Git submodules)
tools/                  offline SPIR-V translation tool
momotenConfig.cmake.in installed CMake package configuration template
tests/shaders/          Vulkan GLSL fixtures
tests/translator/       device-independent translator tests
tests/runtime/          generic Vulkan/OpenCL runtime tests
tests/api/              ABI, proc registry, profile and architecture tests
integrations/ncnn/      optional ncnn adapter and integration tests
docs/                   architecture, profile, loader and testing contracts
```

The files under `src/` are the only development sources. `tools/amalgamate.py` can generate a copyable single-file distribution into the build tree. Both forms have no ncnn source dependency and still require Vulkan/OpenCL headers, SPIRV-Cross and SPIRV-Tools when compiled. Define `MOMOTEN_TRANSLATOR_ONLY` when compiling the generated file to omit the Vulkan ABI and OpenCL runtime portion. The legacy `NCNN_MOMOTEN_TRANSLATOR_ONLY` spelling remains accepted by the generated file.

The shared library keeps every implemented `vk*` entry point directly exportable. `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` expose the same implementation through the scope, core-version and enabled-extension policy declared in `src/vulkan_procs.inc`; direct calls and proc lookup are both supported ABI paths.

## Standalone build

The source build requires CMake 3.22.1 or newer.

```bash
git clone --recurse-submodules https://github.com/nihui/momoten.git
cmake -S momoten -B build-momoten
cmake --build build-momoten
ctest --test-dir build-momoten --output-on-failure
cmake --install build-momoten
```

For an existing checkout, initialize or update its dependencies once with:

```bash
git submodule update --init --recursive
```

Standalone builds compile only the modular implementation by default. Generate and compile the single-file form explicitly with:

```bash
cmake -S momoten -B build-momoten -DMOMOTEN_BUILD_AMALGAMATION=ON
cmake --build build-momoten --target momoten_amalgamation
```

The generated file is `build-momoten/dist/momoten.cpp`. It is not tracked by Git; release automation can publish it as a GitHub Release asset. The amalgamation step requests Python3 only when enabled; the vendored SPIRV-Tools build may independently require Python3 for generated tables.

SPIRV-Cross, SPIRV-Tools, SPIRV-Headers, OpenCL-Headers and Vulkan-Headers are pinned Git submodules under `third_party/`. CMake never downloads dependencies and does not use a system OpenCL or Vulkan SDK for their headers. Once the submodules are initialized, configuration and compilation work offline. A missing submodule produces an error with the initialization command.

The build-tree CMake targets are:

```text
momoten::driver
momoten::translator
momoten::headers
```

The installed package exports `momoten::driver` and `momoten::headers`; the installed headers target exposes version metadata only. `spv_to_clc.h` and `momoten::translator` are build-tree development interfaces and are deliberately not installed because the translator's private SPIRV dependencies are embedded rather than exposed as a package ABI. When `MOMOTEN_BUILD_TOOLS=ON`, the supported installed translation interface is the `momoten-spv-to-clc` executable.

When `MOMOTEN_BUILD_AMALGAMATION=ON`, the development-only `momoten_amalgamated_driver` and `momoten_amalgamated_translator` targets verify that the generated distribution remains buildable and behaviorally equivalent. The generated source is installed only in this mode.

The installed runtime consists of `libmomoten.so`, `momoten.dll` or the platform equivalent. SPIRV-Cross and SPIRV-Tools are linked into the shared library; it does not link the system OpenCL ICD loader. The selected device vendor's OpenCL runtime remains an external platform component and is opened directly by momoten. See [docs/opencl-loader.md](docs/opencl-loader.md) for Linux/Unix, Windows, Android and macOS discovery details.

## Run ncnn through simplevk

The current ncnn checkout has no `NCNN_BUILD_MOMOTEN` option. Build ncnn and momoten from a small parent project, or add momoten as an embedded subdirectory, then explicitly include the integration adapter. See [`integrations/ncnn/README.md`](integrations/ncnn/README.md) for both layouts.

After building ncnn with Vulkan and `momoten_driver`, select the shared library at runtime:

```bash
NCNN_VULKAN_DRIVER="$PWD/build-ncnn-momoten/momoten/libmomoten.so" \
    ./build-ncnn-momoten/benchmark/benchncnn
```

`NCNN_VULKAN_DRIVER` belongs to simplevk and is intentionally not renamed. Once loaded, ncnn routes calls through the restricted API implementation and does not call any momoten-specific entry point.

Select and configure momoten with:

```text
MOMOTEN_OPENCL_DRIVER  vendor OpenCL library or .icd file
MOMOTEN_CACHE          program-binary cache directory
MOMOTEN_MEMORY_CACHE_ENTRIES
                        maximum retained in-memory OpenCL programs (default 8,
                        0 disables the in-memory cache)
MOMOTEN_DEBUG          vendor, build and cache diagnostics
MOMOTEN_DUMP_DIR       failed pipeline SPIR-V/OpenCL/ABI dumps
```

For existing ncnn scripts, `NCNN_MOMOTEN_OPENCL_DRIVER`, `NCNN_OPENCL_DRIVER`, and the `NCNN_MOMOTEN_CACHE`, `NCNN_MOMOTEN_DEBUG`, and `NCNN_MOMOTEN_DUMP_DIR` variables remain fallback aliases.

## Tests

Generic tests do not include ncnn headers or enumerate ncnn shaders. Optional ncnn tests are registered by `integrations/ncnn/ncnn_momoten.cmake` only when the parent ncnn build requests them. All such tests load the same library through `NCNN_VULKAN_DRIVER` and have a 30-minute per-test timeout.

See [docs/testing.md](docs/testing.md) and the ncnn-specific [integration guide](integrations/ncnn/README.md).

## Supported profile

momoten implements a deliberately narrow, buffer-oriented compute subset of the API. It is not a complete Vulkan implementation. The OpenCL host API and generated kernel source use OpenCL 1.0 as their compatibility baseline. FP32 storage-buffer compute is the baseline; native FP16 storage and arithmetic are exposed only when the selected OpenCL device reports `cl_khr_fp16`. `shaderFloat64` is exposed when the device reports `cl_khr_fp64` together with a nonzero `CL_DEVICE_DOUBLE_FP_CONFIG`; the current restricted fp64 path covers scalar/vector arithmetic and storage buffers, represents storage through exact `ulong` bitcasts, and uses conservative OpenCL compilation. Integer dot product is exposed conditionally through `cl_khr_integer_dot_product`, with acceleration properties copied from the OpenCL device rather than assumed from extension presence. Unsupported Vulkan or SPIR-V features fail explicitly.

See [docs/vulkan-profile.md](docs/vulkan-profile.md) for the current boundary and hardware-validation status.

## Development note

GPT was used extensively during the development of this project, including for initial code generation, implementation experiments, and test development. The resulting code was manually reviewed, corrected, and validated against the full ncnn test suite.

## License

momoten is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
