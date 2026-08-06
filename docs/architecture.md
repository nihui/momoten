# Architecture

momoten has one public runtime boundary: the Vulkan API exported by the shared
library. A consumer such as ncnn's simplevk loader opens that library and calls
`vkGetInstanceProcAddr`; no consumer-specific callback is required.

The canonical implementation is split by dependency direction:

1. `src/entrypoints.cpp` owns the exported C ABI and proc availability policy.
   `src/vulkan_procs.inc` is the only list of supported Vulkan entry points; it
   generates direct exported wrappers, internal `impl_*` declarations, and proc
   metadata. Every wrapper is `noexcept` and calls a templated
   `VulkanAbiGuard` before entering C++ code. Core/KHR aliases use the same
   guarded `impl_*` callable.
2. The object-oriented `src/*.cpp` frontend modules own ordinary C++
   `momoten_detail::impl_*` implementations, split by Vulkan object
   responsibility. They do not export C symbols and do not call one exported
   wrapper from another.
3. `src/translator` owns SPIR-V validation, specialization and OpenCL C
   emission. `spirv_prepare.cpp` normalizes SPIR-V,
   `compiler_opencl.cpp` orchestrates translation, and `opencl_abi.cpp`
   reflects Vulkan resources into kernel arguments. Generic OpenCL lowering
   remains in `opencl_codegen.cpp`; integer dot product, subgroup and matrix
   lowerings have independent implementation units.
4. `src/backend` owns vendor discovery, the explicit `OpenCLApi` function
   table, device discovery, program cache and command replay. The neutral
   `ShaderDeviceProfile` in `device_profile.h` is the single internal source
   for Vulkan feature/property reporting and translator target options.

The resulting dependency direction is:

```text
exported Vulkan wrappers + proc registry
                    |
                    v
            impl_* frontend modules
               |                |
               v                v
        OpenCL backend       translator
               |
               v
      dynamically loaded vendor API
```

`momoten_translator` is a static library used by the offline tool and linked
into `momoten_driver`. Shared translator declarations live in
`src/translator/translator_internal.h`. `src/vulkan_internal.h` contains only
the lightweight handle utilities, common command-buffer state enum, and
generated `impl_*` declarations. Backend object definitions live in
`src/backend/objects.h`; device discovery in `device.h`; command replay and
submissions in `runtime.h`; program-cache keys and disk operations in
`cache.h`; and OpenCL program creation in `program.h`. All of these headers are
private. `vulkan_internal.h` never includes a backend header, so the include
graph does not rely on an include-guard cycle.

`src/source_manifest.txt` is the canonical ordered list of translator and
driver parts. CMake derives modular target sources from its `.cpp` entries and
`tools/amalgamate.py` consumes the same list. Its global top-to-bottom order is
the amalgamation order: all `translator` entries must be contiguous and precede
every `driver` entry. Both parsers reject unknown groups, missing paths, and a
translator entry after the driver boundary. The proc `.inc` is a manifest
dependency and is expanded at each include site rather than emitted as an
independent source chunk.

`tools/amalgamate.py` concatenates the public header and canonical modules into
`dist/momoten.cpp` under the build directory. The generated file is a release
artifact, not a tracked source file or source of truth. It retains
`MOMOTEN_TRANSLATOR_ONLY` so it can also build the offline translator without
OpenCL or Vulkan runtime headers.

SPIRV-Cross, SPIRV-Tools and their headers are build dependencies, linked
privately into `libmomoten`. OpenCL-Headers and Vulkan-Headers provide ABI
declarations only. A vendor OpenCL implementation is selected at runtime and
loaded directly; the system OpenCL ICD loader is not a dependency.

Consumer-specific code belongs under `integrations/`. It may register tests or
provide build examples, but it must not be included by the generic momoten
build. In particular, momoten does not maintain an ncnn shader inventory.

momoten and nikki intentionally use a similar navigation model—guarded ABI
wrappers, `impl_*` frontend modules, backend objects/profile, and an isolated
translator boundary—but do not share runtime objects, command synchronization,
kernel/bind-group ABIs, or translator implementations.
