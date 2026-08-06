# Testing

Tests are separated by ownership and runtime requirements:

- `tests/translator` validates translation, deterministic output, rejected
  features and strict OpenCL C 1.0 syntax without requiring a device.
- `tests/runtime` validates the Vulkan-facing driver, OpenCL vendor loading,
  program caching, integer dot-product translation/dispatch and absence of a
  system OpenCL loader dependency.
- `tests/api` validates the Vulkan 1.1 proc/profile surface, every manifest proc
  scope/version/extension rule, direct symbol exports, core/KHR alias pointer
  identity, and the `VkResult`/void/proc ABI exception guards. It also compiles
  the Vulkan frontend with `VK_USE_64_BIT_PTR_DEFINES=0` and checks source-layer
  dependency invariants.
- Optional amalgamation tests generate `dist/momoten.cpp` in the build tree,
  compile its translator-only and full-driver modes, repeat proc/profile tests,
  and execute a dispatch through the generated driver.
- `integrations/ncnn` registers ncnn model, layer regression and generated
  shader checks only when included by an ncnn parent build.

CTest labels include `translator`, `runtime`, `opencl-1.0`, `cache`, `api`,
`abi`, `proc`, `architecture`, and `integration;ncnn`. Runtime-dependent tests
return 77 when no suitable OpenCL device is available. The optional
direct-OpenCL test is enabled only when `MOMOTEN_TEST_OPENCL_LIBRARY` is
explicitly supplied and never changes the driver's link dependencies.

The fp64 fixture checks capability reflection, guarded extension setup, scalar/vector lowering, `ulong` storage-buffer bitcasts and numerical execution through an actual OpenCL compiler. The integer dot-product fixture covers packed 4x8-bit signed, unsigned and mixed-signedness operations, their accumulating-saturating variants, and 32-bit vector modulo/saturating corner cases. The driver test also enables the Vulkan feature and executes the translated SPIR-V through public Vulkan entry points. Device acceleration properties are tested independently from functional extension support.

Shader fixtures are registered through the `momoten_compile_test_shader`
CMake helper. Direct-OpenCL runtime modes are described by one test-case table,
which keeps device requirements and dispatch callbacks together. The workgroup-split fixture forces a logical `8x16x3` workgroup through a smaller physical OpenCL workgroup and validates reconstructed 3D invocation, workgroup and emulated BASIC subgroup built-ins.

Run the generic suite with:

```bash
ctest --test-dir build-momoten -LE ncnn --output-on-failure
```

For Rusticl llvmpipe, use the explicitly loaded vendor ICD so the test also
exercises momoten's own OpenCL loader:

```bash
LP_NUM_THREADS=8 \
RUSTICL_ENABLE=llvmpipe \
RUSTICL_DEBUG=no_variants \
MOMOTEN_OPENCL_DRIVER=/etc/OpenCL/vendors/rusticl.icd \
ctest --test-dir build-momoten --output-on-failure
```

Run ncnn integration tests serially with the required per-test timeout:

```bash
ctest --test-dir build-ncnn-momoten -L ncnn -j1 --timeout 1800 --output-on-failure
```

The adapter assigns every ncnn test a 1800-second timeout. The ncnn tree owns
its generated test targets and shader registry; momoten discovers those
targets instead of maintaining a second inventory.

Large layer matrices can compile thousands of specializations. The in-memory
OpenCL program cache retains at most 8 entries by default; set
`MOMOTEN_MEMORY_CACHE_ENTRIES=0` to measure the vendor runtime without that
cache, or raise it only when the selected driver has a low per-program memory
cost. This setting does not disable `MOMOTEN_CACHE` binary files.
