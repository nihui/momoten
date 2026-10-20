# Testing

The device-independent `momoten_version_configuration` test checks release-date
validation, UTC defaults and cached reconfiguration. `momoten_driver_version`
queries the core and extended physical-device properties using a synthetic
OpenCL device, so driver-version/date checks run even without an OpenCL runtime.
With amalgamation enabled, `momoten_amalgamation_current` checks that the
single-file source embeds the current generated version header.

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
return 77 when no suitable OpenCL device is available. The CMake cache-test
wrapper preserves this as a distinct CTest skip marker; driver/API errors
remain test failures. The optional
direct-OpenCL test is enabled only when `MOMOTEN_TEST_OPENCL_LIBRARY` is
explicitly supplied and never changes the driver's link dependencies.

The fp16 fixture reproduces ncnn's packed-half workgroup representation with `uint64_t`, validates `ushort4`/`ulong` bitcasts, and exercises the width conversion required by OpenCL `select` for a half-vector result. The fp64 fixture checks capability reflection, guarded extension setup, scalar/vector lowering, `ulong` storage-buffer bitcasts and numerical execution through an actual OpenCL compiler. The integer dot-product fixture covers packed 4x8-bit signed, unsigned and mixed-signedness operations, their accumulating-saturating variants, and 32-bit vector modulo/saturating corner cases. The driver test also enables Vulkan features and executes translated SPIR-V through public Vulkan entry points. Device acceleration properties are tested independently from functional extension support.

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

On macOS, build CI exercises both native architectures and both
modular/amalgamated drivers. Device-independent tests always run. `vulkan_loader_smoke.py` accepts `--loader`
for the Khronos loader dylib and `--probe-opencl` for Apple's framework.
The optional probe returns 77 only when that backend reports no compatible
devices; ICD ABI/manifest/loading checks still run first. The hosted ARM64 job
also supplies `--apple-opencl-unavailable-probe`: only for the system Apple
framework, `CL_INVALID_VALUE` from device enumeration can produce the distinct
exit code 78 after a separately compiled SDK probe reproduces the same error
with both count-only and non-null device-buffer calls. This is reported as an
unavailable backend, not absent hardware or a passed runtime test. All other
API errors remain failures. Without these opt-ins, the smoke test requires a
working OpenCL backend.
The hosted Intel job also compiles `apple_barrier_probe.cpp` against the native
SDK, without momoten. A four-lane control kernel must execute and produce the
correct values. Only when a minimal barrier kernel independently reports a
one-lane maximum and rejects a four-lane dispatch, **and** a two-argument
struct/barrier kernel reports three arguments and rejects dispatch with
`CL_INVALID_KERNEL_ARGS`, CI excludes these five tests:

- `momoten_opencl_simple`: four-lane barrier and by-value push-constant struct
- `momoten_opencl_subgroup_basic`: four-lane subgroup barrier
- `momoten_driver_simple`, `momoten_program_cache`, and
  `momoten_amalgamated_driver_simple`: pipeline creation correctly rejects the
  barrier kernel's workgroup limit before reaching dispatch

The job summary lists them separately as **not run**, with reasons; they are
not passed tests. The probe requires a single Apple CPU device, checks exact
error signatures, and fails on unexpected errors or partially matching
limitations. If the capabilities work or another device topology is present,
the full suite runs. The exclusion applies only to that hosted Intel CI job;
normal CTest runs remain unchanged. All other tests, including actual supported
OpenCL dispatches and system Vulkan Loader checks, remain mandatory.

See [binary builds](binary-builds.md#experimental-macos-builds) for the
runtime-coverage limitations of hosted macOS VMs.
