# Cross-platform binary build

GitHub Actions builds desktop and Android libraries using the pinned Git
submodules. The resulting packages are available as workflow artifacts. Manual
`workflow_dispatch` runs also publish the verified platform archives as GitHub
Release assets, using a date version like `20261020`.

| Target | ABI | Minimum Android API | Build toolchain |
| --- | --- | --- | --- |
| Linux | x86_64 | — | Docker manylinux2014 (glibc 2.17) |
| Linux | x86 (i686) | — | Docker manylinux2014 (glibc 2.17) |
| macOS | x86_64 | — | Xcode 16.4, macos-15-intel (deployment target 10.13) |
| macOS | arm64 | — | Xcode 16.4, macos-15 (deployment target 11.0) |
| Windows | x86_64 | — | MSVC 2022 (x64) |
| Windows | x86 | — | MSVC 2022 (Win32) |
| Android 4.0 | armeabi-v7a | 14 | NDK r17c |
| Android 4.0 | x86 | 14 | NDK r17c |
| Android | arm64-v8a | 21 | GitHub hosted NDK |
| Android | x86_64 | 21 | GitHub hosted NDK |

All platforms install a single unversioned runtime library without
development headers, CMake package exports or versioned .so symlinks.
Android builds use `ANDROID_STL=c++_static` and install `libmomoten.so`. The required vendor OpenCL library
must be provided by the device; momoten opens it dynamically rather than
linking an OpenCL ICD loader.

The Android API 14 builds provide a **build compatibility target**, not a
promise of device compatibility. Older Android devices may lack the required
vendor OpenCL API, online compiler, or extensions. Runtime testing still
requires appropriate physical hardware. In particular, successful cross
compilation does not validate the OpenCL driver on Android 4.0.

For CMake, reproduce the legacy configuration using the r17c toolchain and:

```sh
cmake -S . -B build-android-14 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r17c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=armeabi-v7a -DANDROID_PLATFORM=android-14 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
  -DMOMOTEN_BUILD_TESTS=OFF -DMOMOTEN_BUILD_TOOLS=OFF
cmake --build build-android-14 --target momoten_driver
```

The build uses pinned SPIRV-Tools (C++17 required). NDK r17c's libc++ lacks
`<charconv>`. The legacy Android and Intel macOS jobs apply
`cmake/patches/spirv-tools-charconv.patch`, replacing the optimizer's two
`std::from_chars` calls with a decimal-only parser of identical accepted
syntax and uint32 overflow behavior. Android r17c additionally applies
`cmake/patches/spirv-tools-ndk-r17c.patch` to give three constexpr capability
arrays explicit template parameters, avoiding missing C++17 array deduction
guides. The upstream submodule revision and other platform builds are unchanged. **Do not silently upgrade the minimum
Android API**. Desktop jobs check the Vulkan proc exports,
while cross-compiled Android jobs inspect the ELF dynamic section.

## Smaller Android release binaries

Android CI removes the NDK toolchain's hardcoded `-g` flag before the
CMake configure step, following ncnn's Android release workflow. Both
`android.toolchain.cmake` (NDK r17c) and
`android-legacy.toolchain.cmake` (newer NDKs) are handled when present.
Compile commands are checked for residual `-g` flags.

The installed `libmomoten.so` is also stripped using
`cmake --install build --strip --prefix stage`, without stripping
the dynamic Vulkan entry-point symbols. CI rejects debug sections and
prints the final uncompressed library size. Debug builds outside this
release workflow are unaffected.

## Linux compatibility

The Linux x86_64 and i686 shared libraries are built **directly** in
pinned `quay.io/pypa/manylinux2014_x86_64` and
`quay.io/pypa/manylinux2014_i686` Docker containers (glibc 2.17,
GCC 10). The image digests are fixed in the workflow and originate from
a passing cibuildwheel run. No Python package, dummy `setup.py`, or
cibuildwheel wrapper is needed.

The repository is bind-mounted into the container at `/project`. CMake
compiles and installs directly to `/project/stage`, which is accessible
on the host without any intermediate copy or `/host/tmp` mount. Docker
uses the runner's UID and GID to avoid root-owned output files. The
`i686` build selects Docker platform `linux/386`. Setting
`CMAKE_INSTALL_LIBDIR=lib` makes both architectures install under
`stage/lib/` rather than using `lib64/` on x86_64.

Both builds aim to run on glibc 2.17 and later (for example, RHEL/CentOS 7).
The Linux release links `libstdc++` and `libgcc` statically to avoid
depending on a newer system C++ runtime. CI verifies the ELF class,
exports `vkGetInstanceProcAddr`, rejects dynamic OpenCL ICD/C++ runtime
dependencies, and checks that required `GLIBC_*` versions do not exceed
2.17. It also strips debug symbols and checks that no `.debug_*`
sections remain.

A compatible vendor OpenCL implementation is still required at runtime;
these checks do not prove functionality on every old GPU or driver.

## CI matrix and cancellation

The workflow produces distinct artifacts for Linux x86_64/x86,
macOS x86_64/arm64, Windows x86_64/x86, and Android armeabi-v7a/x86 (API 14) plus
arm64-v8a/x86_64 (API 21). The Android and Linux shared libraries
are checked for the expected ELF class. The Windows generator explicitly
selects MSVC x64 or Win32.

Workflow-level concurrency cancels superseded CI runs for earlier commits on
the same branch or pull request. Manual release runs use a separate concurrency
group and are not cancelled; jobs within the current run still use
`fail-fast: false`.

## Runtime-only distribution

Linux CI packages `libmomoten.so`, `momoten_icd.json`, and `LICENSE.txt`/
`THIRD_PARTY_NOTICES.md` in one `momoten-<version>-linux-<arch>.tar.gz`.
Windows and Android similarly ship only the runtime library and notices.
No installed CMake SDK, version header, Windows import library, or
versioned `.so` symlinks are required.

The build-tree targets `momoten::driver`, `momoten::translator`, and
`momoten::headers` remain available to embedded/superbuild consumers.
The generated version header remains build-tree metadata and supplies the
driver release version. Linux keeps
`-static-libstdc++ -static-libgcc` for compatibility with older runtimes;
changing that choice can be measured in a separate experiment.

## Experimental macOS builds

Native Intel and Apple Silicon jobs run on `macos-15-intel` and `macos-15`,
respectively, with Xcode 16.4. Intel builds target macOS 10.13 (High Sierra);
Apple Silicon builds target macOS 11.0.
They produce separate `momoten-<version>-macos-x86_64.tar.gz` and
`momoten-<version>-macos-arm64.tar.gz` archives containing:

- `libmomoten.dylib`
- `momoten_icd.json`
- `LICENSE.txt` and `THIRD_PARTY_NOTICES.md`

These are **build compatibility baselines**, not verified old-OS runtime
support. CI executes on macOS 15, not 10.13 or 11, and cannot establish
compatibility with old Mac GPUs or their OpenCL drivers. Apple lists 10.13 as
the lower end of the deployment range for the pinned Xcode 16.4 in its
[toolchain table](https://developer.apple.com/xcode/system-requirements).
That is the conservative baseline used here, not proof that every earlier
macOS version is technically impossible. Earlier targets require a separate
legacy toolchain, SDK, dependency, and physical-runtime investigation.

The dylib uses the system libc++ and libSystem; neither runtime is bundled.
Apple's [C++ availability table](https://developer.apple.com/xcode/cpp/) puts
integral `std::from_chars` at macOS 10.15. The Intel build therefore applies
the same narrow decimal-parser patch used for Android r17c, without the
Android-only deduction-guide changes. The optimizer flag regression covers
zero, uint32 limits, overflow, signs, whitespace, and malformed separators.
The pinned SPIRV-Tools still requires C++17; changing a deployment flag alone
would not address its library API requirements.

All compiled sources, including static SPIRV dependencies and both driver
variants, receive the selected deployment target and hard availability errors.
CI checks their compile commands, native Mach-O architecture, exact minimum-OS
load command, `@rpath` install name, Vulkan/ICD exports, and exact link dependencies
(`/usr/lib/libc++.1.dylib` and `/usr/lib/libSystem.B.dylib`). Build-machine or
Homebrew libraries, OpenCL/Vulkan/MoltenVK links, embedded rpaths, and weak dylib
loads are rejected. The full undefined-symbol inventory is printed for both
drivers; an Intel guard rejects known newer libc++/libSystem API families.
This guard is not an exhaustive old-OS symbol database or an old-OS execution
test. CI also extracts the archive into a different directory and loads it.
These archives are not Developer ID signed or notarized.

To reproduce the Intel build with the pinned Xcode, initialize submodules,
apply `cmake/patches/spirv-tools-charconv.patch` from `third_party/spirv-tools`,
and configure with `CMAKE_OSX_ARCHITECTURES=x86_64` and
`CMAKE_OSX_DEPLOYMENT_TARGET=10.13`, as shown in the workflow. Do not bypass
availability diagnostics to force an older target.

The backend dynamically opens Apple's
`/System/Library/Frameworks/OpenCL.framework/OpenCL`; Apple OpenCL is
deprecated and compatible devices/online compilers may be unavailable.
No MoltenVK or Metal backend is involved. Applications may load the dylib
and its Vulkan entry points directly. Applications using the Khronos Vulkan
Loader need that loader installed separately and must opt in with an absolute
`VK_DRIVER_FILES=/path/to/momoten_icd.json`. Keep the JSON next to the dylib.
The manifest is not installed globally. momoten remains an experimental,
non-conformant compute-only Vulkan implementation.

Both modular and generated amalgamated drivers are compiled and tested;
only the modular runtime is packaged, matching the other platforms.
Homebrew `glslang` and `vulkan-loader` are test tools, not shipped dependencies.
The optional direct-OpenCL tests link to the SDK's framework stub; the momoten
dylib itself still loads OpenCL dynamically.

Hosted macOS VMs may not expose an OpenCL device, particularly on Apple
Silicon. Device-independent translation/ABI/load checks always run.
Hardware-dependent CTest cases are explicitly skipped when no compatible
device exists, and their names/counts appear in the job summary. A separate
Apple OpenCL probe guards the system Vulkan Loader smoke test: no compatible
device is a reported skip. The hosted ARM64 Apple framework can instead return
`CL_INVALID_VALUE` for valid device-enumeration calls. Only on that hosted
architecture and system framework, CI reports this as an unavailable backend
if an independent native SDK probe reproduces the error with both count-only
and non-null device-buffer calls. This distinct skip does not claim hardware
is absent. Other API errors, library-loading errors, and ICD failures remain
failures. Passing such a build is not evidence of a successful GPU dispatch.

The hosted Intel Apple CPU backend executes the fp64, int16, packed-atomic,
and virtual-workgroup direct OpenCL tests, but currently cannot execute the
four-lane barrier kernels used by the simple-buffer and subgroup tests. A
separate native SDK reproducer also exposes a by-value struct-argument ABI
mismatch. CI requires a correct four-lane control dispatch and both exact
failure signatures before excluding five affected runtime tests; their names
and reasons appear as **not run**, separately from passing and skipped tests.
The remaining tests and modular/amalgamated Vulkan Loader smoke tests still
run. See [testing](testing.md) for the precise exclusion conditions. These
artifacts are experimental build/release deliverables, not a claim of complete
Apple OpenCL shader, push-constant, or barrier compatibility.

## Date-versioned releases

Like [pnnx](https://github.com/pnnx/pnnx/blob/main/.github/workflows/release.yml),
releases are explicitly requested with `workflow_dispatch`, not by pushing a tag
or on a schedule. Choose **Run workflow** for `build` and select the branch to release. A manual run publishes after all platform builds and package
validation succeed; it is not a build-only dry run. Pushes to `main` and pull
requests run the same build and packaging checks without publishing or creating
tags.

The `setup` job generates one UTC date version in `YYYYMMDD` format for the
entire run. Every platform uses that same value, including jobs that finish
on the following day. Packages are named `momoten-<version>-<target>.tar.gz`
(Linux/macOS) or `momoten-<version>-<target>.zip` (Windows/Android), for example
`momoten-20261020-linux-x86_64.tar.gz`.

The `package-release` job checks all ten platform artifacts and their
runtime-only archive contents, then creates `SHA256SUMS`. The aggregate is also
available as the `momoten-<version>-release` workflow artifact, so packaging is
tested on every CI run. Each platform build also receives the same date as
`-DMOMOTEN_RELEASE_VERSION=<version>`. `VkPhysicalDeviceProperties::driverVersion`
encodes it as `VK_MAKE_VERSION(year - 2000, month, day)`, so `20261020` displays
as `26.10.20` with the standard Vulkan version helpers. `driverName` is `momoten`
and `driverInfo` uses the concise format `momoten 20261020; OpenCL backend: ...`.
`deviceName` keeps the underlying OpenCL device name with a `momoten: ` prefix.
Both the modular and generated single-file driver use the same configured version.

For local builds, `MOMOTEN_RELEASE_VERSION` defaults to the UTC configure date
and is cached for subsequent reconfiguration. Set it explicitly to reproduce a
release. CMake rejects malformed or impossible dates and years outside 2000-2099;
this range keeps all packed Vulkan fields in bounds. The date is a driver release
identifier, not the Vulkan API or ABI version: the CMake project version remains
`0.1.0`, the ICD API version remains `1.1.0`, and runtime filenames stay unversioned.

Only the final publishing job receives `contents: write`. It creates the date
tag (without a `v` prefix) at the exact commit selected when the workflow was
started and publishes the ten archives plus `SHA256SUMS` under
`Release <version>`. A pre-existing date tag for a different commit is rejected.
An existing release for that date is not overwritten: `gh release create` fails
rather than silently replacing its assets. There is at most one release per UTC
day. Manual release runs are not cancelled by newer CI runs.
