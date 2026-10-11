# Cross-platform binary build

GitHub Actions builds desktop and Android libraries using the pinned Git
submodules. The resulting packages are available as workflow artifacts.

| Target | ABI | Minimum Android API | Build toolchain |
| --- | --- | --- | --- |
| Linux | x86_64 | — | Docker manylinux2014 (glibc 2.17) |
| Linux | x86 (i686) | — | Docker manylinux2014 (glibc 2.17) |
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
`<charconv>`. For the legacy job only, the workflow applies the narrow
`cmake/patches/spirv-tools-ndk-r17c.patch`, replacing the optimizer's two
`std::from_chars` calls with a decimal-only parser of identical accepted
syntax and uint32 overflow behavior. The patch also gives three constexpr capability arrays explicit template
parameters, avoiding C++17 array deduction guides missing from r17c's libc++.
The upstream submodule and all other platform builds are unmodified. **Do not silently upgrade the minimum
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
Windows x86_64/x86, and Android armeabi-v7a/x86 (API 14) plus
arm64-v8a/x86_64 (API 21). The Android and Linux shared libraries
are checked for the expected ELF class. The Windows generator explicitly
selects MSVC x64 or Win32.

Workflow-level concurrency with `cancel-in-progress: true` cancels
superseded runs for earlier commits on the same branch or pull request;
jobs within the current run still use `fail-fast: false`.

## Runtime-only distribution

Linux CI packages `libmomoten.so` and `LICENSE.txt`/
`THIRD_PARTY_NOTICES.md` in one `momoten-linux-<arch>.tar.gz`.
Windows and Android similarly ship only the runtime library and notices.
No installed CMake SDK, version header, Windows import library, or
versioned `.so` symlinks are required.

The build-tree targets `momoten::driver`, `momoten::translator`, and
`momoten::headers` remain available to embedded/superbuild consumers.
The generated version header remains build-tree metadata. Linux keeps
`-static-libstdc++ -static-libgcc` for compatibility with older runtimes;
changing that choice can be measured in a separate experiment.
