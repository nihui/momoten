# Cross-platform binary build

GitHub Actions builds desktop and Android libraries using the pinned Git
submodules. The resulting packages are available as workflow artifacts.

| Target | ABI | Minimum Android API | Build toolchain |
| --- | --- | --- | --- |
| Linux | x86_64 | — | cibuildwheel manylinux2014 (glibc 2.17) |
| Windows | x86_64 | — | MSVC 2022 |
| Android 4.0 | armeabi-v7a | 14 | NDK r17c |
| Android | arm64-v8a | 21 | GitHub hosted NDK |
| Android | x86_64 | 21 | GitHub hosted NDK |

Android builds use `ANDROID_STL=c++_static` and install an **unversioned**
`libmomoten.so`, including its SONAME. The required vendor OpenCL library
must be provided by the device; momoten opens it dynamically rather than
linking an OpenCL ICD loader.

The Android API 14 build provides a **build compatibility target**, not a
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

The Linux x86_64 shared library is built inside cibuildwheel's pinned
`manylinux2014` container, following qwenimage-ncnn-vulkan's release
approach. It targets glibc 2.17 (e.g. CentOS 7 or newer glibc-based
distributions) instead of the Ubuntu runner's glibc.

This is a native `libmomoten.so` distribution, **not a Python wheel**. A
temporary setuptools placeholder only provides cibuildwheel with a build
entry point; the real library is built with CMake in `CIBW_BEFORE_ALL`
and installed to the host-mounted `stage/` directory.

The Linux release links `libstdc++` and `libgcc` statically, to avoid
newer libstdc++ runtime version requirements. CI verifies that
`vkGetInstanceProcAddr` is still exported, that there are no dynamic
dependencies on a system OpenCL ICD loader or C++ runtime, and that all
recorded `GLIBC_*` versions are at most 2.17. It also strips the binary
and checks that no `.debug_*` sections remain.

The system still needs a compatible vendor OpenCL implementation at
runtime. This compatibility check does not claim functional compatibility
with every old GPU/driver combination.
