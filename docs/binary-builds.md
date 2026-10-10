# Cross-platform binary build

GitHub Actions builds desktop and Android libraries using the pinned Git
submodules. The resulting packages are available as workflow artifacts.

| Target | ABI | Minimum Android API | Build toolchain |
| --- | --- | --- | --- |
| Linux | x86_64 | — | Ubuntu GCC |
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

The build uses contemporary pinned SPIRV-Tools (C++17 required). If the
legacy NDK compiler cannot build it, its CI job must be treated as failed and
the dependency/compiler compatibility addressed; **do not silently upgrade
the minimum Android API**. Desktop jobs check the Vulkan proc exports,
while cross-compiled Android jobs inspect the ELF dynamic section.
