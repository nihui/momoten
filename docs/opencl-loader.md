# OpenCL driver loading

momoten does not link `libOpenCL`, `OpenCL.dll`, or an OpenCL ICD loader.
Instead it opens a vendor implementation and resolves either its ICD dispatch
table or directly exported OpenCL functions.

Selection starts with `MOMOTEN_OPENCL_DRIVER`; the legacy
`NCNN_MOMOTEN_OPENCL_DRIVER` and `NCNN_OPENCL_DRIVER` names are checked in
that order only when the generic variable is unset.

- Linux and desktop Unix parse `OCL_ICD_VENDORS` and common OpenCL vendor
  directories, then probe known vendor library names.
- Windows reads the Khronos OpenCL Vendors registry key for the matching
  32/64-bit view, then probes known vendor DLL names.
- Android checks vendor, ODM and system ICD locations and common Qualcomm,
  Mali and PowerVR library paths.
- macOS opens the system OpenCL framework directly.

The first vendor containing an available, little-endian 32- or 64-bit device
with an online compiler is selected. The vendor runtime remains an external
platform component; all source-level translation dependencies are linked into
`libmomoten`.

## Mesa Rusticl

`RUSTICL_ENABLE` enables Mesa devices, but it does not select Rusticl when
another installed vendor is discovered first by momoten's direct loader. Pin
the Rusticl ICD explicitly when testing llvmpipe on Linux:

```bash
RUSTICL_ENABLE=llvmpipe \
MOMOTEN_OPENCL_DRIVER=/etc/OpenCL/vendors/rusticl.icd \
NCNN_VULKAN_DRIVER=/absolute/path/to/libmomoten.so \
    ./your_ncnn_program
```

The ICD file names the vendor library; momoten still opens that library
directly and does not acquire a runtime dependency on the system OpenCL ICD
loader. Use `MOMOTEN_DEBUG=1` to verify that the selected platform and device
are Rusticl and llvmpipe.

## Vulkan Loader ICD manifest (desktop)

The Vulkan ICD JSON is separate from OpenCL's `.icd` text files. Desktop
builds enable `MOMOTEN_BUILD_ICD_MANIFEST=ON` by default and install
`momoten_icd.json` alongside `libmomoten.so` (or `momoten.dll`).
The JSON uses the relative library path, so move the pair together.
The Vulkan Loader discovers this ICD only when explicitly requested:

```sh
VK_DRIVER_FILES=/absolute/path/to/momoten_icd.json vulkaninfo --summary
```

To use a particular OpenCL implementation at the same time, set
`MOMOTEN_OPENCL_DRIVER` to that vendor library or its OpenCL `.icd` file.
Vulkan ICD manifests are **not** OpenCL vendor manifests.

The ICD entry points negotiate Loader/ICD interface version 5 and initialize
all dispatchable handles for loader dispatch. This is an experimental,
nonconformant, compute-only Vulkan implementation: never copy the manifest
into `/usr/share/vulkan/icd.d/` for automatic system discovery. Android
continues using direct dynamic loading; desktop JSON registration is not
the Android Vulkan HAL discovery mechanism.

## Android Vulkan HAL

Android 7.0 (API 24) and newer use a **hardware HAL module**, not the
desktop Vulkan ICD JSON manifest. Build the optional adapter with
`-DMOMOTEN_BUILD_ANDROID_HAL=ON`. Installation produces:

```text
lib/libmomoten.so
lib/hw/vulkan.momoten.so
```

The small `vulkan.momoten.so` adapter exports the Android `HMI` hardware
module symbol, supports `open("vk0")`, and provides the
`EnumerateInstanceExtensionProperties`, `CreateInstance`, and
`GetInstanceProcAddr` function pointers. It links to the same
`libmomoten.so` used by ncnn's direct Vulkan loading path.

AOSP chooses the hardware module name from device properties; the
`vulkan.momoten.so` suffix is selected only when the device's HAL discovery
configuration resolves to `momoten`. Simply placing a file in
`/vendor/lib[64]/hw/` on a stock device is not sufficient. System
integration may require a device image configuration, partition permissions,
SELinux policies, and correct linker namespaces; it is not achieved by an
unprivileged APK. These binaries implement a restricted, nonconformant
Vulkan subset and should not replace a conformant system Vulkan HAL by default.

Android 4.0 (API 14) builds intentionally remain **direct-load-only**:
they still provide just `libmomoten.so`, preserving the older Android
compatibility target. Android CI builds the optional HAL on its newer NDK
targets, but a successful cross-compile does not establish that a particular
device image will load it.
