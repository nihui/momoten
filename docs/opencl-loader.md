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

The same `libmomoten.so` includes the Android Vulkan HAL `HMI` symbol,
the `open("vk0")` callback and Vulkan function pointers. There is no separate
`vulkan.momoten.so` adapter dependency.

For direct use by ncnn/simplevk, load `libmomoten.so` as before.
For Android 7.0+ system HAL deployment, a device-image integrator may rename
the **same binary** to `vulkan.<device-variant>.so` and install it in the
device's vendor HAL directory (typically `/vendor/lib[64]/hw/`). The
`<device-variant>` must match the device's HAL discovery configuration; it is
not necessarily `momoten`. Device SELinux and linker namespace configuration
may also need updating, and an unprivileged APK cannot replace a system HAL.

The Android 4.0 (API 14) binaries retain their legacy direct-load path. The
HAL symbol is present but does not make Android 4.0 support the Android 7+
system Vulkan HAL loader. This remains a restricted, non-conformant compute
implementation and should not be installed as the system default driver on
an ordinary device.
