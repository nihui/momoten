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
