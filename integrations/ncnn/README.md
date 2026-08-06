# ncnn integration

momoten is loaded through ncnn's existing simplevk path. The current ncnn
checkout does not define `NCNN_BUILD_MOMOTEN`; do not pass that option.

For sibling checkouts, create a small parent `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(ncnn_momoten_superbuild LANGUAGES C CXX)
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MOMOTEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(/path/to/ncnn ncnn)
add_subdirectory(/path/to/momoten momoten)
include(/path/to/momoten/integrations/ncnn/ncnn_momoten.cmake)
ncnn_momoten_register_ncnn_tests()
```

For an embedded checkout, replace the absolute paths with
`add_subdirectory(ncnn)`, `add_subdirectory(momoten)`, and
`include(momoten/integrations/ncnn/ncnn_momoten.cmake)`.

Configure and run only the adapter-owned tests:

```bash
cmake -S parent -B build -DNCNN_VULKAN=ON -DNCNN_BUILD_TESTS=ON
cmake --build build --target momoten_driver momoten_ncnn_model_test
ctest --test-dir build -L ncnn --output-on-failure
```

The minimal model test compares CPU and Vulkan execution of a tiny in-memory
network. Other adapter tests cover generated fp16 shaders and selected ncnn
layer regressions. Generic momoten tests never include ncnn headers.

Current ncnn detects buffer-only drivers through
`GpuInfo::support_image_storage()`. momoten returns
`VK_ERROR_FORMAT_NOT_SUPPORTED` from Vulkan image-format queries, so ncnn
automatically avoids image allocation, image descriptors and sampler setup.
No momoten-specific ncnn build option is required.

The repository's integration superbuild keeps the upstream ncnn suite off for
the default smoke build. Configure it with
`-DMOMOTEN_NCNN_BUILD_ALL_TESTS=ON` to register each current ncnn test again
with `NCNN_VULKAN_DRIVER` pointing at momoten and a 30-minute timeout. Run the
full suite serially on memory-constrained systems because several ncnn layer
tests create thousands of specialized OpenCL programs and concurrent JIT
compilation can otherwise trigger the host OOM killer:

```bash
ctest --test-dir build -L ncnn -j1 --timeout 1800 --output-on-failure
```

For ordinary applications, build ncnn and `momoten_driver`, then run with:

```bash
NCNN_VULKAN_DRIVER=/absolute/path/to/libmomoten.so ./your_ncnn_program
```

For Mesa Rusticl with the software llvmpipe device, enable the device and pin
the Rusticl vendor because momoten intentionally bypasses the system ICD
loader and another installed vendor may otherwise be selected first:

```bash
RUSTICL_ENABLE=llvmpipe \
MOMOTEN_OPENCL_DRIVER=/etc/OpenCL/vendors/rusticl.icd \
NCNN_VULKAN_DRIVER=/absolute/path/to/libmomoten.so \
    ./your_ncnn_program
```
