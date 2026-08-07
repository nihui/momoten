# Restricted API compute profile

momoten is an experimental, non-conformant implementation of a restricted
compute subset of the Vulkan API. The current baseline covers device discovery,
buffers and memory binding, mapped host shadows, push descriptors and update
templates, push constants, compute pipelines, command recording and reuse,
buffer copies, dispatch, queue-ordering barriers, submission, fences and
pipeline-cache stubs.

`VK_KHR_driver_properties` reports conformance version `0.0.0.0`.
`deviceName`, `driverName`, and `driverInfo` explicitly identify momoten as
experimental and non-conformant.

Physical-device identity is derived from the selected OpenCL device rather than
from one implementation-wide placeholder. `vendorID` mirrors
`CL_DEVICE_VENDOR_ID`, `deviceID` is a stable hash of the OpenCL vendor, product
name and device type, and CPU/GPU devices are reported with the corresponding
Vulkan physical-device category. OpenCL 1.1 unified-memory information is used
when available to distinguish integrated and discrete GPUs; OpenCL 1.0 devices
remain supported and conservatively report a GPU without unified-memory
information as discrete.

The OpenCL device-wide maximum workgroup size is not a sufficient Vulkan limit because a translated kernel can have a lower `CL_KERNEL_WORK_GROUP_SIZE` after OpenCL resource allocation. The restricted compute profile therefore reports at most 256 `maxComputeWorkGroupInvocations`, allowing clients such as ncnn to choose a stable local size before pipeline compilation. Pipeline creation always queries the compiled kernel limit and local-memory use. If a shader has neither workgroup storage nor a control or memory barrier, momoten may preserve its logical Vulkan workgroup while splitting it into smaller, independent physical OpenCL workgroups. The generated kernel reconstructs the Vulkan workgroup and invocation built-ins, and the physical chunk size is chosen from the kernel-specific limit, the OpenCL X-dimension limit and the advertised profile limit. Chunk boundaries are subgroup-aligned whenever the subgroup size is greater than one. Shaders with cross-invocation workgroup state remain on the exact one-to-one path and are rejected when the compiled OpenCL kernel cannot support their requested local size.

The translator accepts Vulkan GLCompute SPIR-V and emits conservative OpenCL C
1.0 source. The supported shader subset includes:

- 32-bit scalar and vector arithmetic;
- structured control flow and compute builtins;
- storage buffers, push constants and specialization constants;
- common GLSL.std.450 operations;
- fixed 32-bit workgroup arrays and workgroup barriers;
- ncnn-style 32-bit-carrier packed fp16/int8 forms;
- conditional native half storage and arithmetic with `cl_khr_fp16`;
- conditional 64-bit integer scalar/vector storage and arithmetic, including fixed workgroup arrays, from the OpenCL profile, extension and feature declarations;
- conditional double scalar/vector storage and arithmetic with `cl_khr_fp64`;
- Vulkan 1.1 subgroup BASIC operations and built-ins through an exact size-one subgroup fallback on OpenCL C 1.0;
- optional BASIC emulation using either the width returned by the standard `cl_khr_subgroups` kernel query or a logical power-of-two width derived from `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE`; the latter is explicitly an emulation choice rather than a native subgroup contract;
- optional native subgroup BASIC lowering when `cl_khr_subgroups`, `cl_khr_subgroup_non_uniform_vote` and the kernel subgroup query are all available and the probed subgroup width satisfies the Vulkan contract;
- the float32/float16 matrix subset currently used by ncnn Vulkan shaders;
- optional global 32-bit compare-exchange with
  `cl_khr_global_int32_base_atomics`;
- optional `VK_KHR_shader_integer_dot_product` through
  `cl_khr_integer_dot_product`, including packed and unpacked 4x8-bit native
  forms and exact OpenCL C 1.0 software lowering for 32-bit vector forms.

Creation entry points validate their Vulkan structure type, supported `pNext`
chain, flags, layers, queue parameters and object-specific fields. Custom
allocation callbacks and create-info extensions outside this restricted profile
are rejected explicitly instead of being ignored.

When `cl_khr_fp16` is present, the implementation reports only the relevant
`storageBuffer16BitAccess` and `shaderFloat16` Vulkan features. It does not
report uniform, push-constant or I/O 16-bit storage.

The core `shaderInt64` feature is reported for OpenCL `FULL_PROFILE` devices, for `EMBEDDED_PROFILE` devices reporting `cles_khr_int64`, or when the OpenCL 3.0 `CL_DEVICE_OPENCL_C_FEATURES` list contains `__opencl_c_int64`. Querying the OpenCL 3.0 property is opportunistic and does not raise the OpenCL 1.0 host API baseline; older drivers simply reject the property query. Device discovery does not invoke the online compiler for int64 detection. The translator accepts SPIR-V `Int64`, maps signed and unsigned scalar/vector values to OpenCL `long` and `ulong`, and supports fixed int64/uint64 workgroup arrays and representable storage-buffer layouts. This path supplies ncnn's packed-half local-memory representation when fp16 storage and arithmetic are available but fp16 uniforms are not. It does not imply 64-bit atomic support.

When `cl_khr_fp64` is present and `CL_DEVICE_DOUBLE_FP_CONFIG` is nonzero, the implementation reports the core `shaderFloat64` feature. The translator accepts SPIR-V `Float64` scalar, vec2, vec3 and vec4 arithmetic in its restricted compute profile; matrix operations and fp64 atomics remain outside the profile. Double storage-buffer values are read and written through layout-identical `ulong` bit patterns with `as_double` and `as_ulong`, avoiding typed global-pointer aliasing while preserving the Vulkan buffer representation. OpenCL C 3.0 implementations with `__opencl_c_fp64` use the core feature directly, while older language modes retain a guarded `cl_khr_fp64` pragma. fp64 programs currently use the standard `-cl-opt-disable` build option because some online compilers are unstable when optimizing long double-precision arithmetic chains; this is a conservative functional baseline rather than an acceleration claim.

`VK_KHR_shader_integer_dot_product` is exposed only after the selected OpenCL
device reports packed 4x8-bit support and successfully compiles every required
signed, unsigned, mixed-signedness and accumulating-saturating builtin. The
Vulkan 8-bit and packed 4x8-bit acceleration properties mirror the two OpenCL
acceleration-property queries; they are never inferred merely from extension
support. Vulkan 16-, 32- and 64-bit acceleration properties remain false. The
translator still accepts 32-bit vector dot-product instructions and implements
their Vulkan result and saturation semantics in OpenCL C 1.0, but does not
advertise that software path as accelerated.

This distinction matters to ncnn: its packed int8 shaders select the integer
dot-product branch only when
`integerDotProduct4x8BitPackedSignedAccelerated` is true. A software OpenCL
implementation may therefore expose the feature while ncnn deliberately keeps
using its existing fallback branch.

Only `VK_SUBGROUP_FEATURE_BASIC_BIT` is reported. Arithmetic, ballot, shuffle,
shuffle-relative, clustered and quad subgroup operations remain unreported and
their SPIR-V instructions are rejected until exact lowerings exist. Images,
samplers, native 8-bit/integer-16-bit storage, device-scope barriers, general
atomics, graphics pipelines and general matrix operations are likewise outside
the current profile and must be rejected rather than silently approximated.

The emulated BASIC profile derives `SubgroupLocalInvocationId`, `SubgroupId`, `NumSubgroups` and election from the linear local invocation ID. A subgroup control barrier is lowered to a core OpenCL workgroup barrier only when the complete workgroup is one logical subgroup. A shader that combines an emulated subgroup control barrier with multiple logical subgroups per workgroup is rejected during translation because OpenCL C 1.0 cannot express that barrier exactly. Subgroup-scoped memory barriers remain per-invocation memory fences.

Image-format queries return `VK_ERROR_FORMAT_NOT_SUPPORTED`. Current ncnn uses
that result for `GpuInfo::support_image_storage()` and automatically selects
its buffer-only Vulkan path.

The OpenCL host API and baseline source target OpenCL 1.0. Validation on real
OpenCL 1.0 hardware and FP16-capable physical devices remains an acceptance
item; newer PoCL runtimes are useful development targets but do not close it.
