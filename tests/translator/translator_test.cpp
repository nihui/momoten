// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

static bool read_spirv(const char* path, std::vector<uint32_t>& words)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return false;
    const std::streamoff size = stream.tellg();
    if (size < 0 || size % 4 != 0)
        return false;
    words.resize(static_cast<size_t>(size) / 4);
    stream.seekg(0, std::ios::beg);
    return words.empty() || static_cast<bool>(stream.read(reinterpret_cast<char*>(words.data()), size));
}

static int require(bool condition, const char* message)
{
    if (!condition)
    {
        fprintf(stderr, "translator_test: %s\n", message);
        return 1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 9)
        return require(false, "expected positive, unsupported, atomic, fp16, int16, subgroup, integer-dot and workgroup-split SPIR-V input paths");

    std::vector<uint32_t> words;
    if (!read_spirv(argv[1], words))
        return require(false, "failed to read test SPIR-V");

    momoten::TranslationOptions options;
    momoten::SpecializationValue specialization;
    specialization.constant_id = 0;
    specialization.data.resize(4);
    const uint32_t element_count = 8;
    memcpy(specialization.data.data(), &element_count, 4);
    options.specializations.push_back(specialization);

    momoten::TranslationResult result;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, result))
    {
        fprintf(stderr, "translator_test: translation failed: %s\n", result.diagnostics.c_str());
        return 1;
    }

    if (require(result.abi.local_size[0] == 4 && result.abi.local_size[1] == 1 && result.abi.local_size[2] == 1,
                "local size reflection failed")
        || require(!result.abi.workgroup_splittable, "workgroup-storage kernel was marked splittable")
        || require(result.abi.buffers.size() == 2, "expected two storage buffers") || require(result.abi.buffers[0].binding == 0 && result.abi.buffers[1].binding == 1, "storage buffers are not ordered by binding") || require(result.abi.buffers[0].buffer_arg_index == 0 && result.abi.buffers[0].offset_arg_index == 1 && result.abi.buffers[0].size_arg_index == 2 && result.abi.buffers[1].buffer_arg_index == 3 && result.abi.buffers[1].offset_arg_index == 4 && result.abi.buffers[1].size_arg_index == 5, "storage-buffer pointer/offset/size ABI is incorrect") || require(result.abi.push_constant_arg_index == 6 && result.abi.push_constant_size == 4, "push-constant ABI is incorrect") || require(result.abi.entry_point == "momo_main", "translated kernel name is incorrect") || require(result.source.find("#define round(x) rint(x)") != std::string::npos, "GLSL nearest-even round compatibility macro was not emitted") || require(result.source.find("#define SPIRV_CROSS_UNROLL\n") != std::string::npos && result.source.find("#define SPIRV_CROSS_LOOP\n") != std::string::npos, "portable SPIR-V loop-control hint macros were not emitted") || require(result.source.find("SPIRV_CROSS_UNROLL\n        for") != std::string::npos && result.source.find("SPIRV_CROSS_LOOP\n        for") != std::string::npos, "SPIR-V loop-control hints were not preserved in the fixture") || require(result.source.find("__local float scratch[4]") != std::string::npos, "workgroup storage was not localized into the OpenCL kernel") || require(result.source.find("momo_workgroup_barrier(CLK_LOCAL_MEM_FENCE)") != std::string::npos, "workgroup barrier was not lowered to OpenCL C 1.0") || require(result.source.find("any((valid) != int4(0))") != std::string::npos && result.source.find("all((valid) != int4(0))") != std::string::npos, "boolean-vector any/all was not canonicalized for OpenCL sign-bit semantics") || require(result.source.find("__kernel __attribute__((reqd_work_group_size(4, 1, 1))) void momo_main") != std::string::npos, "OpenCL kernel declaration is missing") || require(result.source.find("momo_buffer_offset_0") != std::string::npos, "buffer offset argument is missing") || require(result.source.find("momo_buffer_size_0") != std::string::npos, "buffer byte-size argument is missing") || require(result.source.find("(__global uchar *)&(") == std::string::npos, "scalar storage load was wrapped in an unsupported robust-access guard") || require(result.source.find("volatile int momo_active = 1") != std::string::npos && result.source.find("if (momo_active != 0)") != std::string::npos, "workgroup return convergence and storage-store predication are missing") || require(result.source.find("apply_scale") == std::string::npos, "SPIR-V helper function was not inlined into kernel scope") || require(result.source.find("float load_scale(") != std::string::npos && result.source.find("__global const uchar* momo_push_constants") != std::string::npos, "retained helper did not receive the kernel resource ABI") || require(result.source.find("inout ") == std::string::npos, "GLSL inout qualifier leaked into OpenCL source") || require(result.source.find("layout(") == std::string::npos, "Vulkan GLSL layout syntax leaked into OpenCL source"))
        return 1;

    momoten::TranslationResult repeated;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, repeated) || require(repeated.source == result.source, "translation is not deterministic"))
        return 1;

    std::vector<uint32_t> invalid = words;
    invalid[0] = 0;
    momoten::TranslationResult rejected;
    if (require(!momoten::translate_spirv_to_opencl_c(
                    invalid.data(), invalid.size(), options, rejected),
                "invalid SPIR-V was accepted")
        || require(!rejected.diagnostics.empty(), "invalid SPIR-V did not produce a diagnostic"))
        return 1;

    std::vector<uint32_t> unsupported;
    if (!read_spirv(argv[2], unsupported))
        return require(false, "failed to read unsupported test SPIR-V");
    momoten::TranslationResult unsupported_result;
    if (require(!momoten::translate_spirv_to_opencl_c(
                    unsupported.data(), unsupported.size(), options, unsupported_result),
                "unsupported storage-image SPIR-V was accepted")
        || require(unsupported_result.diagnostics.find("storage buffers") != std::string::npos,
                   "storage-image rejection was not actionable"))
        return 1;

    std::vector<uint32_t> atomic;
    if (!read_spirv(argv[3], atomic))
        return require(false, "failed to read atomic test SPIR-V");
    momoten::TranslationOptions atomic_options;
    momoten::TranslationResult atomic_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    atomic.data(), atomic.size(), atomic_options, atomic_result),
                "supported global int32/uint32 compare-exchange was rejected")
        || require(atomic_result.abi.workgroup_splittable,
                   "independent-work-item kernel was not marked splittable")
        || require(atomic_result.source.find("reqd_work_group_size") == std::string::npos
                       && atomic_result.source.find("momo_virtual_local_invocation_index") != std::string::npos,
                   "splittable kernel did not receive the virtual-workgroup ABI")
        || require(atomic_result.abi.required_extensions.size() == 1 && atomic_result.abi.required_extensions[0] == "cl_khr_global_int32_base_atomics",
                   "atomic extension requirement was not reflected")
        || require(atomic_result.source.find(
                       "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable")
                       != std::string::npos,
                   "atomic extension pragma was not emitted")
        || require(atomic_result.source.find("momo_atomic_cmpxchg_uint(") != std::string::npos,
                   "unsigned atomic compare exchange was not emitted")
        || require(atomic_result.source.find("momo_atomic_cmpxchg_int(") != std::string::npos,
                   "signed atomic compare exchange was not emitted")
        || require(atomic_result.source.find("momo_mat4") != std::string::npos && atomic_result.source.find("dot(") != std::string::npos,
                   "std430 mat4/vector multiplication was not lowered"))
        return 1;

    std::vector<uint32_t> fp16;
    if (!read_spirv(argv[4], fp16))
        return require(false, "failed to read fp16 test SPIR-V");
    momoten::TranslationResult fp16_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    fp16.data(), fp16.size(), options, fp16_result),
                "fp16 storage/arithmetic SPIR-V was rejected")
        || require(fp16_result.abi.required_extensions.size() == 1 && fp16_result.abi.required_extensions[0] == "cl_khr_fp16",
                   "fp16 extension requirement was not reflected")
        || require(fp16_result.source.find("#pragma OPENCL EXTENSION cl_khr_fp16 : enable") != std::string::npos,
                   "fp16 extension pragma was not emitted")
        || require(fp16_result.abi.buffers.size() == 3,
                   "fp16 fixture storage-buffer reflection is incomplete")
        || require(fp16_result.source.find("__global half4 *") != std::string::npos,
                   "float16 storage buffer was not lowered to half4")
        || require(fp16_result.source.find("__global momo_hmat4 *") != std::string::npos,
                   "float16 mat4 storage buffer was not lowered")
        || require(fp16_result.source.find("__local half4 scratch[4]") != std::string::npos,
                   "float16 workgroup storage was not lowered")
        || require(fp16_result.source.find("typedef struct { half4 c[4]; } momo_hmat4") != std::string::npos,
                   "float16 mat4 representation was not emitted")
        || require(fp16_result.source.find("half(0.5f)") != std::string::npos && fp16_result.source.find("half(0.5)") == std::string::npos,
                   "float16 constants retained an accidental fp64 dependency"))
        return 1;

    std::vector<uint32_t> int16;
    if (!read_spirv(argv[5], int16))
        return require(false, "failed to read int16 test SPIR-V");
    momoten::TranslationResult int16_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    int16.data(), int16.size(), options, int16_result),
                "core OpenCL C int16 SPIR-V was rejected")
        || require(int16_result.abi.required_extensions.empty(),
                   "core OpenCL C int16 unexpectedly required an extension")
        || require(int16_result.source.find("__global ushort4 *") != std::string::npos,
                   "uint16 storage buffer was not lowered to ushort4")
        || require(int16_result.source.find("as_uint(") != std::string::npos && int16_result.source.find("as_ushort2(") != std::string::npos,
                   "equal-size scalar/vector bitcasts were not lowered with OpenCL as_type builtins"))
        return 1;

    std::vector<uint32_t> subgroup;
    if (!read_spirv(argv[6], subgroup))
        return require(false, "failed to read subgroup BASIC test SPIR-V");
    momoten::TranslationOptions singleton_options;
    momoten::TranslationResult singleton_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    subgroup.data(), subgroup.size(), singleton_options, singleton_result),
                "subgroup BASIC SPIR-V was rejected by the singleton path")
        || require(singleton_result.abi.subgroup_mode == momoten::SubgroupModeSingleton && singleton_result.abi.subgroup_size == 1,
                   "singleton subgroup ABI was not reflected")
        || require(singleton_result.abi.required_extensions.empty(),
                   "singleton subgroup unexpectedly requires an OpenCL extension")
        || require(singleton_result.source.find("#pragma OPENCL EXTENSION cl_khr_subgroups") == std::string::npos,
                   "singleton subgroup enabled a native OpenCL extension")
        || require(singleton_result.source.find("gl_LocalInvocationIndex") != std::string::npos && singleton_result.source.find("10000u") != std::string::npos && singleton_result.source.find("mem_fence(CLK_GLOBAL_MEM_FENCE)") != std::string::npos && singleton_result.source.find("sub_group_barrier(") == std::string::npos,
                   "singleton subgroup built-ins or election were not lowered"))
        return 1;

    momoten::TranslationOptions emulated_options;
    emulated_options.subgroup_mode = momoten::SubgroupModeEmulatedBasic;
    emulated_options.subgroup_size = 8;
    momoten::TranslationResult emulated_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    subgroup.data(), subgroup.size(), emulated_options,
                    emulated_result),
                "subgroup BASIC SPIR-V was rejected by the emulated path")
        || require(emulated_result.abi.subgroup_mode == momoten::SubgroupModeEmulatedBasic && emulated_result.abi.subgroup_size == 8,
                   "emulated subgroup ABI was not reflected")
        || require(emulated_result.abi.required_extensions.empty(),
                   "emulated subgroup unexpectedly requires an OpenCL extension")
        || require(emulated_result.source.find("#pragma OPENCL EXTENSION cl_khr_subgroups") == std::string::npos,
                   "emulated subgroup enabled a native OpenCL extension")
        || require(emulated_result.source.find("% 8u") != std::string::npos && emulated_result.source.find("/ 8u") != std::string::npos && emulated_result.source.find("== 0u") != std::string::npos && emulated_result.source.find("barrier(") != std::string::npos && emulated_result.source.find("sub_group_barrier(") == std::string::npos,
                   "emulated subgroup BASIC operations were not lowered"))
        return 1;

    momoten::TranslationOptions split_emulated_options;
    split_emulated_options.subgroup_mode = momoten::SubgroupModeEmulatedBasic;
    split_emulated_options.subgroup_size = 2;
    momoten::TranslationResult split_emulated_result;
    if (require(!momoten::translate_spirv_to_opencl_c(
                    subgroup.data(), subgroup.size(), split_emulated_options,
                    split_emulated_result),
                "a multi-subgroup emulated control barrier was accepted")
        || require(split_emulated_result.diagnostics.find("one logical subgroup per workgroup") != std::string::npos,
                   "emulated control-barrier rejection was not diagnostic"))
        return 1;

    momoten::TranslationOptions native_options;
    native_options.subgroup_mode = momoten::SubgroupModeNative;
    native_options.subgroup_size = 8;
    momoten::TranslationResult native_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    subgroup.data(), subgroup.size(), native_options, native_result),
                "subgroup BASIC SPIR-V was rejected by the native path")
        || require(native_result.abi.subgroup_mode == momoten::SubgroupModeNative && native_result.abi.subgroup_size == 8,
                   "native subgroup ABI was not reflected")
        || require(native_result.abi.required_extensions.size() == 2 && native_result.abi.required_extensions[0] == "cl_khr_subgroups" && native_result.abi.required_extensions[1] == "cl_khr_subgroup_non_uniform_vote",
                   "native subgroup extension requirements are incomplete")
        || require(native_result.source.find(
                       "#pragma OPENCL EXTENSION cl_khr_subgroups : enable")
                           != std::string::npos
                       && native_result.source.find(
                              "#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable")
                              != std::string::npos,
                   "native subgroup extension pragmas were not emitted")
        || require(native_result.source.find("get_sub_group_local_id()") != std::string::npos && native_result.source.find("get_sub_group_id()") != std::string::npos && native_result.source.find("get_num_sub_groups()") != std::string::npos && native_result.source.find("sub_group_elect()") != std::string::npos && native_result.source.find("sub_group_barrier(") != std::string::npos,
                   "native subgroup BASIC operations were not lowered"))
        return 1;

    std::vector<uint32_t> integer_dot;
    if (!read_spirv(argv[7], integer_dot))
        return require(false, "failed to read integer dot-product test SPIR-V");
    momoten::TranslationResult integer_dot_rejected;
    if (require(!momoten::translate_spirv_to_opencl_c(
                    integer_dot.data(), integer_dot.size(), options,
                    integer_dot_rejected),
                "integer dot-product SPIR-V was accepted without a probed device profile")
        || require(integer_dot_rejected.diagnostics.find("packed 4x8-bit") != std::string::npos,
                   "integer dot-product rejection did not identify the missing profile"))
        return 1;

    momoten::TranslationOptions integer_dot_options;
    integer_dot_options.integer_dot_product = true;
    integer_dot_options.integer_dot_product_input_4x8bit = true;
    integer_dot_options.integer_dot_product_input_4x8bit_packed = true;
    momoten::TranslationResult integer_dot_result;
    const bool integer_dot_translated = momoten::translate_spirv_to_opencl_c(
        integer_dot.data(), integer_dot.size(), integer_dot_options,
        integer_dot_result);
    if (!integer_dot_translated)
        fprintf(stderr, "translator_test: integer dot translation failed: %s\n",
                integer_dot_result.diagnostics.c_str());
    if (require(integer_dot_translated,
                "integer dot-product SPIR-V was rejected by the native profile")
        || require(integer_dot_result.abi.integer_dot_product,
                   "integer dot-product ABI requirement was not reflected")
        || require(integer_dot_result.abi.required_extensions.size() == 1 && integer_dot_result.abi.required_extensions[0] == "cl_khr_integer_dot_product",
                   "integer dot-product OpenCL extension requirement is incomplete")
        || require(integer_dot_result.source.find(
                       "#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable")
                       != std::string::npos,
                   "integer dot-product extension pragma was not emitted")
        || require(integer_dot_result.source.find("dot_4x8packed_ss_int(") != std::string::npos && integer_dot_result.source.find("dot_4x8packed_uu_uint(") != std::string::npos && integer_dot_result.source.find("dot_4x8packed_su_int(") != std::string::npos && integer_dot_result.source.find("dot_acc_sat_4x8packed_ss_int(") != std::string::npos && integer_dot_result.source.find("dot_acc_sat_4x8packed_uu_uint(") != std::string::npos && integer_dot_result.source.find("dot_acc_sat_4x8packed_su_int(") != std::string::npos,
                   "integer dot-product instructions were not lowered to OpenCL builtins")
        || require(integer_dot_result.source.find("spirv_instruction") == std::string::npos,
                   "SPIRV-Cross GLSL integer-dot polyfill leaked into OpenCL source"))
        return 1;

    std::vector<uint32_t> workgroup_split;
    if (!read_spirv(argv[8], workgroup_split))
        return require(false, "failed to read workgroup-split test SPIR-V");
    momoten::TranslationOptions split_options;
    split_options.subgroup_mode = momoten::SubgroupModeEmulatedBasic;
    split_options.subgroup_size = 8;
    momoten::TranslationResult split_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    workgroup_split.data(), workgroup_split.size(), split_options,
                    split_result),
                "workgroup-split SPIR-V was rejected")
        || require(split_result.abi.workgroup_splittable,
                   "independent 3D workgroup was not marked splittable")
        || require(split_result.source.find("reqd_work_group_size") == std::string::npos
                       && split_result.source.find("momo_virtual_local_invocation_id") != std::string::npos,
                   "3D virtual-workgroup source ABI is incomplete"))
        return 1;

    momoten::TranslationOptions native_split_options;
    native_split_options.subgroup_mode = momoten::SubgroupModeNative;
    native_split_options.subgroup_size = 8;
    momoten::TranslationResult native_split_result;
    if (require(momoten::translate_spirv_to_opencl_c(
                    workgroup_split.data(), workgroup_split.size(), native_split_options,
                    native_split_result),
                "native-subgroup workgroup-split SPIR-V was rejected")
        || require(native_split_result.source.find("momo_workgroup_chunk * ((uint)get_local_size(0) / 8u)") != std::string::npos,
                   "native subgroup ID was not rebased across workgroup chunks")
        || require(native_split_result.source.find("48u") != std::string::npos,
                   "logical native subgroup count was not preserved"))
        return 1;

    return 0;
}
