// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"
#include "integer_dot_product.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace momoten {

OpenCLAddressSpace opencl_address_space(StorageClass storage)
{
    switch (storage)
    {
    case StorageClassFunction:
    case StorageClassPrivate:
    case StorageClassPushConstant:
        return OpenCLAddressSpace::Private;
    case StorageClassWorkgroup:
        return OpenCLAddressSpace::Local;
    case StorageClassUniform:
    case StorageClassStorageBuffer:
    case StorageClassCrossWorkgroup:
        return OpenCLAddressSpace::Global;
    default:
        return OpenCLAddressSpace::Unsupported;
    }
}

const char* opencl_address_space_qualifier(OpenCLAddressSpace address_space)
{
    switch (address_space)
    {
    case OpenCLAddressSpace::Private:
        return "__private";
    case OpenCLAddressSpace::Local:
        return "__local";
    case OpenCLAddressSpace::Global:
        return "__global";
    default:
        return 0;
    }
}

bool needs_volatile_private_vector_array(const SPIRType& type, StorageClass storage)
{
    // Adreno OpenCL optimizers can lose lane updates or crash when function
    // arrays of 32-bit integer vectors are accessed through dynamic indices.
    // Keep every such array volatile: translation is device independent and
    // does not prove which accesses will remain static after optimization.
    return storage == StorageClassFunction && !type.array.empty()
           && type.columns == 1 && type.vecsize > 1 && type.width == 32
           && (type.basetype == SPIRType::Int || type.basetype == SPIRType::UInt);
}

void CompilerOpenCL::emit_entry_point_declarations()
{
    CompilerGLSL::emit_entry_point_declarations();
    if (push_constant_variable_id != 0)
    {
        const SPIRType& push_type = get_variable_data_type(get<SPIRVariable>(push_constant_variable_id));
        std::string values;
        const size_t count = raw_push_constants ? push_constant_bytes / 4 : push_type.member_types.size();
        for (size_t i = 0; i < count; i++)
        {
            if (i != 0)
                values += ", ";
            values += "momo_push_constant_" + std::to_string(i);
        }
        if (raw_push_constants)
            statement("const uint momo_push_constants[", count, "] = {", values, "};");
        else
            statement(type_to_glsl(push_type), " momo_push_constants = {", values, "};");
    }
    if (translation_options.workgroup_mode == WorkgroupModeVirtual)
    {
        const uint64_t invocation_count = static_cast<uint64_t>(local_size[0]) * local_size[1] * local_size[2];
        const uint64_t xy_size = static_cast<uint64_t>(local_size[0]) * local_size[1];
        statement("const uint momo_physical_local_size = (uint)get_local_size(0);");
        statement("const uint momo_workgroup_chunk_count = (", invocation_count, "u + momo_physical_local_size - 1u) / momo_physical_local_size;");
        statement("const uint momo_workgroup_chunk = (uint)(get_group_id(0) % (size_t)momo_workgroup_chunk_count);");
        statement("const uint momo_virtual_local_invocation_index = momo_workgroup_chunk * momo_physical_local_size + (uint)get_local_id(0);");
        statement("if (momo_virtual_local_invocation_index >= ", invocation_count, "u) return;");
        statement("const uint3 momo_virtual_local_invocation_id = (uint3)(momo_virtual_local_invocation_index % ", local_size[0], "u, (momo_virtual_local_invocation_index / ", local_size[0], "u) % ", local_size[1], "u, momo_virtual_local_invocation_index / ", xy_size, "u);");
        statement("const uint3 momo_virtual_workgroup_id = (uint3)((uint)(get_group_id(0) / (size_t)momo_workgroup_chunk_count), (uint)get_group_id(1), (uint)get_group_id(2));");
        statement("const uint3 momo_virtual_num_workgroups = (uint3)((uint)(get_num_groups(0) / (size_t)momo_workgroup_chunk_count), (uint)get_num_groups(1), (uint)get_num_groups(2));");
        statement("const uint3 momo_virtual_global_invocation_id = momo_virtual_workgroup_id * (uint3)(", local_size[0], "u, ", local_size[1], "u, ", local_size[2], "u) + momo_virtual_local_invocation_id;");
    }
    if (translation_options.workgroup_mode == WorkgroupModeRepartitioned)
    {
        statement("const uint3 momo_virtual_global_invocation_id = (uint3)((uint)get_global_id(0), (uint)get_global_id(1), (uint)get_global_id(2));");
        statement("const uint3 momo_virtual_local_invocation_id = (uint3)((uint)(get_global_id(0) % ", local_size[0], "u), (uint)(get_global_id(1) % ", local_size[1], "u), (uint)(get_global_id(2) % ", local_size[2], "u));");
        statement("const uint momo_virtual_local_invocation_index = momo_virtual_local_invocation_id.x + ", local_size[0], "u * (momo_virtual_local_invocation_id.y + ", local_size[1], "u * momo_virtual_local_invocation_id.z);");
        statement("const uint3 momo_virtual_workgroup_id = (uint3)((uint)(get_global_id(0) / ", local_size[0], "u), (uint)(get_global_id(1) / ", local_size[1], "u), (uint)(get_global_id(2) / ", local_size[2], "u));");
        statement("const uint3 momo_virtual_num_workgroups = (uint3)((uint)(get_global_size(0) / ", local_size[0], "u), (uint)(get_global_size(1) / ", local_size[1], "u), (uint)(get_global_size(2) / ", local_size[2], "u));");
    }
    if (uses_workgroup_storage)
        statement("volatile int momo_active = 1;");
}

void CompilerOpenCL::emit_header()
{
    // CompilerGLSL supplies the structured control-flow and expression
    // emitter. These switches make its output C-like before the virtual
    // type/resource hooks below finish the OpenCL-specific lowering.
    backend.allow_precision_qualifiers = false;
    backend.supports_extensions = false;
    backend.use_initializer_list = true;
    backend.use_typed_initializer_list = false;
    backend.use_array_constructor = false;
    // C struct tags have their own namespace. Use the same explicit type in
    // definitions and references, after the emitter has resolved its name.
    backend.explicit_struct_type = true;
    backend.array_is_value_type = false;
    backend.array_is_value_type_in_buffer_blocks = false;
    backend.implicit_c_integer_promotion_rules = true;
    backend.native_row_major_matrix = false;
    backend.support_precise_qualifier = false;
    backend.unsized_array_supported = true;
    backend.float_literal_suffix = true;
    // OpenCL C double constants use the unsuffixed form. The GLSL "lf"
    // suffix emitted by the default SPIRV-Cross backend is not valid here.
    backend.double_literal_suffix = false;
    // GLSL accepts the 16-bit integer literal suffixes "s" and "us", but
    // OpenCL C does not. Leaving the suffixes empty makes the generic
    // constant emitter use explicit short(...) and ushort(...) conversions,
    // which are lowered to OpenCL C casts after emission.
    backend.int16_t_literal_suffix = "";
    backend.uint16_t_literal_suffix = "";
    backend.boolean_mix_function = "select";

    statement("/* Generated from Vulkan Shader SPIR-V by momoten. */");
    statement("/* Baseline language: OpenCL C 1.0. */");
    if (!(floating_point_controls.fast_math_flags & FPFastMathModeAllowContractMask))
        statement("#pragma OPENCL FP_CONTRACT OFF");
    if (requires_global_int32_atomics)
        statement("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable");
    if (requires_fp16)
        statement("#pragma OPENCL EXTENSION cl_khr_fp16 : enable");
    if (translation_options.byte_addressable_store)
    {
        statement("#if defined(cl_khr_byte_addressable_store) && (!defined(__OPENCL_C_VERSION__) || __OPENCL_C_VERSION__ < 110)");
        statement("#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable");
        statement("#endif");
    }
    if (requires_fp64)
    {
        statement("#if !defined(__opencl_c_fp64)");
        statement("#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
        statement("#endif");
    }
    if (requires_integer_dot_product)
        statement("#pragma OPENCL EXTENSION ", integer_dot_product_extension(translation_options.integer_dot_product_backend), " : enable");
    if (translation_options.subgroup_mode == SubgroupModeNative)
    {
        statement("#pragma OPENCL EXTENSION cl_khr_subgroups : enable");
        statement("#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable");
    }
    statement("");

    if (requires_fp64)
        emit_fp64_vector_helpers();
    if (requires_scalar_pow)
    {
        statement("float momo_pow(float base, float exponent)");
        begin_scope();
        statement("volatile float runtime_base = base;");
        statement("return pow(runtime_base, exponent);");
        end_scope();
    }
    // Vulkan std430 mat4 has 16-byte alignment and 64-byte stride. An
    // OpenCL float16 would impose 64-byte alignment, so preserve layout
    // with a value-copyable struct containing four float4 columns.
    statement("typedef struct { float4 c[4]; } momo_mat4;");
    statement("typedef char momo_mat4_size_must_be_64[(sizeof(momo_mat4) == 64) ? 1 : -1];");
    if (!translation_options.native_fp16)
        statement("typedef struct { ushort4 c[4]; } momo_hmat4;");
    if (requires_fp16)
    {
        // A Vulkan f16mat4 is four tightly packed half4 columns (32 bytes).
        // OpenCL's half4 has the same 8-byte column alignment when
        // cl_khr_fp16 is present.
        statement("typedef struct { half4 c[4]; } momo_hmat4;");
        statement("typedef char momo_hmat4_size_must_be_32[(sizeof(momo_hmat4) == 32) ? 1 : -1];");
        if (translation_options.packed_fp16_vector_loads)
        {
            statement("#if !defined(__has_attribute)");
            statement("#error Packed half loads require may_alias and aligned attributes");
            statement("#elif !__has_attribute(may_alias) || !__has_attribute(aligned)");
            statement("#error Packed half loads require may_alias and aligned attributes");
            statement("#endif");
            statement("typedef uint2 momo_alias_uint2 __attribute__((may_alias, aligned(4)));");
            statement("typedef uint4 momo_alias_uint4 __attribute__((may_alias, aligned(4)));");
        }
    }
    if (requires_integer_dot_product)
        emit_integer_dot_product_helpers();

    statement("#define momo_mod(x, y) ((x) - (y) * floor((x) / (y)))");
    statement("#define mod(x, y) momo_mod((x), (y))");
    statement("#define inversesqrt(x) rsqrt(x)");
    // Preserve Vulkan/GLSL nearest-even behavior for ROUND. OpenCL
    // round() instead rounds halfway cases away from
    // zero; rint() is the OpenCL C 1.0 nearest-even equivalent.
    statement("#ifdef round");
    statement("#undef round");
    statement("#endif");
    statement("#define round(x) rint(x)");
    statement("#define roundEven(x) rint(x)");
    statement("#define floatBitsToInt(x) as_int(x)");
    statement("#define floatBitsToUint(x) as_uint(x)");
    statement("#define intBitsToFloat(x) as_float(x)");
    statement("#define uintBitsToFloat(x) as_float(x)");
    statement("#define lessThan(x, y) ((x) < (y))");
    statement("#define lessThanEqual(x, y) ((x) <= (y))");
    statement("#define greaterThan(x, y) ((x) > (y))");
    statement("#define greaterThanEqual(x, y) ((x) >= (y))");
    statement("#define equal(x, y) ((x) == (y))");
    statement("#define notEqual(x, y) ((x) != (y))");
    // SPIR-V Unroll and DontUnroll are optimization requests. OpenCL C
    // 1.0 has no portable source spelling for either request, so retain
    // the loop semantics and intentionally discard only the hint.
    statement("#define SPIRV_CROSS_UNROLL");
    statement("#define SPIRV_CROSS_LOOP");
    statement("#define memoryBarrier() mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)");
    statement("#define memoryBarrierBuffer() mem_fence(CLK_GLOBAL_MEM_FENCE)");
    statement("#define memoryBarrierShared() mem_fence(CLK_LOCAL_MEM_FENCE)");
    statement("#define groupMemoryBarrier() mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)");
    statement("");
    // Keep generated function helpers as ordinary definitions. OpenCL C 1.0
    // rejects static functions, while C99-style plain inline definitions can
    // leave unresolved external references on drivers that do not inline.
    statement("float momo_half_to_float_scalar(ushort h)");
    begin_scope();
    statement("uint sign = ((uint)h & 0x8000u) << 16;");
    statement("uint exponent = ((uint)h >> 10) & 0x1fu;");
    statement("uint mantissa = (uint)h & 0x3ffu;");
    statement("uint bits;");
    statement("if (exponent == 0u)");
    begin_scope();
    statement("if (mantissa == 0u)");
    statement("    bits = sign;");
    statement("else");
    begin_scope();
    statement("uint shift = clz(mantissa) - 21u;");
    statement("mantissa = (mantissa << shift) & 0x3ffu;");
    statement("bits = sign | ((113u - shift) << 23) | (mantissa << 13);");
    end_scope();
    end_scope();
    statement("else if (exponent == 31u)");
    statement("    bits = sign | 0x7f800000u | (mantissa << 13);");
    statement("else");
    statement("    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);");
    statement("return as_float(bits);");
    end_scope();
    statement("");
    statement("ushort momo_float_to_half_scalar(float value)");
    begin_scope();
    statement("uint bits = as_uint(value);");
    statement("uint sign = (bits >> 16) & 0x8000u;");
    statement("uint exponent = (bits >> 23) & 0xffu;");
    statement("uint mantissa = bits & 0x7fffffu;");
    statement("if (exponent == 255u)");
    statement("    return (ushort)(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));");
    statement("int half_exponent = (int)exponent - 112;");
    statement("if (half_exponent >= 31)");
    statement("    return (ushort)(sign | 0x7c00u);");
    statement("if (half_exponent <= 0)");
    begin_scope();
    statement("if (half_exponent < -10)");
    statement("    return (ushort)sign;");
    statement("mantissa |= 0x800000u;");
    statement("uint shift = (uint)(14 - half_exponent);");
    statement("uint half_mantissa = mantissa >> shift;");
    statement("uint remainder = mantissa & ((1u << shift) - 1u);");
    statement("uint halfway = 1u << (shift - 1u);");
    statement("if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u)))");
    statement("    half_mantissa++;");
    statement("return (ushort)(sign | half_mantissa);");
    end_scope();
    statement("uint half_mantissa = mantissa >> 13;");
    statement("uint remainder = mantissa & 0x1fffu;");
    statement("if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u)))");
    begin_scope();
    statement("half_mantissa++;");
    statement("if (half_mantissa == 0x400u)");
    begin_scope();
    statement("half_mantissa = 0u;");
    statement("half_exponent++;");
    statement("if (half_exponent >= 31)");
    statement("    return (ushort)(sign | 0x7c00u);");
    end_scope();
    end_scope();
    statement("return (ushort)(sign | ((uint)half_exponent << 10) | half_mantissa);");
    end_scope();
    statement("");
    statement("ushort momo_float_to_half_rtz_scalar(float value)");
    begin_scope();
    statement("uint bits = as_uint(value);");
    statement("uint sign = (bits >> 16) & 0x8000u;");
    statement("uint exponent = (bits >> 23) & 0xffu;");
    statement("uint mantissa = bits & 0x7fffffu;");
    statement("if (exponent == 255u)");
    statement("    return (ushort)(sign | (mantissa == 0u ? 0x7c00u : 0x7e00u));");
    statement("int half_exponent = (int)exponent - 112;");
    statement("if (half_exponent >= 31)");
    statement("    return (ushort)(sign | 0x7bffu);");
    statement("if (half_exponent <= 0)");
    begin_scope();
    statement("if (half_exponent < -10)");
    statement("    return (ushort)sign;");
    statement("return (ushort)(sign | ((mantissa | 0x800000u) >> (uint)(14 - half_exponent)));");
    end_scope();
    statement("return (ushort)(sign | ((uint)half_exponent << 10) | (mantissa >> 13));");
    end_scope();
    statement("");
    statement("float2 momo_unpack_half2x16(uint value)");
    begin_scope();
    statement("return (float2)(momo_half_to_float_scalar((ushort)value), momo_half_to_float_scalar((ushort)(value >> 16)));");
    end_scope();
    statement("uint momo_pack_half2x16(float2 value)");
    begin_scope();
    statement("return (uint)momo_float_to_half_scalar(value.x) | ((uint)momo_float_to_half_scalar(value.y) << 16);");
    end_scope();
    statement("#define unpackHalf2x16(x) momo_unpack_half2x16((uint)(x))");
    statement("#define packHalf2x16(x) momo_pack_half2x16((float2)(x))");
    if (requires_global_int32_atomics)
    {
        // OpenCL C 1.0 exposes separate signed and unsigned overloads. Keep
        // the pointee type explicit instead of routing signed atomics
        // through uint, which would make the returned value conversion
        // implementation-defined above INT_MAX. A scalar fp16 storage
        // write is lowered by glslang to a compare-exchange loop. For a
        // converged workgroup kernel, inactive tail lanes must make that
        // loop report success without touching global memory, just like
        // the guarded storage-buffer OpStore path.
        if (uses_workgroup_storage)
        {
            statement("#define momo_atomic_cmpxchg_int(object, compare, value) (momo_active != 0 ? atomic_cmpxchg((volatile __global int *)&(object), (int)(compare), (int)(value)) : (int)(compare))");
            statement("#define momo_atomic_cmpxchg_uint(object, compare, value) (momo_active != 0 ? atomic_cmpxchg((volatile __global uint *)&(object), (uint)(compare), (uint)(value)) : (uint)(compare))");
        }
        else
        {
            statement("#define momo_atomic_cmpxchg_int(object, compare, value) atomic_cmpxchg((volatile __global int *)&(object), (int)(compare), (int)(value))");
            statement("#define momo_atomic_cmpxchg_uint(object, compare, value) atomic_cmpxchg((volatile __global uint *)&(object), (uint)(compare), (uint)(value))");
        }
    }
    statement("");

    if (translation_options.workgroup_mode != WorkgroupModeDirect)
    {
        statement("#define gl_NumWorkGroups momo_virtual_num_workgroups");
        statement("#define gl_WorkGroupID momo_virtual_workgroup_id");
        statement("#define gl_LocalInvocationID momo_virtual_local_invocation_id");
        statement("#define gl_GlobalInvocationID momo_virtual_global_invocation_id");
        statement("#define gl_LocalInvocationIndex momo_virtual_local_invocation_index");
    }
    else
    {
        statement("#define gl_NumWorkGroups ((uint3)((uint)get_num_groups(0), (uint)get_num_groups(1), (uint)get_num_groups(2)))");
        statement("#define gl_WorkGroupID ((uint3)((uint)get_group_id(0), (uint)get_group_id(1), (uint)get_group_id(2)))");
        statement("#define gl_LocalInvocationID ((uint3)((uint)get_local_id(0), (uint)get_local_id(1), (uint)get_local_id(2)))");
        statement("#define gl_GlobalInvocationID ((uint3)((uint)get_global_id(0), (uint)get_global_id(1), (uint)get_global_id(2)))");
        const uint64_t xy_size = static_cast<uint64_t>(local_size[0]) * local_size[1];
        statement("#define gl_LocalInvocationIndex ((uint)(get_local_id(2) * ", xy_size, "u + get_local_id(1) * ", local_size[0], "u + get_local_id(0)))");
    }
    statement("#define gl_WorkGroupSize ((uint3)(", local_size[0], "u, ", local_size[1], "u, ", local_size[2], "u))");
    statement("");
}

std::string CompilerOpenCL::type_to_glsl(const SPIRType& type, uint32_t id)
{
    (void)id;

    if (type.columns > 1)
    {
        if (type.basetype == SPIRType::Float && type.width == 32 && type.vecsize == 4 && type.columns == 4)
            return "momo_mat4";
        if (type.basetype == SPIRType::Half && type.width == 16 && type.vecsize == 4 && type.columns == 4)
            return "momo_hmat4";
        throw std::runtime_error("only float32 or float16 mat4 values are supported by the OpenCL C profile");
    }

    std::string base;
    switch (type.basetype)
    {
    case SPIRType::Void:
        base = "void";
        break;
    case SPIRType::Boolean:
        base = type.vecsize == 1 ? "bool" : "int";
        break;
    case SPIRType::SByte:
        base = "char";
        break;
    case SPIRType::UByte:
        base = "uchar";
        break;
    case SPIRType::Short:
        base = "short";
        break;
    case SPIRType::UShort:
        base = "ushort";
        break;
    case SPIRType::Int:
        base = "int";
        break;
    case SPIRType::UInt:
        base = "uint";
        break;
    case SPIRType::Int64:
        base = "long";
        break;
    case SPIRType::UInt64:
        base = "ulong";
        break;
    case SPIRType::Half:
        base = translation_options.native_fp16 ? "half" : "ushort";
        break;
    case SPIRType::Float:
        base = "float";
        break;
    case SPIRType::Double:
        base = "double";
        break;
    case SPIRType::Struct:
        return "struct " + to_name(type.self);
    default:
        throw std::runtime_error("SPIR-V type cannot be represented by the initial OpenCL C profile");
    }

    if (type.vecsize > 1)
        base += std::to_string(type.vecsize);
    return base;
}

std::string CompilerOpenCL::layout_for_member(const SPIRType&, uint32_t)
{
    return std::string();
}

const char* CompilerOpenCL::to_storage_qualifiers_glsl(const SPIRVariable&)
{
    return "";
}

std::string CompilerOpenCL::to_qualifiers_glsl(uint32_t id)
{
    const SPIRVariable* variable = maybe_get<SPIRVariable>(id);
    if (variable && variable->storage == StorageClassWorkgroup)
        return "__local ";
    if (variable && needs_volatile_private_vector_array(get_variable_data_type(*variable), variable->storage))
        return "volatile ";
    return std::string();
}

std::string CompilerOpenCL::bitcast_glsl_op(const SPIRType& result_type, const SPIRType& argument_type)
{
    // OpenCL as_type builtins require equal total storage size, not identical
    // scalar width or vector length. This covers SPIR-V bitcasts such as
    // uint <-> ushort2 emitted for packed 16-bit ncnn storage.
    const uint32_t result_bits = result_type.width * result_type.vecsize;
    const uint32_t argument_bits = argument_type.width * argument_type.vecsize;
    if (result_type.columns != 1 || argument_type.columns != 1 || result_type.vecsize == 3 || argument_type.vecsize == 3 || result_bits != argument_bits)
        throw std::runtime_error("unsupported OpenCL bitcast shape");

    std::string type = type_to_glsl(result_type);
    // Mixed-signedness arithmetic can request an identity bitcast for one
    // operand. Older PowerVR compilers omit these as_type overloads.
    if (type_to_glsl(result_type) == type_to_glsl(argument_type) && result_type.width == argument_type.width && result_type.vecsize == argument_type.vecsize)
        return "(" + type + ")";
    return "as_" + type;
}

std::string CompilerOpenCL::builtin_to_glsl(BuiltIn builtin, StorageClass storage)
{
    const uint64_t invocation_count = static_cast<uint64_t>(local_size[0]) * local_size[1] * local_size[2];

    switch (builtin)
    {
    case BuiltInSubgroupSize:
    case BuiltInSubgroupMaxSize:
        // Vulkan exposes one fixed subgroupSize for the physical device,
        // including a partially occupied final subgroup. OpenCL's current
        // subgroup size may be smaller there, so use the validated profile
        // constant instead of get_sub_group_size().
        return std::to_string(translation_options.subgroup_size) + "u";
    case BuiltInSubgroupLocalInvocationId:
        if (translation_options.subgroup_mode == SubgroupModeNative)
            return "((uint)get_sub_group_local_id())";
        if (translation_options.subgroup_mode == SubgroupModeEmulatedBasic)
            return "(gl_LocalInvocationIndex % " + std::to_string(translation_options.subgroup_size) + "u)";
        return "0u";
    case BuiltInSubgroupId:
        if (translation_options.subgroup_mode == SubgroupModeNative)
        {
            if (translation_options.workgroup_mode == WorkgroupModeVirtual)
                return "(momo_workgroup_chunk * ((uint)get_local_size(0) / " + std::to_string(translation_options.subgroup_size) + "u) + (uint)get_sub_group_id())";
            return "((uint)get_sub_group_id())";
        }
        if (translation_options.subgroup_mode == SubgroupModeEmulatedBasic)
            return "(gl_LocalInvocationIndex / " + std::to_string(translation_options.subgroup_size) + "u)";
        return "gl_LocalInvocationIndex";
    case BuiltInNumSubgroups:
    case BuiltInNumEnqueuedSubgroups:
        if (translation_options.subgroup_mode == SubgroupModeNative)
        {
            if (translation_options.workgroup_mode == WorkgroupModeVirtual)
            {
                const uint64_t subgroup_count = (invocation_count + translation_options.subgroup_size - 1) / translation_options.subgroup_size;
                return std::to_string(subgroup_count) + "u";
            }
            return "((uint)get_num_sub_groups())";
        }
        if (translation_options.subgroup_mode == SubgroupModeEmulatedBasic)
        {
            const uint64_t subgroup_count = (invocation_count + translation_options.subgroup_size - 1) / translation_options.subgroup_size;
            return std::to_string(subgroup_count) + "u";
        }
        return std::to_string(invocation_count) + "u";
    default:
        return CompilerGLSL::builtin_to_glsl(builtin, storage);
    }
}

std::string CompilerOpenCL::to_member_reference(uint32_t base, const SPIRType& type, uint32_t index,
                                                bool ptr_chain_is_resolved)
{
    if (storage_block_types.count(type.self) != 0)
    {
        if (index != 0)
            throw std::runtime_error("internal error: flattened storage block member index is not zero");
        return std::string();
    }
    return CompilerGLSL::to_member_reference(base, type, index, ptr_chain_is_resolved);
}

bool CompilerOpenCL::emit_fp64_vector_instruction(
    const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    static const char components[] = "xyzw";

    if (opcode == OpDot && instruction.length >= 4)
    {
        const SPIRType& operand_type = expression_type(ops[2]);
        if (operand_type.basetype != SPIRType::Double || operand_type.width != 64
            || operand_type.columns != 1 || operand_type.vecsize < 2
            || operand_type.vecsize > 4)
            return false;

        const std::string suffix = std::to_string(operand_type.vecsize);
        const std::string expression = "momo_fp64_dot" + suffix + "(" + to_expression(ops[2]) + ", " + to_expression(ops[3]) + ")";
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        return true;
    }

    switch (opcode)
    {
    case OpCompositeConstruct:
    case OpFAdd:
    case OpFSub:
    case OpFMul:
    case OpFDiv:
    case OpFNegate:
    case OpVectorTimesScalar:
        break;
    default:
        return false;
    }

    if (instruction.length < 3)
        return false;
    const SPIRType& result_type = get<SPIRType>(ops[0]);
    if (result_type.basetype != SPIRType::Double || result_type.width != 64
        || result_type.columns != 1 || result_type.vecsize < 2
        || result_type.vecsize > 4)
        return false;

    const std::string suffix = std::to_string(result_type.vecsize);
    if (opcode == OpCompositeConstruct)
    {
        std::vector<std::string> arguments;
        bool forward = true;
        const uint32_t constituent_count = instruction.length - 2;
        for (uint32_t i = 0; i < constituent_count; i++)
        {
            const uint32_t constituent = ops[2 + i];
            const SPIRType& constituent_type = expression_type(constituent);
            forward = forward && should_forward(constituent);
            if (constituent_type.columns != 1 || constituent_type.vecsize > 4)
                throw std::runtime_error("fp64 vector construction requires scalar or vector constituents");
            const std::string value = to_expression(constituent);
            if (constituent_type.vecsize == 1)
            {
                arguments.push_back(value);
            }
            else
            {
                for (uint32_t component = 0; component < constituent_type.vecsize; component++)
                    arguments.push_back("(" + value + ")." + components[component]);
            }
            inherit_expression_dependencies(ops[1], constituent);
        }
        if (arguments.size() != result_type.vecsize)
            throw std::runtime_error("fp64 vector construction has an inconsistent component count");

        std::string expression = "momo_make_double" + suffix + "(";
        for (size_t i = 0; i < arguments.size(); i++)
        {
            if (i != 0)
                expression += ", ";
            expression += arguments[i];
        }
        expression += ")";
        emit_op(ops[0], ops[1], expression, forward);
        return true;
    }

    const char* operation = 0;
    switch (opcode)
    {
    case OpFAdd:
        operation = "add";
        break;
    case OpFSub:
        operation = "sub";
        break;
    case OpFMul:
        operation = "mul";
        break;
    case OpFDiv:
        operation = "div";
        break;
    default:
        break;
    }
    if (operation)
    {
        const std::string expression = std::string("momo_fp64_") + operation + suffix + "(" + to_expression(ops[2]) + ", " + to_expression(ops[3]) + ")";
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        return true;
    }

    if (opcode == OpFNegate)
    {
        const std::string expression = "momo_fp64_neg" + suffix + "(" + to_expression(ops[2]) + ")";
        emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
        inherit_expression_dependencies(ops[1], ops[2]);
        return true;
    }

    if (opcode == OpVectorTimesScalar)
    {
        const std::string expression = "momo_fp64_scale" + suffix + "(" + to_expression(ops[2]) + ", " + to_expression(ops[3]) + ")";
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        return true;
    }

    return false;
}

void CompilerOpenCL::emit_instruction(const Instruction& instruction)
{
    const Op opcode = static_cast<Op>(instruction.op);
    const uint32_t* ops = stream(instruction);

    if (emit_memory_instruction(instruction, ops))
        return;

    if ((opcode == OpAccessChain || opcode == OpInBoundsAccessChain) && emit_matrix_access_chain(instruction, ops))
        return;

    if (emit_fp64_vector_instruction(instruction, ops))
        return;

    if (emit_integer_conversion_instruction(instruction, ops))
        return;

    if (emit_int16_arithmetic_instruction(instruction, ops))
        return;

    if (emit_integer_dot_product_instruction(instruction, ops))
        return;

    if (emit_basic_subgroup_instruction(instruction, ops))
        return;

    switch (opcode)
    {
    case OpAtomicLoad:
    case OpAtomicStore:
    case OpAtomicExchange:
    case OpAtomicCompareExchangeWeak:
    case OpAtomicIIncrement:
    case OpAtomicIDecrement:
    case OpAtomicIAdd:
    case OpAtomicISub:
    case OpAtomicSMin:
    case OpAtomicUMin:
    case OpAtomicSMax:
    case OpAtomicUMax:
    case OpAtomicAnd:
    case OpAtomicOr:
    case OpAtomicXor:
        throw std::runtime_error("SPIR-V atomics require optional OpenCL extensions and are deferred");
    default:
        break;
    }

    if (emit_barrier_instruction(instruction, ops))
        return;

    if (opcode == OpVectorExtractDynamic)
    {
        const SPIRType& vector_type = expression_type(ops[2]);
        if (vector_type.columns != 1 || vector_type.vecsize < 2 || vector_type.vecsize > 4)
            throw std::runtime_error("dynamic vector extraction requires a vec2, vec3 or vec4");

        // Older OpenCL compilers also reject subscripts on vector rvalues,
        // which cannot use the private-pointer access-chain lowering.
        static const char components[] = "xyzw";
        const std::string vector = "(" + to_expression(ops[2]) + ")";
        const std::string index = "(" + to_expression(ops[3]) + ")";
        std::string expression = vector + "." + components[vector_type.vecsize - 1];
        for (uint32_t i = vector_type.vecsize - 1; i > 0; i--)
            expression = "(" + index + " == " + std::to_string(i - 1) + "u ? " + vector + "." + components[i - 1] + " : " + expression + ")";
        if (vector_type.basetype == SPIRType::Boolean)
            expression = "(" + expression + " != 0)";
        // Scalar conditional operands undergo integer promotion in OpenCL C.
        // Restore the SPIR-V result type before forwarding into a bitcast.
        expression = "((" + type_to_glsl(get<SPIRType>(ops[0])) + ")(" + expression + "))";
        emit_op(ops[0], ops[1], expression, should_forward(ops[2]) && should_forward(ops[3]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        return;
    }

    if (opcode == OpCompositeExtract && instruction.length == 4)
    {
        const SPIRType& type = expression_type(ops[2]);
        if (type.basetype == SPIRType::Boolean && type.array.empty() && type.columns == 1
            && type.vecsize > 1 && type.vecsize <= 4 && ops[3] < type.vecsize)
        {
            static const char components[] = "xyzw";
            const std::string expression = "((" + to_expression(ops[2]) + ")." + components[ops[3]] + " != 0)";
            emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
            inherit_expression_dependencies(ops[1], ops[2]);
            return;
        }
    }

    if (opcode == OpAny || opcode == OpAll)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const SPIRType& vector_type = expression_type(ops[2]);
        if (result_type.basetype != SPIRType::Boolean || result_type.vecsize != 1 || vector_type.basetype != SPIRType::Boolean || vector_type.columns != 1 || (vector_type.vecsize != 2 && vector_type.vecsize != 3 && vector_type.vecsize != 4))
            throw std::runtime_error("OpAny/OpAll requires a bvec2/bvec3/bvec4 in the OpenCL C profile");

        // OpenCL any()/all() inspect the sign bit of each integer lane.
        // SPIR-V boolean vectors can also contain scalar stores converted
        // to 0/1 (rather than comparison masks containing 0/-1), so first
        // compare every lane against zero to create canonical masks.
        const std::string vector = "(" + to_expression(ops[2]) + ")";
        const std::string vector_type_name = type_to_glsl(vector_type);
        const std::string expression = std::string(opcode == OpAny ? "any" : "all") + "(" + vector + " != " + vector_type_name + "(0))";
        emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
        inherit_expression_dependencies(ops[1], ops[2]);
        return;
    }

    if (opcode == OpIEqual || opcode == OpINotEqual
        || opcode == OpUGreaterThan || opcode == OpSGreaterThan
        || opcode == OpUGreaterThanEqual || opcode == OpSGreaterThanEqual
        || opcode == OpULessThan || opcode == OpSLessThan
        || opcode == OpULessThanEqual || opcode == OpSLessThanEqual
        || opcode == OpFOrdEqual || opcode == OpFUnordEqual
        || opcode == OpFOrdNotEqual || opcode == OpFUnordNotEqual
        || opcode == OpFOrdLessThan || opcode == OpFUnordLessThan
        || opcode == OpFOrdGreaterThan || opcode == OpFUnordGreaterThan
        || opcode == OpFOrdLessThanEqual || opcode == OpFUnordLessThanEqual
        || opcode == OpFOrdGreaterThanEqual || opcode == OpFUnordGreaterThanEqual)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        if (result_type.basetype == SPIRType::Boolean && result_type.columns == 1 && result_type.vecsize > 1)
        {
            const std::string left = "(" + to_expression(ops[2]) + ")";
            const std::string right = "(" + to_expression(ops[3]) + ")";
            const bool unordered = opcode == OpFUnordEqual || opcode == OpFUnordNotEqual
                                   || opcode == OpFUnordLessThan || opcode == OpFUnordGreaterThan
                                   || opcode == OpFUnordLessThanEqual || opcode == OpFUnordGreaterThanEqual;
            const bool ordered_not_equal = opcode == OpFOrdNotEqual;

            const char* operation = 0;
            switch (opcode)
            {
            case OpIEqual:
            case OpFOrdEqual:
            case OpFUnordEqual:
                operation = "==";
                break;
            case OpINotEqual:
            case OpFOrdNotEqual:
            case OpFUnordNotEqual:
                operation = "!=";
                break;
            case OpUGreaterThan:
            case OpSGreaterThan:
            case OpFOrdGreaterThan:
            case OpFUnordGreaterThan:
                operation = ">";
                break;
            case OpUGreaterThanEqual:
            case OpSGreaterThanEqual:
            case OpFOrdGreaterThanEqual:
            case OpFUnordGreaterThanEqual:
                operation = ">=";
                break;
            case OpULessThan:
            case OpSLessThan:
            case OpFOrdLessThan:
            case OpFUnordLessThan:
                operation = "<";
                break;
            case OpULessThanEqual:
            case OpSLessThanEqual:
            case OpFOrdLessThanEqual:
            case OpFUnordLessThanEqual:
                operation = "<=";
                break;
            default:
                break;
            }

            std::string comparison = left + " " + operation + " " + right;
            if (ordered_not_equal)
                comparison = "(~isunordered(" + left + ", " + right + ") & (" + comparison + "))";
            else if (unordered && opcode != OpFUnordNotEqual)
                comparison = "(isunordered(" + left + ", " + right + ") | (" + comparison + "))";

            const std::string expression = "convert_int" + std::to_string(result_type.vecsize) + "(" + comparison + ")";
            emit_op(ops[0], ops[1], expression,
                    should_forward(ops[2]) && should_forward(ops[3]));
            inherit_expression_dependencies(ops[1], ops[2]);
            inherit_expression_dependencies(ops[1], ops[3]);
            return;
        }
    }

    if (opcode == OpVectorTimesMatrix)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const SPIRType& vector_type = expression_type(ops[2]);
        const SPIRType& matrix_type = expression_type(ops[3]);
        const bool supported_base = (result_type.basetype == SPIRType::Float && result_type.width == 32) || (result_type.basetype == SPIRType::Half && result_type.width == 16);
        if (!supported_base || result_type.vecsize != 4 || result_type.columns != 1 || vector_type.basetype != result_type.basetype || vector_type.width != result_type.width || vector_type.vecsize != 4 || vector_type.columns != 1 || matrix_type.basetype != result_type.basetype || matrix_type.width != result_type.width || matrix_type.vecsize != 4 || matrix_type.columns != 4)
            throw std::runtime_error("only matching float4/half4 times mat4 is supported by the OpenCL C profile");

        const std::string vector = "(" + to_expression(ops[2]) + ")";
        const std::string matrix = "(" + to_expression(ops[3]) + ")";
        std::string expression = type_to_glsl(result_type) + "(dot(" + vector + ", " + matrix + ".c[0]), "
                                                                                                "dot("
                                 + vector + ", " + matrix + ".c[1]), "
                                                            "dot("
                                 + vector + ", " + matrix + ".c[2]), "
                                                            "dot("
                                 + vector + ", " + matrix + ".c[3]))";
        // Keep the four fp16 dot products in vector arithmetic.
        if (result_type.basetype == SPIRType::Half)
        {
            expression.clear();
            for (size_t lane = 0; lane < 4; lane++)
            {
                const std::string component(1, "xyzw"[lane]);
                if (lane)
                    expression += " + ";
                expression += vector + "." + component + " * ((half4)(";
                for (size_t column = 0; column < 4; column++)
                {
                    if (column)
                        expression += ", ";
                    expression += matrix + ".c[" + std::to_string(column) + "]." + component;
                }
                expression += "))";
            }
            expression = "(" + expression + ")";
        }
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        return;
    }

    if (opcode == OpMatrixTimesVector)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const SPIRType& matrix_type = expression_type(ops[2]);
        const SPIRType& vector_type = expression_type(ops[3]);
        const bool supported_base = (result_type.basetype == SPIRType::Float && result_type.width == 32) || (result_type.basetype == SPIRType::Half && result_type.width == 16);
        if (!supported_base || result_type.vecsize != 4 || result_type.columns != 1 || matrix_type.basetype != result_type.basetype || matrix_type.width != result_type.width || matrix_type.vecsize != 4 || matrix_type.columns != 4 || vector_type.basetype != result_type.basetype || vector_type.width != result_type.width || vector_type.vecsize != 4 || vector_type.columns != 1)
            throw std::runtime_error("only matching float32/float16 mat4 times vector is supported by the OpenCL C profile");

        const std::string matrix = "(" + to_expression(ops[2]) + ")";
        const std::string vector = "(" + to_expression(ops[3]) + ")";
        const std::string expression = "(" + matrix + ".c[0] * " + vector + ".x + " + matrix + ".c[1] * " + vector + ".y + " + matrix + ".c[2] * " + vector + ".z + " + matrix + ".c[3] * " + vector + ".w)";
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        return;
    }

    if (opcode == OpCompositeExtract)
    {
        const SPIRType& composite_type = expression_type(ops[2]);
        if (composite_type.columns > 1)
        {
            const bool supported_base = (composite_type.basetype == SPIRType::Float && composite_type.width == 32) || (composite_type.basetype == SPIRType::Half && composite_type.width == 16);
            if (!supported_base || composite_type.vecsize != 4 || composite_type.columns != 4 || (instruction.length != 4 && instruction.length != 5) || ops[3] >= 4 || (instruction.length == 5 && ops[4] >= 4))
                throw std::runtime_error("only literal column or scalar extraction from float32/float16 mat4 is supported");

            std::string expression = "(" + to_expression(ops[2]) + ").c[" + std::to_string(ops[3]) + "]";
            if (instruction.length == 5)
                expression += "." + std::string(1, "xyzw"[ops[4]]);
            emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
            inherit_expression_dependencies(ops[1], ops[2]);
            return;
        }
    }

    if (opcode == OpMatrixTimesMatrix || opcode == OpMatrixTimesScalar || opcode == OpOuterProduct || opcode == OpTranspose)
        throw std::runtime_error("SPIR-V matrix operation is outside the float4-times-mat4 OpenCL profile");

    if (opcode == OpAtomicCompareExchange)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const SPIRType& pointer_type = expression_type(ops[2]);
        if (!requires_global_int32_atomics || (result_type.basetype != SPIRType::Int && result_type.basetype != SPIRType::UInt) || result_type.width != 32 || result_type.vecsize != 1 || (pointer_type.storage != StorageClassUniform && pointer_type.storage != StorageClassStorageBuffer))
        {
            std::ostringstream message;
            message << "only global int32/uint32 OpAtomicCompareExchange is supported by the OpenCL 1.0 extension path"
                    << " (result basetype=" << static_cast<unsigned>(result_type.basetype)
                    << " width=" << result_type.width
                    << " vecsize=" << result_type.vecsize
                    << " pointer storage=" << static_cast<unsigned>(pointer_type.storage) << ')';
            throw std::runtime_error(message.str());
        }

        const char* function_name = result_type.basetype == SPIRType::Int
                                        ? "momo_atomic_cmpxchg_int"
                                        : "momo_atomic_cmpxchg_uint";
        const SPIRVariable* backing = maybe_get_backing_variable(ops[2]);
        if (backing && uses_byte_addressing(backing->self))
        {
            require_statement_continue();
            flush_variable_declaration(ops[2]);
            to_expression(ops[2], true);
            const std::string object = "*((volatile __global " + type_to_glsl(result_type) + " *)(" + byte_address(ops[2]) + "))";
            forced_temporaries.insert(ops[1]);
            emit_op(ops[0], ops[1], std::string(function_name) + "(" + object + ", " + to_unpacked_expression(ops[7]) + ", " + to_unpacked_expression(ops[6]) + ")", false);
            flush_all_atomic_capable_variables();
        }
        else
            emit_atomic_func_op(ops[0], ops[1], ops[2], ops[7], ops[6], function_name);
        return;
    }

    if (opcode == OpSelect)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const SPIRType& condition_type = expression_type(ops[2]);
        if (result_type.pointer)
            throw std::runtime_error("pointer selection is not supported by the OpenCL C profile");

        std::string expression;
        if (condition_type.vecsize == 1)
        {
            expression = "(" + to_expression(ops[2]) + " ? " + to_expression(ops[3]) + " : " + to_expression(ops[4]) + ")";
            if (result_type.width == 16 && (result_type.basetype == SPIRType::Short || result_type.basetype == SPIRType::UShort || (result_type.basetype == SPIRType::Half && !translation_options.native_fp16)))
                expression = "((" + type_to_glsl(result_type) + ")" + expression + ")";
        }
        else
        {
            // OpenCL select requires an integer mask whose element width
            // matches the selected value. SPIRV-Cross represents every
            // SPIR-V boolean vector as intN, so normalize both its truth
            // representation and width before passing it to select. This is
            // especially important for halfN and doubleN values, where an
            // intN mask has no matching OpenCL overload.
            const std::string condition = "(" + to_expression(ops[2]) + ")";
            const std::string condition_type_name = type_to_glsl(condition_type);
            const uint32_t result_width = result_type.basetype == SPIRType::Boolean ? 32 : result_type.width;
            const char* mask_base = result_width == 8    ? "char"
                                    : result_width == 16 ? "short"
                                    : result_width == 32 ? "int"
                                    : result_width == 64 ? "long"
                                                         : 0;
            if (!mask_base || result_type.columns != 1 || result_type.vecsize != condition_type.vecsize)
                throw std::runtime_error("vector OpSelect requires matching scalar/vector values with 8, 16, 32 or 64-bit elements");
            const std::string mask_type = std::string(mask_base) + std::to_string(condition_type.vecsize);
            const std::string mask = "convert_" + mask_type + "(" + condition + " != " + condition_type_name + "(0))";
            expression = "select(" + to_expression(ops[4]) + ", " + to_expression(ops[3]) + ", " + mask + ")";
        }
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]) && should_forward(ops[4]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        inherit_expression_dependencies(ops[1], ops[4]);
        return;
    }

    if (opcode == OpFConvert)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const SPIRType& source_type = expression_type(ops[2]);
        const bool to_half = result_type.basetype == SPIRType::Half;
        const uint32_t rounding = to_half && has_decoration(ops[1], DecorationFPRoundingMode) ? get_decoration(ops[1], DecorationFPRoundingMode) : uint32_t(FPRoundingModeRTE);
        if (to_half && rounding != FPRoundingModeRTE && rounding != FPRoundingModeRTZ)
            throw std::runtime_error("fp16 storage conversion supports RTE and RTZ rounding only");
        if (!translation_options.native_fp16 && (source_type.basetype == SPIRType::Half || result_type.basetype == SPIRType::Half))
        {
            if (source_type.columns != 1 || result_type.columns != 1 || source_type.vecsize != result_type.vecsize || (result_type.basetype == SPIRType::Half && source_type.width != 32))
                throw std::runtime_error("storage-only fp16 conversion requires matching scalar/vector float32 values");
            const std::string source = to_expression(ops[2]);
            const char* conversion = result_type.basetype == SPIRType::Half ? (rounding == FPRoundingModeRTZ ? "momo_float_to_half_rtz_scalar" : "momo_float_to_half_scalar") : "momo_half_to_float_scalar";
            std::string expression;
            for (uint32_t i = 0; i < source_type.vecsize; i++)
            {
                if (i)
                    expression += ", ";
                expression += std::string(conversion) + "((" + source + ")" + (source_type.vecsize == 1 ? "" : ".s" + std::to_string(i)) + ")";
            }
            if (source_type.vecsize > 1)
            {
                SPIRType converted = result_type;
                if (converted.basetype == SPIRType::Double)
                {
                    converted.basetype = SPIRType::Float;
                    converted.width = 32;
                }
                expression = "((" + type_to_glsl(converted) + ")(" + expression + "))";
            }
            if (result_type.basetype == SPIRType::Double)
                expression = "convert_" + type_to_glsl(result_type) + "(" + expression + ")";
            emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
            inherit_expression_dependencies(ops[1], ops[2]);
            return;
        }
        const std::string conversion = "convert_" + type_to_glsl(result_type) + (to_half ? (rounding == FPRoundingModeRTZ ? "_rtz" : "_rte") : "");
        std::string expression = conversion + "(" + to_expression(ops[2]) + ")";
        emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
        inherit_expression_dependencies(ops[1], ops[2]);
        return;
    }

    CompilerGLSL::emit_instruction(instruction);
}

void CompilerOpenCL::emit_glsl_op(uint32_t result_type, uint32_t result_id, uint32_t op,
                                  const uint32_t* args, uint32_t count)
{
    const SPIRType& result = get<SPIRType>(result_type);
    if (op == GLSLstd450Pow && count == 2
        && result.basetype == SPIRType::Float && result.width == 32 && result.vecsize == 1)
    {
        // Adreno folds constant-base scalar pow with insufficient accuracy.
        // A local load can become constant only in the OpenCL compiler, so
        // materialize every scalar float base before calling the builtin.
        if (!requires_scalar_pow)
        {
            requires_scalar_pow = true;
            force_recompile();
        }
        emit_binary_func_op(result_type, result_id, args[0], args[1], "momo_pow");
        return;
    }
    if (op == GLSLstd450SAbs && count == 1)
    {
        const SPIRType& type = get<SPIRType>(result_type);
        const std::string value = to_expression(args[0]);
        std::string expression;
        if (type.vecsize == 1)
            expression = "((" + value + ") < 0 ? -(" + value + ") : (" + value + "))";
        else
            expression = "select((" + value + "), -(" + value + "), (" + value + ") < (" + type_to_glsl(type) + ")(0))";
        if (type.width == 16)
        {
            SPIRType wide = type;
            wide.basetype = SPIRType::Int;
            wide.width = 32;
            const std::string signed_value = bitcast_expression(SPIRType::Short, args[0]);
            expression = narrow_int16(type, "abs(" + convert_value(wide, signed_value) + ")");
        }
        emit_op(result_type, result_id, expression, should_forward(args[0]));
        inherit_expression_dependencies(result_id, args[0]);
        return;
    }
    if (op == GLSLstd450SSign && count == 1 && result.width == 16)
    {
        SPIRType wide = result;
        wide.basetype = SPIRType::Int;
        wide.width = 32;
        const std::string value = convert_value(wide, bitcast_expression(SPIRType::Short, args[0]));
        const std::string expression = narrow_int16(result, "clamp(" + value + ", (" + type_to_glsl(wide) + ")(-1), (" + type_to_glsl(wide) + ")(1))");
        emit_op(result_type, result_id, expression, should_forward(args[0]));
        inherit_expression_dependencies(result_id, args[0]);
        return;
    }
    if (op == GLSLstd450FAbs && count == 1)
    {
        emit_unary_func_op(result_type, result_id, args[0], "fabs");
        return;
    }
    if (op == GLSLstd450Atan2 && count == 2)
    {
        emit_binary_func_op(result_type, result_id, args[0], args[1], "atan2");
        return;
    }
    if (op == GLSLstd450Fract && count == 1)
    {
        const std::string expression = to_expression(args[0]);
        emit_op(result_type, result_id,
                "((" + expression + ") - floor(" + expression + "))",
                should_forward(args[0]));
        return;
    }
    CompilerGLSL::emit_glsl_op(result_type, result_id, op, args, count);
}

std::string CompilerOpenCL::constant_op_expression(const SPIRConstantOp& operation)
{
    if (operation.opcode == OpCompositeExtract && operation.arguments.size() == 2)
    {
        const SPIRType& composite_type = expression_type(operation.arguments[0]);
        const bool supported_base = (composite_type.basetype == SPIRType::Float && composite_type.width == 32) || (composite_type.basetype == SPIRType::Half && composite_type.width == 16);
        if (supported_base && composite_type.vecsize == 4 && composite_type.columns == 4 && operation.arguments[1] < 4)
        {
            return "(" + to_expression(operation.arguments[0]) + ").c[" + std::to_string(operation.arguments[1]) + "]";
        }
    }
    return CompilerGLSL::constant_op_expression(operation);
}

std::string CompilerOpenCL::constant_expression_vector(const SPIRConstant& constant, uint32_t vector)
{
    const SPIRType& type = get<SPIRType>(constant.constant_type);
    if (type.basetype == SPIRType::Half && !translation_options.native_fp16)
    {
        std::string expression;
        for (uint32_t i = 0; i < type.vecsize; i++)
        {
            if (i)
                expression += ", ";
            expression += std::to_string(constant.scalar_u16(vector, i));
        }
        SPIRType vector_type = type;
        vector_type.columns = 1;
        return "((" + type_to_glsl(vector_type) + ")(" + expression + "))";
    }
    std::string expression = CompilerGLSL::constant_expression_vector(constant, vector);
    if (type.basetype == SPIRType::Double && type.width == 64 && type.columns == 1 && type.vecsize >= 2 && type.vecsize <= 4)
    {
        const std::string constructor = "double" + std::to_string(type.vecsize) + "(";
        if (expression.compare(0, constructor.size(), constructor) == 0)
        {
            const size_t arguments_begin = constructor.size();
            const size_t arguments_end = expression.rfind(')');
            if (arguments_end != std::string::npos)
            {
                const std::string arguments = expression.substr(arguments_begin, arguments_end - arguments_begin);
                const bool splat = arguments.find(',') == std::string::npos;
                expression.replace(0, constructor.size(),
                                   std::string(splat ? "momo_splat_double" : "momo_make_double")
                                       + std::to_string(type.vecsize) + "(");
            }
        }
        return expression;
    }
    if (type.basetype != SPIRType::Half)
        return expression;

    // SPIRV-Cross spells half constants as half(1.0). In OpenCL C, 1.0
    // is still a double literal and therefore requires cl_khr_fp64 even
    // when immediately converted to half. Suffix decimal literals so the
    // generated cl_khr_fp16 kernel has no accidental fp64 dependency.
    for (size_t i = 0; i < expression.size();)
    {
        if (expression[i] < '0' || expression[i] > '9')
        {
            i++;
            continue;
        }
        const size_t begin = i;
        bool decimal = false;
        while (i < expression.size())
        {
            const char c = expression[i];
            if ((c >= '0' && c <= '9') || c == '.')
            {
                decimal = decimal || c == '.';
                i++;
                continue;
            }
            if (c == 'e' || c == 'E')
            {
                decimal = true;
                i++;
                if (i < expression.size() && (expression[i] == '+' || expression[i] == '-'))
                    i++;
                continue;
            }
            break;
        }
        if (decimal && (i == expression.size() || (expression[i] != 'f' && expression[i] != 'F')))
        {
            expression.insert(i, 1, 'f');
            i++;
        }
        if (i == begin)
            i++;
    }
    return expression;
}

const char* CompilerOpenCL::function_pointer_address_space(StorageClass storage)
{
    const char* qualifier = opencl_address_space_qualifier(opencl_address_space(storage));
    if (!qualifier)
        throw std::runtime_error("unsupported function-pointer storage class in the OpenCL C profile");
    return qualifier;
}

} // namespace momoten
