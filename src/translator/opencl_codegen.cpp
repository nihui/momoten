// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace momoten {

void CompilerOpenCL::emit_entry_point_declarations()
{
    CompilerGLSL::emit_entry_point_declarations();
    if (workgroup_splittable)
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
    backend.array_is_value_type = false;
    backend.array_is_value_type_in_buffer_blocks = false;
    backend.implicit_c_integer_promotion_rules = true;
    backend.native_row_major_matrix = false;
    backend.support_precise_qualifier = false;
    backend.unsized_array_supported = true;
    backend.float_literal_suffix = true;
    backend.boolean_mix_function = "select";

    statement("/* Generated from Vulkan Shader SPIR-V by momoten. */");
    statement("/* Baseline language: OpenCL C 1.0. */");
    if (requires_global_int32_atomics)
        statement("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable");
    if (requires_fp16)
        statement("#pragma OPENCL EXTENSION cl_khr_fp16 : enable");
    if (requires_integer_dot_product)
        statement("#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable");
    if (translation_options.subgroup_mode == SubgroupModeNative)
    {
        statement("#pragma OPENCL EXTENSION cl_khr_subgroups : enable");
        statement("#pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable");
    }
    statement("");

    emit_constructor_macros("char");
    emit_constructor_macros("uchar");
    emit_constructor_macros("short");
    emit_constructor_macros("ushort");
    emit_constructor_macros("int");
    emit_constructor_macros("uint");
    emit_constructor_macros("long");
    emit_constructor_macros("ulong");
    emit_constructor_macros("float");
    if (requires_fp16)
        emit_constructor_macros("half");
    // Vulkan std430 mat4 has 16-byte alignment and 64-byte stride. An
    // OpenCL float16 would impose 64-byte alignment, so preserve layout
    // with a value-copyable struct containing four float4 columns.
    statement("typedef struct { float4 c[4]; } momo_mat4;");
    statement("typedef char momo_mat4_size_must_be_64[(sizeof(momo_mat4) == 64) ? 1 : -1];");
    statement("#define momo_mat4(...) ((momo_mat4){{__VA_ARGS__}})");
    if (requires_fp16)
    {
        // A Vulkan f16mat4 is four tightly packed half4 columns (32 bytes).
        // OpenCL's half4 has the same 8-byte column alignment when
        // cl_khr_fp16 is present.
        statement("typedef struct { half4 c[4]; } momo_hmat4;");
        statement("typedef char momo_hmat4_size_must_be_32[(sizeof(momo_hmat4) == 32) ? 1 : -1];");
        statement("#define momo_hmat4(...) ((momo_hmat4){{__VA_ARGS__}})");
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
    // Define the core OpenCL call before introducing the zero-argument
    // GLSL compatibility macro so internal lowering can pass explicit flags
    // without recursively expanding that macro.
    statement("inline void momo_workgroup_barrier(uint flags)");
    begin_scope();
    statement("barrier(flags);");
    end_scope();
    // SPIR-V generated from GLSL barrier() carries WorkgroupMemory
    // semantics. A global fence is neither required nor implied here.
    statement("#define barrier() momo_workgroup_barrier(CLK_LOCAL_MEM_FENCE)");
    statement("");
    statement("inline float momo_half_to_float_scalar(ushort h)");
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
    statement("int unbiased_exponent = -14;");
    statement("while ((mantissa & 0x400u) == 0u)");
    begin_scope();
    statement("mantissa <<= 1;");
    statement("unbiased_exponent--;");
    end_scope();
    statement("mantissa &= 0x3ffu;");
    statement("bits = sign | ((uint)(unbiased_exponent + 127) << 23) | (mantissa << 13);");
    end_scope();
    end_scope();
    statement("else if (exponent == 31u)");
    statement("    bits = sign | 0x7f800000u | (mantissa << 13);");
    statement("else");
    statement("    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);");
    statement("return as_float(bits);");
    end_scope();
    statement("");
    statement("inline ushort momo_float_to_half_scalar(float value)");
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
    statement("#if defined(__clang__)");
    statement("__attribute__((noinline))");
    statement("#endif");
    statement("ushort momo_load_ushort(volatile __global const ushort* value, __global const uchar* base, ulong size)");
    begin_scope();
    statement("__global const uchar* address = (__global const uchar*)value;");
    statement("ulong offset = (ulong)(address - base);");
    statement("if (size < (ulong)sizeof(ushort) || offset > size - (ulong)sizeof(ushort))");
    statement("    return (ushort)0;");
    // Do not emit a native 16-bit global load here. LLVM's CPU OpenCL
    // vectorizer may turn it into an unmasked gather for inactive work-item
    // lanes at the edge of an ncnn dispatch. Load the containing aligned
    // 32-bit word and extract the requested half instead.
    statement("ulong word_offset = offset & ~(ulong)3;");
    statement("if (word_offset + (ulong)sizeof(uint) > size)");
    begin_scope();
    statement("uchar lo = base[offset];");
    statement("uchar hi = base[offset + 1];");
    statement("return (ushort)((ushort)lo | ((ushort)hi << 8));");
    end_scope();
    statement("uint word = *((volatile __global const uint*)(base + word_offset));");
    statement("return (ushort)(word >> ((uint)(offset - word_offset) * 8u));");
    end_scope();
    statement("#if defined(__clang__)");
    statement("__attribute__((noinline))");
    statement("#endif");
    statement("short momo_load_short(volatile __global const short* value, __global const uchar* base, ulong size)");
    begin_scope();
    statement("__global const uchar* address = (__global const uchar*)value;");
    statement("ulong offset = (ulong)(address - base);");
    statement("if (size < (ulong)sizeof(short) || offset > size - (ulong)sizeof(short))");
    statement("    return (short)0;");
    statement("return (short)momo_load_ushort((volatile __global const ushort*)value, base, size);");
    end_scope();
    statement("");
    statement("inline float2 momo_unpack_half2x16(uint value)");
    begin_scope();
    statement("return (float2)(momo_half_to_float_scalar((ushort)value), momo_half_to_float_scalar((ushort)(value >> 16)));");
    end_scope();
    statement("inline uint momo_pack_half2x16(float2 value)");
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

    if (workgroup_splittable)
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
        statement("#define gl_LocalInvocationIndex ((uint)(get_local_id(2) * get_local_size(1) * get_local_size(0) + get_local_id(1) * get_local_size(0) + get_local_id(0)))");
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
        base = "half";
        break;
    case SPIRType::Float:
        base = "float";
        break;
    case SPIRType::Double:
        base = "double";
        break;
    case SPIRType::Struct:
        return to_name(type.self);
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
            if (workgroup_splittable)
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
            if (workgroup_splittable)
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

void CompilerOpenCL::emit_instruction(const Instruction& instruction)
{
    const Op opcode = static_cast<Op>(instruction.op);
    const uint32_t* ops = stream(instruction);

    if (opcode == OpLoad && emit_robust_buffer_load(instruction, ops))
        return;

    if ((opcode == OpAccessChain || opcode == OpInBoundsAccessChain) && emit_matrix_access_chain(instruction, ops))
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

    if (opcode == OpControlBarrier || opcode == OpMemoryBarrier)
    {
        if (emit_subgroup_barrier_instruction(instruction, ops))
            return;

        validate_barrier_instruction(opcode, ops);
        // CompilerGLSL performs the required expression flushes around
        // workgroup synchronization. The header maps its small fixed set
        // of GLSL barrier names to OpenCL C 1.0 fences.
        CompilerGLSL::emit_instruction(instruction);
        return;
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
        const std::string expression = type_to_glsl(result_type) + "(dot(" + vector + ", " + matrix + ".c[0]), "
                                                                                                      "dot("
                                       + vector + ", " + matrix + ".c[1]), "
                                                                  "dot("
                                       + vector + ", " + matrix + ".c[2]), "
                                                                  "dot("
                                       + vector + ", " + matrix + ".c[3]))";
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
            if (!supported_base || composite_type.vecsize != 4 || composite_type.columns != 4 || instruction.length != 4 || ops[3] >= 4)
                throw std::runtime_error("only literal column extraction from float32 mat4 is supported");

            const std::string expression = "(" + to_expression(ops[2]) + ").c[" + std::to_string(ops[3]) + "]";
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
        }
        else
        {
            // OpenCL select consumes the all-zero/all-one integer masks
            // produced by vector comparisons and preserves the result
            // vector type without GLSL-style bvec constructors.
            expression = "select(" + to_expression(ops[4]) + ", " + to_expression(ops[3]) + ", " + to_expression(ops[2]) + ")";
        }
        emit_op(ops[0], ops[1], expression,
                should_forward(ops[2]) && should_forward(ops[3]) && should_forward(ops[4]));
        inherit_expression_dependencies(ops[1], ops[2]);
        inherit_expression_dependencies(ops[1], ops[3]);
        inherit_expression_dependencies(ops[1], ops[4]);
        return;
    }

    if (opcode == OpSConvert || opcode == OpUConvert || opcode == OpConvertSToF || opcode == OpConvertUToF || opcode == OpConvertFToS || opcode == OpConvertFToU || opcode == OpFConvert)
    {
        const SPIRType& result_type = get<SPIRType>(ops[0]);
        const std::string expression = "convert_" + type_to_glsl(result_type) + "(" + to_expression(ops[2]) + ")";
        emit_op(ops[0], ops[1], expression, should_forward(ops[2]));
        inherit_expression_dependencies(ops[1], ops[2]);
        return;
    }

    CompilerGLSL::emit_instruction(instruction);
}

void CompilerOpenCL::emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression)
{
    SPIRVariable* backing = maybe_get_backing_variable(lhs_expression);
    const bool resource_store = backing && resource_indices.find(backing->self) != resource_indices.end();

    bool scalar_half_store = false;
    bool vector_store = false;
    if (resource_store)
    {
        const SPIRType& value_type = expression_type(rhs_expression);
        scalar_half_store = value_type.basetype == SPIRType::Half && value_type.vecsize == 1 && value_type.columns == 1 && value_type.array.empty();
        vector_store = value_type.columns == 1 && value_type.array.empty() && (value_type.vecsize == 2 || value_type.vecsize == 3 || value_type.vecsize == 4);

        // OpenCL C does not permit taking the address of a vector element.
        // Leave those uncommon component stores to SPIRV-Cross.
        const SPIRType& block_type = get<SPIRType>(backing->basetype);
        if (scalar_half_store && block_type.basetype == SPIRType::Struct && block_type.member_types.size() == 1)
        {
            const SPIRType& storage_element = get<SPIRType>(block_type.member_types[0]);
            if (storage_element.vecsize > 1 || storage_element.columns > 1)
                scalar_half_store = false;
        }
    }

    const bool guarded_store = uses_workgroup_storage && resource_store;
    if (guarded_store)
    {
        statement("if (momo_active != 0)");
        begin_scope();
    }

    if (scalar_half_store)
    {
        // Some OpenCL 1.x CPU backends speculate native scalar-half stores
        // for inactive tail lanes and can generate invalid host accesses.
        // Store the Vulkan f16 bit pattern through ushort instead. This is
        // layout-identical and keeps rounding independent of the host's
        // native half-store implementation.
        const std::string lhs = to_dereferenced_expression(lhs_expression);
        const std::string rhs = to_expression(rhs_expression);
        statement("*((volatile __global ushort *)&(", lhs,
                  ")) = momo_float_to_half_scalar(convert_float(", rhs, "));");
        register_write(lhs_expression);
    }
    else if (vector_store)
    {
        // Scalarize descriptor-vector stores. Some CPU backends incorrectly
        // widen/speculate native vector stores (including vstoreN) for tail
        // elements even when the descriptor access is valid.
        const SPIRType& value_type = expression_type(rhs_expression);
        SPIRType element_type = value_type;
        element_type.vecsize = 1;
        const std::string storage_type = value_type.basetype == SPIRType::Half
                                             ? "ushort"
                                             : type_to_glsl(element_type);
        flush_variable_declaration(rhs_expression);
        const std::string lhs = to_dereferenced_expression(lhs_expression);
        const std::string rhs = to_expression(rhs_expression);
        static const char components[] = "xyzw";
        for (uint32_t component = 0; component < value_type.vecsize; component++)
        {
            std::string component_value = "(" + rhs + ")." + components[component];
            if (value_type.basetype == SPIRType::Half)
                component_value = "as_ushort(" + component_value + ")";
            statement("*((volatile __global ", storage_type, " *)&(", lhs,
                      ") + ", component, ") = ", component_value, ";");
        }
        register_write(lhs_expression);
    }
    else
    {
        CompilerGLSL::emit_store_statement(lhs_expression, rhs_expression);
    }

    if (guarded_store)
        end_scope();
}

void CompilerOpenCL::emit_glsl_op(uint32_t result_type, uint32_t result_id, uint32_t op,
                                  const uint32_t* args, uint32_t count)
{
    if (op == GLSLstd450SAbs && count == 1)
    {
        const SPIRType& type = get<SPIRType>(result_type);
        const std::string value = to_expression(args[0]);
        std::string expression;
        if (type.vecsize == 1)
            expression = "((" + value + ") < 0 ? -(" + value + ") : (" + value + "))";
        else
            expression = "select((" + value + "), -(" + value + "), (" + value + ") < (" + type_to_glsl(type) + ")(0))";
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
    std::string expression = CompilerGLSL::constant_expression_vector(constant, vector);
    const SPIRType& type = get<SPIRType>(constant.constant_type);
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

bool CompilerOpenCL::emit_robust_buffer_load(const Instruction& instruction, const uint32_t* ops)
{
    if (instruction.length < 3)
        return false;

    const uint32_t result_type = ops[0];
    const uint32_t result_id = ops[1];
    const uint32_t pointer_id = ops[2];
    SPIRVariable* backing = maybe_get_backing_variable(pointer_id);
    if (!backing)
        return false;
    const std::map<uint32_t, size_t>::const_iterator resource = resource_indices.find(backing->self);
    if (resource == resource_indices.end())
        return false;

    const SPIRType& type = get<SPIRType>(result_type);
    if (!type.array.empty())
        throw std::runtime_error("robust storage-buffer array loads are not supported");

    // OpenCL C does not permit taking the address of a vector element. Work
    // from the enclosing vector for component loads so 16-bit components can
    // use the same safe scalar load path as scalar descriptor elements.
    const SPIRType& block_type = get<SPIRType>(backing->basetype);
    if (block_type.basetype == SPIRType::Struct && block_type.member_types.size() == 1)
    {
        const SPIRType& storage_element = get<SPIRType>(block_type.member_types[0]);
        if ((storage_element.vecsize > 1 || storage_element.columns > 1) && type.vecsize == 1 && type.columns == 1)
        {
            // OpenCL C forbids taking the address of a vector component, but
            // emitting the component lvalue directly lets some CPU backends
            // speculate a widened vector access past a descriptor tail. Use
            // the enclosing vector address, then load just the requested
            // scalar component through the scalar storage representation.
            flush_variable_declaration(pointer_id);
            const std::string lvalue = to_dereferenced_expression(pointer_id, false);
            const size_t dot = lvalue.rfind('.');
            if (storage_element.columns == 1 && (storage_element.vecsize == 2 || storage_element.vecsize == 3 || storage_element.vecsize == 4) && dot != std::string::npos && dot + 2 == lvalue.size() && std::string("xyzw").find(lvalue[dot + 1]) < storage_element.vecsize)
            {
                SPIRType element_type = storage_element;
                element_type.vecsize = 1;
                const std::string storage_type = storage_element.basetype == SPIRType::Half
                                                     ? "ushort"
                                                     : type_to_glsl(element_type);
                const size_t component = std::string("xyzw").find(lvalue[dot + 1]);
                std::string expression;
                if (storage_element.width == 16)
                {
                    const std::string load_function = storage_element.basetype == SPIRType::Short || storage_element.basetype == SPIRType::Int
                                                          ? "momo_load_short"
                                                          : "momo_load_ushort";
                    const std::string index = std::to_string(resource->second);
                    expression = load_function + "((volatile __global const " + storage_type + " *)&(" + lvalue.substr(0, dot) + ") + " + std::to_string(component) + ", momo_buffer_" + index + " + momo_buffer_offset_" + index + ", momo_buffer_size_" + index + ")";
                }
                else
                {
                    expression = "*((volatile __global const " + storage_type + " *)&(" + lvalue.substr(0, dot) + ") + " + std::to_string(component) + ")";
                }
                if (storage_element.basetype == SPIRType::Half)
                    expression = "as_half(" + expression + ")";
                cast_from_variable_load(pointer_id, expression, type);
                emit_op(result_type, result_id, expression, false);
                register_read(result_id, pointer_id, false);
                inherit_expression_dependencies(result_id, pointer_id);
                return true;
            }
            return false;
        }
    }

    // momoten does not advertise robustBufferAccess, so valid Vulkan shaders
    // do not require descriptor loads to be bounds checked here. Address-based
    // conditional loads are also miscompiled by multiple OpenCL 1.x backends,
    // including Rusticl/llvmpipe, for both scalar and vector storage types.
    // Keep only representation/code-generation workarounds below and let
    // SPIRV-Cross emit ordinary valid scalar loads directly.
    if (type.vecsize == 1 && type.columns == 1)
    {
        if (type.width != 16 || (type.basetype != SPIRType::Half && type.basetype != SPIRType::UShort && type.basetype != SPIRType::Short && type.basetype != SPIRType::UInt && type.basetype != SPIRType::Int))
            return false;

        flush_variable_declaration(pointer_id);
        const std::string lvalue = to_dereferenced_expression(pointer_id, false);
        const bool signed_type = type.basetype == SPIRType::Short || type.basetype == SPIRType::Int;
        const std::string storage_type = signed_type ? "short" : "ushort";
        const std::string load_function = signed_type
                                              ? "momo_load_short"
                                              : "momo_load_ushort";
        const std::string index = std::to_string(resource->second);
        std::string expression = load_function + "((volatile __global const " + storage_type + " *)&(" + lvalue + "), " + "momo_buffer_" + index + " + momo_buffer_offset_" + index + ", " + "momo_buffer_size_" + index + ")";
        if (type.basetype == SPIRType::Half)
            expression = "as_half(" + expression + ")";
        cast_from_variable_load(pointer_id, expression, type);
        emit_op(result_type, result_id, expression, false);
        register_read(result_id, pointer_id, false);
        inherit_expression_dependencies(result_id, pointer_id);
        return true;
    }

    if (type.columns == 1 && (type.vecsize == 2 || type.vecsize == 3 || type.vecsize == 4))
    {
        flush_variable_declaration(pointer_id);
        const std::string lvalue = to_dereferenced_expression(pointer_id, false);
        SPIRType element_type = type;
        element_type.vecsize = 1;
        const std::string storage_type = type.basetype == SPIRType::Half
                                             ? "ushort"
                                             : type_to_glsl(element_type);
        std::string expression = type_to_glsl(type) + "(";
        for (uint32_t component = 0; component < type.vecsize; component++)
        {
            if (component != 0)
                expression += ", ";
            std::string component_load;
            if (type.width == 16)
            {
                const bool signed_type = type.basetype == SPIRType::Short || type.basetype == SPIRType::Int;
                const std::string load_function = signed_type
                                                      ? "momo_load_short"
                                                      : "momo_load_ushort";
                const std::string index = std::to_string(resource->second);
                component_load = load_function + "((volatile __global const " + storage_type + " *)&(" + lvalue + ") + " + std::to_string(component) + ", momo_buffer_" + index + " + momo_buffer_offset_" + index + ", momo_buffer_size_" + index + ")";
            }
            else
            {
                component_load = "*((volatile __global const " + storage_type + " *)&(" + lvalue + ") + " + std::to_string(component) + ")";
            }
            if (type.basetype == SPIRType::Half)
                component_load = "as_half(" + component_load + ")";
            expression += component_load;
        }
        expression += ")";
        cast_from_variable_load(pointer_id, expression, type);
        emit_op(result_type, result_id, expression, false);
        register_read(result_id, pointer_id, false);
        inherit_expression_dependencies(result_id, pointer_id);
        return true;
    }
    return false;
}

const char* CompilerOpenCL::function_pointer_address_space(StorageClass storage)
{
    switch (storage)
    {
    case StorageClassFunction:
    case StorageClassPrivate:
        return "__private";
    case StorageClassWorkgroup:
        return "__local";
    case StorageClassUniform:
    case StorageClassStorageBuffer:
    case StorageClassPushConstant:
    case StorageClassCrossWorkgroup:
        return "__global";
    default:
        throw std::runtime_error("unsupported function-pointer storage class in the OpenCL C profile");
    }
}

void CompilerOpenCL::emit_struct_typedef(const SPIRType& type)
{
    if (emitted_struct_typedefs.insert(type.self).second)
    {
        const std::string name = type_to_glsl(type);
        statement("typedef struct ", name, " ", name, ";");
    }
}

} // namespace momoten
