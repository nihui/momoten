// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <stdexcept>

namespace momoten {

bool CompilerOpenCL::emit_memory_instruction(const Instruction& instruction, const uint32_t* ops)
{
    const Op opcode = static_cast<Op>(instruction.op);
    if (opcode == OpAccessChain || opcode == OpInBoundsAccessChain)
        track_private_vector_access(instruction, ops);
    if (opcode == OpAccessChain || opcode == OpInBoundsAccessChain || opcode == OpCopyObject)
        track_aliased_buffer_access(instruction, ops);
    if ((opcode == OpLoad || opcode == OpStore) && emit_private_vector_access(instruction, ops))
        return true;

    if ((opcode == OpLoad || opcode == OpStore) && emit_aliased_buffer_access(instruction, ops))
        return true;

    if (opcode == OpLoad && (emit_coalesced_buffer_load(instruction, ops) || emit_storage_buffer_load(instruction, ops)))
        return true;

    if (opcode == OpStore && emit_storage_buffer_store(instruction, ops))
        return true;

    return false;
}

void CompilerOpenCL::require_statement_continue()
{
    // SPIRV-Cross normally joins a continue block into the for-loop update
    // expression. Declarations and control flow must instead stay in the body,
    // including on branches that reach it through an explicit continue.
    if (current_continue_block && !current_continue_block->complex_continue)
    {
        get<SPIRBlock>(current_continue_block->self).complex_continue = true;
        force_recompile();
    }
}

std::string CompilerOpenCL::buffer_scalar_type(const SPIRType& type)
{
    if (type.basetype == SPIRType::Double && type.width == 64)
        return "ulong";
    SPIRType scalar_type = type;
    scalar_type.vecsize = 1;
    return type_to_glsl(scalar_type);
}

void CompilerOpenCL::emit_buffer_load_result(uint32_t result_type, uint32_t result_id, uint32_t pointer, const std::string& expression)
{
    emit_op(result_type, result_id, expression, false);
    register_read(result_id, pointer, false);
    inherit_expression_dependencies(result_id, pointer);
}

bool CompilerOpenCL::prepare_access_chain_for_scalar_access(std::string& expression, const SPIRType& scalar_type,
                                                            StorageClass storage, bool& packed)
{
    // Older OpenCL compilers reject vector subscripting. Index the contiguous
    // scalar components through a pointer in the original address space.
    const OpenCLAddressSpace address_space = opencl_address_space(storage);
    const char* qualifier = opencl_address_space_qualifier(address_space);
    if (!qualifier)
        return false;
    const bool global_storage = address_space == OpenCLAddressSpace::Global;
    // OpenCL has no bool vectors: bvecN is represented by intN, so its
    // component stride is sizeof(int), not sizeof(bool).
    const std::string scalar_name = scalar_type.basetype == SPIRType::Boolean
                                        ? "int"
                                    : global_storage
                                        ? buffer_scalar_type(scalar_type)
                                        : type_to_glsl(scalar_type);
    // Preserve scalar accesses at descriptor tails instead of allowing a
    // compiler to speculate a wider vector load/store.
    const std::string qualifiers = (global_storage ? "volatile " : "") + std::string(qualifier);
    expression = "((" + qualifiers + " " + scalar_name + " *)&(" + expression + "))";
    packed = true;
    return true;
}

void CompilerOpenCL::cast_from_variable_load(uint32_t source_id, std::string& expression, const SPIRType& type)
{
    CompilerGLSL::cast_from_variable_load(source_id, expression, type);
    if (type.basetype == SPIRType::Boolean && type.array.empty() && type.columns == 1 && type.vecsize == 1)
    {
        // Vector comparisons use 0/-1 masks, while scalar bool uses 0/1.
        // Normalize even forwarded loads before scalar logical comparisons.
        expression = "(" + expression + " != 0)";
    }
}

void CompilerOpenCL::track_private_vector_access(const Instruction& instruction, const uint32_t* ops)
{
    if (instruction.length != 4)
        return;
    const SPIRVariable* variable = maybe_get<SPIRVariable>(ops[2]);
    if (!variable || variable->storage != StorageClassFunction || variable->parameter)
        return;
    const SPIRType& type = expression_type(ops[2]);
    if (!type.array.empty() || type.columns != 1 || type.vecsize < 2 || type.vecsize > 4
        || type.basetype == SPIRType::Double
        || has_decoration(ops[2], DecorationVolatile) || has_decoration(ops[2], DecorationCoherent)
        || has_decoration(ops[1], DecorationVolatile) || has_decoration(ops[1], DecorationCoherent))
        return;
    PrivateVectorAccess access = {ops[2], ops[3]};
    private_vector_accesses[ops[1]] = access;
}

bool CompilerOpenCL::emit_private_vector_access(const Instruction& instruction, const uint32_t* ops)
{
    const bool load = instruction.op == OpLoad;
    const uint32_t pointer = ops[load ? 2 : 0];
    const auto found = private_vector_accesses.find(pointer);
    if (found == private_vector_accesses.end())
        return false;
    const uint32_t memory_operand = load ? 3 : 2;
    if (instruction.length > memory_operand && ops[memory_operand] != MemoryAccessMaskNone)
        return false;

    const PrivateVectorAccess& access = found->second;
    const SPIRType& vector_type = expression_type(access.variable);
    const SPIRConstant* constant = maybe_get<SPIRConstant>(access.index);
    const bool fixed = constant && !constant->specialization;
    if (fixed && constant->scalar() >= vector_type.vecsize)
        return false;

    flush_variable_declaration(pointer);
    // Keep the original pointer expression and its dependency tracking for any
    // escaping uses. Only direct loads/stores become scalar lane operations.
    to_expression(pointer, !load);
    const std::string vector = "(" + to_expression(access.variable, false) + ")";
    const std::string index = "(" + to_expression(access.index, false) + ")";
    static const char components[] = "xyzw";
    if (load)
    {
        std::string expression = vector + "." + components[fixed ? constant->scalar() : vector_type.vecsize - 1];
        if (!fixed)
        {
            for (uint32_t i = vector_type.vecsize - 1; i > 0; i--)
                expression = "(" + index + " == " + std::to_string(i - 1) + "u ? " + vector + "." + components[i - 1] + " : " + expression + ")";
            // Conditional expressions promote narrow integer scalar operands.
            expression = "((" + type_to_glsl(get<SPIRType>(ops[0])) + ")(" + expression + "))";
        }
        cast_from_variable_load(pointer, expression, get<SPIRType>(ops[0]));
        const bool forward = should_forward(pointer) && forced_temporaries.count(ops[1]) == 0;
        SPIRExpression& result = emit_op(ops[0], ops[1], expression, forward);
        register_read(ops[1], pointer, forward);
        inherit_expression_dependencies(ops[1], pointer);
        if (forward)
            add_implied_read_expression(result, pointer);
    }
    else
    {
        flush_variable_declaration(ops[1]);
        const std::string value = to_unpacked_expression(ops[1]);
        if (fixed)
            statement(vector, ".", components[constant->scalar()], " = ", value, ";");
        else
        {
            require_statement_continue();
            // A conditional component store neither takes the vector's address
            // nor reads its other lanes, which may still be uninitialized.
            for (uint32_t i = 0; i < vector_type.vecsize; i++)
            {
                statement(i == 0 ? "if (" : "else if (", index, " == ", i, "u)");
                begin_scope();
                statement(vector, ".", components[i], " = ", value, ";");
                end_scope();
            }
        }
        register_write(pointer);
    }
    return true;
}

bool CompilerOpenCL::emit_coalesced_buffer_load(const Instruction& instruction, const uint32_t* ops)
{
    if (instruction.length < 3)
        return false;
    const auto found = coalesced_buffer_loads.find(ops[1]);
    if (found == coalesced_buffer_loads.end())
        return false;

    const CoalescedBufferLoad& load = found->second;
    const SPIRType& type = get<SPIRType>(ops[0]);
    const std::string temporary = "momo_buffer_load_" + std::to_string(load.first_result);
    if (load.first_result == ops[1])
    {
        require_statement_continue();
        flush_variable_declaration(ops[2]);
        const std::string lvalue = to_dereferenced_expression(ops[2], false);
        // Float reads retain their access type. Packed half reads use a
        // probed may_alias type so half stores invalidate the wide load.
        if (type.basetype == SPIRType::Float)
            statement("uint4 ", temporary, " = as_uint4(vload4(0, (__global const float *)&(", lvalue, ")));");
        else if (type.basetype == SPIRType::Half)
            statement("uint4 ", temporary, " = *((__global const momo_alias_uint4 *)&(", lvalue, "));");
        else
            statement("uint4 ", temporary, " = vload4(0, (__global const uint *)&(", lvalue, "));");
    }

    const uint32_t words = type.width * type.vecsize / 32;
    std::string expression = temporary + ".s";
    for (uint32_t i = 0; i < words; i++)
        expression += static_cast<char>('0' + load.word_offset + i);
    if (type.basetype != SPIRType::UInt)
        expression = "as_" + type_to_glsl(type) + "(" + expression + ")";
    cast_from_variable_load(ops[2], expression, type);
    // Materialize each original result at its original instruction. The wide
    // temporary snapshots the whole proven-safe group at the first load.
    emit_buffer_load_result(ops[0], ops[1], ops[2], expression);
    return true;
}

bool CompilerOpenCL::emit_storage_buffer_load(const Instruction& instruction, const uint32_t* ops)
{
    if (instruction.length < 3)
        return false;

    const uint32_t result_type = ops[0];
    const uint32_t result_id = ops[1];
    const uint32_t pointer_id = ops[2];
    SPIRVariable* backing = maybe_get_backing_variable(pointer_id);
    if (!backing)
        return false;
    if (resource_indices.find(backing->self) == resource_indices.end())
        return false;

    const SPIRType& type = get<SPIRType>(result_type);
    if (!type.array.empty())
        throw std::runtime_error("storage-buffer array loads are not supported");

    // OpenCL C does not permit taking the address of a vector element. Work
    // from the enclosing vector and load only the requested scalar component.
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
                const std::string storage_type = buffer_scalar_type(storage_element);
                const size_t component = std::string("xyzw").find(lvalue[dot + 1]);
                std::string expression = "*((volatile __global const " + storage_type + " *)&(" + lvalue.substr(0, dot) + ") + " + std::to_string(component) + ")";
                if (storage_element.basetype == SPIRType::Double && storage_element.width == 64)
                    expression = "as_double(" + expression + ")";
                cast_from_variable_load(pointer_id, expression, type);
                emit_buffer_load_result(result_type, result_id, pointer_id, expression);
                return true;
            }
            // Pointer-based scalar access chains already retain their address
            // space. Fall through so fp64 storage still gets its bitcast.
        }
    }

    // Let SPIRV-Cross emit ordinary scalar loads. Only fp64 needs a storage
    // representation conversion; half, short and ushort use native types.
    if (type.vecsize == 1 && type.columns == 1)
    {
        if (type.basetype == SPIRType::Double && type.width == 64)
        {
            flush_variable_declaration(pointer_id);
            std::string expression = "as_double(" + to_dereferenced_expression(pointer_id, false) + ")";
            cast_from_variable_load(pointer_id, expression, type);
            emit_buffer_load_result(result_type, result_id, pointer_id, expression);
            return true;
        }
        return false;
    }

    if (type.columns == 1 && (type.vecsize == 2 || type.vecsize == 3 || type.vecsize == 4))
    {
        flush_variable_declaration(pointer_id);
        const std::string lvalue = to_dereferenced_expression(pointer_id, false);
        const std::string storage_type = buffer_scalar_type(type);
        const Bitset flags = get_buffer_block_flags(backing->self);
        const bool volatile_load = flags.get(DecorationVolatile) || (instruction.length > 3 && (ops[3] & MemoryAccessVolatileMask) != 0);
        if (translation_options.native_buffer_vector_loads && !volatile_load
            && type.basetype != SPIRType::Double && type.basetype != SPIRType::Boolean
            && (type.vecsize == 2 || type.vecsize == 4))
        {
            // vloadN requires only scalar alignment and reads the complete
            // SPIR-V vector. Partial-vector accesses keep the scalar path.
            std::string expression = "vload" + std::to_string(type.vecsize) + "(0, (__global const " + storage_type + " *)&(" + lvalue + "))";
            // Dereference the may_alias typedef directly: passing it to
            // vloadN would discard the attribute in the builtin parameter.
            if (translation_options.packed_fp16_vector_loads && type.basetype == SPIRType::Half && type.vecsize == 4)
                expression = "as_half4(*((__global const momo_alias_uint2 *)&(" + lvalue + ")))";
            cast_from_variable_load(pointer_id, expression, type);
            emit_buffer_load_result(result_type, result_id, pointer_id, expression);
            return true;
        }
        std::string expression = type.basetype == SPIRType::Double && type.width == 64
                                     ? "momo_make_double" + std::to_string(type.vecsize) + "("
                                     : type_to_glsl(type) + "(";
        for (uint32_t component = 0; component < type.vecsize; component++)
        {
            if (component != 0)
                expression += ", ";
            std::string component_load = "*((volatile __global const " + storage_type + " *)&(" + lvalue + ") + " + std::to_string(component) + ")";
            if (type.basetype == SPIRType::Double && type.width == 64)
                component_load = "as_double(" + component_load + ")";
            expression += component_load;
        }
        expression += ")";
        cast_from_variable_load(pointer_id, expression, type);
        emit_buffer_load_result(result_type, result_id, pointer_id, expression);
        return true;
    }
    return false;
}

bool CompilerOpenCL::emit_storage_buffer_store(const Instruction& instruction, const uint32_t* ops)
{
    if (!translation_options.native_buffer_vector_stores)
        return false;
    SPIRVariable* backing = maybe_get_backing_variable(ops[0]);
    if (!backing || resource_indices.find(backing->self) == resource_indices.end())
        return false;

    const SPIRType& value_type = expression_type(ops[1]);
    if (value_type.columns != 1 || !value_type.array.empty()
        || (value_type.vecsize != 2 && value_type.vecsize != 4)
        || value_type.basetype == SPIRType::Double || value_type.basetype == SPIRType::Boolean)
        return false;
    const Bitset flags = get_buffer_block_flags(backing->self);
    if (flags.get(DecorationVolatile) || (instruction.length > 2 && (ops[2] & MemoryAccessVolatileMask) != 0))
        return false;

    flush_variable_declaration(ops[0]);
    flush_variable_declaration(ops[1]);
    const std::string lhs = to_dereferenced_expression(ops[0]);
    const std::string rhs = to_expression(ops[1]);
    if (uses_workgroup_storage)
    {
        require_statement_continue();
        statement("if (momo_active != 0)");
        begin_scope();
    }
    // vstoreN requires only scalar alignment. Scalar component stores and
    // vec3 tails continue through the conservative volatile lowering.
    statement("vstore", value_type.vecsize, "(", rhs, ", 0, (__global ", buffer_scalar_type(value_type), " *)&(", lhs, "));");
    register_write(ops[0]);
    if (uses_workgroup_storage)
        end_scope();
    return true;
}

void CompilerOpenCL::emit_store_statement(uint32_t lhs_expression, uint32_t rhs_expression)
{
    SPIRVariable* backing = maybe_get_backing_variable(lhs_expression);
    const bool resource_store = backing && resource_indices.find(backing->self) != resource_indices.end();

    bool scalar_double_store = false;
    bool vector_store = false;
    if (resource_store)
    {
        const SPIRType& value_type = expression_type(rhs_expression);
        scalar_double_store = value_type.basetype == SPIRType::Double && value_type.width == 64 && value_type.vecsize == 1 && value_type.columns == 1 && value_type.array.empty();
        vector_store = value_type.columns == 1 && value_type.array.empty() && (value_type.vecsize == 2 || value_type.vecsize == 3 || value_type.vecsize == 4);
    }

    const bool guarded_store = uses_workgroup_storage && resource_store;
    if (guarded_store)
    {
        require_statement_continue();
        statement("if (momo_active != 0)");
        begin_scope();
    }

    if (scalar_double_store)
    {
        // Treat fp64 descriptor storage as its exact 64-bit representation.
        // This avoids address-space casts from the byte-oriented kernel ABI
        // to double pointers while preserving the Vulkan buffer bit layout.
        const std::string lhs = to_dereferenced_expression(lhs_expression);
        const std::string rhs = to_expression(rhs_expression);
        statement(lhs, " = as_ulong(", rhs, ");");
        register_write(lhs_expression);
    }
    else if (vector_store)
    {
        // Scalarize descriptor-vector stores. Some CPU backends incorrectly
        // widen/speculate native vector stores (including vstoreN) for tail
        // elements even when the descriptor access is valid.
        const SPIRType& value_type = expression_type(rhs_expression);
        const std::string storage_type = buffer_scalar_type(value_type);
        flush_variable_declaration(rhs_expression);
        const std::string lhs = to_dereferenced_expression(lhs_expression);
        const std::string rhs = to_expression(rhs_expression);
        static const char components[] = "xyzw";
        for (uint32_t component = 0; component < value_type.vecsize; component++)
        {
            std::string component_value = "(" + rhs + ")." + components[component];
            if (value_type.basetype == SPIRType::Double && value_type.width == 64)
                component_value = "as_ulong(" + component_value + ")";
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

} // namespace momoten
