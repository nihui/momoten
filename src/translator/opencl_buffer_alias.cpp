// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <stdexcept>

namespace momoten {

void CompilerOpenCL::track_aliased_buffer_access(const Instruction& instruction, const uint32_t* ops)
{
    if (!get<SPIRType>(ops[0]).pointer)
        return;
    const SPIRVariable* backing = maybe_get_backing_variable(ops[2]);
    if (!backing || !resource_indices.count(backing->self) || !has_decoration(backing->self, DecorationAliased))
        return;
    BufferAccessChain& access = buffer_access_chains[ops[1]];
    access.base = ops[2];
    access.indices.assign(ops + 3, ops + instruction.length);
}

std::string CompilerOpenCL::aliased_buffer_address(uint32_t pointer)
{
    const auto found = buffer_access_chains.find(pointer);
    if (found == buffer_access_chains.end())
    {
        const auto resource = resource_indices.find(pointer);
        if (resource == resource_indices.end())
            throw std::runtime_error("unsupported aliased storage-buffer pointer");
        const std::string index = std::to_string(resource->second);
        return "momo_buffer_" + index + " + momo_buffer_offset_" + index;
    }

    const BufferAccessChain& access = found->second;
    std::string address = aliased_buffer_address(access.base);
    uint32_t type_id = expression_type(access.base).parent_type;
    for (uint32_t index : access.indices)
    {
        const SPIRType* type = &get<SPIRType>(type_id);
        uint32_t stride;
        if (!type->array.empty())
        {
            stride = get_decoration(type_id, DecorationArrayStride);
        }
        else if (type->basetype == SPIRType::Struct)
        {
            const uint32_t member = evaluate_constant_u32(index);
            if (member >= type->member_types.size())
                throw std::runtime_error("aliased buffer member index is out of bounds");
            address += " + " + std::to_string(get_member_decoration(type->self, member, DecorationOffset)) + "u";
            type_id = type->member_types[member];
            continue;
        }
        else
        {
            // Validated buffer layouts are tightly packed scalar/vector
            // arrays or column-major mat4 arrays. Preserve the byte address
            // structurally, without forming an address of a vector lane.
            stride = type->width / 8 * (type->columns > 1 ? type->vecsize : 1);
        }
        const char* offset_type = translation_options.address_bits == 64 ? "ulong" : "uint";
        address += " + ((" + std::string(offset_type) + ")(" + to_unpacked_expression(index) + ") * " + std::to_string(stride) + "u)";
        type_id = type->parent_type;
    }
    return address;
}

std::string CompilerOpenCL::aliased_scalar_load(const SPIRType& type, const std::string& address, uint32_t offset)
{
    const uint32_t bytes = type.width / 8;
    std::string expression = "as_" + type_to_glsl(type) + "((uchar" + std::to_string(bytes) + ")(";
    for (uint32_t i = 0; i < bytes; i++)
    {
        if (i)
            expression += ", ";
        expression += "((volatile __global const uchar *)(" + address + "))[" + std::to_string(offset + i) + "]";
    }
    return expression + "))";
}

void CompilerOpenCL::emit_aliased_scalar_store(const SPIRType& type, const std::string& address, uint32_t offset,
                                               const std::string& value, const std::string& temporary)
{
    const uint32_t unit_bytes = type.width == 16 ? 1 : 4;
    const uint32_t count = type.width / 8 / unit_bytes;
    const std::string unit_type = unit_bytes == 1 ? "uchar" : "uint";
    const std::string packed_type = unit_type + (count == 1 ? "" : std::to_string(count));
    statement(packed_type, " ", temporary, " = as_", packed_type, "(", value, ");");
    for (uint32_t i = 0; i < count; i++)
        statement("((volatile __global ", unit_type, " *)(", address, " + ", offset, "u))[", i, "] = ", temporary,
                  count == 1 ? "" : ".s" + std::to_string(i), ";");
}

bool CompilerOpenCL::emit_aliased_buffer_access(const Instruction& instruction, const uint32_t* ops)
{
    const bool load = instruction.op == OpLoad;
    const uint32_t pointer = ops[load ? 2 : 0];
    const SPIRVariable* backing = maybe_get_backing_variable(pointer);
    if (!backing || resource_indices.count(backing->self) == 0 || !has_decoration(backing->self, DecorationAliased))
        return false;
    const SPIRType& type = load ? get<SPIRType>(ops[0]) : expression_type(ops[1]);
    if (!type.array.empty())
        throw std::runtime_error("aliased storage-buffer array accesses are not supported");
    if (type.pointer || (type.width != 16 && type.width != 32 && type.width != 64) || type.vecsize > 4)
        throw std::runtime_error("aliased storage-buffer access has an unsupported value type");

    if (!load && type.width == 16 && !translation_options.byte_addressable_store)
        throw std::runtime_error("16-bit aliased buffer stores require OpenCL C 1.1 or cl_khr_byte_addressable_store");

    require_statement_continue();
    flush_variable_declaration(pointer);
    // Retain SPIRV-Cross's pointer dependency and write tracking even though
    // the memory operation uses the original access chain's byte address.
    to_expression(pointer, !load);
    const std::string address = aliased_buffer_address(pointer);

    // Aliased descriptors may view the same bytes through unrelated SPIR-V
    // types; SPIR-V requires every overlapping declaration to be Aliased.
    // Character reads alias every store type. All stores of at least
    // 32 bits use the same uint representation, which also works in OpenCL
    // C 1.0 without byte-addressable stores. Narrow stores require that feature.
    // Never touch neighboring lanes or descriptor tails.
    const std::string temporary = "momo_aliased_" + std::to_string(ops[1]);
    if (load)
        statement(type_to_glsl(type), " ", temporary, ";");
    else
    {
        flush_variable_declaration(ops[1]);
        const std::string value = to_expression(ops[1]);
        if (uses_workgroup_storage)
            statement("if (momo_active != 0)");
        begin_scope();
        statement(type_to_glsl(type), " ", temporary, " = ", value, ";");
    }
    SPIRType scalar_type = type;
    scalar_type.vecsize = scalar_type.columns = 1;
    const uint32_t bytes = type.width / 8;
    static const char components[] = "xyzw";
    for (uint32_t column = 0; column < type.columns; column++)
        for (uint32_t lane = 0; lane < type.vecsize; lane++)
        {
            std::string scalar = temporary;
            if (type.columns > 1)
                scalar += ".c[" + std::to_string(column) + "]";
            if (type.vecsize > 1)
                scalar += "." + std::string(1, components[lane]);
            const uint32_t offset = (column * type.vecsize + lane) * bytes;
            if (load)
                statement(scalar, " = ", aliased_scalar_load(scalar_type, address, offset), ";");
            else
            {
                const std::string packed = temporary + "_bits_" + std::to_string(column) + "_" + std::to_string(lane);
                emit_aliased_scalar_store(scalar_type, address, offset, scalar, packed);
            }
        }
    if (load)
    {
        emit_buffer_load_result(ops[0], ops[1], pointer, temporary);
    }
    else
    {
        end_scope();
        register_write(pointer);
    }
    return true;
}

} // namespace momoten
