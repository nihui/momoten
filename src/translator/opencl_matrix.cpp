// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <stdexcept>

namespace momoten {

bool CompilerOpenCL::access_chain_contains_matrix(
    uint32_t base, const uint32_t* indices, uint32_t count)
{
    const SPIRType* type = &get_pointee_type(expression_type(base));
    for (uint32_t i = 0; i < count; i++)
    {
        if (type->basetype == SPIRType::Struct)
        {
            const uint32_t member = evaluate_constant_u32(indices[i]);
            if (member >= type->member_types.size())
                return false;
            type = &get<SPIRType>(type->member_types[member]);
        }
        else if (!type->array.empty())
        {
            type = &get<SPIRType>(type->parent_type);
        }
        else if (type->columns > 1)
        {
            return true;
        }
        else if (type->vecsize > 1)
        {
            type = &get<SPIRType>(type->parent_type);
        }
        else
        {
            return false;
        }
    }
    return false;
}

bool CompilerOpenCL::emit_matrix_access_chain(
    const Instruction& instruction, const uint32_t* ops)
{
    if (instruction.length <= 3)
        return false;
    const uint32_t result_type = ops[0];
    const uint32_t result_id = ops[1];
    const uint32_t base = ops[2];
    const uint32_t* indices = ops + 3;
    const uint32_t count = instruction.length - 3;
    if (!access_chain_contains_matrix(base, indices, count))
        return false;

    SPIRVariable* variable = maybe_get<SPIRVariable>(base);
    if (variable)
        flush_variable_declaration(variable->self);

    std::string expression = to_enclosed_expression(base);
    const SPIRType* type = &get_pointee_type(expression_type(base));
    for (uint32_t i = 0; i < count; i++)
    {
        const std::string index = to_unpacked_expression(indices[i]);
        if (type->basetype == SPIRType::Struct)
        {
            const uint32_t member = evaluate_constant_u32(indices[i]);
            if (member >= type->member_types.size())
                throw std::runtime_error("matrix access-chain struct index is out of bounds");
            if (storage_block_types.count(type->self) == 0)
                expression += to_member_reference(base, *type, member, i != 0);
            type = &get<SPIRType>(type->member_types[member]);
        }
        else if (!type->array.empty())
        {
            expression += "[" + index + "]";
            type = &get<SPIRType>(type->parent_type);
        }
        else if (type->columns > 1)
        {
            expression += ".c[" + index + "]";
            type = &get<SPIRType>(type->parent_type);
        }
        else if (type->vecsize > 1)
        {
            expression += "[" + index + "]";
            type = &get<SPIRType>(type->parent_type);
        }
        else
        {
            throw std::runtime_error("unsupported scalar tail in matrix access chain");
        }
    }

    SPIRExpression& result = set<SPIRExpression>(
        result_id, expression, result_type, should_forward(base));
    SPIRVariable* backing = maybe_get_backing_variable(base);
    result.loaded_from = backing ? backing->self : ID(base);
    result.access_chain = true;
    inherit_expression_dependencies(result_id, base);
    add_implied_read_expression(result, base);
    for (uint32_t i = 0; i < count; i++)
    {
        inherit_expression_dependencies(result_id, indices[i]);
        add_implied_read_expression(result, indices[i]);
    }
    return true;
}

} // namespace momoten
