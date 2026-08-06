// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace momoten {

bool resource_less(const ResourceInfo& a, const ResourceInfo& b)
{
    if (a.descriptor_set != b.descriptor_set)
        return a.descriptor_set < b.descriptor_set;
    if (a.binding != b.binding)
        return a.binding < b.binding;
    return a.variable_id < b.variable_id;
}

void CompilerOpenCL::emit_buffer_block(const SPIRVariable& var)
{
    const std::map<uint32_t, size_t>::const_iterator it = resource_indices.find(var.self);
    if (it == resource_indices.end())
        throw std::runtime_error("internal error: storage buffer is missing from kernel ABI");

    add_resource_name(var.self);
    SPIRType& type = get<SPIRType>(var.basetype);

    const std::string instance_name = to_name(var.self);
    const SPIRType& element_type = get<SPIRType>(type.member_types[0]);
    const std::string type_name = type_to_glsl(element_type);
    const size_t index = it->second;
    statement_no_indent("#define ", instance_name,
                        " ((__global ", type_name, " *)(momo_buffer_", index,
                        " + momo_buffer_offset_", index, "))");
    statement("");
}

void CompilerOpenCL::emit_push_constant_block(const SPIRVariable& var)
{
    if (push_constant_variable_id == 0 || var.self != push_constant_variable_id)
        throw std::runtime_error("internal error: unexpected push-constant block");

    add_resource_name(var.self);
    SPIRType& type = get<SPIRType>(var.basetype);
    emit_struct_typedef(type);
    emit_struct(type);

    const std::string instance_name = to_name(var.self);
    const std::string type_name = type_to_glsl(type);
    statement_no_indent("#define ", instance_name,
                        " (*((__global const struct ", type_name, " *)momo_push_constants))");
    statement("");
}

void CompilerOpenCL::emit_uniform(const SPIRVariable&)
{
    throw std::runtime_error("plain uniforms, images and samplers are not supported by the OpenCL 1.0 profile");
}

void CompilerOpenCL::emit_function_prototype(SPIRFunction& func, const Bitset& return_flags)
{
    if (func.self != ir.default_entry_point)
    {
        add_function_overload(func);
        local_variable_names = resource_names;

        const SPIRType& return_type = get<SPIRType>(func.return_type);
        std::ostringstream declaration;
        declaration << flags_to_qualifiers_glsl(return_type, 0, return_flags)
                    << type_to_glsl(return_type) << type_to_array_glsl(return_type, 0)
                    << " " << to_name(func.self) << "(";

        bool first_argument = true;
        for (size_t i = 0; i < func.arguments.size() + func.shadow_arguments.size(); i++)
        {
            SPIRFunction::Parameter& argument = i < func.arguments.size()
                                                    ? func.arguments[i]
                                                    : func.shadow_arguments[i - func.arguments.size()];
            if (skip_argument(argument.id))
                continue;

            if (!first_argument)
                declaration << ", ";
            first_argument = false;

            add_local_variable_name(argument.id);
            const SPIRType& argument_type = expression_type(argument.id);
            if (is_pointer(argument_type))
            {
                if (argument_type.parent_type == 0)
                    throw std::runtime_error("function pointer parameter has no pointee type");
                const SPIRType& value_type = get<SPIRType>(argument_type.parent_type);
                if (!value_type.array.empty() || value_type.basetype == SPIRType::Struct)
                    throw std::runtime_error("array and struct function-pointer parameters are deferred in the OpenCL C profile");

                const char* address_space = function_pointer_address_space(argument_type.storage);
                const std::string pointer_name = "momo_param_" + std::to_string(argument.id);
                declaration << address_space << " ";
                if (argument.write_count == 0)
                    declaration << "const ";
                declaration << type_to_glsl(value_type) << "* " << pointer_name;

                // SPIRV-Cross's GLSL emitter models Function pointers as
                // in/out value parameters. OpenCL C requires a real private
                // pointer, so keep the body lvalue semantics by remapping
                // every reference to an explicit dereference.
                set_name(argument.id, "(*" + pointer_name + ")");
                local_variable_names.insert(pointer_name);
            }
            else
            {
                declaration << argument_decl(argument);
            }

            SPIRVariable* variable = maybe_get<SPIRVariable>(argument.id);
            if (variable)
                variable->parameter = &argument;
        }

        // Descriptor and push-constant globals are represented by OpenCL
        // kernel arguments. Pass the same ABI values through every retained
        // helper so resource macros remain valid when SPIR-V helpers cannot
        // be inlined (for example, early-return helpers in barrier kernels).
        const char* offset_type = translation_options.address_bits == 64 ? "ulong" : "uint";
        for (size_t i = 0; i < resources.size(); i++)
        {
            if (!first_argument)
                declaration << ", ";
            first_argument = false;
            declaration << "__global uchar* momo_buffer_" << i
                        << ", " << offset_type << " momo_buffer_offset_" << i
                        << ", " << offset_type << " momo_buffer_size_" << i;
        }
        if (push_constant_variable_id != 0)
        {
            if (!first_argument)
                declaration << ", ";
            declaration << "__global const uchar* momo_push_constants";
        }

        declaration << ")";
        statement(declaration.str());
        return;
    }

    if (!func.arguments.empty() || !func.shadow_arguments.empty())
        throw std::runtime_error("compute entry points with explicit SPIR-V function parameters are unsupported");

    local_variable_names = resource_names;

    std::vector<std::string> args;
    const char* offset_type = translation_options.address_bits == 64 ? "ulong" : "uint";
    for (size_t i = 0; i < resources.size(); i++)
    {
        std::ostringstream buffer_arg;
        buffer_arg << "__global uchar* momo_buffer_" << i;
        args.push_back(buffer_arg.str());

        std::ostringstream offset_arg;
        offset_arg << offset_type << " momo_buffer_offset_" << i;
        args.push_back(offset_arg.str());

        std::ostringstream size_arg;
        size_arg << offset_type << " momo_buffer_size_" << i;
        args.push_back(size_arg.str());
    }
    if (push_constant_variable_id != 0)
        args.push_back("__global const uchar* momo_push_constants");

    std::ostringstream declaration;
    declaration << "__kernel __attribute__((reqd_work_group_size(" << local_size[0] << ", "
                << local_size[1] << ", " << local_size[2] << "))) void "
                << get_entry_point().name << "(";
    for (size_t i = 0; i < args.size(); i++)
    {
        if (i != 0)
            declaration << ", ";
        declaration << args[i];
    }
    declaration << ")";
    statement(declaration.str());
    processing_entry_point = true;
}

std::string CompilerOpenCL::to_func_call_arg(const SPIRFunction::Parameter& argument, uint32_t id)
{
    const SPIRType& argument_type = get<SPIRType>(argument.type);
    if (is_pointer(argument_type))
    {
        uint32_t name_id = id;
        SPIRVariable* variable = maybe_get<SPIRVariable>(id);
        if (variable && variable->basevariable)
            name_id = variable->basevariable;
        return "&(" + to_unpacked_expression(name_id) + ")";
    }
    return CompilerGLSL::to_func_call_arg(argument, id);
}

void CompilerOpenCL::append_global_func_args(const SPIRFunction& func, uint32_t index,
                                             SmallVector<std::string>& arglist)
{
    CompilerGLSL::append_global_func_args(func, index, arglist);

    if (func.self == ir.default_entry_point)
        return;

    for (size_t i = 0; i < resources.size(); i++)
    {
        arglist.push_back("momo_buffer_" + std::to_string(i));
        arglist.push_back("momo_buffer_offset_" + std::to_string(i));
        arglist.push_back("momo_buffer_size_" + std::to_string(i));
    }
    if (push_constant_variable_id != 0)
        arglist.push_back("momo_push_constants");
}

void CompilerOpenCL::validate_storage_block(const Resource& resource)
{
    const SPIRType& block = get_type(resource.base_type_id);
    if (block.basetype != SPIRType::Struct || block.member_types.size() != 1)
        throw std::runtime_error("initial momoten storage buffers must contain one runtime-array member");

    if (!has_member_decoration(block.self, 0, DecorationOffset) || get_member_decoration(block.self, 0, DecorationOffset) != 0)
        throw std::runtime_error("initial momoten storage-buffer member must have byte offset zero");

    const SPIRType& member = get_type(block.member_types[0]);
    if (member.op != OpTypeRuntimeArray)
        throw std::runtime_error("initial momoten storage-buffer member must be a runtime array");

    const uint32_t stride = get_decoration(block.member_types[0], DecorationArrayStride);
    if (member.basetype == SPIRType::Half && member.width == 16)
        requires_fp16 = true;
    const bool supported_float_matrix = member.basetype == SPIRType::Float && member.width == 32 && member.vecsize == 4 && member.columns == 4;
    const bool supported_half_matrix = member.basetype == SPIRType::Half && member.width == 16 && member.vecsize == 4 && member.columns == 4;
    const bool supported_scalar_vector = member.columns == 1 && ((member.width == 32 && (member.basetype == SPIRType::Int || member.basetype == SPIRType::UInt || member.basetype == SPIRType::Float)) || (member.width == 16 && (member.basetype == SPIRType::Short || member.basetype == SPIRType::UShort || member.basetype == SPIRType::Int || member.basetype == SPIRType::UInt)) || (member.width == 16 && member.basetype == SPIRType::Half));
    if ((!supported_float_matrix && !supported_half_matrix && !supported_scalar_vector) || member.vecsize == 3 || member.vecsize > 4)
        throw std::runtime_error("momoten storage buffers require scalar, vec2, vec4 or mat4 float/int elements with a representable OpenCL layout");

    const uint32_t expected_stride = supported_float_matrix ? 64u : supported_half_matrix ? 32u
                                                                                          : (member.vecsize == 3 ? 16u : (member.width / 8) * member.vecsize);
    if (stride != expected_stride)
        throw std::runtime_error("storage-buffer ArrayStride is not representable by the initial OpenCL ABI");

    storage_block_types.insert(block.self);
}

void CompilerOpenCL::validate_push_constant_block(const Resource& resource, size_t& size)
{
    const SPIRType& block = get_type(resource.base_type_id);
    if (block.basetype != SPIRType::Struct)
        throw std::runtime_error("push constants must be a struct");

    for (uint32_t i = 0; i < block.member_types.size(); i++)
    {
        const SPIRType& member = get_type(block.member_types[i]);
        if (member.width != 32 || member.vecsize != 1 || member.columns != 1 || !member.array.empty())
            throw std::runtime_error("initial momoten push constants require packed 32-bit scalar members");
        if (!has_member_decoration(block.self, i, DecorationOffset) || get_member_decoration(block.self, i, DecorationOffset) != i * 4)
            throw std::runtime_error("initial momoten push constants must be tightly packed 32-bit scalars");
    }

    size = get_declared_struct_size(block);
}

void CompilerOpenCL::reflect_resources(KernelABI& abi)
{
    const ShaderResources shader_resources = get_shader_resources();
    if (!shader_resources.uniform_buffers.empty() || !shader_resources.storage_images.empty() || !shader_resources.sampled_images.empty() || !shader_resources.separate_images.empty() || !shader_resources.separate_samplers.empty() || !shader_resources.subpass_inputs.empty() || !shader_resources.atomic_counters.empty() || !shader_resources.gl_plain_uniforms.empty())
        throw std::runtime_error("the initial OpenCL C profile supports storage buffers and push constants only");

    resources.clear();
    for (size_t i = 0; i < shader_resources.storage_buffers.size(); i++)
    {
        const Resource& resource = shader_resources.storage_buffers[i];
        validate_storage_block(resource);

        ResourceInfo info;
        info.variable_id = resource.id;
        info.descriptor_set = get_decoration(resource.id, DecorationDescriptorSet);
        info.binding = get_decoration(resource.id, DecorationBinding);
        if (info.descriptor_set != 0)
            throw std::runtime_error("the initial OpenCL C profile supports descriptor set 0 only");

        const Bitset flags = get_buffer_block_flags(resource.id);
        const bool nonwritable = flags.get(DecorationNonWritable);
        const bool nonreadable = flags.get(DecorationNonReadable);
        info.access = nonwritable ? BufferAccessReadOnly : (nonreadable ? BufferAccessWriteOnly : BufferAccessReadWrite);

        info.instance_name = get_name(resource.id);
        if (info.instance_name.empty())
        {
            info.instance_name = "momo_resource_" + std::to_string(resource.id);
            set_name(resource.id, info.instance_name);
        }
        resources.push_back(info);
    }
    std::sort(resources.begin(), resources.end(), resource_less);

    resource_indices.clear();
    abi.buffers.clear();
    uint32_t arg_index = 0;
    for (size_t i = 0; i < resources.size(); i++)
    {
        resource_indices[resources[i].variable_id] = i;

        BufferArgument argument;
        argument.variable_id = resources[i].variable_id;
        argument.descriptor_set = resources[i].descriptor_set;
        argument.binding = resources[i].binding;
        argument.buffer_arg_index = arg_index++;
        argument.offset_arg_index = arg_index++;
        argument.size_arg_index = arg_index++;
        argument.access = resources[i].access;
        argument.name = resources[i].instance_name;
        abi.buffers.push_back(argument);
    }

    if (shader_resources.push_constant_buffers.size() > 1)
        throw std::runtime_error("only one push-constant block is supported");

    push_constant_variable_id = 0;
    abi.push_constant_arg_index = -1;
    abi.push_constant_size = 0;
    if (!shader_resources.push_constant_buffers.empty())
    {
        const Resource& resource = shader_resources.push_constant_buffers[0];
        validate_push_constant_block(resource, abi.push_constant_size);
        push_constant_variable_id = resource.id;
        if (get_name(resource.id).empty())
            set_name(resource.id, "momo_push_block");
        abi.push_constant_arg_index = static_cast<int>(arg_index);
    }
}

} // namespace momoten
