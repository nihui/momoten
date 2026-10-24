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
    if (raw_buffer_variables.count(var.self))
    {
        statement_no_indent("#define ", instance_name, " (momo_buffer_", it->second, " + momo_buffer_offset_", it->second, ")");
        statement("");
        return;
    }
    const SPIRType& element_type = get<SPIRType>(type.member_types[0]);
    std::string type_name = type_to_glsl(element_type);
    if (element_type.basetype == SPIRType::Double && element_type.width == 64 && element_type.columns == 1)
    {
        type_name = "ulong";
        if (element_type.vecsize > 1)
            type_name += std::to_string(element_type.vecsize);
    }
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
    if (!raw_push_constants)
        emit_struct(type);

    const std::string instance_name = to_name(var.self);
    statement_no_indent("#define ", instance_name, " momo_push_constants");
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
                        << ", " << offset_type << " momo_buffer_offset_" << i;
        }
        if (push_constant_variable_id != 0)
        {
            if (!first_argument)
                declaration << ", ";
            const SPIRType& push_type = get_variable_data_type(get<SPIRVariable>(push_constant_variable_id));
            declaration << (raw_push_constants ? "__private const uint*" : type_to_glsl(push_type)) << " momo_push_constants";
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
    }
    if (push_constant_variable_id != 0)
    {
        const SPIRType& push_type = get_variable_data_type(get<SPIRVariable>(push_constant_variable_id));
        // Scalar kernel arguments avoid driver-specific by-value struct ABIs.
        const size_t count = raw_push_constants ? push_constant_bytes / 4 : push_type.member_types.size();
        for (size_t i = 0; i < count; i++)
            args.push_back((raw_push_constants ? "uint" : type_to_glsl(get<SPIRType>(push_type.member_types[i])))
                           + " momo_push_constant_" + std::to_string(i));
    }

    std::ostringstream declaration;
    declaration << "__kernel ";
    // Independent invocations can be split when the compiler's natural
    // workgroup limit is smaller than the Vulkan shape. Requiring that shape
    // here can instead force register spilling and hide the lower limit.
    if (translation_options.workgroup_mode == WorkgroupModeDirect && !workgroup_splittable)
    {
        declaration << "__attribute__((reqd_work_group_size(" << local_size[0] << ", "
                    << local_size[1] << ", " << local_size[2] << "))) ";
    }
    declaration << "void " << get_entry_point().name << "(";
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
        const SPIRVariable* backing = maybe_get_backing_variable(id);
        if (backing && (raw_buffer_variables.count(backing->self) || (raw_push_constants && backing->self == push_constant_variable_id)))
            throw std::runtime_error("passing raw-layout block pointers to helper functions is not supported");
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
    if (member.basetype == SPIRType::Struct)
    {
        if (stride == 0)
            throw std::runtime_error("storage-buffer struct arrays require an explicit stride");
        validate_raw_block_type(member.parent_type, StorageClassStorageBuffer);
        raw_buffer_variables.insert(resource.id);
        return;
    }
    if (member.basetype == SPIRType::Half && member.width == 16 && translation_options.native_fp16)
        requires_fp16 = true;
    if (member.basetype == SPIRType::Double && member.width == 64)
        requires_fp64 = true;
    if ((member.basetype == SPIRType::Int64 || member.basetype == SPIRType::UInt64) && member.width == 64)
        requires_int64 = true;
    const bool supported_float_matrix = member.basetype == SPIRType::Float && member.width == 32 && member.vecsize == 4 && member.columns == 4;
    const bool supported_half_matrix = member.basetype == SPIRType::Half && member.width == 16 && member.vecsize == 4 && member.columns == 4;
    const bool byte_integer = member.width == 8 && (member.basetype == SPIRType::SByte || member.basetype == SPIRType::UByte);
    if (member.width == 16)
    {
        if (!requires_storage_buffer16 && !requires_uniform_buffer16 && !requires_storage_buffer8)
            throw std::runtime_error("16-bit storage buffers require a SPIR-V storage capability");
        requires_storage_buffer16 = true;
    }
    if (byte_integer && !requires_storage_buffer8)
        throw std::runtime_error("8-bit storage buffers require SPIR-V StorageBuffer8BitAccess");
    const bool supported_scalar_vector = member.columns == 1 && ((member.width == 64 && (member.basetype == SPIRType::Double || member.basetype == SPIRType::Int64 || member.basetype == SPIRType::UInt64)) || (member.width == 32 && (member.basetype == SPIRType::Int || member.basetype == SPIRType::UInt || member.basetype == SPIRType::Float)) || (member.width == 16 && (member.basetype == SPIRType::Short || member.basetype == SPIRType::UShort || member.basetype == SPIRType::Int || member.basetype == SPIRType::UInt)) || (member.width == 16 && member.basetype == SPIRType::Half) || byte_integer);
    if ((!supported_float_matrix && !supported_half_matrix && !supported_scalar_vector) || member.vecsize == 3 || member.vecsize > 4)
        throw std::runtime_error("momoten storage buffers require scalar, vec2, vec4 or mat4 float/int elements with a representable OpenCL layout, including scalar/vector int64, uint64 and fp64");

    const uint32_t expected_stride = supported_float_matrix ? 64u : supported_half_matrix ? 32u
                                                                                          : (member.vecsize == 3 ? 16u : (member.width / 8) * member.vecsize);
    if (stride != expected_stride)
        throw std::runtime_error("storage-buffer ArrayStride is not representable by the initial OpenCL ABI");

    storage_block_types.insert(block.self);
}

void CompilerOpenCL::validate_raw_block_type(uint32_t type_id, StorageClass storage)
{
    const SPIRType& type = get_type(type_id);
    if (!type.array.empty())
    {
        if (!type.array_size_literal.back() || type.array.back() == 0 || !has_decoration(type_id, DecorationArrayStride) || get_decoration(type_id, DecorationArrayStride) == 0)
            throw std::runtime_error("buffer and push-constant arrays require fixed sizes and an explicit stride");
        validate_raw_block_type(type.parent_type, storage);
        return;
    }
    if (type.basetype == SPIRType::Struct)
    {
        for (uint32_t i = 0; i < type.member_types.size(); i++)
        {
            if (!has_member_decoration(type.self, i, DecorationOffset) || has_member_decoration(type.self, i, DecorationRowMajor))
                throw std::runtime_error("buffer and push-constant members require explicit offsets and column-major matrices");
            const SPIRType& member = get_type(type.member_types[i]);
            if (member.columns > 1 && !has_member_decoration(type.self, i, DecorationMatrixStride))
                throw std::runtime_error("buffer and push-constant matrices require an explicit stride");
            validate_raw_block_type(type.member_types[i], storage);
        }
        return;
    }
    if ((type.width != 16 && type.width != 32 && type.width != 64) || type.vecsize > 4
        || (type.columns > 1 && (type.columns != 4 || type.vecsize != 4 || (type.basetype != SPIRType::Float && type.basetype != SPIRType::Half))))
        throw std::runtime_error("unsupported buffer or push-constant scalar, vector or matrix type");
    if (type.basetype == SPIRType::Half && translation_options.native_fp16)
        requires_fp16 = true;
    if (type.width == 16 && ((storage == StorageClassUniform && !requires_uniform_buffer16) || (storage == StorageClassPushConstant && !requires_push_constant16) || (storage == StorageClassStorageBuffer && !requires_storage_buffer16 && !requires_uniform_buffer16 && !requires_storage_buffer8)))
        throw std::runtime_error("16-bit block member requires its corresponding SPIR-V storage capability");
    if (type.width == 16 && storage == StorageClassStorageBuffer)
        requires_storage_buffer16 = true;
}

void CompilerOpenCL::validate_push_constant_block(const Resource& resource, size_t& size)
{
    const SPIRType& block = get_type(resource.base_type_id);
    if (block.basetype != SPIRType::Struct)
        throw std::runtime_error("push constants must be a struct");

    validate_raw_block_type(block.self, StorageClassPushConstant);
    raw_push_constants = false;
    for (uint32_t i = 0; i < block.member_types.size(); i++)
    {
        const SPIRType& member = get_type(block.member_types[i]);
        if (member.width != 32 || member.vecsize != 1 || member.columns != 1 || !member.array.empty() || get_member_decoration(block.self, i, DecorationOffset) != i * 4)
            raw_push_constants = true;
    }

    size = (get_declared_struct_size(block) + 3) / 4 * 4;
    if (size == 0 || size > 256)
        throw std::runtime_error("push constants exceed the supported 256-byte range");
    push_constant_bytes = size;
}

void CompilerOpenCL::reflect_resources(KernelABI& abi)
{
    const ShaderResources shader_resources = get_shader_resources();
    if (!shader_resources.storage_images.empty() || !shader_resources.sampled_images.empty() || !shader_resources.separate_images.empty() || !shader_resources.separate_samplers.empty() || !shader_resources.subpass_inputs.empty() || !shader_resources.atomic_counters.empty() || !shader_resources.gl_plain_uniforms.empty())
        throw std::runtime_error("the OpenCL C profile supports storage buffers, uniform buffers and push constants only");

    resources.clear();
    raw_buffer_variables.clear();
    SmallVector<Resource> buffers = shader_resources.storage_buffers;
    buffers.insert(buffers.end(), shader_resources.uniform_buffers.begin(), shader_resources.uniform_buffers.end());
    for (size_t i = 0; i < buffers.size(); i++)
    {
        const Resource& resource = buffers[i];
        const bool uniform = i >= shader_resources.storage_buffers.size();
        if (uniform)
        {
            validate_raw_block_type(resource.base_type_id, StorageClassUniform);
            raw_buffer_variables.insert(resource.id);
        }
        else
        {
            validate_storage_block(resource);
            const SPIRType& block = get_type(resource.base_type_id);
            if (!translation_options.native_fp16 && get_type(block.member_types[0]).basetype == SPIRType::Half)
                raw_buffer_variables.insert(resource.id);
        }

        ResourceInfo info;
        info.uniform = uniform;
        info.variable_id = resource.id;
        info.descriptor_set = get_decoration(resource.id, DecorationDescriptorSet);
        info.binding = get_decoration(resource.id, DecorationBinding);
        if (info.descriptor_set != 0)
            throw std::runtime_error("the initial OpenCL C profile supports descriptor set 0 only");

        const Bitset flags = get_buffer_block_flags(resource.id);
        const bool nonwritable = flags.get(DecorationNonWritable);
        const bool nonreadable = flags.get(DecorationNonReadable);
        info.access = uniform || nonwritable ? BufferAccessReadOnly : (nonreadable ? BufferAccessWriteOnly : BufferAccessReadWrite);

        info.name = get_name(resource.id);
        if (info.name.empty())
            info.name = "momo_resource_" + std::to_string(resource.id);
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
        argument.access = resources[i].access;
        argument.uniform = resources[i].uniform;
        argument.name = resources[i].name;
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
        abi.push_constant_arg_index = static_cast<int>(arg_index);
    }
}

} // namespace momoten
