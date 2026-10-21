// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"

#include <spirv-tools/libspirv.hpp>

#include <stdio.h>

#include <fstream>
#include <sstream>
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

static std::string rename_debug_symbols(const std::string& assembly, const std::string& name)
{
    std::istringstream input(assembly);
    std::ostringstream output;
    std::string line;
    while (std::getline(input, line))
    {
        const size_t opcode = line.find_first_not_of(" \t");
        if (opcode != std::string::npos
            && (line.compare(opcode, 7, "OpName ") == 0 || line.compare(opcode, 13, "OpMemberName ") == 0))
        {
            if (name.empty())
                continue;
            line.erase(line.find('"', opcode));
            line += '"' + name + '"';
        }
        output << line << '\n';
    }
    return output.str();
}

int main(int argc, char** argv)
{
    if (argc < 2)
        return 1;

    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    // Deliberately duplicate names across every symbol and struct member.
    // Include unknown names, backend identifiers and stripped debug metadata.
    const char* names[] = {"future_vendor_macro", "momo_main", "momo_buffer_0", "momo_spv_1", "momo_member_0", "to_global", "MAX_WORK_DIM", ""};
    for (int arg = 1; arg < argc; arg++)
    {
        std::vector<uint32_t> original;
        std::string assembly;
        momoten::TranslationOptions options;
        momoten::TranslationResult reference;
        if (!read_spirv(argv[arg], original)
            || !tools.Disassemble(original, &assembly, SPV_BINARY_TO_TEXT_OPTION_NO_HEADER)
            || !momoten::translate_spirv_to_opencl_c(original.data(), original.size(), options, reference))
        {
            fprintf(stderr, "opencl_symbols_test: failed to prepare %s: %s\n", argv[arg], reference.diagnostics.c_str());
            return 1;
        }

        for (const char* name : names)
        {
            std::vector<uint32_t> renamed;
            momoten::TranslationResult result;
            if (!tools.Assemble(rename_debug_symbols(assembly, name), &renamed, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS)
                || !tools.Validate(renamed)
                || !momoten::translate_spirv_to_opencl_c(renamed.data(), renamed.size(), options, result)
                || result.source != reference.source)
            {
                fprintf(stderr, "opencl_symbols_test: debug name '%s' changed source for %s: %s\n", name, argv[arg], result.diagnostics.c_str());
                return 1;
            }
            if (result.abi.buffers.size() != reference.abi.buffers.size())
                return 1;
            for (size_t i = 0; i < result.abi.buffers.size(); i++)
            {
                const momoten::BufferArgument& buffer = result.abi.buffers[i];
                const momoten::BufferArgument& expected = reference.abi.buffers[i];
                const std::string debug_name = *name ? std::string(name) : "momo_resource_" + std::to_string(buffer.variable_id);
                if (buffer.name != debug_name || buffer.variable_id != expected.variable_id
                    || buffer.binding != expected.binding || buffer.descriptor_set != expected.descriptor_set
                    || buffer.buffer_arg_index != expected.buffer_arg_index || buffer.offset_arg_index != expected.offset_arg_index
                    || buffer.access != expected.access)
                {
                    fprintf(stderr, "opencl_symbols_test: debug name '%s' changed buffer reflection for %s\n", name, argv[arg]);
                    return 1;
                }
            }
        }
    }
    return 0;
}
