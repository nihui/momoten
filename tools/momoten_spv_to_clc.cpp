// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

static void print_usage(const char* executable)
{
    fprintf(stderr,
            "Usage: %s [--entry name] [--address-bits 32|64] "
            "[--local-size x,y,z] [--specialize id=value] input.spv [output.cl]\n",
            executable);
}

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

static bool parse_u32(const char* text, uint32_t& value)
{
    errno = 0;
    char* end = 0;
    const unsigned long parsed = strtoul(text, &end, 0);
    if (errno != 0 || !end || *end != '\0' || parsed > 0xfffffffful)
        return false;
    value = static_cast<uint32_t>(parsed);
    return true;
}

static bool parse_local_size(const char* text, uint32_t values[3])
{
    char trailing = 0;
    return sscanf(text, "%u,%u,%u%c", &values[0], &values[1], &values[2], &trailing) == 3 && values[0] != 0 && values[1] != 0 && values[2] != 0;
}

static bool parse_specialization(const char* text, momoten::SpecializationValue& value)
{
    const char* equals = strchr(text, '=');
    if (!equals)
        return false;

    std::string id_text(text, static_cast<size_t>(equals - text));
    uint32_t bits = 0;
    if (!parse_u32(id_text.c_str(), value.constant_id) || !parse_u32(equals + 1, bits))
        return false;

    value.data.resize(4);
    memcpy(value.data.data(), &bits, 4);
    return true;
}

int main(int argc, char** argv)
{
    momoten::TranslationOptions options;
    const char* input_path = 0;
    const char* output_path = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc)
        {
            options.entry_point = argv[++i];
        }
        else if (strcmp(argv[i], "--address-bits") == 0 && i + 1 < argc)
        {
            if (!parse_u32(argv[++i], options.address_bits))
            {
                print_usage(argv[0]);
                return 2;
            }
        }
        else if (strcmp(argv[i], "--local-size") == 0 && i + 1 < argc)
        {
            if (!parse_local_size(argv[++i], options.local_size))
            {
                print_usage(argv[0]);
                return 2;
            }
            options.override_local_size = true;
        }
        else if (strcmp(argv[i], "--specialize") == 0 && i + 1 < argc)
        {
            momoten::SpecializationValue value;
            if (!parse_specialization(argv[++i], value))
            {
                print_usage(argv[0]);
                return 2;
            }
            options.specializations.push_back(value);
        }
        else if (!input_path)
        {
            input_path = argv[i];
        }
        else if (!output_path)
        {
            output_path = argv[i];
        }
        else
        {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!input_path)
    {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<uint32_t> words;
    if (!read_spirv(input_path, words))
    {
        fprintf(stderr, "momoten-spv-to-clc: failed to read SPIR-V file %s\n", input_path);
        return 1;
    }

    momoten::TranslationResult result;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, result))
    {
        fprintf(stderr, "momoten-spv-to-clc: translation failed: %s\n", result.diagnostics.c_str());
        return 1;
    }

    FILE* output = stdout;
    if (output_path)
    {
        output = fopen(output_path, "wb");
        if (!output)
        {
            fprintf(stderr, "momoten-spv-to-clc: failed to open output file %s\n", output_path);
            return 1;
        }
    }

    fwrite(result.source.data(), 1, result.source.size(), output);
    if (output_path)
        fclose(output);

    fprintf(stderr, "kernel=%s local_size=%u,%u,%u buffers=%u push_constants=%u\n",
            result.abi.entry_point.c_str(), result.abi.local_size[0], result.abi.local_size[1],
            result.abi.local_size[2], static_cast<unsigned>(result.abi.buffers.size()),
            static_cast<unsigned>(result.abi.push_constant_size));
    for (size_t i = 0; i < result.abi.buffers.size(); i++)
    {
        const momoten::BufferArgument& argument = result.abi.buffers[i];
        fprintf(stderr, "  set=%u binding=%u buffer_arg=%u offset_arg=%u size_arg=%u name=%s access=%d\n",
                argument.descriptor_set, argument.binding, argument.buffer_arg_index,
                argument.offset_arg_index, argument.size_arg_index, argument.name.c_str(), static_cast<int>(argument.access));
    }

    return 0;
}
