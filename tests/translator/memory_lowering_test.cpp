// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"
#include "../spirv_test_helpers.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <string>

static bool translate(const std::vector<uint32_t>& words, const momoten::TranslationOptions& options,
                      const std::string& path, unsigned fixture)
{
    const bool narrow_store = fixture == 1;
    momoten::TranslationResult result;
    const bool success = momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, result);
    if (narrow_store && !options.byte_addressable_store)
    {
        if (!success && result.diagnostics.find("cl_khr_byte_addressable_store") != std::string::npos)
            return true;
        fprintf(stderr, "memory_lowering_test: narrow store did not enforce device capability\n");
        return false;
    }
    if (!success)
    {
        fprintf(stderr, "memory_lowering_test: %s\n", result.diagnostics.c_str());
        return false;
    }
    // Clang does not diagnose the old OpenCL C 1.0 narrow-store restriction.
    // Check that ordinary FP32 introduces no writable byte pointer as well.
    if (!narrow_store && result.source.find("volatile __global uchar *") != std::string::npos)
    {
        fprintf(stderr, "memory_lowering_test: FP32 unexpectedly requires byte stores\n");
        return false;
    }
    if (fixture == 0 && options.native_buffer_vector_loads
        && result.source.find("uint4 momo_buffer_load_") == std::string::npos)
    {
        fprintf(stderr, "memory_lowering_test: continue block lost coalesced loads\n");
        return false;
    }
    if (!path.empty())
    {
        std::ofstream output(path.c_str());
        output << result.source;
        if (!output)
            return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5)
        return 1;
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        fprintf(stderr, "memory_lowering_test: %s\n", message);
    });
    const char* names[] = {"float", "half", "double", "copied"};
    for (unsigned fixture = 0; fixture < 4; fixture++)
    {
        std::vector<uint32_t> words;
        if (!read_spirv(argv[fixture == 3 ? 1 : fixture + 1], words) || !decorate_aliased_buffers(words, 2))
            return 1;
        if (fixture == 3)
            copy_access_chain_pointers(words);
        if (!tools.Validate(words))
            return 1;
        for (unsigned bits : {32u, 64u})
            for (bool native : {false, true})
                for (bool byte_stores : {false, true})
                {
                    momoten::TranslationOptions options;
                    options.address_bits = bits;
                    options.native_buffer_vector_loads = options.native_buffer_vector_stores = native;
                    options.byte_addressable_store = byte_stores;
                    const std::string name = std::string(names[fixture]) + "_" + std::to_string(bits)
                                             + "_" + std::to_string(native) + "_" + std::to_string(byte_stores);
                    const std::string path = argc == 5 ? std::string(argv[4]) + "/" + name + ".cl" : "";
                    if (!translate(words, options, path, fixture))
                        return 1;
                }
    }
    return 0;
}
