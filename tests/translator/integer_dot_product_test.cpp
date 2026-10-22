// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"
#include "integer_dot_product_fixture.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>

int main()
{
    const char* extensions[] = {
        "cl_khr_integer_dot_product",
        "cl_arm_integer_dot_product_int8",
        "cl_arm_integer_dot_product_accumulate_int8",
        "cl_arm_integer_dot_product_accumulate_saturate_int8",
        "cl_qcom_dot_product8"};
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        fprintf(stderr, "integer_dot_product_test: %s\n", message);
    });
    for (bool packed : {false, true})
    {
        std::vector<uint32_t> words;
        if (!tools.Assemble(integer_dot_product_fixture(packed), &words) || !tools.Validate(words))
            return 1;
        for (int backend = 0; backend < 5; backend++)
        {
            momoten::TranslationOptions options;
            options.integer_dot_product = true;
            options.integer_dot_product_backend = static_cast<momoten::IntegerDotProductBackend>(backend);
            options.integer_dot_product_input_4x8bit = !packed;
            options.integer_dot_product_input_4x8bit_packed = packed;
            // Vendor code must not inherit KHR driver workarounds.
            options.integer_dot_product_split_packed_acc_sat = backend != 0;
            options.integer_dot_product_swap_mixed_acc_sat = backend != 0;
            momoten::TranslationResult translated;
            if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
            {
                fprintf(stderr, "integer_dot_product_test: backend %d packed %d: %s\n", backend, packed, translated.diagnostics.c_str());
                return 1;
            }
            if (!translated.abi.integer_dot_product
                || translated.abi.required_extensions != std::vector<std::string>{extensions[backend]}
                || translated.source.find(std::string("#pragma OPENCL EXTENSION ") + extensions[backend] + " : enable") == std::string::npos
                || (backend != 0 && translated.source.find("dot_4x8packed_") != std::string::npos))
                return 1;
            options.integer_dot_product_input_4x8bit = false;
            options.integer_dot_product_input_4x8bit_packed = false;
            if (momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
                return 1;
        }
    }
    return 0;
}
