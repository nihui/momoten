// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "momoten/spv_to_clc.h"
#include "../spirv_test_helpers.h"
#include "../integer_conversion_fixture.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

static int check_integer_conversions(const char* output = 0)
{
    std::vector<uint32_t> words;
    if (!make_integer_conversion_spirv(words))
        return 1;
    for (uint32_t address_bits : {32u, 64u})
    {
        momoten::TranslationOptions options;
        options.address_bits = address_bits;
        momoten::TranslationResult translated;
        if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated) || !translated.abi.int16)
        {
            fprintf(stderr, "integer_storage_test: conversion fixture failed: %s\n", translated.diagnostics.c_str());
            return 1;
        }
        if (output && address_bits == 64)
        {
            std::ofstream stream(output);
            if (!(stream << translated.source))
                return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && strcmp(argv[1], "--conversion-output") == 0)
        return check_integer_conversions(argv[2]);
    if (argc != 6)
        return 1;
    struct Fixture
    {
        const char* path;
        bool storage8;
        bool int16;
        bool storage16;
    };
    const Fixture fixtures[] = {{argv[1], true, false, false}, {argv[2], true, false, false}, {argv[3], false, true, false}, {argv[4], false, false, true}};
    for (const Fixture& fixture : fixtures)
    {
        std::vector<uint32_t> words;
        if (!read_spirv(fixture.path, words))
            return 1;
        momoten::TranslationOptions options;
        momoten::TranslationResult translated;
        if (fixture.storage8 && momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
        {
            fprintf(stderr, "integer_storage_test: byte storage accepted without byte-addressable stores\n");
            return 1;
        }
        options.byte_addressable_store = true;
        for (int native = 0; native != 2; native++)
        {
            options.native_fp16 = !fixture.storage16 || native != 0;
            options.native_buffer_vector_loads = native != 0;
            options.native_buffer_vector_stores = native != 0;
            if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
            {
                fprintf(stderr, "integer_storage_test: %s native=%d: %s\n", fixture.path, native, translated.diagnostics.c_str());
                return 1;
            }
            if (!options.native_fp16 && std::find(translated.abi.required_extensions.begin(), translated.abi.required_extensions.end(), "cl_khr_fp16") != translated.abi.required_extensions.end())
            {
                fprintf(stderr, "integer_storage_test: storage-only fp16 requires native fp16\n");
                return 1;
            }
            if (translated.abi.int16 != fixture.int16 || translated.abi.storage_buffer8 != fixture.storage8
                || translated.abi.storage_buffer16 != fixture.storage16 || translated.abi.uniform_buffer16 != fixture.storage16 || translated.abi.push_constant16 != fixture.storage16)
            {
                fprintf(stderr, "integer_storage_test: required Vulkan features were not reflected\n");
                return 1;
            }
            if (fixture.storage16)
            {
                std::vector<uint32_t> rtz_words = words;
                if (!set_fp16_rounding(rtz_words, spv::FPRoundingModeRTZ)
                    || !momoten::translate_spirv_to_opencl_c(rtz_words.data(), rtz_words.size(), options, translated))
                {
                    fprintf(stderr, "integer_storage_test: fp16 RTZ conversion failed: %s\n", translated.diagnostics.c_str());
                    return 1;
                }
            }
        }
    }
    std::vector<uint32_t> arithmetic;
    if (!read_spirv(argv[5], arithmetic))
        return 1;
    momoten::TranslationOptions options;
    options.native_fp16 = false;
    options.byte_addressable_store = true;
    momoten::TranslationResult translated;
    if (momoten::translate_spirv_to_opencl_c(arithmetic.data(), arithmetic.size(), options, translated))
    {
        fprintf(stderr, "integer_storage_test: fp16 arithmetic accepted without native fp16\n");
        return 1;
    }
    return check_integer_conversions();
}
