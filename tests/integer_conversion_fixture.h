// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_INTEGER_CONVERSION_FIXTURE_H
#define MOMOTEN_INTEGER_CONVERSION_FIXTURE_H

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <sstream>
#include <vector>

static const uint32_t integer_conversion_words = 24;

inline bool make_integer_conversion_spirv(std::vector<uint32_t>& words)
{
    // mismatched integer signedness is legal SPIR-V but uncommon in GLSL output
    std::ostringstream source;
    source << "OpCapability Shader\nOpCapability Int16\n"
              "OpMemoryModel Logical GLSL450\n"
              "OpEntryPoint GLCompute %main \"main\" %global_id\n"
              "OpExecutionMode %main LocalSize 64 1 1\n"
              "OpDecorate %global_id BuiltIn GlobalInvocationId\n"
              "OpDecorate %array ArrayStride 4\n"
              "OpMemberDecorate %block 0 Offset 0\n"
              "OpDecorate %block BufferBlock\n"
              "OpDecorate %data DescriptorSet 0\nOpDecorate %data Binding 0\n"
              "%void = OpTypeVoid\n%function = OpTypeFunction %void\n"
              "%uint = OpTypeInt 32 0\n%int = OpTypeInt 32 1\n"
              "%ushort = OpTypeInt 16 0\n%short = OpTypeInt 16 1\n"
              "%float = OpTypeFloat 32\n%uint3 = OpTypeVector %uint 3\n";
    for (const char* type : {"uint", "int", "ushort", "short", "float"})
        source << '%' << type << "4 = OpTypeVector %" << type << " 4\n";
    source << "%input_pointer = OpTypePointer Input %uint3\n"
              "%global_id = OpVariable %input_pointer Input\n"
              "%array = OpTypeRuntimeArray %uint\n%block = OpTypeStruct %array\n"
              "%block_pointer = OpTypePointer Uniform %block\n"
              "%uint_pointer = OpTypePointer Uniform %uint\n"
              "%data = OpVariable %block_pointer Uniform\n";
    for (uint32_t i = 0; i <= integer_conversion_words; i++)
        source << "%c" << i << " = OpConstant %uint " << i << '\n';
    source << "%main = OpFunction %void None %function\n%entry = OpLabel\n"
              "%global = OpLoad %uint3 %global_id\n"
              "%index = OpCompositeExtract %uint %global 0\n"
              "%base = OpIMul %uint %index %c24\n";
    for (uint32_t i = 0; i < 4; i++)
    {
        source << "%offset" << i << " = OpIAdd %uint %base %c" << i << '\n'
               << "%pointer" << i << " = OpAccessChain %uint_pointer %data %c0 %offset" << i << '\n'
               << "%input" << i << " = OpLoad %uint %pointer" << i << '\n'
               << "%u" << i << " = OpUConvert %ushort %input" << i << '\n'
               << "%s" << i << " = OpSConvert %short %input" << i << '\n';
    }
    source << "%uv = OpCompositeConstruct %ushort4 %u0 %u1 %u2 %u3\n"
              "%sv = OpCompositeConstruct %short4 %s0 %s1 %s2 %s3\n";
    struct Conversion
    {
        const char* opcode;
        const char* input;
        const char* type;
    };
    const Conversion conversions[] = {
        {"OpUConvert", "s", "uint"},
        {"OpSConvert", "u", "int"},
        {"OpConvertUToF", "s", "float"},
        {"OpConvertSToF", "u", "float"}};
    uint32_t output = 4;
    for (uint32_t width : {1u, 4u})
    {
        for (const Conversion& conversion : conversions)
        {
            const std::string id = "result" + std::to_string(output);
            source << '%' << id << " = " << conversion.opcode << " %" << conversion.type << (width == 1 ? "" : "4")
                   << " %" << conversion.input << (width == 1 ? "0" : "v") << '\n';
            for (uint32_t lane = 0; lane < width; lane++, output++)
            {
                std::string value = id;
                if (width != 1)
                {
                    value += "_" + std::to_string(lane);
                    source << '%' << value << " = OpCompositeExtract %" << conversion.type << " %" << id << ' ' << lane << '\n';
                }
                if (std::string(conversion.type) != "uint")
                {
                    source << '%' << value << "_bits = OpBitcast %uint %" << value << '\n';
                    value += "_bits";
                }
                source << "%offset" << output << " = OpIAdd %uint %base %c" << output << '\n'
                       << "%pointer" << output << " = OpAccessChain %uint_pointer %data %c0 %offset" << output << '\n'
                       << "OpStore %pointer" << output << " %" << value << '\n';
            }
        }
    }
    source << "OpReturn\nOpFunctionEnd\n";
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "integer conversion fixture at %zu: %s\n", position.line, message);
    });
    return tools.Assemble(source.str(), &words) && tools.Validate(words);
}

#endif // MOMOTEN_INTEGER_CONVERSION_FIXTURE_H
