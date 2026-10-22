// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator/translator_internal.h"

#include <spirv-tools/libspirv.hpp>

#include <stdio.h>
#include <sstream>

static void replace_all(std::string& text, const std::string& before, const std::string& after)
{
    size_t offset = 0;
    while ((offset = text.find(before, offset)) != std::string::npos)
    {
        text.replace(offset, before.size(), after);
        offset += after.size();
    }
}

struct LoadFixture
{
    std::string scalar = "OpTypeInt 32 0";
    unsigned width = 32;
    unsigned lanes = 2;
    bool signed_index = false;
    bool private_index = false;
    bool loop = false;
    bool branch = false;
    bool packed_half = true;
    bool native = true;
    bool reverse = false;
    bool split_buffers = false;
    bool duplicate = false;
    unsigned padding = 0;
    unsigned first_offset = 0;
    std::string scale;
    std::string decorations;
    std::string memory_access;
    std::string between;
    std::string functions;

    unsigned count() const { return 128 / (width * lanes); }

    std::string assembly() const
    {
        const unsigned n = count();
        const bool half = width == 16 && scalar == "OpTypeFloat 16";
        std::ostringstream s;
        s << "OpCapability Shader\n";
        if (half)
            s << "OpCapability Float16\nOpCapability StorageBuffer16BitAccess\nOpExtension \"SPV_KHR_16bit_storage\"\n";
        s << "OpMemoryModel Logical GLSL450\n"
             "OpEntryPoint GLCompute %main \"main\" %gid\n"
             "OpExecutionMode %main LocalSize 1 1 1\n"
             "OpDecorate %gid BuiltIn GlobalInvocationId\n"
             "OpDecorate %array ArrayStride " << width * lanes / 8 + padding << "\n"
             "OpMemberDecorate %block 0 Offset 0\n"
             "OpDecorate %block Block\n"
             "OpDecorate %input DescriptorSet 0\nOpDecorate %input Binding 0\n"
             "OpDecorate %output DescriptorSet 0\nOpDecorate %output Binding 1\n"
          << decorations
          << "%void = OpTypeVoid\n%bool = OpTypeBool\n%uint = OpTypeInt 32 0\n"
             "%integer = OpTypeInt 32 " << signed_index << "\n"
             "%uint3 = OpTypeVector %uint 3\n"
             "%ptr_input_uint3 = OpTypePointer Input %uint3\n"
             "%gid = OpVariable %ptr_input_uint3 Input\n"
             "%scalar = " << scalar << "\n";
        if (lanes > 1)
            s << "%element = OpTypeVector %scalar " << lanes << "\n";
        const char* element = lanes > 1 ? "%element" : "%scalar";
        s << "%array = OpTypeRuntimeArray " << element << "\n"
             "%block = OpTypeStruct %array\n"
             "%ptr_block = OpTypePointer StorageBuffer %block\n"
             "%ptr_element = OpTypePointer StorageBuffer " << element << "\n"
             "%ptr_private = OpTypePointer Function %integer\n"
             "%input = OpVariable %ptr_block StorageBuffer\n"
             "%output = OpVariable %ptr_block StorageBuffer\n"
             "%function = OpTypeFunction %void\n"
             "%true = OpConstantTrue %bool\n"
             "%scope = OpConstant %uint 2\n"
             "%semantics = OpConstant %uint 72\n";
        for (unsigned i = 0; i <= 8; i++)
            s << "%c" << i << " = OpConstant %integer " << i << "\n";
        s << "%main = OpFunction %void None %function\n%entry = OpLabel\n";
        if (private_index || loop)
            s << "%local = OpVariable %ptr_private Function\n";
        s << "%invocation = OpLoad %uint3 %gid\n"
             "%x = OpCompositeExtract %uint %invocation 0\n"
             "%signed_x = OpBitcast %integer %x\n";
        if (loop)
        {
            s << "OpStore %local %c0\nOpBranch %header\n%header = OpLabel\n"
                 "%iteration = OpLoad %integer %local\n"
                 "%condition = " << (signed_index ? "OpSLessThan" : "OpULessThan") << " %bool %iteration %c8\n"
                 "OpLoopMerge %merge %continue None\n"
                 "OpBranchConditional %condition %body %merge\n%body = OpLabel\n"
                 "%base = OpLoad %integer %local\n";
        }
        else
        {
            s << (scale.empty() ? "%base = OpIMul %integer %signed_x %c" + std::to_string(n) + "\n" : scale);
            if (private_index)
                s << "OpStore %local %base\n";
        }
        for (unsigned i = 0; i < n; i++)
        {
            if (i == 1)
            {
                s << between;
                if (branch)
                    s << "%take = OpIEqual %bool %signed_x %c0\n"
                         "OpSelectionMerge %merge None\nOpBranchConditional %take %tail %merge\n%tail = OpLabel\n";
            }
            if (private_index)
                s << "%index_base" << i << " = OpLoad %integer %local\n";
            const std::string base = private_index ? "%index_base" + std::to_string(i) : "%base";
            const unsigned offset = first_offset + (duplicate && i == n - 1 ? 0 : reverse ? n - i - 1 : i);
            s << "%index" << i << " = OpIAdd %integer " << base << " %c" << offset << "\n"
                 "%pointer" << i << " = OpAccessChain %ptr_element " << (split_buffers && i == n - 1 ? "%output" : "%input") << " %c0 %index" << i << "\n"
                 "%value" << i << " = OpLoad " << element << " %pointer" << i << memory_access << "\n";
        }
        for (unsigned i = 0; i < n; i++)
            s << "%out" << i << " = OpAccessChain %ptr_element %output %c0 %c" << i << "\nOpStore %out" << i << " %value" << i << "\n";
        if (loop)
            s << "OpBranch %continue\n%continue = OpLabel\n"
                 "%old = OpLoad %integer %local\n%next = OpIAdd %integer %old %c" << n << "\n"
                 "OpStore %local %next\nOpBranch %header\n%merge = OpLabel\n";
        else if (branch)
            s << "OpBranch %merge\n%merge = OpLabel\n";
        s << "OpReturn\nOpFunctionEnd\n" << functions;
        std::string text = s.str();
        if (scalar == "OpTypeInt 32 0")
        {
            replace_all(text, "%scalar = OpTypeInt 32 0\n", "");
            replace_all(text, "%scalar", "%uint");
        }
        else if (scalar == "OpTypeInt 32 1" && signed_index)
        {
            replace_all(text, "%scalar = OpTypeInt 32 1\n", "");
            replace_all(text, "%scalar", "%integer");
        }
        if (!signed_index)
        {
            replace_all(text, "%integer = OpTypeInt 32 0\n", "");
            replace_all(text, "%integer", "%uint");
        }
        return text;
    }
};

static int check(const char* name, const LoadFixture& fixture, unsigned expected, bool emit = true)
{
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([name](spv_message_level_t, const char*, const spv_position_t& p, const char* message) {
        fprintf(stderr, "buffer_loads_test: %s SPIR-V at %zu:%zu: %s\n", name, p.line, p.column, message);
    });
    std::vector<uint32_t> words;
    if (!tools.Assemble(fixture.assembly(), &words) || !tools.Validate(words))
        return 1;
    momoten::TranslationOptions options;
    options.native_buffer_vector_loads = fixture.native;
    options.packed_fp16_vector_loads = fixture.packed_half;
    const auto groups = momoten::plan_coalesced_buffer_loads(words.data(), words.size(), options);
    if (groups.size() != expected)
    {
        fprintf(stderr, "buffer_loads_test: %s coalesced loads=%zu expected=%u\n", name, groups.size(), expected);
        return 1;
    }
    if (!emit)
        return 0;
    momoten::TranslationResult result;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, result))
    {
        fprintf(stderr, "buffer_loads_test: %s translation: %s\n", name, result.diagnostics.c_str());
        return 1;
    }
    const bool wide_load = result.source.find("uint4 momo_buffer_load_") != std::string::npos;
    if (wide_load != (expected != 0))
    {
        fprintf(stderr, "buffer_loads_test: %s unexpected emitted load width\n%s\n", name, result.source.c_str());
        return 1;
    }
    return 0;
}

int main()
{
    int failures = 0;
    LoadFixture f;
    failures += check("uint2 pair", f, 2);
    f.scalar = "OpTypeInt 32 1";
    failures += check("int2 pair", f, 2);
    f.scalar = "OpTypeFloat 32";
    failures += check("float2 pair", f, 2);
    f.lanes = 1;
    failures += check("four scalar floats", f, 4);
    f.scalar = "OpTypeInt 32 0";
    failures += check("four scalar uints", f, 4);
    f.scalar = "OpTypeInt 32 1";
    failures += check("four scalar ints", f, 4);
    f = LoadFixture();
    f.scalar = "OpTypeFloat 16";
    f.width = 16;
    f.lanes = 4;
    failures += check("packed half4 pair", f, 2);
    f.packed_half = false;
    failures += check("half mapping not probed", f, 0);
    f = LoadFixture();
    f.private_index = true;
    failures += check("private index copies", f, 2);
    f.loop = true;
    failures += check("loop-carried aligned index", f, 2);
    f.signed_index = true;
    failures += check("signed aligned loop index", f, 2);
    f = LoadFixture();
    f.scale = "%base = OpShiftLeftLogical %integer %signed_x %c1\n";
    failures += check("shifted index", f, 2);
    f.first_offset = 2;
    failures += check("aligned pair after possible base wrap", f, 2);
    f.signed_index = true;
    failures += check("signed aligned pair after possible base wrap", f, 2);
    f.first_offset = 1;
    failures += check("pair crosses possible signed wrap", f, 0);
    f.signed_index = false;
    failures += check("pair crosses possible unsigned wrap", f, 0);
    f.first_offset = 0;
    f.scale = "%base = OpCopyObject %integer %signed_x\n";
    failures += check("possible unsigned index wrap", f, 0);
    f.signed_index = true;
    failures += check("possible signed index wrap", f, 0);
    f = LoadFixture();
    f.private_index = true;
    f.between = "OpStore %local %c0\n";
    failures += check("private index changes between loads", f, 0);
    f = LoadFixture();
    f.native = false;
    failures += check("native loads disabled", f, 0);
    f = LoadFixture();
    f.memory_access = " Volatile";
    failures += check("volatile memory access", f, 0);
    f.memory_access = " Aligned 8";
    failures += check("explicit alignment", f, 2);
    f.memory_access = " Nontemporal";
    failures += check("memory access hints retained", f, 0);
    f = LoadFixture();
    f.decorations = "OpDecorate %input Volatile\n";
    failures += check("volatile buffer", f, 0);
    f.decorations = "OpDecorate %input Aliased\nOpDecorate %output Aliased\n";
    failures += check("aliased buffer", f, 0);
    f.scalar = "OpTypeFloat 16";
    f.width = 16;
    f.lanes = 4;
    failures += check("aliased packed half buffer", f, 0);
    f = LoadFixture();
    f.decorations = "OpMemberDecorate %block 0 Coherent\n";
    failures += check("coherent block member", f, 0);
    f = LoadFixture();
    f.branch = true;
    failures += check("conditional tail stays conditional", f, 0);
    f = LoadFixture();
    f.between = "OpMemoryBarrier %scope %semantics\n";
    failures += check("memory barrier", f, 0);
    f.between = "%alias = OpAccessChain %ptr_element %output %c0 %c0\nOpStore %alias %value0\n";
    failures += check("other descriptor may alias", f, 0);
    f.lanes = 1;
    f.between = "%alias = OpAccessChain %ptr_element %output %c0 %c0\n%atomic = OpAtomicIAdd %scalar %alias %scope %semantics %c1\n";
    failures += check("atomic separates loads", f, 0, false);
    f.lanes = 2;
    f.between = "%call = OpFunctionCall %void %helper\n";
    f.functions = "%helper = OpFunction %void None %function\n%helper_entry = OpLabel\nOpReturn\nOpFunctionEnd\n";
    failures += check("unknown call", f, 0, false);
    f = LoadFixture();
    f.reverse = true;
    failures += check("reverse addresses are not speculated", f, 0);
    f = LoadFixture();
    f.split_buffers = true;
    failures += check("different buffers", f, 0);
    f = LoadFixture();
    f.duplicate = true;
    failures += check("duplicate address is not a wider access", f, 0);
    f = LoadFixture();
    f.padding = 8;
    failures += check("array stride gap", f, 0, false);
    return failures ? 1 : 0;
}
