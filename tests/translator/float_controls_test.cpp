// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "backend/device_profile.h"
#include "backend/objects.h"
#include "momoten/spv_to_clc.h"

#include <spirv-tools/libspirv.hpp>

#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

namespace momoten_detail {

OpenCLApi g_opencl = {};

} // namespace momoten_detail

static const uint32_t full_flags = 0x7000f;
static const char full_default[] = "OpExecutionModeId %main FPFastMathDefault %float %full\n";
static const char zero_default[] = "OpExecutionModeId %main FPFastMathDefault %float %none\n";
static const char add_body[] = "%result = OpFAdd %float %value %one\n";
static const char full_decoration[] = "OpDecorate %result FPFastMathMode NotNaN|NotInf|NSZ|AllowRecip|AllowContract|AllowReassoc|AllowTransform\n";

struct Fixture
{
    std::string capabilities;
    std::string entries;
    std::string modes;
    std::string decorations;
    std::string types;
    std::string body;
    std::string functions;
    bool integer_buffer;
    bool float_controls2;

    Fixture()
        : modes(full_default), body(add_body), integer_buffer(false), float_controls2(true)
    {
    }

    std::string assembly() const
    {
        const char* scalar = integer_buffer ? "%uint" : "%float";
        std::ostringstream source;
        source << "OpCapability Shader\n";
        if (float_controls2)
            source << "OpCapability FloatControls2\n";
        source << capabilities;
        if (float_controls2)
            source << "OpExtension \"SPV_KHR_float_controls2\"\n";
        source << "OpExtension \"SPV_KHR_float_controls\"\n"
                  "%glsl = OpExtInstImport \"GLSL.std.450\"\n"
                  "OpMemoryModel Logical GLSL450\n"
                  "OpEntryPoint GLCompute %main \"main\"\n"
               << entries
               << "OpExecutionMode %main LocalSize 1 1 1\n"
               << modes
               << "OpName %data \"data_buffer\"\n"
                  "OpDecorate %array ArrayStride 4\n"
                  "OpMemberDecorate %block 0 Offset 0\n"
                  "OpDecorate %block Block\n"
                  "OpDecorate %data DescriptorSet 0\n"
                  "OpDecorate %data Binding 0\n"
               << decorations
               << "%void = OpTypeVoid\n"
                  "%uint = OpTypeInt 32 0\n"
                  "%float = OpTypeFloat 32\n"
                  "%void_function = OpTypeFunction %void\n"
                  "%float_function = OpTypeFunction %float\n"
                  "%zero = OpConstant %uint 0\n"
                  "%full = OpConstant %uint 458767\n"
                  "%partial = OpConstant %uint 65543\n"
                  "%finite = OpConstant %uint 3\n"
                  "%none = OpConstant %uint 0\n"
                  "%one = OpConstant %float 1\n"
               << types
               << "%array = OpTypeRuntimeArray " << scalar << "\n"
               << "%block = OpTypeStruct %array\n"
                  "%block_pointer = OpTypePointer StorageBuffer %block\n"
               << "%scalar_pointer = OpTypePointer StorageBuffer " << scalar << "\n"
               << "%data = OpVariable %block_pointer StorageBuffer\n"
                  "%main = OpFunction %void None %void_function\n"
                  "%main_label = OpLabel\n"
                  "%pointer = OpAccessChain %scalar_pointer %data %zero %zero\n"
               << "%value = OpLoad " << scalar << " %pointer\n"
               << body
               << "OpStore %pointer %result\n"
                  "OpReturn\n"
                  "OpFunctionEnd\n"
               << functions;
        return source.str();
    }
};

enum BuildOption
{
    FiniteMath = 1,
    NoSignedZeros = 2,
    MadEnable = 4,
    FastRelaxedMath = 8,
    OptimizationDisabled = 16
};

struct ExpectedControls
{
    uint32_t flags;
    uint32_t widths;
    uint32_t denorm_preserve;
    uint32_t round_to_nearest;
    uint32_t signed_zero_inf_nan_preserve;
    bool builtin_accuracy;
    bool float_controls2;
    unsigned int build_options;
    bool contraction_off;

    ExpectedControls(uint32_t flags_, unsigned int build_options_)
        : flags(flags_), widths(momoten::FloatingPointWidth32), denorm_preserve(0), round_to_nearest(0), signed_zero_inf_nan_preserve(0), builtin_accuracy(false), float_controls2(true), build_options(build_options_), contraction_off((flags_ & 0x10000u) == 0)
    {
    }
};

static bool assemble(const char* name, const Fixture& fixture, std::vector<uint32_t>& words)
{
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([name](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        fprintf(stderr, "float_controls_test: %s SPIR-V at %zu:%zu: %s\n", name, position.line, position.column, message);
    });
    return tools.Assemble(fixture.assembly(), &words) && tools.Validate(words);
}

static int check_translation(const char* name, const Fixture& fixture, const ExpectedControls& expected, const char* entry_point = "main", const char* relaxed_math_standard = "-cl-std=CL3.0")
{
    std::vector<uint32_t> words;
    if (!assemble(name, fixture, words))
        return 1;

    momoten::TranslationOptions options;
    options.entry_point = entry_point;
    momoten::TranslationResult translated;
    if (!momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated))
    {
        fprintf(stderr, "float_controls_test: %s translation failed: %s\n", name, translated.diagnostics.c_str());
        return 1;
    }

    const momoten::FloatingPointControls& actual = translated.abi.floating_point;
    if (translated.abi.float_controls2 != expected.float_controls2 || actual.fast_math_flags != expected.flags || actual.widths != expected.widths
        || actual.denorm_preserve_widths != expected.denorm_preserve || actual.round_to_nearest_widths != expected.round_to_nearest
        || actual.signed_zero_inf_nan_preserve_widths != expected.signed_zero_inf_nan_preserve
        || actual.requires_builtin_accuracy != expected.builtin_accuracy)
    {
        fprintf(stderr, "float_controls_test: %s controls mismatch: feature=%d flags=0x%x widths=0x%x denorm=0x%x rte=0x%x signed_zero_inf_nan=0x%x builtin_accuracy=%d\n", name, translated.abi.float_controls2, actual.fast_math_flags, actual.widths, actual.denorm_preserve_widths, actual.round_to_nearest_widths, actual.signed_zero_inf_nan_preserve_widths, actual.requires_builtin_accuracy);
        return 1;
    }

    const bool contraction_off = translated.source.find("#pragma OPENCL FP_CONTRACT OFF") != std::string::npos;
    if (contraction_off != expected.contraction_off)
    {
        fprintf(stderr, "float_controls_test: %s FP_CONTRACT OFF does not match the selected entry point's permissions\n", name);
        return 1;
    }

    momoten_detail::ShaderDeviceProfile profile;
    profile.relaxed_math_standard = relaxed_math_standard;
    const std::string build_options = momoten_detail::opencl_build_options(translated.abi, profile);
    const bool fast_relaxed = build_options.find("-cl-fast-relaxed-math") != std::string::npos;
    const struct
    {
        const char* option;
        unsigned int flag;
        bool implied_by_fast_relaxed;
    } option_checks[] = {
        {"-cl-finite-math-only", FiniteMath, true},
        {"-cl-no-signed-zeros", NoSignedZeros, true},
        {"-cl-mad-enable", MadEnable, true},
        {"-cl-fast-relaxed-math", FastRelaxedMath, false},
        {"-cl-opt-disable", OptimizationDisabled, false}};
    for (size_t i = 0; i < sizeof(option_checks) / sizeof(option_checks[0]); i++)
    {
        const bool enabled = build_options.find(option_checks[i].option) != std::string::npos || (fast_relaxed && option_checks[i].implied_by_fast_relaxed);
        if (enabled != ((expected.build_options & option_checks[i].flag) != 0))
        {
            fprintf(stderr, "float_controls_test: %s unexpected selection for %s: '%s'\n", name, option_checks[i].option, build_options.c_str());
            return 1;
        }
    }
    if (((expected.build_options & (MadEnable | FastRelaxedMath)) && build_options.find(relaxed_math_standard) == std::string::npos)
        || (!*relaxed_math_standard && build_options.find("-cl-std=") != std::string::npos))
    {
        fprintf(stderr, "float_controls_test: %s relaxed math did not select the required language standard: '%s'\n", name, build_options.c_str());
        return 1;
    }
    return 0;
}

static int check_rejection(const char* name, const Fixture& fixture, const char* diagnostic)
{
    std::vector<uint32_t> words;
    if (!assemble(name, fixture, words))
        return 1;

    momoten::TranslationOptions options;
    momoten::TranslationResult translated;
    if (momoten::translate_spirv_to_opencl_c(words.data(), words.size(), options, translated)
        || translated.diagnostics.find(diagnostic) == std::string::npos)
    {
        fprintf(stderr, "float_controls_test: %s was not rejected with a %s diagnostic: %s\n", name, diagnostic, translated.diagnostics.c_str());
        return 1;
    }
    return 0;
}

static int check_pipeline_case(const char* name, VkDevice device, const momoten::KernelABI& abi, const char* expected_error = 0)
{
    std::string diagnostic;
    const bool accepted = momoten_detail::validate_pipeline_abi(device, abi, diagnostic);
    if (accepted != (expected_error == 0) || (expected_error && diagnostic.find(expected_error) == std::string::npos))
    {
        fprintf(stderr, "float_controls_test: %s pipeline validation mismatch: accepted=%d diagnostic=%s\n", name, accepted, diagnostic.c_str());
        return 1;
    }
    return 0;
}

static int check_pipeline_validation()
{
    VkPhysicalDevice_T physical = {};
    physical.address_bits = 32;
    physical.max_parameter_size = 1024;
    physical.max_workgroup_size = 1;
    for (size_t d = 0; d < 3; d++)
        physical.max_work_item_sizes[d] = 1;
    VkDevice_T device = {};
    device.physical_device = &physical;

    momoten::KernelABI abi;
    abi.address_bits = 32;
    abi.float_controls2 = true;
    int status = check_pipeline_case("float controls 2 not enabled", &device, abi, "float controls 2");
    device.enabled_shader_profile.float_controls2 = true;
    status |= check_pipeline_case("float controls 2 enabled", &device, abi);

    abi.floating_point.denorm_preserve_widths = momoten::FloatingPointWidth32;
    status |= check_pipeline_case("denorm preservation unavailable", &device, abi, "DenormPreserve");
    physical.shader_profile.denorm_preserve_widths = momoten::FloatingPointWidth32;
    status |= check_pipeline_case("denorm preservation supported", &device, abi);

    abi.floating_point.round_to_nearest_widths = momoten::FloatingPointWidth32;
    status |= check_pipeline_case("round-to-nearest unavailable", &device, abi, "RoundingModeRTE");
    physical.shader_profile.round_to_nearest_widths = momoten::FloatingPointWidth32;
    status |= check_pipeline_case("round-to-nearest supported", &device, abi);

    abi.float_controls2 = false;
    device.enabled_shader_profile.float_controls2 = false;
    abi.floating_point.signed_zero_inf_nan_preserve_widths = momoten::FloatingPointWidth32;
    physical.shader_profile.signed_zero_inf_nan_preserve_widths = 0;
    status |= check_pipeline_case("signed zero inf nan preservation unavailable", &device, abi, "SignedZeroInfNanPreserve");
    physical.shader_profile.signed_zero_inf_nan_preserve_widths = momoten::FloatingPointWidth32;
    status |= check_pipeline_case("legacy signed zero inf nan preservation supported", &device, abi);
    return status;
}

int main()
{
    const unsigned int relaxed_options = FiniteMath | NoSignedZeros | MadEnable | FastRelaxedMath;
    const unsigned int limited_options = FiniteMath | NoSignedZeros | MadEnable;
    int status = 0;
    Fixture fixture;
    status |= check_translation("complete default", fixture, ExpectedControls(full_flags, relaxed_options));
    status |= check_translation("complete default with OpenCL C 2.0", fixture, ExpectedControls(full_flags, relaxed_options), "main", "-cl-std=CL2.0");
    status |= check_translation("old or embedded profile avoids arbitrary-accuracy mad", fixture, ExpectedControls(full_flags, FiniteMath | NoSignedZeros), "main", "");

    fixture.modes = "OpExecutionModeId %main FPFastMathDefault %float %partial\n";
    status |= check_translation("partial default", fixture, ExpectedControls(0x10007, limited_options));
    fixture.modes = "OpExecutionModeId %main FPFastMathDefault %float %finite\n";
    status |= check_translation("finite-only default", fixture, ExpectedControls(3, FiniteMath));
    fixture.modes = zero_default;
    status |= check_translation("zero default", fixture, ExpectedControls(0, 0));

    fixture = Fixture();
    fixture.decorations = "OpDecorate %result FPFastMathMode NotNaN\n";
    status |= check_translation("instruction tightens default", fixture, ExpectedControls(1, 0));
    fixture.modes = zero_default;
    fixture.decorations = full_decoration;
    status |= check_translation("instruction cannot relax unannotated memory operations", fixture, ExpectedControls(0, 0));

    fixture = Fixture();
    fixture.capabilities = "OpCapability Float64\n";
    fixture.types = "%double = OpTypeFloat 64\n";
    fixture.integer_buffer = true;
    fixture.body = "%wide = OpConvertUToF %double %value\n"
                   "%narrow = OpFConvert %float %wide\n"
                   "%result = OpConvertFToU %uint %narrow\n";
    ExpectedControls mixed_strict(0, OptimizationDisabled);
    mixed_strict.widths = momoten::FloatingPointWidth32 | momoten::FloatingPointWidth64;
    status |= check_translation("omitted type default is zero", fixture, mixed_strict);
    fixture.modes = "OpExecutionModeId %main FPFastMathDefault %float %none\n"
                    "OpExecutionModeId %main FPFastMathDefault %double %full\n";
    fixture.body = "%wide = OpConvertUToF %double %value\n"
                   "%narrow = OpFConvert %float %wide\n"
                   "%back = OpFConvert %double %narrow\n"
                   "%result = OpConvertFToU %uint %back\n";
    ExpectedControls mixed_relaxed(full_flags, OptimizationDisabled);
    mixed_relaxed.widths = momoten::FloatingPointWidth32 | momoten::FloatingPointWidth64;
    status |= check_translation("conversion combines operand and result permissions", fixture, mixed_relaxed);

    fixture = Fixture();
    fixture.capabilities = "OpCapability Float16\n";
    fixture.types = "%half = OpTypeFloat 16\n";
    fixture.integer_buffer = true;
    fixture.modes = "OpExecutionModeId %main FPFastMathDefault %half %full\n";
    fixture.body = "%half_value = OpConvertUToF %half %value\n"
                   "%result = OpConvertFToU %uint %half_value\n";
    ExpectedControls half_controls(full_flags, FiniteMath | NoSignedZeros);
    half_controls.widths = momoten::FloatingPointWidth16;
    status |= check_translation("half avoids unsafe math and mad options", fixture, half_controls);

    fixture = Fixture();
    fixture.body = "%absolute = OpExtInst %float %glsl FAbs %value\n"
                   "%fma = OpExtInst %float %glsl Fma %absolute %one %one\n"
                   "%minimum = OpExtInst %float %glsl FMin %fma %one\n"
                   "%result = OpExtInst %float %glsl FMax %minimum %one\n";
    status |= check_translation("safe builtins permit relaxed math", fixture, ExpectedControls(full_flags, relaxed_options));
    fixture.body = "%result = OpExtInst %float %glsl Exp %value\n";
    ExpectedControls accurate_builtin(full_flags, limited_options);
    accurate_builtin.builtin_accuracy = true;
    status |= check_translation("exp preserves builtin accuracy", fixture, accurate_builtin);

    fixture = Fixture();
    fixture.entries = "OpEntryPoint GLCompute %other \"other\"\n";
    fixture.modes += "OpExecutionMode %other LocalSize 1 1 1\n"
                     "OpExecutionModeId %other FPFastMathDefault %float %none\n";
    fixture.functions = "%other = OpFunction %void None %void_function\n"
                        "%other_label = OpLabel\n"
                        "%other_pointer = OpAccessChain %scalar_pointer %data %zero %zero\n"
                        "%other_value = OpLoad %float %other_pointer\n"
                        "%other_result = OpExtInst %float %glsl Exp %other_value\n"
                        "OpStore %other_pointer %other_result\n"
                        "OpReturn\n"
                        "OpFunctionEnd\n";
    status |= check_translation("selected entry ignores another entry", fixture, ExpectedControls(full_flags, relaxed_options));
    ExpectedControls other_controls(0, 0);
    other_controls.builtin_accuracy = true;
    status |= check_translation("other entry uses its own default", fixture, other_controls, "other");

    fixture = Fixture();
    fixture.decorations = "OpDecorate %helper_sum FPFastMathMode None\n";
    fixture.functions = "%helper = OpFunction %float None %float_function\n"
                        "%helper_label = OpLabel\n"
                        "%helper_pointer = OpAccessChain %scalar_pointer %data %zero %zero\n"
                        "%helper_value = OpLoad %float %helper_pointer\n"
                        "%helper_sum = OpFAdd %float %helper_value %one\n"
                        "%helper_result = OpExtInst %float %glsl Exp %helper_sum\n"
                        "OpReturnValue %helper_result\n"
                        "OpFunctionEnd\n";
    status |= check_translation("unreachable helper is isolated", fixture, ExpectedControls(full_flags, relaxed_options));
    fixture.body = "%result = OpFunctionCall %float %helper\n";
    status |= check_translation("reachable helper constrains the entry", fixture, other_controls);

    fixture = Fixture();
    fixture.capabilities = "OpCapability DenormPreserve\n";
    fixture.modes = std::string(zero_default) + "OpExecutionMode %main DenormPreserve 32\n";
    ExpectedControls preserve_strict(0, 0);
    preserve_strict.denorm_preserve = momoten::FloatingPointWidth32;
    status |= check_translation("denorm preservation and strict contraction", fixture, preserve_strict);
    fixture.modes = std::string(full_default) + "OpExecutionMode %main DenormPreserve 32\n";
    ExpectedControls preserve_relaxed(full_flags, limited_options);
    preserve_relaxed.denorm_preserve = momoten::FloatingPointWidth32;
    status |= check_translation("denorm preservation limits relaxed options", fixture, preserve_relaxed);

    fixture = Fixture();
    fixture.capabilities = "OpCapability RoundingModeRTE\n";
    fixture.modes += "OpExecutionMode %main RoundingModeRTE 32\n";
    ExpectedControls nearest(full_flags, relaxed_options);
    nearest.round_to_nearest = momoten::FloatingPointWidth32;
    status |= check_translation("round-to-nearest reflection", fixture, nearest);

    fixture = Fixture();
    fixture.float_controls2 = false;
    fixture.capabilities = "OpCapability SignedZeroInfNanPreserve\n";
    fixture.modes = "OpExecutionMode %main SignedZeroInfNanPreserve 32\n";
    ExpectedControls legacy_preservation(0x70008, MadEnable);
    legacy_preservation.float_controls2 = false;
    legacy_preservation.signed_zero_inf_nan_preserve = momoten::FloatingPointWidth32;
    status |= check_translation("legacy signed zero inf nan preservation", fixture, legacy_preservation);

    fixture.capabilities += "OpCapability Float64\n";
    fixture.types = "%double = OpTypeFloat 64\n";
    fixture.integer_buffer = true;
    fixture.body = "%wide = OpConvertUToF %double %value\n"
                   "%narrow = OpFConvert %float %wide\n"
                   "%back = OpFConvert %double %narrow\n"
                   "%result = OpConvertFToU %uint %back\n";
    ExpectedControls legacy_mixed_preservation(0x70008, OptimizationDisabled);
    legacy_mixed_preservation.float_controls2 = false;
    legacy_mixed_preservation.widths = momoten::FloatingPointWidth32 | momoten::FloatingPointWidth64;
    legacy_mixed_preservation.signed_zero_inf_nan_preserve = momoten::FloatingPointWidth32;
    status |= check_translation("legacy preservation survives cross-width conversion", fixture, legacy_mixed_preservation);

    fixture = Fixture();
    fixture.float_controls2 = false;
    fixture.modes.clear();
    ExpectedControls legacy_default(full_flags, relaxed_options);
    legacy_default.float_controls2 = false;
    status |= check_translation("legacy unconstrained arithmetic", fixture, legacy_default);
    fixture.decorations = "OpDecorate %result NoContraction\n";
    ExpectedControls legacy_no_contraction(7, FiniteMath | NoSignedZeros);
    legacy_no_contraction.float_controls2 = false;
    status |= check_translation("legacy NoContraction restricts all transform permissions", fixture, legacy_no_contraction);

    fixture = Fixture();
    fixture.capabilities = "OpCapability DenormFlushToZero\n";
    fixture.modes += "OpExecutionMode %main DenormFlushToZero 32\n";
    status |= check_rejection("flush-to-zero unsupported", fixture, "DenormFlushToZero");
    fixture = Fixture();
    fixture.capabilities = "OpCapability RoundingModeRTZ\n";
    fixture.modes += "OpExecutionMode %main RoundingModeRTZ 32\n";
    status |= check_rejection("round-toward-zero unsupported", fixture, "RoundingModeRTZ");
    return status | check_pipeline_validation();
}
