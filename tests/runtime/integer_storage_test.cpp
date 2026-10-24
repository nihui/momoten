// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include <vulkan/vulkan.h>
#include "../spirv_test_helpers.h"
#include "../integer_conversion_fixture.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>

enum TestMode
{
    Int8Storage,
    Int8VectorStorage,
    Int8Conversion,
    Int16Arithmetic,
    Int16Conversion,
    StructAtomic,
    Storage16,
    InvalidMode
};

static const size_t storage16_uniform_bytes = 65536 * 48 + 128;
static const size_t storage16_output_bytes = 65536 * 8 * 4;

template<typename T>
static void write_value(std::vector<unsigned char>& bytes, size_t offset, T value)
{
    memcpy(bytes.data() + offset, &value, sizeof(value));
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t half_float_bits(uint16_t value)
{
    const int exponent = (value >> 10) & 31;
    const int mantissa = value & 1023;
    if (exponent == 31)
        return (uint32_t(value & 32768) << 16) | 0x7f800000u | (uint32_t(mantissa) << 13);
    const float magnitude = exponent ? std::ldexp(1.f + mantissa / 1024.f, exponent - 15) : std::ldexp(float(mantissa), -24);
    return float_bits(value & 32768 ? -magnitude : magnitude);
}

static uint16_t float_half_bits(float value, bool round_to_zero = false)
{
    const uint16_t sign = std::signbit(value) ? 0x8000u : 0;
    const double magnitude = std::fabs(double(value));
    if (magnitude == 0)
        return sign;
    int exponent;
    std::frexp(magnitude, &exponent);
    exponent--;
    double scaled = std::ldexp(magnitude, exponent < -14 ? 24 : 10 - exponent);
    uint32_t rounded = uint32_t(std::floor(scaled));
    const double remainder = scaled - rounded;
    if (!round_to_zero && (remainder > 0.5 || (remainder == 0.5 && (rounded & 1))))
        rounded++;
    if (exponent < -14)
        return sign | uint16_t(rounded);
    if (rounded == 2048)
    {
        exponent++;
        rounded = 1024;
    }
    if (exponent > 15)
        return sign | (round_to_zero ? 0x7bffu : 0x7c00u);
    return sign | uint16_t((exponent + 15) * 1024 + rounded - 1024);
}

static std::vector<unsigned char> storage16_push_data()
{
    std::vector<unsigned char> bytes(64, 0xa5);
    write_value<int16_t>(bytes, 0, -1234);
    write_value<uint16_t>(bytes, 2, 60000);
    write_value<uint16_t>(bytes, 8, 0xc300);
    const int16_t values[] = {32767, -32768, 7, -9};
    memcpy(bytes.data() + 16, values, sizeof(values));
    write_value<uint32_t>(bytes, 32, 17);
    write_value<float>(bytes, 40, 0.25f);
    write_value<uint16_t>(bytes, 48, 0x3800);
    write_value<uint16_t>(bytes, 50, 0xbc00);
    return bytes;
}

static int fail(const char* operation, VkResult result)
{
    fprintf(stderr, "integer_storage_test: %s failed with Vulkan error %d\n", operation, result);
    return 1;
}

static int signed16(uint32_t value)
{
    value &= 65535u;
    return value < 32768u ? int(value) : int(value) - 65536;
}

static int arithmetic_shift(int value, uint32_t shift)
{
    const int divisor = 1 << shift;
    return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

static int signed_mod(int a, int b)
{
    const int remainder = a % b;
    return remainder && ((remainder < 0) != (b < 0)) ? remainder + b : remainder;
}

static void make_storage16_data(std::vector<unsigned char>& input, std::vector<unsigned char>& expected)
{
    const uint32_t count = 65536;
    const bool round_to_zero = std::getenv("MOMOTEN_TEST_FP16_RTZ") != 0;
    input.assign(storage16_uniform_bytes + storage16_output_bytes + count * 8, 0xa5);
    for (uint32_t i = 0; i < count; i++)
    {
        write_value<uint16_t>(input, i * 48, i);
        write_value<uint16_t>(input, i * 48 + 16, 65535u - i);
        write_value<uint16_t>(input, i * 48 + 32, (i * 40503u + 13u) & 65535u);
        write_value<uint16_t>(input, i * 48 + 34, i);
        write_value<uint16_t>(input, i * 48 + 36, 0x3c00);
        write_value<uint16_t>(input, i * 48 + 38, 0x4000);
        for (uint32_t lane = 0; lane < 4; lane++)
            write_value<uint16_t>(input, i * 48 + 40 + lane * 2, i * (lane + 1) + lane * 23);
    }
    for (uint32_t matrix = 0; matrix < 2; matrix++)
        for (uint32_t column = 0; column < 4; column++)
            for (uint32_t row = 0; row < 4; row++)
                write_value<float>(input, 65536 * 48 + matrix * 64 + column * 16 + row * 4,
                                   float(1 + matrix * 16 + column * 4 + row));
    expected = input;
    const int16_t pushed[] = {32767, -32768, 7, -9};
    for (uint32_t i = 0; i < count; i++)
    {
        const uint32_t lane = i & 3u;
        const uint32_t values[] = {uint32_t(signed16(i) - 1234), uint32_t(signed16(65535u - i) - pushed[lane]),
                                   ((i * 40503u + 13u) & 65535u) ^ 60000u, half_float_bits(i),
                                   (i * (lane + 1) + lane * 23) & 65535u, float_bits(i & 1u ? 24.5f : 7.5f),
                                   ((((i + 17u) & 65535u) * 40503u + 13u) & 65535u), float_bits(i & 1u ? -0.75f : 0.75f)};
        memcpy(expected.data() + storage16_uniform_bytes + i * 32, values, sizeof(values));
        const size_t half_offset = storage16_uniform_bytes + storage16_output_bytes + i * 8;
        write_value<uint16_t>(expected, half_offset, i);
        write_value<uint16_t>(expected, half_offset + 2, i);
        volatile float product = float(i) * 0.1f;
        const float value = i < 64 ? std::ldexp(float(int(i) - 32), -27) : i < 128 ? float(int(i) - 96) * 2048.5f
                                                                                   : product - 5000.25f;
        write_value<uint16_t>(expected, half_offset + 4, float_half_bits(value, round_to_zero));
        write_value<uint16_t>(expected, half_offset + 6, float_half_bits(-value + 0.03125f, round_to_zero));
    }
}

static void make_int8_data(std::vector<unsigned char>& input, std::vector<unsigned char>& expected, bool vector_storage)
{
    const uint32_t count = 65536;
    const size_t lanes = !vector_storage ? 1 : 4;
    input.assign((count + 2) * lanes, 0xa5);
    for (uint32_t i = 0; i < count; i++)
        for (size_t lane = 0; lane < lanes; lane++)
            input[(i + 1) * lanes + lane] = static_cast<unsigned char>(i * 37 + lane * 61 + 13);
    expected = input;
    for (uint32_t i = 0; i < count; i++)
        for (size_t lane = 0; lane < lanes; lane++)
        {
            const size_t index = (i + 1) * lanes + lane;
            const uint32_t value = input[index];
            if (!vector_storage)
            {
                const int v = value < 128 ? int(value) : int(value) - 256;
                expected[index] = static_cast<unsigned char>(v * 3 + (v < 0 ? 31 : 3) + int(i & 255u));
            }
            else
            {
                const uint32_t bias = lane == 0 ? 1 : lane == 1 ? 127
                                                  : lane == 2   ? 255
                                                                : i;
                expected[index] = static_cast<unsigned char>((value + bias) ^ i);
            }
        }
}

static void make_int8_conversion_data(std::vector<unsigned char>& input, std::vector<unsigned char>& expected)
{
    const uint32_t count = 65536;
    input.assign(count * 70, 0xa5);
    const size_t widths[] = {1, 2, 4};
    const size_t input_offsets[] = {0, count, count * 3, count * 7, count * 8, count * 10};
    const size_t output_offsets[] = {0, 1, 3, 7, 8, 10};
    for (uint32_t i = 0; i < count; i++)
        for (size_t conversion = 0; conversion < 6; conversion++)
            for (size_t lane = 0; lane < widths[conversion % 3]; lane++)
                input[input_offsets[conversion] + i * widths[conversion % 3] + lane] = static_cast<unsigned char>(i * 37 + conversion * 13 + lane * 61);
    expected = input;
    for (uint32_t i = 0; i < count; i++)
        for (size_t conversion = 0; conversion < 6; conversion++)
            for (size_t lane = 0; lane < widths[conversion % 3]; lane++)
            {
                const int value = input[input_offsets[conversion] + i * widths[conversion % 3] + lane];
                const int result = conversion < 3 && value >= 128 ? value - 256 : value;
                write_value<uint32_t>(expected, count * 14 + (i * 14 + output_offsets[conversion] + lane) * 4, uint32_t(result));
            }
}

static void make_int16_data(std::vector<unsigned char>& input, std::vector<unsigned char>& expected)
{
    const uint32_t count = 65536;
    std::vector<uint32_t> words(count * 32, 0xa5a5a5a5u);
    for (uint32_t i = 0; i < count; i++)
    {
        words[i * 32] = i;
        words[i * 32 + 1] = (i * 40503u + 13u) & 65535u;
    }
    input.resize(words.size() * sizeof(uint32_t));
    memcpy(input.data(), words.data(), input.size());
    for (uint32_t i = 0; i < count; i++)
    {
        uint32_t* out = words.data() + i * 32;
        const uint32_t a = out[0];
        const uint32_t b = out[1];
        const int sa = signed16(a);
        const int sb = signed16(b);
        out[2] = (a + b) & 65535u;
        out[3] = (a - b) & 65535u;
        out[4] = (a * b) & 65535u;
        out[5] = static_cast<uint32_t>(signed16(sa + sb));
        out[6] = static_cast<uint32_t>(signed16(-sa));
        out[7] = (a << (i & 15u)) & 65535u;
        out[8] = static_cast<uint32_t>(arithmetic_shift(sa, i & 15u));
        out[9] = a / (b | 1u);
        out[10] = sa < sb;
        out[11] = signed16(a * b) < 0;
        out[12] = ((a + b) & 65535u) | (((65535u - a + 32768u) & 65535u) << 16);
        out[13] = ((a * b) & 65535u) | ((((a ^ b) * 65535u) & 65535u) << 16);
        out[14] = static_cast<uint32_t>(signed16(signed16(a ^ b) + sb));
        out[15] = b >= 32768u ? (a + b) & 65535u : b;
        const int divisor = signed16(b | 1u);
        const int c = signed16(a ^ b);
        out[16] = static_cast<uint32_t>(signed_mod(sa, divisor));
        out[17] = a % (b | 1u);
        out[18] = static_cast<uint32_t>(signed16(sa / divisor));
        out[19] = static_cast<uint32_t>(sa < sb ? sa : sb);
        out[20] = static_cast<uint32_t>(signed16(sa < 0 ? -sa : sa));
        out[21] = static_cast<uint32_t>((sa > 0) - (sa < 0));
        out[22] = out[19];
        out[23] = static_cast<uint32_t>(sa < -1000 ? -1000 : sa > 1000 ? 1000
                                                                       : sa);
        out[24] = (a * 987655u) & 65535u;
        out[25] = static_cast<uint32_t>(signed16(a * 987655u));
        out[26] = ((a ^ b) << (i & 15u)) & 65535u;
        out[27] = static_cast<uint32_t>(arithmetic_shift(c, i & 15u));
        out[28] = static_cast<uint32_t>(signed16(c < 0 ? -c : c));
        out[29] = static_cast<uint32_t>((c > 0) - (c < 0));
        out[30] = static_cast<uint32_t>(signed_mod(c, divisor));
        const uint32_t neighbor = (i & ~63u) | ((i + 1u) & 63u);
        out[31] = (neighbor ^ ((neighbor * 40503u + 13u) & 65535u)) >> (i & 15u);
        out[0] = static_cast<uint32_t>(sa);
        out[1] = static_cast<uint32_t>(sb);
    }
    expected.resize(input.size());
    memcpy(expected.data(), words.data(), expected.size());
}

static void make_integer_conversion_data(std::vector<unsigned char>& input, std::vector<unsigned char>& expected)
{
    const uint32_t count = 65536;
    std::vector<uint32_t> words(count * integer_conversion_words, 0xa5a5a5a5u);
    for (uint32_t i = 0; i < count; i++)
    {
        uint32_t* values = words.data() + i * integer_conversion_words;
        values[0] = i;
        values[1] = 65535u - i;
        values[2] = i ^ 32768u;
        values[3] = (i * 40503u + 13u) & 65535u;
    }
    input.resize(words.size() * sizeof(uint32_t));
    memcpy(input.data(), words.data(), input.size());
    for (uint32_t i = 0; i < count; i++)
    {
        uint32_t* values = words.data() + i * integer_conversion_words;
        uint32_t output = 4;
        for (uint32_t width : {1u, 4u})
            for (uint32_t conversion = 0; conversion < 4; conversion++)
                for (uint32_t lane = 0; lane < width; lane++)
                {
                    const uint32_t bits = values[lane];
                    const int signed_value = signed16(bits);
                    const uint32_t results[] = {bits, uint32_t(signed_value), float_bits(float(bits)), float_bits(float(signed_value))};
                    values[output++] = results[conversion];
                }
    }
    expected.resize(input.size());
    memcpy(expected.data(), words.data(), expected.size());
}

static void make_struct_atomic_data(std::vector<unsigned char>& input, std::vector<unsigned char>& expected)
{
    std::vector<uint32_t> words(65536 * 8, 0xa5a5a5a5u);
    for (uint32_t i = 0; i < 65536; i++)
    {
        words[i * 8] = i * 40503u + 13u;
        words[i * 8 + 1] = uint32_t(int(i) - 32768);
    }
    input.resize(words.size() * sizeof(uint32_t));
    memcpy(input.data(), words.data(), input.size());
    for (uint32_t i = 0; i < 65535; i++)
    {
        uint32_t* item = words.data() + i * 8;
        const uint32_t neighbor = (i & ~63u) | ((i + 1u) & 63u);
        const uint32_t before = item[0];
        item[0] ^= 0x80000000u | neighbor;
        item[4] = item[1];
        item[1] = uint32_t(int(neighbor) - 131072);
        item[2] = before;
        item[3] = item[0];
        item[5] = item[1];
        item[6] = before;
        item[7] = item[0];
    }
    expected.resize(input.size());
    memcpy(expected.data(), words.data(), expected.size());
}

static void make_data(TestMode mode, std::vector<unsigned char>& input, std::vector<unsigned char>& expected)
{
    switch (mode)
    {
    case Int8Storage: make_int8_data(input, expected, false); break;
    case Int8VectorStorage: make_int8_data(input, expected, true); break;
    case Int8Conversion: make_int8_conversion_data(input, expected); break;
    case Int16Arithmetic: make_int16_data(input, expected); break;
    case Int16Conversion: make_integer_conversion_data(input, expected); break;
    case StructAtomic: make_struct_atomic_data(input, expected); break;
    case Storage16: make_storage16_data(input, expected); break;
    default: break;
    }
}

static int run_kernel(VkPhysicalDevice physical, VkDevice device, VkQueue queue, const std::vector<uint32_t>& words, TestMode mode, bool enabled, bool native_fp16, bool use_template = false)
{
    const uint32_t binding_count = mode == Int8Conversion ? 7 : mode == Storage16 ? 3
                                                                                  : 1;
    VkDescriptorSetLayoutBinding bindings[7] = {};
    for (uint32_t i = 0; i < binding_count; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = mode == Storage16 && i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    set_info.bindingCount = binding_count;
    set_info.pBindings = bindings;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkResult result = vkCreateDescriptorSetLayout(device, &set_info, 0, &set_layout);
    if (result != VK_SUCCESS) return fail("vkCreateDescriptorSetLayout", result);
    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    const VkPushConstantRange push_range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    if (mode == Storage16)
    {
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
    }
    VkPipelineLayout layout = VK_NULL_HANDLE;
    result = vkCreatePipelineLayout(device, &layout_info, 0, &layout);
    if (result != VK_SUCCESS) return fail("vkCreatePipelineLayout", result);
    VkShaderModuleCreateInfo module_info = {};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = words.size() * sizeof(uint32_t);
    module_info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    result = vkCreateShaderModule(device, &module_info, 0, &module);
    if (result != VK_SUCCESS) return fail("vkCreateShaderModule", result);
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = layout;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, 0, &pipeline);
    if (!enabled)
    {
        if (result != VK_ERROR_FEATURE_NOT_PRESENT)
            return fail("pipeline without enabled integer feature", result);
        vkDestroyShaderModule(device, module, 0);
        vkDestroyPipelineLayout(device, layout, 0);
        vkDestroyDescriptorSetLayout(device, set_layout, 0);
        return 0;
    }
    if (result != VK_SUCCESS) return fail("vkCreateComputePipelines", result);

    std::vector<unsigned char> input;
    std::vector<unsigned char> expected;
    make_data(mode, input, expected);
    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(physical, &properties);
    const VkDeviceSize alignment = properties.limits.minStorageBufferOffsetAlignment;
    const VkDeviceSize descriptor_offset = (64 + alignment - 1) / alignment * alignment;
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = descriptor_offset + input.size();
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (mode == Storage16 ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : 0);
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    result = vkCreateBuffer(device, &buffer_info, 0, &buffer);
    if (result != VK_SUCCESS) return fail("vkCreateBuffer", result);
    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    const VkDeviceSize bind_offset = (64 + requirements.alignment - 1) / requirements.alignment * requirements.alignment;
    VkPhysicalDeviceMemoryProperties memory_properties = {};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    uint32_t memory_type = 0;
    while (memory_type < memory_properties.memoryTypeCount && (!(requirements.memoryTypeBits & (1u << memory_type)) || !(memory_properties.memoryTypes[memory_type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)))
        memory_type++;
    if (memory_type == memory_properties.memoryTypeCount) return fail("host-visible memory type", VK_ERROR_FEATURE_NOT_PRESENT);
    VkMemoryAllocateInfo allocation = {};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = bind_offset + requirements.size + 64;
    allocation.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    result = vkAllocateMemory(device, &allocation, 0, &memory);
    if (result != VK_SUCCESS) return fail("vkAllocateMemory", result);
    result = vkBindBufferMemory(device, buffer, memory, bind_offset);
    if (result != VK_SUCCESS) return fail("vkBindBufferMemory", result);
    void* mapping = 0;
    result = vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapping);
    if (result != VK_SUCCESS) return fail("vkMapMemory", result);
    memset(mapping, 0xa5, static_cast<size_t>(allocation.allocationSize));
    const size_t offset = static_cast<size_t>(bind_offset + descriptor_offset);
    memcpy(static_cast<unsigned char*>(mapping) + offset, input.data(), input.size());
    std::vector<unsigned char> expected_all(static_cast<size_t>(allocation.allocationSize), 0xa5);
    memcpy(expected_all.data() + offset, expected.data(), expected.size());
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.size = VK_WHOLE_SIZE;
    result = vkFlushMappedMemoryRanges(device, 1, &range);
    if (result != VK_SUCCESS) return fail("vkFlushMappedMemoryRanges", result);
    VkDescriptorBufferInfo descriptors[7] = {{buffer, descriptor_offset, input.size()},
                                             {buffer, descriptor_offset + storage16_uniform_bytes, storage16_output_bytes},
                                             {buffer, descriptor_offset + storage16_uniform_bytes + storage16_output_bytes, 65536 * 8}};
    if (mode == Storage16)
        descriptors[0].range = storage16_uniform_bytes;
    if (mode == Int8Conversion)
    {
        const VkDeviceSize sizes[] = {65536, 65536 * 2, 65536 * 4, 65536, 65536 * 2, 65536 * 4, 65536 * 14 * 4};
        VkDeviceSize position = descriptor_offset;
        for (uint32_t i = 0; i < binding_count; i++)
        {
            descriptors[i] = {buffer, position, sizes[i]};
            position += sizes[i];
        }
    }
    VkWriteDescriptorSet writes[7] = {};
    for (uint32_t i = 0; i < binding_count; i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].descriptorType;
        writes[i].pBufferInfo = &descriptors[i];
    }
    VkDescriptorUpdateTemplate descriptor_template = VK_NULL_HANDLE;
    if (use_template)
    {
        VkDescriptorUpdateTemplateEntry entries[7] = {};
        for (uint32_t i = 0; i < binding_count; i++)
        {
            entries[i].dstBinding = i;
            entries[i].descriptorCount = 1;
            entries[i].descriptorType = bindings[i].descriptorType;
            entries[i].offset = i * sizeof(VkDescriptorBufferInfo);
            entries[i].stride = sizeof(VkDescriptorBufferInfo);
        }
        VkDescriptorUpdateTemplateCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
        info.descriptorUpdateEntryCount = binding_count;
        info.pDescriptorUpdateEntries = entries;
        info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
        info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
        info.pipelineLayout = layout;
        result = vkCreateDescriptorUpdateTemplate(device, &info, 0, &descriptor_template);
        if (result != VK_SUCCESS) return fail("vkCreateDescriptorUpdateTemplate", result);
    }
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    VkCommandPool pool = VK_NULL_HANDLE;
    result = vkCreateCommandPool(device, &pool_info, 0, &pool);
    if (result != VK_SUCCESS) return fail("vkCreateCommandPool", result);
    VkCommandBufferAllocateInfo command_allocation = {};
    command_allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_allocation.commandPool = pool;
    command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocation.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    result = vkAllocateCommandBuffers(device, &command_allocation, &command);
    if (result != VK_SUCCESS) return fail("vkAllocateCommandBuffers", result);
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    result = vkBeginCommandBuffer(command, &begin);
    if (result != VK_SUCCESS) return fail("vkBeginCommandBuffer", result);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (use_template)
        vkCmdPushDescriptorSetWithTemplateKHR(command, descriptor_template, layout, 0, descriptors);
    else
        vkCmdPushDescriptorSetKHR(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, binding_count, writes);
    if (mode == Storage16)
    {
        const std::vector<unsigned char> pushed = storage16_push_data();
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushed.size(), pushed.data());
    }
    vkCmdDispatch(command, 1024, 1, 1);
    result = vkEndCommandBuffer(command);
    if (result != VK_SUCCESS) return fail("vkEndCommandBuffer", result);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    result = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) return fail("vkQueueSubmit", result);
    result = vkQueueWaitIdle(queue);
    if (result != VK_SUCCESS) return fail("vkQueueWaitIdle", result);
    result = vkInvalidateMappedMemoryRanges(device, 1, &range);
    if (result != VK_SUCCESS) return fail("vkInvalidateMappedMemoryRanges", result);
    const unsigned char* actual = static_cast<const unsigned char*>(mapping);
    if (mode == Storage16)
    {
        for (uint32_t i = 0; i < 65536; i++)
        {
            const size_t position = offset + storage16_uniform_bytes + i * 32 + 12;
            uint32_t value;
            memcpy(&value, actual + position, sizeof(value));
            const bool nan = (i & 0x7c00u) == 0x7c00u && (i & 1023u);
            const bool allowed_nan = nan && (value & 0x7f800000u) == 0x7f800000u && (value & 0x7fffffu);
            const bool allowed_flush = native_fp16 && (i & 0x7c00u) == 0 && value == (i & 32768u) << 16;
            if (allowed_nan || allowed_flush)
                memcpy(expected_all.data() + position, &value, sizeof(value));
            const size_t half_position = offset + storage16_uniform_bytes + storage16_output_bytes + i * 8 + 2;
            uint16_t half_value;
            memcpy(&half_value, actual + half_position, sizeof(half_value));
            if ((nan && (half_value & 0x7c00u) == 0x7c00u && (half_value & 1023u))
                || (native_fp16 && (i & 0x7c00u) == 0 && half_value == (i & 32768u)))
                memcpy(expected_all.data() + half_position, &half_value, sizeof(half_value));
        }
    }
    int status = 0;
    for (size_t i = 0; i < expected_all.size(); i++)
        if (actual[i] != expected_all[i])
        {
            fprintf(stderr, "integer_storage_test: mode %d byte %zu is %02x, expected %02x\n", mode, i, actual[i], expected_all[i]);
            if (mode == Int16Arithmetic && i >= offset && i < offset + input.size())
            {
                const size_t word = (i - offset) / sizeof(uint32_t);
                uint32_t value = 0;
                uint32_t reference = 0;
                memcpy(&value, actual + offset + word * sizeof(uint32_t), sizeof(uint32_t));
                memcpy(&reference, expected.data() + word * sizeof(uint32_t), sizeof(uint32_t));
                fprintf(stderr, "integer_storage_test: record %zu word %zu is %08x, expected %08x\n", word / 32, word % 32, value, reference);
            }
            status = 1;
            break;
        }
    vkUnmapMemory(device, memory);
    vkDestroyCommandPool(device, pool, 0);
    if (descriptor_template)
        vkDestroyDescriptorUpdateTemplate(device, descriptor_template, 0);
    vkDestroyBuffer(device, buffer, 0);
    vkFreeMemory(device, memory, 0);
    vkDestroyPipeline(device, pipeline, 0);
    vkDestroyShaderModule(device, module, 0);
    vkDestroyPipelineLayout(device, layout, 0);
    vkDestroyDescriptorSetLayout(device, set_layout, 0);
    return status;
}

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3)
        return 1;
    const char* name = argv[argc - 1];
    const bool aliased_int8 = strcmp(name, "int8-aliased") == 0 || strcmp(name, "int8-vector-aliased") == 0;
    TestMode mode = InvalidMode;
    if (strcmp(name, "int8") == 0 || strcmp(name, "int8-aliased") == 0) mode = Int8Storage;
    if (strcmp(name, "int8-vector") == 0 || strcmp(name, "int8-vector-aliased") == 0) mode = Int8VectorStorage;
    if (strcmp(name, "int8-conversion") == 0) mode = Int8Conversion;
    if (strcmp(name, "int16") == 0) mode = Int16Arithmetic;
    if (strcmp(name, "int16-conversion") == 0) mode = Int16Conversion;
    if (strcmp(name, "struct-atomic") == 0) mode = StructAtomic;
    if (strcmp(name, "storage16") == 0) mode = Storage16;
    if (mode == InvalidMode)
        return 1;
    std::vector<uint32_t> words;
    if (mode == Int16Conversion)
    {
        if (!make_integer_conversion_spirv(words))
            return 1;
    }
    else if (argc != 3 || !read_spirv(argv[1], words))
        return 1;
    if (aliased_int8 && !decorate_aliased_buffers(words, 1))
        return 1;
    const bool integer_arithmetic = mode == Int16Arithmetic || mode == Int16Conversion || mode == Int8Conversion;
    const bool byte_storage = mode == Int8Storage || mode == Int8VectorStorage || mode == Int8Conversion;
    if (std::getenv("MOMOTEN_TEST_FP16_RTZ") && !set_fp16_rounding(words, spv::FPRoundingModeRTZ))
        return 1;
    VkApplicationInfo application = {};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&instance_info, 0, &instance);
    if (result != VK_SUCCESS) return fail("vkCreateInstance", result);
    uint32_t count = 0;
    result = vkEnumeratePhysicalDevices(instance, &count, 0);
    if (result != VK_SUCCESS) return fail("vkEnumeratePhysicalDevices", result);
    if (!count)
    {
        vkDestroyInstance(instance, 0);
        return 77;
    }
    std::vector<VkPhysicalDevice> physicals(count);
    result = vkEnumeratePhysicalDevices(instance, &count, physicals.data());
    if (result != VK_SUCCESS) return fail("vkEnumeratePhysicalDevices", result);
    VkPhysicalDevice physical = physicals[0];
    VkPhysicalDevice8BitStorageFeatures storage = {};
    storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage16 = {};
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features arithmetic = {};
    arithmetic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    storage.pNext = &storage16;
    storage16.pNext = &arithmetic;
    VkPhysicalDeviceFeatures2 available = {};
    available.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    available.pNext = &storage;
    vkGetPhysicalDeviceFeatures2(physical, &available);
    if (std::getenv("MOMOTEN_TEST_NO_NATIVE_FP16") && arithmetic.shaderFloat16)
        return fail("fp16 arithmetic was not disabled", VK_ERROR_FEATURE_NOT_PRESENT);
    if ((integer_arithmetic && !available.features.shaderInt16) || (byte_storage && !storage.storageBuffer8BitAccess)
        || (mode == Storage16 && (!storage16.storageBuffer16BitAccess || !storage16.uniformAndStorageBuffer16BitAccess || !storage16.storagePushConstant16)))
    {
        vkDestroyInstance(instance, 0);
        return 77;
    }
    struct FeatureCase
    {
        const char* name;
        bool int16;
        bool storage8;
        bool storage16;
        bool uniform16;
        bool push16;
        bool succeeds;
    };
    std::vector<FeatureCase> cases;
    if (mode == Storage16)
    {
        cases = {{"missing storage buffer feature", false, false, false, true, true, false},
                 {"missing uniform buffer feature", false, false, true, false, true, false},
                 {"missing push constant feature", false, false, true, true, false, false},
                 {"all storage16 features", false, false, true, true, true, true},
                 {"storage8 permits storage16 buffers", false, true, false, true, true, true}};
    }
    else if (mode == Int8Conversion)
    {
        cases = {{"missing int16 feature", false, true, false, false, false, false},
                 {"missing storage8 feature", true, false, false, false, false, false},
                 {"enabled integer features", true, true, false, false, false, true}};
    }
    else if (mode == StructAtomic)
    {
        cases.push_back({"global compare-exchange", false, false, false, false, false, true});
    }
    else
    {
        cases = {{"missing feature", false, false, false, false, false, false},
                 {"enabled feature", integer_arithmetic, !integer_arithmetic, false, false, false, true}};
    }
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures requested = {};
    storage.uniformAndStorageBuffer8BitAccess = VK_FALSE;
    storage.storagePushConstant8 = VK_FALSE;
    storage.pNext = 0;
    storage16.pNext = 0;
    for (const FeatureCase& test : cases)
    {
        requested.shaderInt16 = test.int16;
        storage.storageBuffer8BitAccess = test.storage8;
        storage16.storageBuffer16BitAccess = test.storage16;
        storage16.uniformAndStorageBuffer16BitAccess = test.uniform16;
        storage16.storagePushConstant16 = test.push16;
        storage16.pNext = test.storage8 ? &storage : 0;
        std::vector<const char*> extensions = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
        if (mode == Storage16)
            extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        if (test.storage8 || byte_storage)
            extensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
        VkDeviceCreateInfo device_info = {};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.pEnabledFeatures = &requested;
        device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        device_info.ppEnabledExtensionNames = extensions.data();
        if (byte_storage || mode == Storage16)
            device_info.pNext = mode == Storage16 ? static_cast<void*>(&storage16) : static_cast<void*>(&storage);
        VkDevice device = VK_NULL_HANDLE;
        result = vkCreateDevice(physical, &device_info, 0, &device);
        if (result != VK_SUCCESS) return fail("vkCreateDevice", result);
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, 0, 0, &queue);
        int status = run_kernel(physical, device, queue, words, mode, test.succeeds, arithmetic.shaderFloat16 != 0);
        if (status == 0 && mode == Storage16 && test.succeeds)
            status = run_kernel(physical, device, queue, words, mode, true, arithmetic.shaderFloat16 != 0, true);
        vkDestroyDevice(device, 0);
        if (status)
        {
            fprintf(stderr, "integer_storage_test: %s failed\n", test.name);
            return status;
        }
    }
    vkDestroyInstance(instance, 0);
    return 0;
}
