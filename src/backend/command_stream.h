// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_COMMAND_STREAM_H
#define MOMOTEN_BACKEND_COMMAND_STREAM_H

#include "../vulkan_internal.h"

#include <map>
#include <memory>
#include <vector>

struct Buffer;
struct Pipeline;

struct DescriptorValue
{
    std::shared_ptr<Buffer> buffer;
    VkDeviceSize offset;
    VkDeviceSize range;

    DescriptorValue()
        : offset(0), range(VK_WHOLE_SIZE)
    {
    }
};

struct RecordedMemoryBarrier
{
    VkAccessFlags src_access_mask;
    VkAccessFlags dst_access_mask;
};

struct RecordedBufferBarrier
{
    VkAccessFlags src_access_mask;
    VkAccessFlags dst_access_mask;
    std::shared_ptr<Buffer> buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
};

struct RecordedCommand
{
    enum Type
    {
        CopyBuffer,
        Dispatch,
        Barrier
    } type;

    std::shared_ptr<Buffer> src_buffer;
    std::shared_ptr<Buffer> dst_buffer;
    std::vector<VkBufferCopy> copy_regions;
    VkPipelineStageFlags src_stage_mask;
    VkPipelineStageFlags dst_stage_mask;
    std::vector<RecordedMemoryBarrier> memory_barriers;
    std::vector<RecordedBufferBarrier> buffer_barriers;
    std::shared_ptr<Pipeline> pipeline;
    std::map<uint32_t, DescriptorValue> descriptors;
    std::vector<unsigned char> push_constants;
    uint32_t group_count[3];

    RecordedCommand()
        : type(CopyBuffer), src_stage_mask(0), dst_stage_mask(0)
    {
        group_count[0] = group_count[1] = group_count[2] = 0;
    }
};

#endif // MOMOTEN_BACKEND_COMMAND_STREAM_H
