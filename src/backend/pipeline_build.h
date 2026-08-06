// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_PIPELINE_BUILD_H
#define MOMOTEN_BACKEND_PIPELINE_BUILD_H

#include "../vulkan_internal.h"

namespace momoten_detail {

VkResult build_compute_pipelines(
    VkDevice device, VkPipelineCache pipeline_cache, uint32_t count,
    const VkComputePipelineCreateInfo* infos,
    const VkAllocationCallbacks* allocator, VkPipeline* pipelines);

} // namespace momoten_detail

#endif // MOMOTEN_BACKEND_PIPELINE_BUILD_H
