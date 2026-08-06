// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_MEMORY_SYNC_H
#define MOMOTEN_BACKEND_MEMORY_SYNC_H

#include "objects.h"

struct ReplayState;

void prepare_mapped_memory_for_submit(VkDevice device);
void mark_host_range(const std::shared_ptr<DeviceMemory>& memory,
                     VkDeviceSize offset, VkDeviceSize size);
cl_int synchronize_memory_to_host(VkDevice device,
                                  const std::shared_ptr<DeviceMemory>& memory,
                                  VkDeviceSize offset, VkDeviceSize size);

cl_int upload_host_ranges(ReplayState& state,
                          const std::shared_ptr<DeviceMemory>& memory,
                          size_t offset, size_t size);
void mark_device_range(ReplayState& state,
                       const std::shared_ptr<DeviceMemory>& memory,
                       VkDeviceSize offset, VkDeviceSize size);
cl_int finish_mapped_memory_readbacks(ReplayState& state);

#endif // MOMOTEN_BACKEND_MEMORY_SYNC_H
