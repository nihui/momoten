// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_BACKEND_CACHE_H
#define MOMOTEN_BACKEND_CACHE_H

#include "opencl_loader.h"

#include <stdint.h>
#include <string>
#include <vector>

struct ProgramCacheKey
{
    uint64_t first;
    uint64_t second;
    std::string path;
};

uint64_t fnv1a64(const void* data, size_t size, uint64_t hash);
uint64_t fnv1a64_string(const std::string& value, uint64_t hash);
bool ensure_cache_directory(const std::string& path);
ProgramCacheKey make_program_cache_key(
    cl_device_id device, const std::string& source);
bool read_program_binary(const ProgramCacheKey& key, std::vector<unsigned char>& binary);
void write_program_binary(const ProgramCacheKey& key, const std::vector<unsigned char>& binary);

#endif // MOMOTEN_BACKEND_CACHE_H
