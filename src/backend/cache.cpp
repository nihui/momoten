// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "cache.h"
#include "device.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <direct.h>
#endif
#include <sys/stat.h>

uint64_t fnv1a64(const void* data, size_t size, uint64_t hash)
{
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; i++)
    {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t fnv1a64_string(const std::string& value, uint64_t hash)
{
    hash = fnv1a64(value.data(), value.size(), hash);
    const unsigned char separator = 0;
    return fnv1a64(&separator, 1, hash);
}

struct ProgramBinaryHeader
{
    char magic[8];
    uint64_t first;
    uint64_t second;
    uint64_t binary_size;
};

bool ensure_cache_directory(const std::string& path)
{
    if (path.empty())
        return false;
#if defined(_WIN32)
    const int result = _mkdir(path.c_str());
#else
    const int result = mkdir(path.c_str(), 0755);
#endif
    if (result == 0)
        return true;

    struct stat status;
    return stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
}

ProgramCacheKey make_program_cache_key(
    cl_device_id device, const std::string& source)
{
    ProgramCacheKey key = {};
    key.first = 1469598103934665603ull;
    key.second = 1099511628211ull;

    const cl_device_info identity_fields[] = {
        CL_DEVICE_VENDOR, CL_DEVICE_NAME, CL_DEVICE_VERSION, CL_DRIVER_VERSION};
    for (size_t i = 0; i < sizeof(identity_fields) / sizeof(identity_fields[0]); i++)
    {
        const std::string value = opencl_device_string(device, identity_fields[i]);
        key.first = fnv1a64_string(value, key.first);
        key.second = fnv1a64_string(value, key.second);
    }
    key.first = fnv1a64_string(source, key.first);
    key.second = fnv1a64_string(source, key.second);
    const char* directory_value = getenv("MOMOTEN_CACHE");
    if (!directory_value || !directory_value[0])
        directory_value = getenv("NCNN_MOMOTEN_CACHE");
    if (!directory_value || !directory_value[0])
        return key;
    const std::string directory(directory_value);
    if (!ensure_cache_directory(directory))
        return key;

    std::ostringstream filename;
    filename << directory;
    if (directory[directory.size() - 1] != '/' && directory[directory.size() - 1] != '\\')
        filename << '/';
    filename << "momo-" << std::hex << std::setfill('0')
             << std::setw(16) << key.first << std::setw(16) << key.second << ".bin";
    key.path = filename.str();
    return key;
}

bool read_program_binary(const ProgramCacheKey& key, std::vector<unsigned char>& binary)
{
    if (key.path.empty())
        return false;
    std::ifstream stream(key.path.c_str(), std::ios::binary);
    ProgramBinaryHeader header = {};
    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header)))
        return false;
    static const char magic[8] = {'M', 'O', 'M', 'O', 'T', 'E', 'N', '1'};
    if (memcmp(header.magic, magic, sizeof(magic)) != 0 || header.first != key.first || header.second != key.second || header.binary_size == 0 || header.binary_size > static_cast<uint64_t>(SIZE_MAX) || header.binary_size > (1ull << 32))
        return false;
    binary.resize(static_cast<size_t>(header.binary_size));
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(binary.data()), binary.size()));
}

void write_program_binary(const ProgramCacheKey& key, const std::vector<unsigned char>& binary)
{
    if (key.path.empty() || binary.empty())
        return;
    ProgramBinaryHeader header = {};
    static const char magic[8] = {'M', 'O', 'M', 'O', 'T', 'E', 'N', '1'};
    memcpy(header.magic, magic, sizeof(magic));
    header.first = key.first;
    header.second = key.second;
    header.binary_size = binary.size();

    std::ostringstream temporary_name;
    temporary_name << key.path << ".tmp-" << std::hex
                   << std::hash<std::thread::id>()(std::this_thread::get_id()) << '-'
                   << std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string temporary = temporary_name.str();
    std::ofstream stream(temporary.c_str(), std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char*>(&header), sizeof(header)) || !stream.write(reinterpret_cast<const char*>(binary.data()), binary.size()))
    {
        stream.close();
        std::remove(temporary.c_str());
        return;
    }
    stream.close();
#if defined(_WIN32)
    // C rename cannot replace an existing file on Windows.
    std::remove(key.path.c_str());
#endif
    if (std::rename(temporary.c_str(), key.path.c_str()) != 0)
        std::remove(temporary.c_str());
}
