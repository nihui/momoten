// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_ANDROID7_LOADER_H
#define MOMOTEN_ANDROID7_LOADER_H

#if defined(__ANDROID__)
#include <android/dlext.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

namespace momoten_detail {

// Shared Android loader diagnostics. A null error denotes a successful load.
void log_android_opencl_load(const char* path, const char* operation, const char* error);
void* open_android7_library(const char* path, int flags);

class Android7LoadedElf
{
public:
    explicit Android7LoadedElf(const char* path);
    ~Android7LoadedElf();
    void* symbol(const char* name, unsigned type, size_t min_size = 1) const;

private:
    bool within(size_t offset, size_t length) const;
    Android7LoadedElf(const Android7LoadedElf&) = delete;
    Android7LoadedElf& operator=(const Android7LoadedElf&) = delete;
    const unsigned char* data;
    size_t size;
    uintptr_t base;
};

struct Android7Linker
{
    typedef void* (*DoDlopen)(const char*, int, const android_dlextinfo*, void*);
    typedef char* (*GetError)();
    DoDlopen do_dlopen;
    pthread_mutex_t* mutex;
    void* caller;
    GetError get_error;
    bool supported;

    Android7Linker();
    void* load(const char* path, int flags) const;
};

} // namespace momoten_detail
#endif // __ANDROID__
#endif // MOMOTEN_ANDROID7_LOADER_H
