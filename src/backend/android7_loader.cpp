// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0
// ELF lookup follows the approach in nihui/mesa-turnip-android-driver's
// mesa-implement-android_stub.patch. Its original lookup code is MIT licensed:
// Copyright (c) 2016 avs333
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "android7_loader.h"

#if defined(__ANDROID__)
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/system_properties.h>
#include <unistd.h>

namespace momoten_detail {

static uintptr_t loaded_address(const char* path)
{
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps)
        return 0;
    char line[1024];
    uintptr_t address = 0;
    while (fgets(line, sizeof(line), maps))
    {
        unsigned long begin = 0, end = 0, offset = 0;
        char mapped_path[768] = {};
        if (sscanf(line, "%lx-%lx %*s %lx %*s %*s %767s", &begin, &end, &offset, mapped_path) == 4
            && begin < end && offset == 0 && strcmp(mapped_path, path) == 0)
        {
            address = begin;
            break;
        }
    }
    fclose(maps);
    return address;
}

Android7LoadedElf::Android7LoadedElf(const char* path)
    : data(0), size(0), base(loaded_address(path))
{
    if (!base)
        return;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return;
    struct stat status;
    if (fstat(fd, &status) == 0 && status.st_size >= (off_t)sizeof(ElfW(Ehdr)))
    {
        size = status.st_size;
        void* mapping = mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping != MAP_FAILED)
            data = static_cast<const unsigned char*>(mapping);
    }
    close(fd);
}

Android7LoadedElf::~Android7LoadedElf()
{
    if (data)
        munmap(const_cast<unsigned char*>(data), size);
}

bool Android7LoadedElf::within(size_t offset, size_t length) const
{
    return offset <= size && length <= size - offset;
}

void* Android7LoadedElf::symbol(const char* name, unsigned type, size_t min_size) const
{
    if (!data)
        return 0;
    const ElfW(Ehdr)* elf = reinterpret_cast<const ElfW(Ehdr)*>(data);
    if (memcmp(elf->e_ident, ELFMAG, SELFMAG) != 0 || elf->e_type != ET_DYN
        || elf->e_ident[EI_DATA] != ELFDATA2LSB
        || elf->e_ident[EI_CLASS] != (sizeof(void*) == 8 ? ELFCLASS64 : ELFCLASS32)
        || elf->e_shentsize != sizeof(ElfW(Shdr))
        || elf->e_phentsize != sizeof(ElfW(Phdr))
        || elf->e_shoff % alignof(ElfW(Shdr)) || elf->e_phoff % alignof(ElfW(Phdr))
        || !within(elf->e_shoff, size_t(elf->e_shnum) * sizeof(ElfW(Shdr)))
        || !within(elf->e_phoff, size_t(elf->e_phnum) * sizeof(ElfW(Phdr))))
        return 0;
    const ElfW(Phdr)* phdr = reinterpret_cast<const ElfW(Phdr)*>(data + elf->e_phoff);
    uintptr_t bias = 0;
    bool have_bias = false;
    for (size_t i = 0; i < elf->e_phnum; i++)
    {
        if (phdr[i].p_type == PT_LOAD && phdr[i].p_offset == 0 && phdr[i].p_vaddr <= base)
        {
            bias = base - phdr[i].p_vaddr;
            have_bias = true;
            break;
        }
    }
    if (!have_bias)
        return 0;
    const ElfW(Shdr)* sections = reinterpret_cast<const ElfW(Shdr)*>(data + elf->e_shoff);
    for (size_t i = 0; i < elf->e_shnum; i++)
    {
        const ElfW(Shdr)& symbols = sections[i];
        if ((symbols.sh_type != SHT_DYNSYM && symbols.sh_type != SHT_SYMTAB)
            || symbols.sh_entsize != sizeof(ElfW(Sym)) || symbols.sh_link >= elf->e_shnum
            || symbols.sh_offset % alignof(ElfW(Sym))
            || !within(symbols.sh_offset, symbols.sh_size))
            continue;
        const ElfW(Shdr)& strings = sections[symbols.sh_link];
        if (strings.sh_type != SHT_STRTAB || !within(strings.sh_offset, strings.sh_size))
            continue;
        const ElfW(Sym)* entries = reinterpret_cast<const ElfW(Sym)*>(data + symbols.sh_offset);
        for (size_t j = 0; j < symbols.sh_size / sizeof(ElfW(Sym)); j++)
        {
            const ElfW(Sym)& entry = entries[j];
            if (entry.st_shndx == SHN_UNDEF || entry.st_shndx >= elf->e_shnum
                || ELF32_ST_TYPE(entry.st_info) != type || entry.st_name >= strings.sh_size)
                continue;
            const char* candidate = reinterpret_cast<const char*>(data + strings.sh_offset + entry.st_name);
            if (!memchr(candidate, 0, strings.sh_size - entry.st_name) || strcmp(candidate, name) != 0)
                continue;
            uintptr_t address = entry.st_value;
#if defined(__arm__)
            if (type == STT_FUNC)
                address &= ~uintptr_t(1); // Only for bounds checks; preserve the Thumb bit below.
#endif
            for (size_t k = 0; k < elf->e_phnum; k++)
            {
                const ElfW(Phdr)& segment = phdr[k];
                const unsigned flags = type == STT_FUNC ? PF_X : PF_R | PF_W;
                if (segment.p_type == PT_LOAD && (segment.p_flags & flags) == flags
                    && address >= segment.p_vaddr && address - segment.p_vaddr <= segment.p_memsz
                    && min_size <= segment.p_memsz - (address - segment.p_vaddr)
                    && entry.st_value <= UINTPTR_MAX - bias)
                    return reinterpret_cast<void*>(bias + entry.st_value);
            }
        }
    }
    return 0;
}

Android7Linker::Android7Linker()
    : do_dlopen(0), mutex(0), caller(0), get_error(0), supported(false)
{
    char sdk[PROP_VALUE_MAX] = {};
    __system_property_get("ro.build.version.sdk", sdk);
    const int api_level = atoi(sdk);
    supported = api_level == 24 || api_level == 25;
    if (!supported)
        return;
#if defined(__LP64__)
    Android7LoadedElf elf("/system/bin/linker64");
    caller = reinterpret_cast<void*>(loaded_address("/system/lib64/libart.so"));
#else
    Android7LoadedElf elf("/system/bin/linker");
    caller = reinterpret_cast<void*>(loaded_address("/system/lib/libart.so"));
#endif
    do_dlopen = reinterpret_cast<DoDlopen>(elf.symbol("__dl__Z9do_dlopenPKciPK17android_dlextinfoPv", STT_FUNC));
    mutex = reinterpret_cast<pthread_mutex_t*>(elf.symbol("__dl__ZL10g_dl_mutex", STT_OBJECT, sizeof(pthread_mutex_t)));
    get_error = reinterpret_cast<GetError>(elf.symbol("__dl__Z23linker_get_error_bufferv", STT_FUNC));
    // These point into the resident linker/libart, not Android7LoadedElf's temporary file mapping.
}

void* Android7Linker::load(const char* path, int flags) const
{
    if (!do_dlopen || !mutex || !caller)
    {
        log_android_opencl_load(path, "Android 7 loader", "entry, mutex or libart missing");
        return 0;
    }
    // Constructors may recursively enter the loader. Use bionic's recursive
    // mutex, and finish static initialization before calling do_dlopen.
    if (pthread_mutex_lock(mutex) != 0)
        return 0;
    char detail[768] = {};
    char* linker_error = get_error ? get_error() : 0;
    if (linker_error)
        linker_error[0] = 0;
    void* library = do_dlopen(path, flags, 0, caller);
    if (!library && linker_error)
        strncpy(detail, linker_error, sizeof(detail) - 1);
    pthread_mutex_unlock(mutex);
    const char* error = detail[0] ? detail : "no linker error detail";
    log_android_opencl_load(path, "Android 7 do_dlopen", library ? 0 : error);
    return library; // A real dlopen reference, released through the existing dlclose path.
}

void* open_android7_library(const char* path, int flags)
{
    static const Android7Linker linker;
    return linker.supported ? linker.load(path, flags) : 0;
}

} // namespace momoten_detail
#endif // __ANDROID__
