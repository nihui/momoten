// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

// Compile the actual Android helper on Linux with only its three Android
// declarations stubbed. No Android linker, driver installation, or GPU is used.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <android/dlext.h>
#include <android/log.h>
#include <sys/system_properties.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "android7_opencl_loader_test:%d: %s\n", __LINE__, #condition); \
    abort(); } } while (0)

static const char* sdk_version = "0";
static std::string log_text;
static pthread_mutex_t* active_mutex = 0;

// Check ownership from another thread so a recursive mutex cannot disguise a
// missing lock, or a leaked recursion level after the helper returns.
static void* try_lock(void* result)
{
    *static_cast<int*>(result) = pthread_mutex_trylock(active_mutex);
    if (*static_cast<int*>(result) == 0)
        CHECK(pthread_mutex_unlock(active_mutex) == 0);
    return 0;
}
static void check_locked(bool locked)
{
    pthread_t thread;
    int result = -1;
    CHECK(pthread_create(&thread, 0, try_lock, &result) == 0);
    CHECK(pthread_join(thread, 0) == 0);
    CHECK(result == (locked ? EBUSY : 0));
}

extern "C" int __system_property_get(const char* name, char* value)
{
    CHECK(strcmp(name, "ro.build.version.sdk") == 0);
    strcpy(value, sdk_version);
    return strlen(value);
}
extern "C" int __android_log_print(int, const char* tag, const char* format, ...)
{
    CHECK(strcmp(tag, "momoten") == 0);
    if (active_mutex) check_locked(false); // Logging must happen after unlock.
    char buffer[2048];
    va_list args;
    va_start(args, format);
    const int length = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    log_text += buffer;
    return length;
}

#define __ANDROID__ 1
#include "../../src/backend/android7_loader.h"
#undef __ANDROID__

// A real file-backed mapping gives /proc/self/maps an address independent of
// LoadedElf's temporary parsing map. Nonzero virtual addresses test load bias.
struct ElfFixture
{
    char path[64];
    int fd;
    size_t page;
    std::vector<unsigned char> bytes;
    unsigned char* mapping;

    ElfFixture() : fd(-1), page(sysconf(_SC_PAGESIZE)), bytes(page * 2), mapping(0)
    {
        strcpy(path, "/tmp/momoten-android7-XXXXXX");
        fd = mkstemp(path);
        CHECK(fd >= 0 && page >= 4096);
        ElfW(Ehdr)& e = header();
        memcpy(e.e_ident, ELFMAG, SELFMAG);
        e.e_ident[EI_CLASS] = sizeof(void*) == 8 ? ELFCLASS64 : ELFCLASS32;
        e.e_ident[EI_DATA] = ELFDATA2LSB;
        e.e_ident[EI_VERSION] = EV_CURRENT;
        e.e_type = ET_DYN;
        e.e_version = EV_CURRENT;
        e.e_ehsize = sizeof(e);
        e.e_phoff = sizeof(e);
        e.e_phentsize = sizeof(ElfW(Phdr));
        e.e_phnum = 2;
        e.e_shoff = 0x600;
        e.e_shentsize = sizeof(ElfW(Shdr));
        e.e_shnum = 3;
        for (unsigned i = 0; i < 2; i++)
        {
            ElfW(Phdr)& p = segments()[i];
            p.p_type = PT_LOAD;
            p.p_flags = i == 0 ? PF_R | PF_X : PF_R | PF_W;
            p.p_offset = i * page;
            p.p_vaddr = 0x4000 + i * page;
            p.p_filesz = p.p_memsz = page;
            p.p_align = page;
        }
        ElfW(Shdr)& symbols = sections()[1];
        symbols.sh_type = SHT_SYMTAB;
        symbols.sh_offset = 0x200;
        symbols.sh_size = 3 * sizeof(ElfW(Sym));
        symbols.sh_entsize = sizeof(ElfW(Sym));
        symbols.sh_link = 2;
        ElfW(Shdr)& strings = sections()[2];
        strings.sh_type = SHT_STRTAB;
        strings.sh_offset = 0x400;
        const char names[] = "\0function\0object";
        strings.sh_size = sizeof(names);
        memcpy(&bytes[strings.sh_offset], names, sizeof(names));
        entries()[1].st_name = 1;
        entries()[1].st_info = ELF32_ST_INFO(STB_LOCAL, STT_FUNC);
        entries()[1].st_shndx = 1;
        entries()[1].st_value = 0x4900;
        entries()[1].st_size = 16;
        entries()[2].st_name = 10;
        entries()[2].st_info = ELF32_ST_INFO(STB_LOCAL, STT_OBJECT);
        entries()[2].st_shndx = 1;
        entries()[2].st_value = 0x4000 + page + 0x100;
        entries()[2].st_size = sizeof(pthread_mutex_t);
        save();
        void* address = mmap(0, bytes.size(), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        CHECK(address != MAP_FAILED);
        mapping = static_cast<unsigned char*>(address);
        CHECK(mprotect(mapping, page, PROT_READ | PROT_EXEC) == 0);
    }
    ~ElfFixture()
    {
        CHECK(munmap(mapping, page * 2) == 0);
        CHECK(close(fd) == 0);
        CHECK(unlink(path) == 0);
    }
    ElfW(Ehdr)& header() { return *reinterpret_cast<ElfW(Ehdr)*>(&bytes[0]); }
    ElfW(Phdr)* segments() { return reinterpret_cast<ElfW(Phdr)*>(&bytes[sizeof(ElfW(Ehdr))]); }
    ElfW(Shdr)* sections() { return reinterpret_cast<ElfW(Shdr)*>(&bytes[0x600]); }
    ElfW(Sym)* entries() { return reinterpret_cast<ElfW(Sym)*>(&bytes[0x200]); }
    void save()
    {
        CHECK(ftruncate(fd, bytes.size()) == 0);
        CHECK(pwrite(fd, &bytes[0], bytes.size(), 0) == static_cast<ssize_t>(bytes.size()));
    }
};

static void test_elf()
{
    using momoten_android7::LoadedElf;
    CHECK(!LoadedElf("/no/such/momoten-linker").symbol("function", STT_FUNC));
    {
        ElfFixture f;
        void* function;
        unsigned char* object;
        {
            LoadedElf elf(f.path);
            function = elf.symbol("function", STT_FUNC);
            object = static_cast<unsigned char*>(elf.symbol("object", STT_OBJECT, sizeof(pthread_mutex_t)));
            CHECK(function == f.mapping + 0x900);
            CHECK(object == f.mapping + f.page + 0x100);
            CHECK(!elf.symbol("missing", STT_FUNC));
            CHECK(!elf.symbol("function", STT_OBJECT));
            CHECK(!elf.symbol("object", STT_FUNC));
            CHECK(!elf.symbol("object", STT_OBJECT, f.page));
        }
        // The addresses survive destruction of the read-only parsing mapping.
        CHECK(function == f.mapping + 0x900);
        *object = 0x5a;
        CHECK(f.mapping[f.page + 0x100] == 0x5a);
        f.sections()[1].sh_type = SHT_DYNSYM;
        f.save();
        CHECK(LoadedElf(f.path).symbol("function", STT_FUNC) == function);
    }
    {
        ElfFixture f;
        f.header().e_ident[EI_MAG0] = 0;
        f.save();
        CHECK(!LoadedElf(f.path).symbol("function", STT_FUNC));
    }
    {
        ElfFixture f;
        f.header().e_shoff = static_cast<ElfW(Off)>(-1);
        f.save();
        CHECK(!LoadedElf(f.path).symbol("function", STT_FUNC));
    }
    {
        ElfFixture f;
        f.sections()[1].sh_size = f.bytes.size();
        f.save();
        CHECK(!LoadedElf(f.path).symbol("function", STT_FUNC));
    }
    {
        ElfFixture f;
        f.sections()[2].sh_offset = f.bytes.size() - 1;
        f.save();
        CHECK(!LoadedElf(f.path).symbol("function", STT_FUNC));
    }
    {
        ElfFixture f;
        f.sections()[2].sh_size = 9; // "function" has no NUL within this table.
        f.save();
        CHECK(!LoadedElf(f.path).symbol("function", STT_FUNC));
    }
    {
        ElfFixture f;
        f.entries()[1].st_shndx = SHN_UNDEF;
        f.entries()[2].st_name = 999;
        f.save();
        LoadedElf elf(f.path);
        CHECK(!elf.symbol("function", STT_FUNC));
        CHECK(!elf.symbol("object", STT_OBJECT));
    }
    {
        ElfFixture f;
        f.entries()[1].st_value = 0x4000 + f.page; // Non-executable data page.
        f.entries()[2].st_value = 0x4900; // Non-writable text page.
        f.save();
        LoadedElf elf(f.path);
        CHECK(!elf.symbol("function", STT_FUNC));
        CHECK(!elf.symbol("object", STT_OBJECT));
    }
    {
        ElfFixture f;
        f.entries()[2].st_value = 0x4000 + 2 * f.page - 1;
        f.save();
        LoadedElf elf(f.path);
        CHECK(elf.symbol("object", STT_OBJECT, 1) == f.mapping + 2 * f.page - 1);
        CHECK(!elf.symbol("object", STT_OBJECT, sizeof(pthread_mutex_t)));
    }
    {
        ElfFixture f;
        CHECK(ftruncate(f.fd, sizeof(ElfW(Ehdr)) - 1) == 0);
        CHECK(!LoadedElf(f.path).symbol("function", STT_FUNC));
    }
}

static int open_calls = 0, error_calls = 0, caller_token, library_token;
static bool open_success = true;
static bool expect_cleared_error = false;
static bool null_error = false;
static const char* open_error = "missing vendor dependency";
static char error_buffer[1024];
static void* fake_open(const char* path, int flags, const android_dlextinfo* ext, void* caller)
{
    CHECK(strcmp(path, "/vendor/lib/libOpenCL.so") == 0);
    CHECK(flags == (RTLD_LOCAL | RTLD_NOW) && !ext && caller == &caller_token);
    if (expect_cleared_error) CHECK(error_buffer[0] == 0);
    check_locked(true);
    // Android's loader mutex must allow constructors to reenter it.
    CHECK(pthread_mutex_trylock(active_mutex) == 0);
    CHECK(pthread_mutex_unlock(active_mutex) == 0);
    open_calls++;
    if (open_error) strcpy(error_buffer, open_error);
    return open_success ? &library_token : 0;
}
static char* fake_error()
{
    check_locked(true);
    error_calls++;
    return null_error ? 0 : error_buffer;
}

static void test_linker()
{
    using momoten_android7::Linker;
    const char* unsupported[] = {"", "0", "23", "26", "35"};
    for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++)
    {
        sdk_version = unsupported[i];
        Linker absent;
        CHECK(!absent.supported && !absent.open && !absent.mutex && !absent.caller);
    }
    sdk_version = "0";
    Linker linker;
    const char* path = "/vendor/lib/libOpenCL.so";
    const int flags = RTLD_LOCAL | RTLD_NOW;
    CHECK(!momoten_android7::open_library(path, flags, false));
    const char* supported[] = {"24", "25"};
    for (size_t i = 0; i < sizeof(supported) / sizeof(supported[0]); i++)
    {
        sdk_version = supported[i];
        Linker absent;
        CHECK(absent.supported);
        CHECK(!absent.load(path, flags, false)); // No Android linker on this host.
    }
    sdk_version = "0";
    CHECK(!linker.load(path, flags, false));
    linker.open = fake_open;
    CHECK(!linker.load(path, flags, false));
    pthread_mutexattr_t attributes;
    pthread_mutex_t mutex;
    CHECK(pthread_mutexattr_init(&attributes) == 0);
    CHECK(pthread_mutexattr_settype(&attributes, PTHREAD_MUTEX_RECURSIVE) == 0);
    CHECK(pthread_mutex_init(&mutex, &attributes) == 0);
    CHECK(pthread_mutexattr_destroy(&attributes) == 0);
    active_mutex = &mutex;
    linker.mutex = &mutex;
    CHECK(!linker.load(path, flags, false));
    CHECK(open_calls == 0);
    linker.caller = &caller_token;
    linker.open = 0;
    CHECK(!linker.load(path, flags, false));
    linker.open = fake_open;
    linker.mutex = 0;
    CHECK(!linker.load(path, flags, false));
    linker.mutex = &mutex;
    for (int attempt = 0; attempt < 2; attempt++)
    {
        CHECK(linker.load(path, flags, false) == &library_token);
        check_locked(false);
    }
    CHECK(open_calls == 2 && error_calls == 0 && log_text.empty());
    open_success = false;
    CHECK(!linker.load(path, flags, true));
    CHECK(log_text.find("no linker error detail") != std::string::npos);
    log_text.clear();
    linker.error = fake_error;
    expect_cleared_error = true;
    CHECK(!linker.load(path, flags, true));
    CHECK(error_calls == 1 && log_text.find("missing vendor dependency") != std::string::npos);
    check_locked(false);
    log_text.clear();
    open_error = 0; // A failing call that provides no new error must not reuse one.
    strcpy(error_buffer, "stale error from a previous call");
    CHECK(!linker.load(path, flags, true));
    CHECK(error_calls == 2 && log_text.find("no linker error detail") != std::string::npos);
    CHECK(log_text.find("stale error") == std::string::npos);
    log_text.clear();
    CHECK(!linker.load(path, flags, false));
    CHECK(error_calls == 3 && log_text.empty());
    check_locked(false);
    null_error = true;
    expect_cleared_error = false;
    CHECK(!linker.load(path, flags, true));
    CHECK(error_calls == 4 && log_text.find("no linker error detail") != std::string::npos);
    log_text.clear();
    null_error = false;
    expect_cleared_error = true;
    open_success = true;
    CHECK(linker.load(path, flags, true) == &library_token);
    CHECK(log_text.find("loaded") != std::string::npos);
    CHECK(error_calls == 5);
    check_locked(false);
    CHECK(pthread_mutex_destroy(&mutex) == 0);
    active_mutex = 0;
}

int main()
{
    test_elf();
    test_linker();
    return 0;
}
