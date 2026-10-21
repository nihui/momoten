// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <unordered_set>

namespace momoten {

void CompilerOpenCL::replace_illegal_names()
{
    // Use the union of OpenCL C 1.x, 2.0 and 3.0 names. Translation is shared
    // by all compiler profiles, including those without optional features.
    static const std::unordered_set<std::string> keywords = []() {
        std::unordered_set<std::string> names = {
            // C99 keywords in addition to the inherited GLSL name policy.
            "auto", "char", "register", "signed", "_Bool", "_Complex", "_Imaginary",
            // Address spaces, kernel/access qualifiers and OpenCL keywords.
            "__global", "global", "__local", "local", "__constant", "constant",
            "__private", "private", "__generic", "generic", "__kernel", "kernel",
            "__read_only", "read_only", "__write_only", "write_only",
            "__read_write", "read_write", "pipe", "uniform",
            // Scalar, opaque, atomic and reserved data types.
            "uchar", "ushort", "ulong", "size_t", "ptrdiff_t", "intptr_t", "uintptr_t",
            "image1d_t", "image1d_buffer_t", "image1d_array_t", "image2d_t", "image3d_t",
            "image2d_array_t", "image2d_depth_t", "image2d_array_depth_t",
            "image2d_msaa_t", "image2d_array_msaa_t", "image2d_msaa_depth_t", "image2d_array_msaa_depth_t",
            "sampler_t", "event_t", "queue_t", "ndrange_t", "clk_event_t", "reserve_id_t",
            "cl_mem_fence_flags", "memory_scope", "memory_order",
            "atomic_int", "atomic_uint", "atomic_long", "atomic_ulong", "atomic_float", "atomic_double",
            "atomic_intptr_t", "atomic_uintptr_t", "atomic_size_t", "atomic_ptrdiff_t", "atomic_flag",
            "quad", "complex", "imaginary"};
        const char* bases[] = {"char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half", "float", "double", "bool", "quad", "complex", "imaginary"};
        const char* widths[] = {"2", "3", "4", "8", "16"};
        for (const char* base : bases)
            for (const char* width : widths)
                names.insert(std::string(base) + width);
        return names;
    }();

    // Rename SPIR-V aliases before emission so declarations and references
    // use the same names. SPIRV-Cross's normal name caches resolve collisions
    // with existing identifiers such as both pipe and _pipe in one scope.
    CompilerGLSL::replace_illegal_names();
    CompilerGLSL::replace_illegal_names(keywords);

    // The inherited helper covers variables, functions, types and members,
    // but constant arrays and specialization expressions also have aliases.
    ir.for_each_typed_id<SPIRConstant>([&](uint32_t id, const SPIRConstant&) {
        if (keywords.count(get_name(id)))
            set_name(id, "_" + get_name(id));
    });
    ir.for_each_typed_id<SPIRConstantOp>([&](uint32_t id, const SPIRConstantOp&) {
        if (keywords.count(get_name(id)))
            set_name(id, "_" + get_name(id));
    });
}

} // namespace momoten
