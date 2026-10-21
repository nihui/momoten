// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

#include <unordered_set>

namespace momoten {

static bool is_opencl_reserved_name(const std::string& name)
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
            "kernel_enqueue_flags_t", "clk_profiling_info",
            "cl_mem_fence_flags", "memory_scope", "memory_order",
            // Enumerators share C's ordinary identifier namespace with
            // variables, functions and typedef names.
            "memory_scope_work_item", "memory_scope_sub_group", "memory_scope_work_group",
            "memory_scope_device", "memory_scope_all_svm_devices", "memory_scope_all_devices",
            "memory_order_relaxed", "memory_order_acquire", "memory_order_release",
            "memory_order_acq_rel", "memory_order_seq_cst",
            "atomic_int", "atomic_uint", "atomic_long", "atomic_ulong", "atomic_float", "atomic_double",
            "atomic_intptr_t", "atomic_uintptr_t", "atomic_size_t", "atomic_ptrdiff_t", "atomic_flag",
            "ATOMIC_FLAG_INIT", "ATOMIC_VAR_INIT", "MAXFLOAT", "HUGE_VAL", "HUGE_VALF", "INFINITY", "NAN",
            "NULL", "MAX_WORK_DIM",
            // Device enqueue and pipe builtins added by OpenCL C 2.0.
            "enqueue_kernel", "ndrange_1D", "ndrange_2D", "ndrange_3D",
            "retain_event", "release_event", "create_user_event", "is_valid_event",
            "set_user_event_status", "capture_event_profiling_info",
            "read_pipe", "write_pipe", "reserve_read_pipe", "reserve_write_pipe",
            "commit_read_pipe", "commit_write_pipe", "is_valid_reserve_id",
            "quad", "complex", "imaginary"};
        const char* bases[] = {"char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half", "float", "double", "bool", "quad", "complex", "imaginary"};
        const char* widths[] = {"2", "3", "4", "8", "16"};
        for (const char* base : bases)
            for (const char* width : widths)
                names.insert(std::string(base) + width);
        return names;
    }();

    if (keywords.count(name))
        return true;

    // OpenCL headers expose macro families and builtin functions in the
    // shader's namespace. Macros can also expand struct tags and members.
    static const char* prefixes[] = {
        "CL_", "CLK_", "cl_", "M_", "FP_", "FLT_", "DBL_", "HALF_", "CHAR_", "SCHAR_", "UCHAR_",
        "SHRT_", "USHRT_", "INT_", "UINT_", "LONG_", "ULONG_",
        "get_", "sub_group_", "work_group_", "atomic_", "atom_", "async_work_group_"};
    for (const char* prefix : prefixes)
        if (name.compare(0, std::char_traits<char>::length(prefix), prefix) == 0)
            return true;
    return false;
}

void CompilerOpenCL::replace_illegal_names()
{
    // Rewrite aliases before emission so declarations and references agree.
    // Keep SPIRV-Cross's language-independent sanitization and name caches.
    CompilerGLSL::replace_illegal_names();
    const auto rename = [&](uint32_t id) {
        const std::string name = get_name(id);
        if (is_opencl_reserved_name(name))
            set_name(id, "_" + name);
    };
    ir.for_each_typed_id<SPIRVariable>([&](uint32_t, const SPIRVariable& var) {
        if (!is_hidden_variable(var))
            rename(var.self);
    });
    ir.for_each_typed_id<SPIRFunction>([&](uint32_t, const SPIRFunction& func) {
        // Builtins vary by language version, extension and compiler. Give all
        // retained shader helpers internal names instead of enumerating them.
        // Keep the kernel ABI name and let the normal name cache resolve any
        // collisions with shader resources or other generated identifiers.
        if (func.self != ir.default_entry_point)
            set_name(func.self, "momo_function_" + std::to_string(func.self));
    });
    ir.for_each_typed_id<SPIRConstant>([&](uint32_t, const SPIRConstant& constant) {
        rename(constant.self);
    });
    ir.for_each_typed_id<SPIRConstantOp>([&](uint32_t, const SPIRConstantOp& constant) {
        rename(constant.self);
    });
    ir.for_each_typed_id<SPIRType>([&](uint32_t, SPIRType& type) {
        rename(type.self);
        type.member_name_cache.clear();
        for (uint32_t i = 0; i < type.member_types.size(); i++)
        {
            const std::string name = get_member_name(type.self, i);
            if (is_opencl_reserved_name(name))
                set_member_name(type.self, i, "_" + name);
            add_member_name(type, i);
            // Resource aliases become macros, which ignore C member scope.
            // Reserve finalized member names before allocating resource names,
            // including suffixes introduced by collisions such as pipe/_pipe.
            resource_names.insert(to_member_name(type, i));
        }
    });
}

} // namespace momoten
