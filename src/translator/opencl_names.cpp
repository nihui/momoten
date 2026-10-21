// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

namespace momoten {

void CompilerOpenCL::replace_illegal_names()
{
    // OpName and OpMemberName are debug information, not OpenCL identifiers.
    // Allocate shader symbols by SPIR-V ID in a namespace separate from the
    // backend's helpers, ABI arguments and member names. This also covers named
    // SSA results and unnamed IDs, without a compiler/extension keyword list.
    // Resource reflection has already retained the original buffer names.
    for (uint32_t id = 1; id < ir.ids.size(); id++)
    {
        if (id == ir.default_entry_point || has_decoration(id, DecorationBuiltIn))
            continue;
        set_name(id, "momo_spv_" + std::to_string(id));
    }

    ir.for_each_typed_id<SPIRType>([&](uint32_t, SPIRType& type) {
        if (is_builtin_type(type))
            return;
        type.member_name_cache.clear();
        for (uint32_t i = 0; i < type.member_types.size(); i++)
        {
            // Resource aliases are macros, so member names must use a disjoint
            // prefix even though C gives every struct its own member scope.
            set_member_name(type.self, i, "momo_member_" + std::to_string(i));
        }
    });
}

} // namespace momoten
