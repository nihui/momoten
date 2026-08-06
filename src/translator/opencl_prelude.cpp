// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_internal.h"

namespace momoten {

void CompilerOpenCL::emit_constructor_macros(const char* base)
{
    statement("#define ", base, "(...) ((", base, ")(__VA_ARGS__))");
    statement("#define ", base, "2(...) ((", base, "2)(__VA_ARGS__))");
    statement("#define ", base, "3(...) ((", base, "3)(__VA_ARGS__))");
    statement("#define ", base, "4(...) ((", base, "4)(__VA_ARGS__))");
    statement("#define ", base, "8(...) ((", base, "8)(__VA_ARGS__))");
    statement("#define ", base, "16(...) ((", base, "16)(__VA_ARGS__))");
}

} // namespace momoten
