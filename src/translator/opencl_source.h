// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_OPENCL_SOURCE_H
#define MOMOTEN_OPENCL_SOURCE_H

#include <string>

namespace momoten {

// Convert constructors in CompilerOpenCL's generated source to OpenCL C.
// This pass accepts the emitter's syntax, not arbitrary GLSL programs.
void lower_opencl_constructors(std::string& source);

} // namespace momoten

#endif // MOMOTEN_OPENCL_SOURCE_H
