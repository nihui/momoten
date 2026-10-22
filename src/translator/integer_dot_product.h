// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef MOMOTEN_INTEGER_DOT_PRODUCT_H
#define MOMOTEN_INTEGER_DOT_PRODUCT_H

#include "momoten/spv_to_clc.h"

namespace momoten {

const char* integer_dot_product_extension(IntegerDotProductBackend backend);
// Shared by device probes and shader lowering so both validate the same
// signedness corrections and final saturation. KHR uses its builtins directly.
std::string integer_dot_product_helpers(IntegerDotProductBackend backend);

} // namespace momoten

#endif // MOMOTEN_INTEGER_DOT_PRODUCT_H
