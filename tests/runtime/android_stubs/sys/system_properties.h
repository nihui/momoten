// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0
#ifndef MOMOTEN_TEST_SYSTEM_PROPERTIES_H
#define MOMOTEN_TEST_SYSTEM_PROPERTIES_H
#define PROP_VALUE_MAX 92
extern "C" int __system_property_get(const char* name, char* value);
#endif
