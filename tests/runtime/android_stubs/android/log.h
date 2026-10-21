// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0
#ifndef MOMOTEN_TEST_ANDROID_LOG_H
#define MOMOTEN_TEST_ANDROID_LOG_H
#include <stdarg.h>
#define ANDROID_LOG_DEBUG 3
#define ANDROID_LOG_ERROR 6
extern "C" int __android_log_vprint(int priority, const char* tag, const char* format, va_list args);
#endif
